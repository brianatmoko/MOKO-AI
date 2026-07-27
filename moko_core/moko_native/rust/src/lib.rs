// =============================================================================
// MOKO NATIVE ACCELERATION CORE (Rust tier)
// -----------------------------------------------------------------------------
// Implementasi Rust dari jalur panas Anchor-RAG dengan C ABI yang IDENTIK
// dengan tier C++ (moko_native.cpp), sehingga loader ctypes yang sama dapat
// memuat salah satu library tanpa perubahan.
//
// Rust dipilih sebagai lapisan "lebih kuat" (memory-safe + zero-cost abstraction)
// bila C++ dirasa masih kurang. Kontrak paritas dengan Python dipertahankan:
//   - tokenize: lowercase ASCII, ambil deret [a-zA-Z_] panjang >= 2.
//   - retrieve: score = |anchors ∩ focus|; simpan bila >= 1; urut (score desc,
//     index asc); ambil top-limit.
//
// Pemisah serialisasi harus sama dengan tier C++ & native_accel.py:
//   snippet dipisah 0x1e (Record Separator); anchor/token dipisah '\n'.
// =============================================================================

use std::collections::HashSet;
use std::os::raw::{c_char, c_int, c_void};
use std::slice;

const SNIPPET_SEP: u8 = 0x1e; // Record Separator antar-snippet
const ANCHOR_SEP: u8 = b'\n'; // pemisah antar-anchor / antar-token

#[inline]
fn is_word_byte(c: u8) -> bool {
    (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || c == b'_'
}

#[inline]
fn ascii_lower(c: u8) -> u8 {
    if c >= b'A' && c <= b'Z' {
        c - b'A' + b'a'
    } else {
        c
    }
}

/// Tokenisasi byte: kumpulkan deret [a-zA-Z_] panjang >= 2, lowercase ASCII.
fn tokenize_bytes(data: &[u8]) -> Vec<Vec<u8>> {
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for &c in data {
        if is_word_byte(c) {
            current.push(ascii_lower(c));
        } else {
            if current.len() >= 2 {
                tokens.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.len() >= 2 {
        tokens.push(current);
    }
    tokens
}

/// Pecah slice byte berdasarkan pemisah tunggal, membuang bagian kosong.
fn split_nonempty(data: &[u8], sep: u8) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    for part in data.split(|&b| b == sep) {
        if !part.is_empty() {
            out.push(part.to_vec());
        }
    }
    out
}

/// Indeks anchor: satu himpunan anchor per snippet.
struct AnchorIndex {
    snippets: Vec<HashSet<Vec<u8>>>,
}

/// Skoring + peringkat terhadap himpunan fokus. Kembalikan (score, index)
/// terurut menurun skor; seri -> index menaik (paritas Python (score,-index)).
fn rank_focus(index: &AnchorIndex, focus_set: &HashSet<Vec<u8>>) -> Vec<(i32, i32)> {
    let mut scored: Vec<(i32, i32)> = Vec::with_capacity(index.snippets.len());
    for (i, anchors) in index.snippets.iter().enumerate() {
        let mut score: i32 = 0;
        // Iterasi himpunan yang lebih kecil demi efisiensi.
        if focus_set.len() < anchors.len() {
            for t in focus_set {
                if anchors.contains(t) {
                    score += 1;
                }
            }
        } else {
            for a in anchors {
                if focus_set.contains(a) {
                    score += 1;
                }
            }
        }
        if score >= 1 {
            scored.push((score, i as i32));
        }
    }
    // `scored` sudah dalam urutan index menaik; sort_by stabil => seri
    // mempertahankan index terkecil dulu.
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
}

/// Tulis top-`limit` hasil (index, score) ke buffer keluaran. Return jumlah.
unsafe fn write_ranked(
    scored: &[(i32, i32)],
    limit: c_int,
    out_idx: *mut c_int,
    out_score: *mut c_int,
) -> c_int {
    let n = std::cmp::min(scored.len(), limit as usize);
    let out_idx_slice = slice::from_raw_parts_mut(out_idx, n);
    let out_score_slice = slice::from_raw_parts_mut(out_score, n);
    for i in 0..n {
        out_idx_slice[i] = scored[i].1;
        out_score_slice[i] = scored[i].0;
    }
    n as c_int
}

/// Panjang C-string (mencari NUL) secara aman.
unsafe fn cstr_len(ptr: *const c_char) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let mut n = 0usize;
    while *ptr.add(n) != 0 {
        n += 1;
    }
    n
}

#[no_mangle]
pub extern "C" fn moko_native_backend() -> *const c_char {
    // Literal statik dengan NUL terminator.
    b"rust\0".as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn moko_native_abi_version() -> c_int {
    2
}

/// Tokenisasi. Tulis token yang digabung '\n' (NUL-terminated) ke out_buf.
/// Return: jumlah byte yang ditulis (tanpa NUL), atau -1 bila out_cap kurang.
#[no_mangle]
pub extern "C" fn moko_tokenize(text: *const c_char, out_buf: *mut c_char, out_cap: c_int) -> c_int {
    if out_buf.is_null() || out_cap <= 0 {
        return -1;
    }
    let text_bytes: &[u8] = unsafe {
        let len = cstr_len(text);
        if len == 0 {
            &[]
        } else {
            slice::from_raw_parts(text as *const u8, len)
        }
    };
    let tokens = tokenize_bytes(text_bytes);
    let mut joined: Vec<u8> = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        if i != 0 {
            joined.push(ANCHOR_SEP);
        }
        joined.extend_from_slice(tok);
    }
    if (joined.len() as i64) + 1 > out_cap as i64 {
        return -1;
    }
    unsafe {
        let dst = out_buf as *mut u8;
        std::ptr::copy_nonoverlapping(joined.as_ptr(), dst, joined.len());
        *dst.add(joined.len()) = 0; // NUL terminator
    }
    joined.len() as c_int
}

/// Bangun indeks anchor dari korpus serial (snippet dipisah 0x1e, anchor '\n').
#[no_mangle]
pub extern "C" fn moko_index_build(corpus: *const c_char, len: c_int) -> *mut c_void {
    let mut index = Box::new(AnchorIndex {
        snippets: Vec::new(),
    });
    if !corpus.is_null() && len > 0 {
        let data: &[u8] = unsafe { slice::from_raw_parts(corpus as *const u8, len as usize) };
        // Pertahankan boundary snippet apa adanya (termasuk snippet tanpa anchor).
        for snip in data.split(|&b| b == SNIPPET_SEP) {
            let mut anchors: HashSet<Vec<u8>> = HashSet::new();
            for a in split_nonempty(snip, ANCHOR_SEP) {
                anchors.insert(a);
            }
            index.snippets.push(anchors);
        }
    }
    Box::into_raw(index) as *mut c_void
}

/// Kueri top-k. focus = token '\n'-joined. Tulis index & skor terurut ke
/// out_idx/out_score (kapasitas >= limit). Return jumlah hasil (<= limit).
#[no_mangle]
pub extern "C" fn moko_index_query(
    handle: *mut c_void,
    focus: *const c_char,
    focus_len: c_int,
    limit: c_int,
    out_idx: *mut c_int,
    out_score: *mut c_int,
) -> c_int {
    if handle.is_null() || out_idx.is_null() || out_score.is_null() || limit <= 0 {
        return 0;
    }
    let index: &AnchorIndex = unsafe { &*(handle as *const AnchorIndex) };

    let mut focus_set: HashSet<Vec<u8>> = HashSet::new();
    if !focus.is_null() && focus_len > 0 {
        let fdata: &[u8] = unsafe { slice::from_raw_parts(focus as *const u8, focus_len as usize) };
        for t in split_nonempty(fdata, ANCHOR_SEP) {
            focus_set.insert(t);
        }
    }

    let scored = rank_focus(index, &focus_set);
    unsafe { write_ranked(&scored, limit, out_idx, out_score) }
}

/// Kueri gabungan: tokenisasi teks mentah + skoring + peringkat top-k dalam SATU
/// panggilan (menghindari materialisasi daftar token di sisi Python). Jalur panas
/// end-to-end untuk navigasi konteks panjang (gaya Kimi).
/// Setara Python: kb.retrieve(set(tokenize(text)), limit=limit).
#[no_mangle]
pub extern "C" fn moko_index_query_text(
    handle: *mut c_void,
    text: *const c_char,
    limit: c_int,
    out_idx: *mut c_int,
    out_score: *mut c_int,
) -> c_int {
    if handle.is_null() || out_idx.is_null() || out_score.is_null() || limit <= 0 {
        return 0;
    }
    let index: &AnchorIndex = unsafe { &*(handle as *const AnchorIndex) };

    let text_bytes: &[u8] = unsafe {
        let len = cstr_len(text);
        if len == 0 {
            &[]
        } else {
            slice::from_raw_parts(text as *const u8, len)
        }
    };
    let mut focus_set: HashSet<Vec<u8>> = HashSet::new();
    for t in tokenize_bytes(text_bytes) {
        focus_set.insert(t);
    }

    let scored = rank_focus(index, &focus_set);
    unsafe { write_ranked(&scored, limit, out_idx, out_score) }
}

/// Bebaskan indeks yang dibuat moko_index_build.
#[no_mangle]
pub extern "C" fn moko_index_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut AnchorIndex));
        }
    }
}
