// =============================================================================
// MOKO NATIVE ACCELERATION CORE (C++ tier)
// -----------------------------------------------------------------------------
// Implementasi native berkecepatan tinggi untuk jalur panas Anchor-RAG:
//   - moko_tokenize      : tokenisasi setara `re.findall(r"[a-zA-Z_]{2,}", s.lower())`
//   - moko_index_build   : membangun indeks anchor dari korpus snippet
//   - moko_index_query   : skoring + peringkat top-k (paritas dengan Python)
//   - moko_index_free    : membebaskan indeks
//
// Semua fungsi diekspor dengan C linkage (extern "C") agar dapat dimuat oleh
// Python via ctypes DAN agar C ABI-nya identik dengan tier Rust (cdylib).
//
// Kontrak paritas dengan Python (docs/moko_code_knowledge.py):
//   tokenize(text): lowercase seluruh teks, lalu ambil semua deret karakter
//                   [a-zA-Z_] dengan panjang >= 2. Untuk teks ASCII ini
//                   byte-for-byte identik dengan implementasi regex Python.
//   retrieve(focus, limit): score = |anchors ∩ focus|; simpan bila score >= 1;
//                   urut menurun berdasarkan (score, -index) => skor tertinggi
//                   dulu, seri diputus oleh index terkecil; kembalikan top-limit.
// =============================================================================
#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// Pemisah serialisasi (harus sama persis dengan native_accel.py & tier Rust).
constexpr char SNIPPET_SEP = '\x1e';  // Record Separator antar-snippet
constexpr char ANCHOR_SEP = '\n';     // pemisah antar-anchor / antar-token

inline bool is_word_byte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

inline char ascii_lower(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
    return static_cast<char>(c);
}

// Tokenisasi: kembalikan daftar token (panjang >= 2), sudah di-lowercase ASCII.
std::vector<std::string> tokenize_impl(const char* text) {
    std::vector<std::string> tokens;
    if (text == nullptr) return tokens;
    std::string current;
    for (const char* p = text; *p != '\0'; ++p) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (is_word_byte(c)) {
            current.push_back(ascii_lower(c));
        } else {
            if (current.size() >= 2) tokens.push_back(current);
            current.clear();
        }
    }
    if (current.size() >= 2) tokens.push_back(current);
    return tokens;
}

struct AnchorIndex {
    // Setiap snippet menyimpan himpunan anchor untuk pencarian irisan O(1).
    std::vector<std::unordered_set<std::string>> snippets;
};

// Pecah string berdasarkan pemisah tunggal (mempertahankan token kosong->diabaikan).
std::vector<std::string> split_nonempty(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Skoring + peringkat top-k terhadap himpunan fokus. Menulis index & skor
// terurut (score desc, index asc) ke out_idx/out_score. Return jumlah hasil.
int rank_focus(const AnchorIndex& idx, const std::unordered_set<std::string>& focus_set,
               int limit, int* out_idx, int* out_score) {
    std::vector<std::pair<int, int>> scored;  // (score, index)
    scored.reserve(idx.snippets.size());
    for (size_t i = 0; i < idx.snippets.size(); ++i) {
        const std::unordered_set<std::string>& anchors = idx.snippets[i];
        int score = 0;
        // Iterasi himpunan yang lebih kecil demi efisiensi.
        if (focus_set.size() < anchors.size()) {
            for (const std::string& t : focus_set) {
                if (anchors.count(t)) ++score;
            }
        } else {
            for (const std::string& a : anchors) {
                if (focus_set.count(a)) ++score;
            }
        }
        if (score >= 1) scored.emplace_back(score, static_cast<int>(i));
    }

    // Urut: skor menurun; seri -> index menaik (paritas dgn Python (score,-index)).
    std::stable_sort(scored.begin(), scored.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                         if (a.first != b.first) return a.first > b.first;
                         return a.second < b.second;
                     });

    int n = static_cast<int>(scored.size());
    if (n > limit) n = limit;
    for (int i = 0; i < n; ++i) {
        out_idx[i] = scored[i].second;
        out_score[i] = scored[i].first;
    }
    return n;
}

}  // namespace

extern "C" {

// Nama backend untuk pelaporan/diagnostik.
const char* moko_native_backend() { return "cpp"; }

// Versi ABI; dinaikkan bila kontrak fungsi berubah.
int moko_native_abi_version() { return 2; }

// Tokenisasi. Menulis token yang digabung '\n' (NUL-terminated) ke out_buf.
// Return: jumlah byte yang ditulis (tanpa NUL), atau -1 bila out_cap kurang.
int moko_tokenize(const char* text, char* out_buf, int out_cap) {
    if (out_buf == nullptr || out_cap <= 0) return -1;
    std::vector<std::string> tokens = tokenize_impl(text);
    std::string joined;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) joined.push_back(ANCHOR_SEP);
        joined += tokens[i];
    }
    if (static_cast<int>(joined.size()) + 1 > out_cap) return -1;
    std::memcpy(out_buf, joined.data(), joined.size());
    out_buf[joined.size()] = '\0';
    return static_cast<int>(joined.size());
}

// Bangun indeks anchor dari korpus serial:
//   snippet dipisah SNIPPET_SEP ('\x1e'); anchor dalam snippet dipisah '\n'.
// `len` adalah panjang byte korpus. Return handle buram (atau nullptr bila gagal).
void* moko_index_build(const char* corpus, int len) {
    AnchorIndex* idx = new (std::nothrow) AnchorIndex();
    if (idx == nullptr) return nullptr;
    if (corpus == nullptr || len <= 0) return idx;  // indeks kosong valid
    std::string data(corpus, static_cast<size_t>(len));
    // Pertahankan boundary snippet apa adanya (termasuk snippet tanpa anchor).
    std::vector<std::string> raw_snippets;
    std::string cur;
    for (char ch : data) {
        if (ch == SNIPPET_SEP) {
            raw_snippets.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    raw_snippets.push_back(cur);
    idx->snippets.reserve(raw_snippets.size());
    for (const std::string& snip : raw_snippets) {
        std::unordered_set<std::string> anchors;
        for (const std::string& a : split_nonempty(snip, ANCHOR_SEP)) {
            anchors.insert(a);
        }
        idx->snippets.push_back(std::move(anchors));
    }
    return idx;
}

// Kueri top-k: focus = token '\n'-joined. Menulis index & skor terurut ke
// out_idx / out_score (kapasitas >= limit). Return: jumlah hasil (<= limit).
int moko_index_query(void* handle, const char* focus, int focus_len, int limit,
                     int* out_idx, int* out_score) {
    if (handle == nullptr || out_idx == nullptr || out_score == nullptr) return 0;
    if (limit <= 0) return 0;
    AnchorIndex* idx = static_cast<AnchorIndex*>(handle);

    std::unordered_set<std::string> focus_set;
    if (focus != nullptr && focus_len > 0) {
        std::string fdata(focus, static_cast<size_t>(focus_len));
        for (const std::string& t : split_nonempty(fdata, ANCHOR_SEP)) {
            focus_set.insert(t);
        }
    }
    return rank_focus(*idx, focus_set, limit, out_idx, out_score);
}

// Kueri gabungan: tokenisasi teks mentah + skoring + peringkat top-k dalam SATU
// panggilan (menghindari materialisasi daftar token di sisi Python). Ini jalur
// panas end-to-end sesungguhnya untuk navigasi konteks panjang (gaya Kimi).
// Setara Python: kb.retrieve(set(tokenize(text)), limit=limit).
int moko_index_query_text(void* handle, const char* text, int limit,
                          int* out_idx, int* out_score) {
    if (handle == nullptr || out_idx == nullptr || out_score == nullptr) return 0;
    if (limit <= 0) return 0;
    AnchorIndex* idx = static_cast<AnchorIndex*>(handle);

    std::unordered_set<std::string> focus_set;
    for (std::string& t : tokenize_impl(text)) {
        focus_set.insert(std::move(t));
    }
    return rank_focus(*idx, focus_set, limit, out_idx, out_score);
}

// Bebaskan indeks yang dibuat moko_index_build.
void moko_index_free(void* handle) {
    if (handle != nullptr) {
        delete static_cast<AnchorIndex*>(handle);
    }
}

}  // extern "C"
