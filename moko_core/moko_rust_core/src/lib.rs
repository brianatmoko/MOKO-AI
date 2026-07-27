use pyo3::prelude::*;
use blake3;
use xxhash_rust::xxh3::xxh3_64;
use std::collections::HashSet;

// Seed standard untuk mapping bit mix
const XOR_SEED: u64 = 0x517CC1B727220A95;

#[inline]
fn mix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    return h;
}

#[inline]
fn token_to_u64(token_id: &str) -> u64 {
    if token_id.len() >= 16 {
        u64::from_str_radix(&token_id[..16], 16).unwrap_or(0)
    } else {
        xxh3_64(token_id.as_bytes())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BINARY FUSE FILTER RUST
// ──────────────────────────────────────────────────────────────────────────────

#[pyclass]
pub struct BinaryFuseFilterRust {
    fingerprints: Vec<u8>,
    segment_len: usize,
    size: usize,
    seed: u64,
    item_count: usize,
    built: bool,
}

#[pymethods]
impl BinaryFuseFilterRust {
    #[new]
    fn new() -> Self {
        BinaryFuseFilterRust {
            fingerprints: Vec::new(),
            segment_len: 0,
            size: 0,
            seed: XOR_SEED,
            item_count: 0,
            built: false,
        }
    }

    fn build(&mut self, token_ids: Vec<String>) -> bool {
        let n = token_ids.len();
        if n == 0 {
            self.built = true;
            self.item_count = 0;
            return true;
        }

        let keys: Vec<u64> = token_ids.iter().map(|tid| token_to_u64(tid)).collect();
        
        // Setup array size
        let mut array_size = ((1.23 * n as f64) as usize).max(3);
        array_size += (3 - (array_size % 3)) % 3;
        self.segment_len = array_size / 3;
        self.size = array_size;

        let mut fp = vec![0u8; array_size];
        let seed = XOR_SEED;

        // Build: assign fingerprints ke index h0, h1, h2
        for &key in &keys {
            let finger = (mix64(key) & 0xFF) as u8;
            let h0 = (mix64(key ^ seed) % self.segment_len as u64) as usize;
            let h1 = (self.segment_len as u64 + (mix64(key ^ (seed >> 16)) % self.segment_len as u64)) as usize;
            let h2 = (2 * self.segment_len as u64 + (mix64(key ^ (seed >> 32)) % self.segment_len as u64)) as usize;

            fp[h0] ^= finger;
            fp[h1] ^= finger;
            fp[h2] ^= finger;
        }

        self.fingerprints = fp;
        self.item_count = n;
        self.seed = seed;
        self.built = true;
        true
    }

    fn might_contain(&self, token_id: &str) -> bool {
        if !self.built || self.item_count == 0 {
            return true; // default safe
        }

        let key = token_to_u64(token_id);
        let finger = (mix64(key) & 0xFF) as u8;
        
        let h0 = (mix64(key ^ self.seed) % self.segment_len as u64) as usize;
        let h1 = (self.segment_len as u64 + (mix64(key ^ (self.seed >> 16)) % self.segment_len as u64)) as usize;
        let h2 = (2 * self.segment_len as u64 + (mix64(key ^ (self.seed >> 32)) % self.segment_len as u64)) as usize;

        if h0 >= self.size || h1 >= self.size || h2 >= self.size {
            return true;
        }

        let xor_val = self.fingerprints[h0] ^ self.fingerprints[h1] ^ self.fingerprints[h2];
        xor_val == finger
    }

    fn item_count(&self) -> usize {
        self.item_count
    }

    fn is_built(&self) -> bool {
        self.built
    }
}

// ──────────────────────────────────────────────────────────────────────────────
# [pyfunction]
fn compute_simhash_rust(text: &str) -> PyResult<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(0);
    }

    let mut bit_counts = vec![0.0f32; 64];

    for word in words {
        let word_lower = word.to_lowercase();
        let word_hash = xxh3_64(word_lower.as_bytes());
        for bit in 0..64 {
            if ((word_hash >> bit) & 1) == 1 {
                bit_counts[bit] += 1.0;
            } else {
                bit_counts[bit] -= 1.0;
            }
        }
    }

    let mut result: u64 = 0;
    for bit in 0..64 {
        if bit_counts[bit] > 0.0 {
            result |= 1 << bit;
        }
    }

    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
# [pyfunction]
fn compute_blake3_hash(data: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

// ──────────────────────────────────────────────────────────────────────────────
# [pymodule]
fn moko_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BinaryFuseFilterRust>()?;
    m.add_function(wrap_pyfunction!(compute_simhash_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_blake3_hash, m)?)?;
    Ok(())
}
