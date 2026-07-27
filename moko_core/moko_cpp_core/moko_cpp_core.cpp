#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstring>

// Seed deterministik untuk hashing
const uint64_t XOR_SEED = 0x517CC1B727220A95ULL;

// Fast Murmur-like 64-bit mixer
inline uint64_t mix64(uint64_t h) {
    h ^= (h >> 33);
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= (h >> 33);
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= (h >> 33);
    return h;
}

// XXH3-like 64-bit non-cryptographic hash untuk kata
uint64_t xxh3_64(const std::string& word) {
    uint64_t hash = XOR_SEED;
    for (char c : word) {
        hash = (hash ^ c) * 0x100000001B3ULL;
    }
    return mix64(hash);
}

// Konversi token_id hex ke uint64
uint64_t token_to_u64(const char* token_id) {
    std::string s(token_id);
    if (s.length() >= 16) {
        try {
            return std::stoull(s.substr(0, 16), nullptr, 16);
        } catch (...) {
            return xxh3_64(s);
        }
    }
    return xxh3_64(s);
}

// ──────────────────────────────────────────────────────────────────────────────
// HIGH-PERFORMANCE BLOOM FILTER C++ CLASS
// ──────────────────────────────────────────────────────────────────────────────

class BloomFilterCpp {
public:
    std::vector<bool> bit_array;
    size_t size_bits = 0;
    uint64_t seed = XOR_SEED;
    size_t item_count = 0;
    bool built = false;

    bool build(const std::vector<uint64_t>& keys) {
        size_t n = keys.size();
        if (n == 0) {
            built = true;
            item_count = 0;
            return true;
        }

        // Ukuran optimal: 10 bit per item (FPR ~1% untuk 4 hash functions)
        size_bits = n * 10;
        if (size_bits < 64) size_bits = 64;
        
        bit_array.assign(size_bits, false);

        for (uint64_t key : keys) {
            // Hasilkan 4 hash index via double-hashing scheme
            uint64_t hash1 = mix64(key ^ seed);
            uint64_t hash2 = mix64(key ^ (seed >> 16));
            
            for (int i = 0; i < 4; ++i) {
                uint64_t combined_hash = hash1 + i * hash2;
                bit_array[combined_hash % size_bits] = true;
            }
        }

        item_count = n;
        built = true;
        return true;
    }

    bool might_contain(uint64_t key) const {
        if (!built || item_count == 0) {
            return true; // default safe
        }

        uint64_t hash1 = mix64(key ^ seed);
        uint64_t hash2 = mix64(key ^ (seed >> 16));

        for (int i = 0; i < 4; ++i) {
            uint64_t combined_hash = hash1 + i * hash2;
            if (!bit_array[combined_hash % size_bits]) {
                return false; // PASTI tidak ada
            }
        }
        return true; // MUNGKIN ada
    }
};

// Singleton instance global untuk filter
BloomFilterCpp g_filter;

// ──────────────────────────────────────────────────────────────────────────────
// EXPOSED C FUNCTIONS (untuk Ctypes Python)
// ──────────────────────────────────────────────────────────────────────────────

extern "C" {

    // 1. Build Bloom Filter
    bool cpp_fuse_build(const char** token_ids, int count) {
        std::vector<uint64_t> keys;
        keys.reserve(count);
        for (int i = 0; i < count; ++i) {
            keys.push_back(token_to_u64(token_ids[i]));
        }
        return g_filter.build(keys);
    }

    // 2. Query Bloom Filter
    bool cpp_fuse_might_contain(const char* token_id) {
        uint64_t key = token_to_u64(token_id);
        return g_filter.might_contain(key);
    }

    // 3. SimHash C++ Implementation
    uint64_t cpp_compute_simhash(const char* text) {
        std::string s(text);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        
        std::vector<std::string> words;
        std::stringstream ss(s);
        std::string word;
        while (ss >> word) {
            words.push_back(word);
        }

        if (words.empty()) return 0;

        float bit_counts[64] = {0.0f};

        for (const auto& w : words) {
            uint64_t w_hash = xxh3_64(w);
            for (int bit = 0; bit < 64; ++bit) {
                if ((w_hash >> bit) & 1) {
                    bit_counts[bit] += 1.0f;
                } else {
                    bit_counts[bit] -= 1.0f;
                }
            }
        }

        uint64_t result = 0;
        for (int bit = 0; bit < 64; ++bit) {
            if (bit_counts[bit] > 0.0f) {
                result |= (1ULL << bit);
            }
        }
        return result;
    }

    // 4. Shannon Entropy C++ Implementation
    float cpp_shannon_entropy(const char* text) {
        std::string s(text);
        if (s.empty()) return 0.0f;

        std::unordered_map<char, size_t> freq;
        for (char c : s) {
            freq[c]++;
        }

        float entropy = 0.0f;
        float n = (float)s.length();
        for (const auto& pair : freq) {
            float p = (float)pair.second / n;
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    // 5. Homomorphic Obfuscation: Vigenere-based printable ASCII shift cipher
    void cpp_fhe_mask_text(const char* text, const char* key, char* output) {
        std::string s_text(text);
        std::string s_key(key);
        if (s_key.empty()) s_key = "MOKO_DEFAULT_KEY";
        
        for (size_t i = 0; i < s_text.length(); ++i) {
            char c = s_text[i];
            if (c >= 32 && c <= 126) {
                int shift = (int)s_key[i % s_key.length()];
                output[i] = (char)(32 + ((c - 32 + shift) % 95));
            } else {
                output[i] = c;
            }
        }
        output[s_text.length()] = '\0';
    }

    void cpp_fhe_unmask_text(const char* cipher, const char* key, char* output) {
        std::string s_cipher(cipher);
        std::string s_key(key);
        if (s_key.empty()) s_key = "MOKO_DEFAULT_KEY";
        
        for (size_t i = 0; i < s_cipher.length(); ++i) {
            char c = s_cipher[i];
            if (c >= 32 && c <= 126) {
                int shift = (int)s_key[i % s_key.length()];
                int val = (c - 32 - shift) % 95;
                if (val < 0) val += 95;
                output[i] = (char)(32 + val);
            } else {
                output[i] = c;
            }
        }
        output[s_cipher.length()] = '\0';
    }
}

