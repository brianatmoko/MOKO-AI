#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <thread>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include "thread_pool.hpp"
#include "mmap_io.hpp"
#include "simd_math.hpp"

// Static global objects for the Kernel
static moko::ThreadPool* g_thread_pool = nullptr;

extern "C" {

// ---------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------
void moko_kernel_init(int num_threads) {
    moko::init_qev_lut();
    if (!g_thread_pool) {
        // Jika num_threads <= 0, gunakan auto-detect hardware concurrency.
        // Batasi maksimum ke 8 thread untuk mencegah context-switch overhead
        // pada mesin yang padat seperti saat model LLM juga berjalan.
        int hw = (int)std::thread::hardware_concurrency();
        int resolved = (num_threads > 0) ? num_threads : hw;
        // Gunakan setengah core agar tidak berkonflik dengan llama-server,
        // minimum 2, maksimum 8.
        int safe_threads = std::max(2, std::min(8, resolved / 2));
        g_thread_pool = new moko::ThreadPool(safe_threads);
        std::cout << "[MOKO KERNEL] Thread pool diinisialisasi: " << safe_threads 
                  << " threads (HW=" << hw << ")" << std::endl;
    }
}

void moko_kernel_shutdown() {
    if (g_thread_pool) {
        delete g_thread_pool;
        g_thread_pool = nullptr;
    }
}

// ---------------------------------------------------------
// ENCODER (FP32 -> 192 byte QEV)
// ---------------------------------------------------------
void encode_qev_c(float* vector_fp32, uint8_t* out_qev_192) {
    int dim = 768;
    float sum = 0.0f;
    for(int i = 0; i < dim; i++) { sum += std::abs(vector_fp32[i]); }
    float mean = sum / dim;

    float sq_sum = 0.0f;
    for(int i = 0; i < dim; i++) {
        float diff = std::abs(vector_fp32[i]) - mean;
        sq_sum += diff * diff;
    }
    float std_dev = std::sqrt(sq_sum / dim);

    float t_strong = mean + 0.5f * std_dev;
    float t_weak = mean - 0.5f * std_dev;
    if (t_weak < 0) t_weak = 0.0f;

    std::vector<int8_t> qev(dim, 0);
    for(int i = 0; i < dim; i++) {
        float val = vector_fp32[i];
        float abs_val = std::abs(val);

        if (abs_val <= t_weak) {
            qev[i] = 0; 
        } else {
            if (i >= 700 && abs_val >= t_strong) {
                qev[i] = 2; // Emotion Zone
            } else {
                qev[i] = (val > 0) ? 1 : -1;
            }
        }
    }

    for(int i = 0; i < dim / 4; i++) {
        int base = i * 4;
        uint8_t b0 = qev[base + 0] + 1;
        uint8_t b1 = qev[base + 1] + 1;
        uint8_t b2 = qev[base + 2] + 1;
        uint8_t b3 = qev[base + 3] + 1;
        
        out_qev_192[i] = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3;
    }
}

// ---------------------------------------------------------
// THREADED MMAP BATCH SEARCH
// ---------------------------------------------------------
// Membaca file mmap secara langsung, menghitung skor via thread pool,
// mengembalikan top-K hasil.
// query_192: query vector (192 byte)
// filepath:  path ke file route_XX.bin
// top_k:     berapa hasil yang dikembalikan
// out_indices / out_scores: array (dialokasikan Python, ukuran top_k)
int search_mmap_top_k_c(uint8_t* query_192, const char* filepath, int top_k, int* out_indices, int* out_scores) {
    if (!g_thread_pool) return -1; // Not initialized

    moko::MemoryMappedFile mmap_file(filepath);
    if (!mmap_file.isValid()) {
        return -2; // File error
    }

    size_t total_bytes = mmap_file.size();
    int num_records = total_bytes / 192;
    if (num_records == 0) return 0;

    const uint8_t* db_data = mmap_file.data();
    std::vector<int> all_scores(num_records, 0);

    // ── Pembagian kerja dinamis ────────────────────────────────────────────
    // Gunakan semua thread di pool, bukan dikunci ke 2.
    // Python-level _cpp_lock (threading.Lock) sudah mencegah concurrent calls,
    // sehingga aman menggunakan semua thread di dalam satu panggilan.
    // Minimal 1 thread, maksimum = ukuran pool.
    int hw = (int)std::thread::hardware_concurrency();
    int threads_to_use = std::max(1, std::min(hw / 2, std::max(2, (int)(num_records / 500))));
    
    int chunk_size = num_records / threads_to_use;
    if (chunk_size == 0) {
        chunk_size = num_records;
        threads_to_use = 1;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(threads_to_use);
    
    for (int t = 0; t < threads_to_use; t++) {
        int start_idx = t * chunk_size;
        int end_idx = (t == threads_to_use - 1) ? num_records : start_idx + chunk_size;
        
        futures.push_back(g_thread_pool->enqueue([&, start_idx, end_idx]() {
            for (int i = start_idx; i < end_idx; i++) {
                all_scores[i] = moko::compute_qev_dot_product_192(query_192, db_data + (i * 192));
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    // Temukan top-K menggunakan partial_sort
    std::vector<int> indices(num_records);
    for(int i=0; i<num_records; i++) indices[i] = i;

    int actual_k = std::min(top_k, num_records);
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
        [&](int a, int b) {
            return all_scores[a] > all_scores[b];
        });

    for(int i=0; i<actual_k; i++) {
        out_indices[i] = indices[i];
        out_scores[i] = all_scores[indices[i]];
    }

    return actual_k;
}

// ---------------------------------------------------------
// FAST DIRECTORY KEYWORD SEARCH (C++17 filesystem)
// ---------------------------------------------------------
// Mencari kata kunci pada field 'source' (nama file) di meta.jsonl.
// Sangat cepat karena dijalankan native pada level kernel C++.
// keywords_csv: kata kunci yang dipisahkan koma (misal: "binary,search,python")
// ---------------------------------------------------------
// KEYWORD FILESYSTEM SEARCH  (C++17 std::filesystem)
//
// Signature (match Python ctypes binding di disk_manager.py):
//   root_path      : UTF-8 direktori akar RAG
//   keywords_space : Kata kunci dipisahkan spasi (misal "python memory")
//   out_buf        : Buffer output JSON array path (caller-supplied)
//   buf_size       : Ukuran buffer out_buf dalam byte
//   max_results    : Batas jumlah file yang dikembalikan
//
// Menulis JSON array string ke out_buf (null-terminated), contoh:
//   ["/path/a/meta.jsonl","/path/b/meta.jsonl"]
//
// Returns: jumlah file ditemukan (0 jika tidak ada/error/buffer penuh)
// ---------------------------------------------------------
int search_keywords_cpp(
    const char* root_path,
    const char* keywords_space,
    char*       out_buf,
    int         buf_size,
    int         max_results
) {
    namespace fs = std::filesystem;
    if (!root_path || !keywords_space || !out_buf || buf_size < 4) return 0;
    if (!fs::exists(root_path)) { out_buf[0] = '['; out_buf[1] = ']'; out_buf[2] = '\0'; return 0; }

    // Parse keywords (space-separated) → lowercase
    std::vector<std::string> keywords;
    {
        std::stringstream ss(keywords_space);
        std::string tok;
        while (ss >> tok) {
            std::transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
            if (!tok.empty()) keywords.push_back(tok);
        }
    }
    if (keywords.empty()) { out_buf[0] = '['; out_buf[1] = ']'; out_buf[2] = '\0'; return 0; }

    // Kumpulkan path yang cocok
    std::vector<std::string> matched_paths;
    matched_paths.reserve(std::min(max_results, 512));

    try {
        for (auto const& l1 : fs::directory_iterator(root_path)) {
            if (!l1.is_directory()) continue;
            auto l1name = l1.path().filename().string();
            if (l1name.empty() || l1name[0] == '_') continue;

            for (auto const& l2 : fs::directory_iterator(l1.path())) {
                if (!l2.is_directory()) continue;
                if ((int)matched_paths.size() >= max_results) goto done;

                fs::path meta_path = l2.path() / "meta.jsonl";
                if (!fs::exists(meta_path)) continue;

                // Baca baris pertama saja untuk cek topik (lightweight)
                {
                    std::ifstream f(meta_path);
                    if (!f.is_open()) continue;
                    std::string line;
                    if (!std::getline(f, line)) continue;

                    // Lowercase in-place
                    std::transform(line.begin(), line.end(), line.begin(), ::tolower);

                    // ANY keyword harus ada di baris pertama (OR logic)
                    bool found = false;
                    for (const auto& kw : keywords) {
                        if (line.find(kw) != std::string::npos) { found = true; break; }
                    }
                    if (found) {
                        matched_paths.push_back(meta_path.string());
                    }
                }
            }
        }
    } catch (...) {}

done:
    // Serialisasi ke JSON array dengan bound checking ketat
    // Format: ["path1","path2",...]
    int written = 0;
    out_buf[written++] = '[';

    for (size_t i = 0; i < matched_paths.size(); ++i) {
        const std::string& p = matched_paths[i];
        // Estimasi space: '"' + path + '"' + ','  + final ']' + '\0'
        int needed = (int)(1 + p.size() + 1 + 1 + 1 + 1);
        if (written + needed >= buf_size) break;  // buffer hampir penuh → stop

        if (i > 0) out_buf[written++] = ',';
        out_buf[written++] = '"';
        // Copy path, escape backslash dan double-quote
        for (char c : p) {
            if (c == '"' || c == '\\') {
                if (written + 2 >= buf_size) goto flush;
                out_buf[written++] = '\\';
            }
            if (written + 1 >= buf_size) goto flush;
            out_buf[written++] = c;
        }
        out_buf[written++] = '"';
    }

flush:
    if (written < buf_size - 1) out_buf[written++] = ']';
    out_buf[written] = '\0';

    return (int)matched_paths.size();
}

} // extern "C"

