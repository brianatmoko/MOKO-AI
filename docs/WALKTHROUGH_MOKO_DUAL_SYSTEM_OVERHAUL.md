# WALKTHROUGH MOKO DUAL-SYSTEM OVERHAUL: INTEGRASI DEEPSEEK (SISTEM 2) & KIMI AI (SISTEM 1)

## 1. Filosofi & Visi Sistem Ganda (Dual-System)
Moko OS & IDE v5 memadukan kekuatan dua filosofi AI terdepan:
- **Sistem 2 (DeepSeek-R1) - Otak Penalaran & Penjaga (Reasoning & Guard)**: Berjalan lambat, taktis, dan logis menggunakan Chain-of-Thought (CoT). Bertanggung jawab merancang rencana perubahan, menulis unit test otomatis, memverifikasi kesesuaian kode, serta mengevaluasi log compiler/interpreter (via `LLMRuntimeGuard`).
- **Sistem 1 (Kimi K2/K2.5) - Tangan Eksekutor (Agentic Executor)**: Berjalan cepat, otonom, dan menguasai navigasi konteks panjang. Bertanggung jawab mencari berkas relevan di repositori menggunakan Anchor-RAG (`moko_code_knowledge.py`), mengedit berkas kode secara presisi, mengeksekusi perintah terminal, dan melacak status perubahan Git.

## 2. Arsitektur Kognitif & Alur Kerja Loop Agen
Aliran orkestrasi diatur secara otonom oleh `DualSystemOrchestrator`:
1. **User Prompt**: Pengguna memberikan instruksi pemrograman atau penyelesaian bug di Moko IDE.
2. **Planning & Test-Case Generation (Sistem 2 - Brain)**: Menganalisis instruksi secara logis, menghasilkan rencana kerja langkah-demi-langkah, dan menyusun script pengujian unit (`unit test`).
3. **Context-Aware Retrieval (Sistem 1 - Hands)**: Mengambil potongan kode pengetahuan relevan menggunakan pencarian berbasis Anchor-RAG (`moko_code_knowledge.py`) untuk memetakan dependensi file target.
4. **Agentic Code Execution (Sistem 1 - Hands)**: Melakukan modifikasi file kode di dalam workspace secara langsung dan menulis berkas pengujian otomatis.
5. **Terminal-run Verification (Sistem 1 - Hands)**: Menjalankan unit test di subproses terminal lokal untuk menguji keabsahan kode.
6. **Runtime Guard & Review (Sistem 2 - Guard)**: Memeriksa log output terminal (stdout/stderr) menggunakan `LLMRuntimeGuard`. Jika lolos, ia melakukan Git commit otomatis dan melaporkan sukses ke GUI. Jika gagal, ia menganalisis log error, memicu loop koreksi-diri (*self-correction*), dan memerintahkan Sistem 1 melakukan edit ulang.

## 3. Komponen Inti Baru (`moko_core/moko_agents/dual_system/`)
Arsitektur baru ini memiliki struktur internal sebagai berikut:
- `brain_node.py` (System 2): Melakukan penalaran CoT, menghasilkan rencana taktis, dan merancang kasus uji otomatis.
- `executor_node.py` (System 1): Mengelola operasi file (baca, tulis, edit) dan mengeksekusi perintah terminal dengan bantuan modul pencarian Anchor-RAG.
- `runtime_guard.py` (System 2 Guard): Mengintegrasikan `moko_llm_runtime_guard.py` untuk evaluasi kepatuhan runtime, memparsing galat traceback, dan mengeluarkan instruksi perbaikan otomatis.
- `orchestrator.py` (Orchestrator): Sebagai koordinator pusat yang mengikat seluruh node kognitif ke dalam kalang (loop) iteratif otonom.

## 4. Struktur Folder Terpadu Moko OS & IDE v5
Berikut adalah tata letak berkas lengkap di dalam repositori Moko OS Project setelah perombakan:

```
MOKO_OS_Project/
├── bin/                                    # Virtual environment binaries
├── docs/                                   # Dokumentasi & riset terpadu
│   ├── riset/                              # Dokumentasi riset ilmiah (21 file .md)
│   │   ├── 01_KONTRIBUSI_ASLI.md
│   │   ...
│   │   ├── 20_DEEPSEEK_2026_RESEARCH_MASTER.md
│   │   └── 21_KIMI_AI_RESEARCH_MASTER.md
│   ├── WALKTHROUGH_MOKO_DUAL_SYSTEM_OVERHAUL.md  # MASTER WALKTHROUGH BARU
│   ├── moko_code_knowledge.py              # Sistem Anchor-based RAG
│   ├── moko_llm_runtime_guard.py           # Sistem Runtime Guard & Validator
│   └── ...
├── finetune/                               # Fine-tuning & training scripts
│   ├── build_moko_coder_dataset.py         # Pembuat dataset SFT agentic
│   ├── lora_trainer.py                     # Implementasi optimizer MuonClip
│   ├── train_lora.py                       # Pipeline pelatihan LoRA
│   └── ...
├── moko_core/                              # Core system & agents
│   ├── moko_agents/
│   │   ├── dual_system/                    # NEW: Package Sistem Ganda
│   │   │   ├── __init__.py
│   │   │   ├── brain_node.py               # NEW: System 2 Reasoning & Planner
│   │   │   ├── executor_node.py            # NEW: System 1 Agentic Executor
│   │   │   ├── runtime_guard.py            # NEW: System 2 Guard & Reviewer
│   │   │   └── orchestrator.py             # NEW: Dual-System Coordinator Loop
│   │   └── ...
│   ├── moko_native/                        # NEW: Inti akselerasi native (C++/Rust)
│   │   ├── __init__.py
│   │   ├── native_accel.py                 # Loader ctypes + fallback murni-Python
│   │   ├── build.sh                        # Build C++ (g++) & Rust (cargo) cdylib
│   │   ├── bench.py                        # Benchmark native vs Python (paritas)
│   │   ├── cpp/moko_native.cpp             # Tier C++ (C ABI)
│   │   └── rust/                           # Tier Rust (cdylib, C ABI identik)
│   │       ├── Cargo.toml
│   │       └── src/lib.rs
│   ├── moko_ui/
│   │   ├── workers/
│   │   │   └── cognitive_worker.py         # MODIFIED: Integrasi DUAL mode orchestrator
│   │   ├── panels/
│   │   │   └── status_panel.py             # MODIFIED: Indikator status ganda
│   │   └── main_window_v5.py               # MODIFIED: Pengendali sinyal GUI
│   ├── test_dual_system.py                 # NEW: Integration test sistem ganda
│   └── test_native_accel.py                # NEW: Uji paritas & integrasi native
└── moko.sh                                 # Launcher utama Moko OS
```

## 5. Integrasi GUI PyQt6 & Cognitive Worker Thread
- **`cognitive_worker.py`**: Saat pengguna mengaktifkan mode `DUAL`, thread latar belakang ini akan mengimpor `orchestrator.py` dan memicu `DualSystemOrchestrator.run_loop()`.
- **UI Progress Signaling**: Thread mengirimkan sinyal progres dan potongan token stream dari background orchestrator ke PyQt6 main thread secara real-time.
- **`status_panel.py` & `main_window_v5.py`**: Panel status UI akan menampilkan label state kognitif yang sedang aktif secara dinamis:
  - `"🧠 BRAIN PLANNING"`: Saat Sistem 2 sedang menganalisis rencana kerja dan menulis unit test.
  - `"🔧 EXECUTOR ACTING"`: Saat Sistem 1 sedang menelusuri repositori, mengedit kode, atau menjalankan terminal.
  - `"🛡️ GUARD VALIDATING"`: Saat Sistem 2 Guard sedang mengevaluasi log eksekusi dan memverifikasi integritas runtime.

## 6. SFT & Reinforcement Learning (MuonClip & Verifiable Rewards)
Untuk melatih model lokal MOKO-Coder agar memiliki stabilitas numerik dan perilaku agen yang andal, kita menerapkan dua inovasi utama:
- **MuonClip Optimizer**:
  Pembaruan langkah Muon standar dimodifikasi untuk menstabilkan model MoE raksasa dengan mengintegrasikan weight decay orto-konsisten, RMS matching, dan QK-clipping ($C_{qk}$):
  $$\Delta W = \text{orthonormalize}(G)$$
  $$\Delta W_{\text{clip}} = \text{clamp}\left( \Delta W, -\gamma, \gamma \right) \quad \text{di mana} \quad \gamma = \alpha \cdot \text{RMS}(W)$$
  $$W_{t+1} = W_t - \eta \cdot \left( \Delta W_{\text{clip}} + \lambda_{\text{decay}} W_t \right)$$
  Ini dipasang di `lora_trainer.py` and `train_lora.py` untuk menekan logit/gradient explosion.
- **Agentic SFT Dataset Format**:
  Dataset di `build_moko_coder_dataset.py` dilatih menggunakan tag berpasangan khusus untuk memisahkan pikiran kognitif dari tindakan eksekusi:
  ```
  <thought>
  [Analisis Penalaran CoT oleh Sistem 2]
  </thought>
  <action>
  [Pemanggilan alat, pencarian file, atau perubahan kode oleh Sistem 1]
  </action>
  ```
- **Verifiable Rewards**:
  Menggunakan penggabungan Joint RL di mana model diberi imbalan (rewards) positif secara instan berdasarkan keberhasilan kompilasi kode dan kesesuaian unit-test terverifikasi (Verifiable Rewards) serta checklist kepatuhan instruksi.

## 7. Inti Akselerasi Native (C++ / Rust) — "Sistem Super Kuat"
Untuk membuat Dual-System benar-benar tangguh, jalur panas Anchor-RAG yang paling terikat CPU (tokenisasi + skoring anchor + peringkat top-k) dipindahkan dari Python ke **inti native**. Ketika Python menjadi hambatan, sistem otomatis memakai inti berkecepatan tinggi; bila toolchain tidak ada, seluruh jalur mundur mulus ke murni-Python **dengan hasil yang identik** (tidak ada perubahan perilaku, hanya kecepatan).

### 7.1 Dua Tier Native dengan C-ABI Identik
Paket `moko_core/moko_native/` mengekspor satu **C ABI stabil** yang diimplementasikan dua kali:
- **Tier C++** (`cpp/moko_native.cpp`, dibangun dengan `g++ -O3`) → `libmoko_native.so` — tier utama.
- **Tier Rust** (`rust/`, `cargo build --release` → `cdylib`) → `libmoko_native_rs.so` — tier "lebih kuat" (memory-safe, LTO), dipilih bila C++ dirasa kurang.

Fungsi C-ABI yang diekspor (identik di kedua bahasa):
```
const char* moko_native_backend();       // "cpp" | "rust"
int  moko_native_abi_version();          // kontrak fungsi = 2
int  moko_tokenize(text, out_buf, cap);  // tokenisasi [a-zA-Z_]{2,}
void* moko_index_build(corpus, len);     // bangun indeks anchor (sekali)
int  moko_index_query(h, focus, ...);    // skoring + top-k dari token siap pakai
int  moko_index_query_text(h, text, ...);// GABUNGAN: tokenisasi+skoring 1 panggilan
void moko_index_free(h);
```

### 7.2 Pemilihan Backend Otomatis & Fallback
`native_accel.py` (binding `ctypes`, tanpa dependensi pihak ketiga) memilih backend berurutan:
```
Rust (libmoko_native_rs.so)  →  C++ (libmoko_native.so)  →  murni-Python
```
Dapat dipaksa via env `MOKO_NATIVE_LIB=/path/ke/lib.so`. `moko_agents/dual_system/_bridge.py` secara transparan memakai `tokenize`, `retrieve`, dan `retrieve_text` yang berakselerasi native (dengan cache indeks per-`CodeKnowledgeBase`), sehingga `BrainNode` & `ExecutorNode` otomatis lebih cepat tanpa mengubah antarmuka.

### 7.3 Jaminan Paritas & Kinerja
Paritas byte-for-byte diverifikasi oleh `moko_core/test_native_accel.py` (tokenize, query, query_text, dan integrasi `_bridge.retrieve` vs `CodeKnowledgeBase.retrieve`). Benchmark `moko_core/moko_native/bench.py` (korpus 5.000 snippet, konteks ~200K token) menunjukkan jalur gabungan **`query_text` ~1.6–1.75× lebih cepat** dan skoring retrieval hingga **~1.7×** dibanding murni-Python, tanpa satupun perbedaan hasil.

### 7.4 Cara Membangun
```bash
bash moko_core/moko_native/build.sh        # kompilasi C++ + Rust (idempotent)
python moko_core/test_native_accel.py      # verifikasi paritas + integrasi
python moko_core/moko_native/bench.py      # ukur peningkatan kecepatan
```
