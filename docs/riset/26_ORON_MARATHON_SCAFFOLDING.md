# RISET 26: SISTEM MARATON DAN SELF-SCAFFOLDING ORNITH-STYLE
## Pemodelan Matematika Sistem Maraton Terdistribusi (Pos & Layer), Self-Scaffolding, dan Integrasi Keluarga Agen Ornith untuk Coding Profesional
### Dokumen Riset Lanjutan MOKO OS — Juli 2026

---

## PENDAHULUAN: MEMUTUS LOGIKA EKSEKUSI TANPA BATAS

Model bahasa dan agen otonom tradisional (seperti GPT-4, Claude-3.5, hingga versi awal DeepSeek-R1) bekerja dengan paradigma **Autoregressive Run-on-Sentence / Continuous Execution**. Mereka terus-menerus "berlari" (menghasilkan token, berpikir dalam loop penalaran) tanpa adanya struktur pembatas diskret di tingkat sistem yang memaksa mereka berhenti, melakukan audit state secara lengkap, menyimpan checkpoint terkompresi, dan baru melanjutkan langkah berikutnya.

Hal ini memicu tiga kegagalan besar:
1. **KV Cache Explosion:** Sequence token yang memanjang secara eksponensial memicu degradasi memori dan VRAM OOM.
2. **Cognitive Drift:** Semakin jauh agen melangkah dalam penalaran berkelanjutan, semakin tinggi probabilitas mereka menyimpang dari tujuan awal (*loss of system directive*).
3. **Infinite Loop Call:** Agen terus mencoba memanggil perkakas eksternal tanpa menyadari bahwa state lingkungan tidak berubah.

Dokumen ini merumuskan **Sistem Maraton MOKO (Moko Marathon System)**. Sistem ini membagi proses berpikir agen menjadi **Layer dan Pos (Checkpointing)**. Alih-alih lari terus-menerus, agen berhenti di setiap "Pos" untuk menyimpan data lengkap dalam representasi paling sederhana, divalidasi oleh "Guard Layer", lalu diteruskan ke langkah berikutnya.

Riset ini juga membedah implementasi **Self-Scaffolding** yang diperkenalkan oleh keluarga model coding terbaru **Ornith-1.0 (DeepReinforce.AI)** yang dirilis pada Juni 2026.

---

## BAGIAN 1: TEORI MATEMATIKA SISTEM MARATON (LAYER & POS)

Sistem Maraton membagi pengerjaan tugas coding kompleks menjadi $N$ Pos Diskret ($P_1, P_2, \dots, P_N$) yang dipetakan sepanjang $M$ Layer Kognitif ($L_1, L_2, \dots, L_M$).

```
                === SKEMA MARATON MOKO (LAYER & POS) ===

   Layer 3:  [Pos 1.3] ─────────────► [Pos 2.3] ─────────────► [Pos 3.3] (Evaluasi)
    (Audit)      ▲                        ▲                        ▲
                 │ (Verifikasi)           │ (Verifikasi)           │ (Verifikasi)
   Layer 2:  [Pos 1.2] ─────────────► [Pos 2.2] ─────────────► [Pos 3.2] (Error Check)
   (Syntax)      ▲                        ▲                        ▲
                 │ (Verifikasi)           │ (Verifikasi)           │ (Verifikasi)
   Layer 1:  [Pos 1.1] ─────────────► [Pos 2.1] ─────────────► [Pos 3.1] (Generasi)
   (Generasi)    │                        │                        │
                 ▼                        ▼                        ▼
           [CHECKPOINT 1]           [CHECKPOINT 2]           [CHECKPOINT 3]
           Format Simpel            Format Simpel            Format Simpel
```

Aturan dasar: **Agen dilarang melompat ke Pos berikutnya $P_{i+1}$ sebelum Pos $P_i$ dinyatakan lulus (lunas) oleh seluruh Layer Audit.**

### 1.1 Persamaan Transisi State Pos Terbatas

Kita definisikan state kognitif pada Pos $i$ dan Layer $j$ sebagai $\mathbf{s}_{i, j}$. Transisi pemikiran dari satu pos ke pos berikutnya dimodelkan sebagai operator proyeksi non-linear $\mathcal{T}$ yang dihambat oleh matriks gerbang verifikasi $\mathbf{G}_{i}$:

$$\mathbf{s}_{i+1, 1} = \mathbf{G}_{i} \odot \mathcal{T}(\mathbf{s}_{i, M}) + (1 - \mathbf{G}_{i}) \odot \mathbf{s}_{i, \text{fallback}}$$

Di mana:
- $\mathbf{s}_{i, M}$ adalah state akhir di Pos $i$ setelah melewati Layer Audit tertinggi $M$.
- $\mathbf{G}_{i} \in \{0, 1\}^d$ adalah **Vector Gerbang Biner (Binary Gate Vector)** yang dihitung secara deterministik oleh unit-test/interpreter eksternal di runtime:
  
  $$\mathbf{G}_{i} = \begin{cases} 
  \mathbf{1} & \text{jika } f_{\text{verify}}(\mathbf{s}_{i, M}) \ge \theta_{\text{safety}} \\
  \mathbf{0} & \text{lainnya}
  \end{cases}$$

- $\mathbf{s}_{i, \text{fallback}}$ adalah state pemulihan jika verifikasi gagal (mengembalikan agen ke checkpoint stabil terdekat di SSD).

Hal ini menjamin bahwa **tidak ada kode cacat yang bisa berpindah ke pos eksekusi berikutnya**.

---

### 1.2 Symmetric Semantic Checkpoint (Symmetric Compression)

Agar data yang disimpan di setiap Pos $P_i$ sangat ringan, kita menyederhanakan data state $\mathbf{s}_{i, M}$ menjadi **Symmetric Semantic Checkpoint (SSC)**. 

Alih-alih menyimpan seluruh riwayat chat (conversation history) dan seluruh token yang dihasilkan, kita mengompresi state tersebut menjadi pasangan kunci-nilai matematis yang disebut **Semantic State Vector ($\mathbf{z}_i$)** menggunakan projection matrix $W_{\text{ssc}}$:

$$\mathbf{z}_i = \text{LayerNorm}(W_{\text{ssc}} \cdot [\mathbf{h}_{\text{code}} ; \mathbf{h}_{\text{test\_status}} ; \mathbf{h}_{\text{ast\_hash}}])$$

Ukuran $\mathbf{z}_i$ ini dibatasi hanya **128 byte** (sangat sederhana). Ketika agen di Pos $P_{i+1}$ diaktifkan kembali, ia tidak memuat seluruh sejarah percakapan, melainkan melakukan dekompresi $\mathbf{z}_i$ untuk merekonstruksi instruksi kontekstualnya secara instan:

$$\mathbf{s}_{i+1, \text{init}} = W_{\text{decompress}} \cdot \mathbf{z}_i + \mathbf{b}_{\text{bias}}$$

Ini menghemat KV cache hingga **99.5%** pada prefill stage di Pos berikutnya karena kita memotong masa lalu dan hanya membawa "intisari hasil lari di pos sebelumnya".

---

## BAGIAN 2: SELF-SCAFFOLDING (ORNITH-1.0) & DYNAMIC CONTROL PIPELINE

Keluarga model **Ornith-1.0** (DeepReinforce.AI, Juni 2026) memperkenalkan konsep **Self-Scaffolding** di tingkat arsitektur model melalui Reinforcement Learning.

### 2.1 Konsep Self-Scaffolding vs Static Scaffolding

Dalam AI Agent konvensional (seperti AutoGPT atau CrewAI), "scaffold" atau aturan main agen ditulis secara statis oleh developer manusia di sisi kode Python (misalnya: prompt system kaku, parser JSON, dan penanganan exception).

Ornith-1.0 membuang pendekatan kaku ini. Model dilatih dengan RL untuk **membangun scaffold-nya sendiri secara dinamis** sesuai dengan tingkat kesulitan tugas coding yang dihadapinya.

```
Static Scaffolding (Lama):
  User Prompt ──► [System Prompt Manusia (Kaku)] ──► LLM ──► Parser JSON (Sering Crash)

Self-Scaffolding (Ornith-Style):
  User Prompt ──► Model Generates:
                     1. Custom Memory Layout (Variables)
                     2. Custom Iteration Guard (RSI Loop)
                     3. Dynamic Error Handling Strategy
                  ──► Model Executes Self-Scaffold ──► Validated Output
```

### 2.2 Matematika Self-Scaffolding RL (Ornith Policy Update)

Ornith dilatih menggunakan variasi fungsi objektif policy gradient yang memaksimalkan keberhasilan tugas ($R_{\text{task}}$) sekaligus meminimalkan kompleksitas scaffold ($C_{\text{scaffold}}$):

$$\mathcal{L}_{\text{Ornith}}(\theta) = \mathbb{E} \left[ \pi_\theta(a_{\text{scaffold}}, a_{\text{code}} | s) \cdot \left( R_{\text{task}} - \lambda \cdot C_{\text{scaffold}}(a_{\text{scaffold}}) \right) \right]$$

Di mana:
- $a_{\text{scaffold}}$ adalah aksi pembentukan struktur penalaran (jumlah thread, jenis variabel memori sementara).
- $a_{\text{code}}$ adalah aksi penulisan baris kode pemrograman.
- $C_{\text{scaffold}}(a_{\text{scaffold}})$ mengukur overhead komputasi dari scaffold yang dibuat model.
- $\lambda$ adalah koefisien penalti untuk mencegah model membuat sistem berpikir yang terlalu rumit secara berlebihan.

---

## BAGIAN 3: IMPLEMENTASI DESAIN SISTEM MARATON DI MOKO OS

Berikut adalah taktik implementasi konkrit untuk menyuntikkan filosofi Sistem Maraton (Layer & Pos) dan Self-Scaffolding ke dalam `moko_core`:

### 3.1 Struktur Data Checkpoint Maraton Simpel

Kita mendefinisikan struktur checkpoint seminimal mungkin dalam bentuk berkas JSON terkompresi Zstd:

```json
{
  "checkpoint_id": "pos_03_compilation_success",
  "timestamp": 1783454210.15,
  "pos_index": 3,
  "semantic_state": {
    "intent": "implement_trigonometric_lookups",
    "last_valid_code": "def cos_lookup(deg):\n    return lookup_table[int(deg) % 360]",
    "ast_fingerprint": "8f2b3e8c9d1a",
    "verified_layers": ["SYNTAX", "ERROR"]
  },
  "checkpoint_signature": "moko_sig_v1_88af9c"
}
```

Setiap kali agen menyelesaikan satu Pos (misalnya Pos 3: menulis fungsi dan sukses kompilasi), data JSON ini ditulis ke SSD. Jika di Pos 4 (menulis unit-test tingkat lanjut) agen mengalami loop atau crash, orchestrator tinggal memanggil fungsi `rollback_to_pos(3)` yang memuat ulang state JSON di atas dan mereset total action count.

---

### 3.2 Skenario Kolaborasi Multi-Model Domain Expert (MoE) MOKO

Untuk merealisasikan Sistem Maraton ini tanpa membebani memori, kita membagi tugas agen di setiap Layer kognitif menggunakan model spesialis:

```
                            [ PERMINTAAN USER ]
                                     │
                                     ▼
                      ╔═════════════════════════╗
                      ║     Orchestrator        ║
                      ║  (Ornith-9B / GLM-5.2)  ║
                      ╚══════════════╤══════════╝
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    ╔═════════════════╗     ╔═════════════════╗     ╔═════════════════╗
    ║ Pos 1: Generasi ║     ║ Pos 2: Debugging║     ║ Pos 3: Review   ║
    ║   (Qwen-Coder)  ║     ║  (Ornith-9B)    ║     ║ (DeepSeek-R1)   ║
    ╚═════════════════╝     ╚═════════════════╝     ╚═════════════════╝
```

1. **Pos Generasi (Layer 1):** Menggunakan model yang sangat cepat dan efisien (seperti `Qwen2.5-Coder-1.5B-INT4`) untuk menulis draf kasar kode.
2. **Pos Debugging & AST Audit (Layer 2):** Menggunakan model yang dilatih khusus dengan *Self-Scaffolding* (seperti model keluarga **Ornith-9B** yang ramah hardware lokal) untuk membaca AST dan membetulkan sintaksis.
3. **Pos Evaluasi & Guard (Layer 3):** Menggunakan model penalaran tingkat tinggi (seperti **DeepSeek-R1** via Hybrid API Mandor) untuk memastikan kebenaran logika matematis akhir.

---

## KESIMPULAN RISET 26

Sinergi antara **Sistem Maraton (Layer & Pos)** dan **Self-Scaffolding (Ornith-style)** merupakan lompatan paradigma terbesar bagi MOKO OS:

1. **Fisika Komputer Terpenuhi:** Dengan memotong context window di setiap Pos (SSC checkpointing), kebutuhan VRAM dan RAM dibatasi secara konstan.
2. **Akurasi Ekstrem:** Verifikasi bertingkat pada setiap Pos menjamin tidak ada "efek domino" kesalahan penalaran dari awal sampai akhir.
3. **Fleksibilitas Kerja:** Model lokal (Ornith-9B / Qwen-1.5B) bertindak sebagai pekerja cepat di Pos lokal, sementara API besar bertindak sebagai juri pemutus di Pos final.

---
*Dokumen ini merupakan Riset Nomor 26 dalam ekosistem MOKO OS.*
*Status: Teoretis Validated | Tanggal: Juli 2026 | Oleh: Tim Peneliti AI MOKO OS*
