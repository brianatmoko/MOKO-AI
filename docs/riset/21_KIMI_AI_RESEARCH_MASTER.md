# KIMI AI 2026 RESEARCH MASTER: OPEN AGENTIC INTELLIGENCE, MUONCLIP, DAN SISTEM GANDA MOKO IDE
## Dokumen Riset Komprehensif MOKO OS (Kimi K2, Kimi K2.5, & Integrasi Moko IDE)

### Latar Belakang & Visi MOKO AI Agentic
Jika visi MOKO OS yang terinspirasi oleh DeepSeek adalah mewujudkan **otak penalaran yang ultra-efisien dan taktis**, maka visi yang diadaptasi dari **Kimi AI (Moonshot AI)** adalah memberikan **tangan, mata, dan kemampuan eksekusi otonom jangka panjang (Open Agentic Intelligence)** pada ekosistem MOKO.

Sepanjang tahun 2025 hingga awal 2026, Moonshot AI merilis laporan teknis mengenai seri model `Kimi K2` dan `Kimi K2.5`. Model-model ini memecahkan kebuntuan industri dalam penskalaan agen otonom: bagaimana melatih LLM agar mampu memanggil puluhan ribu alat eksternal secara beruntun tanpa terjebak dalam kalang (loop) tak berhingga, serta bagaimana menstabilkan pelatihan model Mixture-of-Experts (MoE) berbobot triliunan parameter tanpa kegagalan numerik.

Dokumen riset ini disusun secara mendalam untuk membedah arsitektur Kimi AI, merumuskan inovasi matematis pengoptimal (optimizer) `MuonClip`, dan merancang integrasi Sistem Ganda (Dual-System) pada MOKO IDE v5 guna menyandingkan keandalan agen Kimi AI dengan kecerdasan penalaran DeepSeek-R1.

---

### Bagian 1: Arsitektur Inti Kimi K2 & Kimi K2.5

Kimi K2 dirancang khusus untuk skenario agen mandiri otonom yang membutuhkan pemrosesan dokumen sangat panjang dan interaksi multi-langkah dengan lingkungan dinamis.

#### 1. Arsitektur Mixture-of-Experts (MoE) Skala Raksasa
Kimi K2 menggunakan arsitektur Sparse MoE dengan konfigurasi parameter ekstrem:
* **Total Parameter**: 1.04 Triliun parameter ($1.04 \times 10^{12}$).
* **Parameter Aktif**: 32 Miliar parameter aktif per token.
* **Jumlah Ahli (Experts)**: total $384$ pakar spesifik, di mana $8$ pakar aktif dipilih per token menggunakan mekanisme routing berbasis Top-$K$ gating.
* **Tujuan**: Memisahkan kapasitas penyimpanan pengetahuan global ke dalam ratusan pakar terisolasi. Hal ini memungkinkan model mempertahankan kecerdasan ensiklopedis tanpa meningkatkan overhead komputasi per token (FLOPs).

#### 2. Multi-head Latent Attention (MLA) untuk Penghematan KV Cache
Sama seperti DeepSeek-V3, Kimi K2 menerapkan **Multi-head Latent Attention (MLA)** untuk mengompresi dimensi kunci (Key) dan nilai (Value) ke dalam ruang laten berdimensi rendah sebelum proyeksi atensi dilakukan.

Formulasi proyeksi MLA untuk query, key, dan value adalah sebagai berikut:
1. **Kompresi Key-Value Laten**:
   $$c_t^{KV} = W^{DKV} h_t$$
   Di mana $h_t \in \mathbb{R}^d$ adalah hidden state pada langkah $t$, $W^{DKV} \in \mathbb{R}^{d_c \times d}$ adalah matriks kompresi, dan $d_c \ll d$ (dimensi laten jauh lebih kecil dibanding dimensi model asli).
2. **Dekonstruksi Key & Value**:
   $$k_t = W^{UK} c_t^{KV} \quad \text{dan} \quad v_t = W^{UV} c_t^{KV}$$
   Di mana $W^{UK}$ dan $W^{UV}$ adalah matriks dekompresi. Dengan taktik ini, memori yang dibutuhkan untuk menyimpan KV Cache di VRAM GPU berkurang hingga **lebih dari 90%**, sehingga model mampu melayani ribuan pengguna secara bersamaan pada panjang konteks ratusan ribu token.

#### 3. Ekstensi Konteks Tanpa Kehilangan Informasi (YaRN)
Kimi K2 memperluas jendela konteks aslinya hingga **128K - 256K token** secara otonom menggunakan metode **YaRN (Yet another RoPE extensioN)**.
* YaRN meregangkan frekuensi atensi pada pengkodean posisi putar (Rotary Position Embedding/RoPE) menggunakan interpolasi frekuensi non-uniform.
* Rumus interpolasi YaRN membagi dimensi embedding posisi menjadi tiga pita frekuensi yang berbeda (high, medium, low) dan menerapkan faktor koreksi interpolasi halus:
  $$f(\theta_i) = \begin{cases} 
  \theta_i & \text{jika } \nu_i < \lambda_b \\
  \frac{\theta_i}{s} & \text{jika } \nu_i > \lambda_e \\
  \text{interpolasi}(\theta_i, s) & \text{lainnya}
  \end{cases}$$
* Hasilnya, Kimi K2 mencapai loss minimum (hampir tanpa degradasi perplexity) dalam skenario pencarian jarum dalam jerami (*Needle In A Haystack*) di sepanjang konteks 256K token.

---

### Bagian 2: Inovasi Optimizer - MuonClip

Salah satu kontribusi teknis terbesar tim Moonshot AI dalam paper Kimi K2 adalah penyelesaian ketidakstabilan pelatihan model skala besar melalui pengenalan pengoptimal **MuonClip**.

#### 1. Masalah Fatal pada Pengoptimal Muon Tradisional
Pengoptimal **Muon** (dikembangkan untuk mempercepat pembelajaran dengan memanfaatkan ortogonalitas gradien) memiliki efisiensi sampel yang luar biasa. Namun, ketika diterapkan pada model berskala ratusan miliar atau triliunan parameter dengan kedalaman tinggi, Muon memicu **ketidakstabilan latihan yang katastrofis**:
* **Pertumbuhan Nilai Singular**: Matriks proyeksi Query dan Key (QK) mengalami akumulasi aditif pada nilai singular tertingginya.
* **Logit Explosion**: Akumulasi ini memicu kenaikan norma hasil perkalian $Q K^T$. Nilai logit pada lapisan perhatian melonjak tajam (*logit explosion*).
* **Gradient Explosion**: Logit yang meledak menghasilkan gradien bernilai sangat besar saat fungsi Softmax dihitung, memicu lonjakan kerugian latihan (*loss spikes*) dan kegagalan total proses latih.

#### 2. Solusi Matematis: MuonClip
Untuk menjinakkan ketidakstabilan ini tanpa kehilangan keunggulan efisiensi Muon, Moonshot AI merancang **MuonClip**. MuonClip menambahkan tiga lapis pengaman:
1. **Weight Decay Orto-Konsisten**: Mengintegrasikan peluruhan bobot yang diselaraskan secara matematis dengan langkah pembaruan ortogonal.
2. **RMS Matching**: Membatasi norma perubahan bobot agar sebanding dengan Root-Mean-Square (RMS) dari bobot aktual.
3. **QK-Clip**: Memotong nilai ekstrem dari perkalian dot-product Query dan Key sebelum Softmax dengan batasan dinamis $C_{qk}$.

Secara matematis, pembaruan langkah Muon standar didefinisikan sebagai:
$$\Delta W = \text{orthonormalize}(G)$$
Di mana $G$ adalah gradien bobot $W$. Pada MuonClip, pembaruan bobot diatur menjadi:
$$\Delta W_{\text{clip}} = \text{clamp}\left( \text{orthonormalize}(G), -\gamma, \gamma \right)$$
$$W_{t+1} = W_t - \eta \cdot \left( \Delta W_{\text{clip}} + \lambda_{\text{decay}} W_t \right)$$
Di mana $\gamma$ dihitung secara adaptif berdasarkan RMS dari bobot saat ini:
$$\gamma = \alpha \cdot \sqrt{\frac{1}{N} \sum_{i,j} W_{ij}^2}$$

Dengan kombinasi ini, Kimi K2 berhasil dilatih pada **15.5 Triliun token** secara penuh tanpa ada satu pun lonjakan kehilangan (*zero loss spike*), menghemat waktu komputasi GPU hingga **28%** dibandingkan AdamW konvensional.

#### 3. Implementasi PyTorch: Lapisan Atensi dengan QK-Clip & Update MuonClip
Berikut adalah contoh implementasi PyTorch bersih yang mensimulasikan mekanisme penstabilan QK-clipping dan optimizer MuonClip:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKClippedAttention(nn.Module):
    def __init__(self, d_model, num_heads, clip_val=2.5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.clip_val = clip_val
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # Proyeksi Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Hitung skor atensi mentah
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Terapkan QK-clipping untuk mencegah logit explosion
        # Ini adalah inti pengaman pada Kimi K2 untuk menjaga stabilitas numerik
        scores_clipped = torch.clamp(scores, min=-self.clip_val, max=self.clip_val)
        
        # Hitung probabilitas atensi dan output
        attn_weights = F.softmax(scores_clipped, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape & proyeksi keluar
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(context)

class MuonClipOptimizer:
    """Simulasi Sederhana Pembaruan Parameter Gaya MuonClip."""
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, alpha=0.1):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha  # Faktor skala RMS clipping

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            G = p.grad.data
            W = p.data
            
            # Hanya lakukan ortogonalisasi jika parameter berbentuk matriks 2D (bobot linear)
            if len(W.shape) == 2:
                # 1. Hitung SVD untuk mendapatkan arah ortogonal (Inti Muon)
                try:
                    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
                    delta_W = torch.matmul(U, Vh)  # Matriks ortogonal terdekat dengan gradien
                except RuntimeError:
                    # Fallback jika SVD gagal konvergen karena masalah numerik
                    delta_W = torch.sign(G)
                
                # 2. Hitung RMS dari Bobot Aktual
                rms_W = torch.sqrt(torch.mean(W ** 2)) + 1e-8
                clip_bound = self.alpha * rms_W
                
                # 3. Terapkan RMS-Matched Clipping (Inti MuonClip)
                delta_W_clipped = torch.clamp(delta_W, min=-clip_bound.item(), max=clip_bound.item())
                
                # 4. Pembaruan dengan Weight Decay terintegrasi
                W.mul_(1.0 - self.lr * self.weight_decay)
                W.add_(delta_W_clipped, alpha=-self.lr)
            else:
                # Fallback ke pembaruan konvensional untuk bias atau vektor 1D
                p.data.add_(G + self.weight_decay * W, alpha=-self.lr)
```

---

### Bagian 3: Post-Training & Agentic Reinforcement Learning

Kekuatan utama Kimi AI terletak pada kemampuan beraksi otonom (*agentic execution*) yang dilatih melalui alur kerja pasca-pelatihan (*post-training*) yang sangat canggih.

#### 1. Agentic Data Synthesis Pipeline (Sintesis Data Agen)
Untuk melatih model agar terampil menggunakan perkakas komputer tanpa kesalahan sintaksis, Moonshot AI tidak mengandalkan data manual manusia yang lambat dan mahal. Mereka membangun **Agentic Data Synthesis Pipeline**:
* Menyimulasikan lingkungan digital dengan lebih dari **20.000 API/perkakas otonom** (terminal bash, editor teks, mesin pencari internet, kalkulator, kompiler, API basis data, dll.).
* Menginstruksikan agen otonom tingkat tinggi (model oracle) untuk menyelesaikan ribuan tugas rumit, lalu merekam log aksi, kesalahan, proses koreksi diri (*self-correction*), dan umpan balik lingkungan.
* Hasilnya adalah jutaan token data demonstrasi langkah-demi-langkah berkualitas tinggi yang mengajarkan model bagaimana bertahan dari kegagalan eksekusi dan memformulasi ulang rencana aksi.

#### 2. Joint Reinforcement Learning (RL) Pipeline
Kimi K2 dilatih menggunakan optimasi RL yang ditenagai oleh kombinasi dua sistem evaluasi ganjaran (*rewards*):

```
                       ┌───────────────────────────────────────┐
                       │           Generasi Kandidat           │
                       └───────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │   Verifiable Rewards   │                    │ Self-Critique Rubric   │
       │   (Sistem Evaluasi 1)  │                    │   (Sistem Evaluasi 2)  │
       ├────────────────────────┤                    ├────────────────────────┤
       │ - Eksekusi Kompiler    │                    │ - Evaluasi Kepatuhan   │
       │ - Interpreter Kode     │                    │ - Penilaian Kualitas   │
       │ - Verifikator Mat/Log  │                    │ - Pemeriksaan Aturan   │
       └────────────────┬───────┘                    └────────┬───────────────┘
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           ▼
                       ┌───────────────────────────────────────┐
                       │          Pembaruan Kebijakan          │
                       │           (Policy Update)             │
                       └───────────────────────────────────────┘
```

1. **Verifiable Rewards (Ganjaran Terverifikasi)**:
   * Diterapkan pada tugas-tugas objektif seperti matematika, logika formal, dan pemrograman.
   * Umpan balik langsung diperoleh dari eksekusi nyata: kompiler (misalnya GCC/Clang), interpreter Python, atau pemverifikasi formal. Model mendapatkan ganjaran positif jika kodenya berhasil dikompilasi, lolos *unit testing*, atau menghasilkan keluaran numerik matematika yang tepat.
2. **Self-Critique Rubric Rewards (Ganjaran Rubrik Kritik Mandiri)**:
   * Diterapkan pada tugas-tugas subjektif atau instruksi rumit yang tidak memiliki metrik evaluasi biner (seperti penulisan kreatif, keandalan gaya bahasa, atau kepatuhan sistem instruksi).
   * Model penilai (*critic model*) dilatih untuk mengevaluasi jawaban berdasarkan rubrik kepatuhan yang ketat ( checklist kualitas).

---

### Bagian 4: Evolusi Kimi K2 Thinking & Kimi K2.5

Evolusi terbaru Kimi AI membawa fungsionalitas penalaran ke tingkat yang belum pernah dicapai sebelumnya.

#### 1. Kimi K2 Thinking (Akhir 2025)
* **Penskalaan Inferensi**: Mirip dengan o1/R1, Kimi K2 Thinking mampu memperpanjang waktu berpikirnya (*inference-time compute*) secara dinamis untuk memecahkan masalah logika yang rumit sebelum menghasilkan jawaban akhir.
* **Stabilitas Rantai Alat**: Mampu mempertahankan stabilitas pemanggilan API dan alat secara berurutan hingga **200–300 langkah panggilan beruntun** tanpa mengalami degradasi perhatian (*loss of focus*) atau kegagalan loop tak berhingga.

#### 2. Kimi K2.5 (Awal 2026)
* **Multimodalitas Asli**: Kimi K2.5 dilatih menggunakan **15 Triliun token multimedia** (teks dan citra visual terintegrasi secara natif). Ini memungkinkan model memahami tangkapan layar (screenshots), diagram arsitektur, dan antarmuka visual IDE secara instan.
* **Empat Mode Interaksi Terpadu**:
  1. `Instant`: Generasi teks ultra-cepat untuk pertanyaan sederhana (latensi rendah).
  2. `Thinking`: Mode penalaran mendalam berbasis Chain-of-Thought untuk kode rumit dan matematika.
  3. `Agent`: Mode otonom satu agen untuk menyelesaikan tugas seperti pencarian web, analisis log, dan pengeditan berkas.
  4. `Agent Swarm`: Kolaborasi multi-agen di mana beberapa agen otonom khusus saling bertukar pesan untuk merampungkan proyek rekayasa perangkat lunak skala besar di dalam repositori.

---

### Bagian 5: Sinergi Sistem Ganda (Dual-System) di Moko IDE

Menggabungkan filosofi Kimi AI dan DeepSeek melahirkan arsitektur **Sistem Ganda (Dual-System)** yang sangat tangguh untuk Moko IDE.

#### 1. Perbandingan Karakteristik Filosofis

| Dimensi | Filosofi DeepSeek (Otak Penalaran) | Filosofi Kimi AI (Tangan & Mata Eksekutor) |
| :--- | :--- | :--- |
| **Kekuatan Utama** | Penalaran logis murni, matematika olimpiade, sintaksis optimal. | Penanganan konteks ultra-panjang, orkestrasi ribuan alat eksternal. |
| **Metode RL** | GRPO (tanpa model critic terpisah, efisiensi memori ekstrem). | Gabungan Verifiable & Self-Critique Rubric Rewards. |
| **Optimasi Inferensi** | Multi-Token Prediction (MTP) untuk kecepatan generasi. | Agent Swarm & Thinking-time scaling untuk stabilitas rantai aksi. |

#### 2. Desain Arsitektur Sistem Ganda Moko IDE

```
                     ┌────────────────────────────────┐
                     │   Permintaan Pengguna (User)   │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │    SISTEM 2 (Otak - DeepSeek)  │
                     │  - Penalaran CoT Mendalam      │
                     │  - Pembuatan Rencana Kerja     │
                     │  - Penyusunan Kasus Uji (Test) │
                     └───────────────┬────────────────┘
                                     │ (Rencana & Test Case)
                                     ▼
 ┌───────────────┐   ┌────────────────────────────────┐
 │  Pengetahuan  │   │    SISTEM 1 (Tangan - Kimi)    │
 │   Kode Moko   │──>│  - Pencarian Berkas (Anchor)   │
 │ (Anchor-RAG)  │   │  - Pengeditan Berkas Kode      │
 └───────────────┘   │  - Eksekusi Perintah Terminal  │
                     └───────────────┬────────────────┘
                                     │ (Umpan Balik / Hasil Eksekusi)
                                     ▼
                     ┌────────────────────────────────┐
                     │    SISTEM 2 (Guard - DeepSeek) │
                     │  - Analisis Log Kesalahan      │
                     │  - Code Review & Verifikasi    │
                     └───────────────┬────────────────┘
                                     │ (Koreksi Logika jika Gagal)
                                     ├──────────────────────────────┐
                                     ▼ (Sukses)                     ▼ (Gagal)
                     ┌────────────────────────────────┐   ┌───────────────────┐
                     │    Commit Git & Laporkan       │   │ Perintahkan Ulang │
                     │       ke Pengguna              │   │   Sistem 1        │
                     └────────────────────────────────┘   └───────────────────┘
```

#### 3. Implementasi Simulasi Kolaborasi Sistem Ganda Moko IDE
Berikut adalah simulasi Python lengkap yang menunjukkan bagaimana Moko IDE menggabungkan komponen nyata proyek (`moko_code_knowledge` untuk penelusuran repositori dan `moko_llm_runtime_guard` sebagai penjamin runtime) ke dalam alur koordinasi ganda ini.

```python
import time
import sys
import os

# Simulasi komponen proyek nyata
from moko_code_knowledge import tokenize, GEOMETRY_SNIPPET, TRIGONOMETRY_SNIPPET
from moko_llm_runtime_guard import LLMRuntimeGuard, normalize_server_status

class MokoDualSystemIDE:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        # Inisialisasi basis pengetahuan moko (Anchor-based RAG)
        self.knowledge_snippets = [GEOMETRY_SNIPPET, TRIGONOMETRY_SNIPPET]
        
    def system2_reason_plan(self, user_prompt: str) -> dict:
        """Sistem 2 (Gaya DeepSeek): Menganalisis masalah logika secara mendalam dan merancang rencana."""
        print("\n[Sistem 2 - OTAK] Memulai Analisis Penalaran (Chain-of-Thought)...")
        time.sleep(0.5)
        
        # Mengekstrak fokus token untuk pencarian pengetahuan (Anchor-based Retrieval)
        focus_tokens = set(tokenize(user_prompt))
        best_snippet = None
        best_score = -1
        
        for snip in self.knowledge_snippets:
            score = snip.score(focus_tokens)
            if score > best_score:
                best_score = score
                best_snippet = snip
                
        print(f"[Sistem 2 - OTAK] Pengetahuan relevan ditemukan: ID={best_snippet.snippet_id if best_snippet else 'None'} (Skor={best_score})")
        
        # Merumuskan rencana tindakan taktis
        plan = {
            "intent": "implement_trigonometry_feature" if "trigonometri" in focus_tokens else "implement_geometry_feature",
            "required_snippet_code": best_snippet.code if best_snippet else "",
            "steps": [
                "1. Cari berkas target matematika di repositori.",
                "2. Sisipkan fungsi pustaka pembantu matematika.",
                "3. Tulis kode pengujian otomatis (unit test) untuk memverifikasi akurasi.",
                "4. Jalankan pengujian di terminal workspace."
            ],
            "expected_test_output": "test_passed"
        }
        return plan

    def system1_execute_agent(self, plan: dict) -> tuple[bool, str]:
        """Sistem 1 (Gaya Kimi AI): Mengambil tindakan otonom di ruang kerja berdasarkan rencana."""
        print("\n[Sistem 1 - TANGAN] Menerima Rencana. Menginisialisasi Siklus Tindakan Agen...")
        time.sleep(0.5)
        
        # Langkah 1 & 2: Pengeditan berkas simulasi
        target_file = os.path.join(self.workspace_dir, "moko_math_runtime.py")
        print(f"[Sistem 1 - TANGAN] Memodifikasi berkas: {target_file}")
        
        # Tulis kode beserta snippet bantuan dari Sistem 2
        with open(target_file, "w") as f:
            f.write("# BERKAS KODE HASIL GENERASI OTOMATIS MOKO IDE\n")
            f.write("import math\n\n")
            f.write(plan["required_snippet_code"] + "\n\n")
            # Tulis fungsi utama fitur baru
            f.write("def hitung_nilai_proyeksi(sudut_derajat: float, amplitudo: float) -> float:\n")
            # Simulasi bug sengaja jika prompt pemicu mengandung kata 'bug'
            f.write("    # Sengaja menulis bug typo matematika jika diperlukan\n")
            f.write("    return amplitudo * hitung_cos_derajat(sudut_derajat)  # Penggunaan library helper\n")
            
        print("[Sistem 1 - TANGAN] Menulis unit test untuk verifikasi...")
        test_file = os.path.join(self.workspace_dir, "test_moko_math.py")
        with open(test_file, "w") as f:
            f.write("from moko_math_runtime import hitung_nilai_proyeksi\n")
            f.write("def test_output():\n")
            f.write("    hasil = hitung_nilai_proyeksi(60.0, 10.0)\n")
            f.write("    # Cos(60) = 0.5. Amplitudo 10. Hasil harus mendekati 5.0\n")
            f.write("    assert abs(hasil - 5.0) < 1e-5\n")
            f.write("    print('UNIT TEST PASSED SUCCESSFUL')\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    test_output()\n")

        # Langkah 4: Eksekusi simulasi run-test di terminal
        print("[Sistem 1 - TANGAN] Mengeksekusi pengujian di terminal lokal...")
        time.sleep(0.5)
        try:
            # Jalankan test otonom di subproses terminal
            import subprocess
            res = subprocess.run([sys.executable, test_file], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                return True, "Eksekusi berhasil:\n" + res.stdout
            else:
                return False, f"Eksekusi gagal (Return Code {res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        except Exception as e:
            return False, f"Kesalahan internal terminal: {str(e)}"

    def run_guard_and_review(self, success: bool, output_log: str) -> str:
        """Sistem 2 bertindak sebagai Runtime Guard (Gaya DeepSeek Guard) untuk memvalidasi keluaran."""
        print("\n[Sistem 2 - GUARD] Memulai Evaluasi Keamanan & Kualitas Kode...")
        
        # Menyiapkan provider status server simulasi
        def mock_status():
            return {"ready": True, "status": "online"}
            
        # Inisialisasi LLMRuntimeGuard untuk memeriksa kepatuhan runtime
        guard = LLMRuntimeGuard(
            status_provider=mock_status,
            llm_generate=lambda prompt: "LLM GUARD VERIFIED: Aman dari kerentanan injeksi kode.",
            fallback_generate=lambda prompt: "TEMPLATE GUARD: Pemeriksaan dasar lolos."
        )
        
        guard_res = guard.generate("Review log eksekusi kode matematika.")
        print(f"[Sistem 2 - GUARD] Status Pengawal: {guard_res.message}")
        
        if success:
            print("[Sistem 2 - GUARD] Hasil validasi: Sukses dan siap untuk dilakukan commit!")
            return "SUCCESS_COMMIT_READY"
        else:
            print("[Sistem 2 - GUARD] Terdeteksi kegagalan eksekusi! Merumuskan strategi perbaikan...")
            # Menganalisis kesalahan log (Self-Correction loop)
            return "REPAIR_INSTRUCTION"

# Demonstrasi Skenario Kerjasama
if __name__ == "__main__":
    import tempfile
    
    print("=== SIMULASI KOORDINASI SISTEM GANDA MOKO IDE v5 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        ide = MokoDualSystemIDE(tmpdir)
        
        # Skenario Permintaan Pengguna: Meminta implementasi trigonometri
        prompt_user = "Buat sistem perhitungan trigonometri sudut derajat untuk pergerakan kamera."
        
        # 1. Perencanaan Penalaran oleh Sistem 2
        rencana_kerja = ide.system2_reason_plan(prompt_user)
        
        # 2. Eksekusi Aksi oleh Sistem 1
        sukses, log_eksekusi = ide.system1_execute_agent(rencana_kerja)
        print(f"Log Output Terminal:\n{log_eksekusi}")
        
        # 3. Peninjauan & Verifikasi oleh Guard Sistem 2
        keputusan = ide.run_guard_and_review(sukses, log_eksekusi)
        
        print(f"\nStatus Akhir Sinergi Sistem Ganda: {keputusan}")
        print("====================================================")
```

Dengan arsitektur sinergis ini, Moko IDE tidak hanya bertindak sebagai asisten penulisan baris kode biasa, melainkan bertransformasi menjadi **insinyur AI otonom** yang mampu merancang, menulis, mengetes, dan menjamin kualitas seluruh basis kode di repositori secara mandiri dan bebas halusinasi.
