# 18 — Data & Teknik Coding Standar Industri untuk MOKO LLM

> **Tujuan:** Mengangkat MOKO IDE agar menulis kode setara "AI raksasa": mencari
> informasi, mengolahnya, lalu menyusunnya menjadi program yang lebih besar dan
> relevan. Dokumen ini merangkum temuan dari GitHub dan komunitas programmer,
> beserta cara MOKO menerapkannya secara konkret di pipeline.

---

## 1. Diagnosis: Kenapa Hasil Belum Setara Industri

Saat user meminta "kalkulator + rumus", log yang dipelajari MOKO sering **tidak
berhubungan** dengan permintaan. Akar masalahnya klasik pada agen coding pemula:

1. **Retrieve-everything antipattern.** Semua token intent dipakai sebagai
   konteks, sehingga sinyal penting (mis. "rumus", "geometri") tenggelam oleh
   kata umum ("buat", "untuk", "yang").
2. **Pengetahuan sempit & hardcoded.** Injeksi hanya untuk satu domain, tanpa
   basis pengetahuan yang bisa diperluas.
3. **Tidak ada dekomposisi.** Permintaan kompleks tidak dipecah menjadi
   sub-kebutuhan (aritmetika + geometri + statistika + finansial).

---

## 2. Temuan dari GitHub & Komunitas Programmer

### 2.1 Teknik Agen Coding (riset & praktik)
- **Task decomposition (Self-Planning, CodeChain).** Pemecah masalah besar
  menjadi sub-tugas adalah pembeda utama antara agen fungsional dan "autocomplete
  mahal". (arXiv 2508.00083 — *Survey on Code Generation with LLM-based Agents*.)
- **Anchor-based retrieval.** Ambil konteks berdasarkan sinyal **terkuat** pada
  query (nama simbol, istilah domain), bukan kemiripan mentah — hindari pencarian
  bising. (arXiv 2603.05344 — *Building AI Coding Agents for the Terminal*.)
- **Self-repair loop (test-driven).** Tulis kode → jalankan test/kompilator →
  baca error → perbaiki → ulang. (AddyOsmani — *Self-Improving Coding Agents*.)
- **Long-term memory / RAG.** Basis pengetahuan eksternal yang bisa di-retrieve
  melampaui batas context window. (arXiv 2508.11126 — *AI Agentic Programming*.)

### 2.2 Sumber Data Pelatihan Kode (open-source)
| Dataset | Penyedia | Kegunaan untuk MOKO |
|---------|----------|---------------------|
| The Stack / StarCoderData | bigcode-project | Korpus kode multi-bahasa untuk pretraining |
| CommitPackFT (OctoPack) | bigcode-project | Commit Git → instruksi berkualitas (SFT) |
| OpenCodeInstruct | NVIDIA | Instruksi + solusi kode skala besar (SFT) |
| Code-Feedback / OSS-Instruct | m-a-p / MagiCoder | Instruksi dengan umpan balik eksekusi |
| HumanEvalPack / MBPP | bigcode / Google | Benchmark evaluasi kualitas kode |

**Prinsip kualitas komunitas:** model kecil pada data **terkurasi berkualitas**
mengalahkan model besar pada data bising — utamakan deduplikasi, filter kualitas,
dan sinyal metadata.

---

## 3. Penerapan Konkret di MOKO (sudah diimplementasikan)

### 3.1 Knowledge Base Lintas-Domain — `moko_code_knowledge.py`
Corpus terkurasi berisi snippet siap-pakai per domain, masing-masing dengan
**anchor eksplisit**, **domain**, **import yang dibutuhkan**, dan **sumber**:
`geometri`, `trigonometri`, `statistika`, `finansial`, `konversi`, `algoritma`.

### 3.2 Anchor-Based Retrieval — `CodeKnowledgeBase.retrieve`
- Snippet hanya diambil bila anchor-nya cocok dengan fokus intent (skor ≥ 1).
- Hasil di-*rank* by anchor overlap dan dibatasi (`limit`) → mencegah
  retrieve-everything, sehingga pengetahuan selalu relevan dengan permintaan.

### 3.3 Komposisi Program (bukan template mentah)
- Engine (`moko_template_learning.py`) menyuntikkan fungsi helper dari beberapa
  domain sekaligus untuk permintaan kompleks, **plus import otomatis** yang
  dibutuhkan (mis. `import statistics`, `import math`).
- Output selalu diverifikasi bisa dikompilasi (syntactically valid).

### 3.4 Jejak Belajar yang Transparan
Metadata output kini mencatat apa yang benar-benar dipelajari & relevan:
```
# retrieval_focus: rumus, geometri, luas, statistika, bunga
# knowledge_sources: geometri, statistika, finansial
```

---

## 4. Roadmap Lanjutan Menuju Level Industri
1. **Perluas corpus** dari sumber di §2.2 (impor pola dari CommitPackFT / OSS-Instruct).
2. **Self-repair loop nyata** — sambungkan ke runner test agar MOKO memperbaiki
   error kompilasi/uji secara otomatis (lihat SHL di dok. 17).
3. **Dekomposisi eksplisit** — pecah intent kompleks menjadi rencana sub-tugas
   sebelum generasi (Self-Planning).
4. **Evaluasi berkelanjutan** dengan HumanEval/MBPP-style harness lokal.
