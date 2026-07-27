# MOKO Research Document 15: Onion Search Integration
====================================================

## 1. Pendahuluan
MOKO OS memerlukan kemampuan untuk melakukan pemindaian data (data scanning) di jaringan Darkweb (.onion) untuk mendukung analisis keamanan dan intelijen ancaman. Dokumen ini merinci implementasi sistem pencarian Onion yang terintegrasi dengan ekosistem MOKO.

## 2. Arsitektur
Sistem ini terdiri dari empat komponen utama:
1.  **OnionSearchTool**: Mesin scraping multi-engine yang mendukung Ahmia, Torch, Haystak, dan Excavator.
2.  **TorBotIntegrator**: Alat "pertajam" informasi yang melakukan verifikasi link, ekstraksi metadata, dan email harvesting.
3.  **Intent Router Expansion**: Penambahan kelas intent `DARKWEB` untuk mendeteksi kueri yang memerlukan akses ke jaringan Tor.
4.  **Integration Layer**: Integrasi ke `RAGAgent` dan `AnalystNode` untuk pengayaan konteks real-time.

## 3. Implementasi Teknis

### 3.1 OnionSearchTool & TorBotIntegrator
- **OnionSearchTool**: Bertanggung jawab mencari link awal dari berbagai search engine darkweb.
- **TorBotIntegrator**: Bertanggung jawab memverifikasi apakah link `.onion` masih aktif dan mengambil intelijen tambahan (seperti email dan judul halaman yang sebenarnya).
- **SOCKS5 Proxy**: Menggunakan `socks5h://127.0.0.1:9050` (konfigurasi default Tor) untuk melakukan request ke jaringan onion secara aman.

### 3.2 Intent Classification
Intent `DARKWEB` dideteksi berdasarkan kata kunci seperti:
- `onion`, `.onion`, `darkweb`, `darknet`
- `ahmia`, `torch`, `haystak`
- `leak data`, `data scanning darkweb`

### 3.3 Alur Kerja (Workflow)
1.  User bertanya: "Cari data bocor untuk domain example.com di darkweb".
2.  `IntentFirstRouter` mengklasifikasikan sebagai `DARKWEB`.
3.  `AnalystNode` memicu `OnionSearchTool.search_all()`.
4.  `OnionSearchTool` mencari link, lalu memanggil `TorBotIntegrator.sharpen()` untuk memverifikasi link aktif.
5.  `TorBot` mengekstrak metadata dan email jika ditemukan.
6.  Hasil yang sudah "tajam" disuntikkan ke konteks LLM.

## 4. Keamanan dan Etika
- Alat ini bersifat pasif (hanya pencarian/scraping).
- Pengguna disarankan menggunakan Tor Browser atau layanan Tor lokal untuk anonimitas maksimal.
- MOKO tidak menyimpan data mentah dari darkweb kecuali jika diinstruksikan untuk di-ingest ke OMNI Memory.

## 5. Pengembangan Masa Depan
- Integrasi dengan alat komunitas yang lebih kompleks (misal: `OnionSearch` oleh megadose).
- Kemampuan untuk melakukan "deep crawl" pada link yang ditemukan.
- Otomatisasi pengunduhan leak data yang ditemukan.
