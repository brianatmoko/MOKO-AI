"""
MOKO OMNI RAG Ingestion Engine
================================
Injects high-quality programming and math reference documents into MOKO's RAG system (.moko_omni/).
Features:
  - 15 detailed programming reference articles (C++, Python, Rust, Go, systems)
  - 15 detailed mathematics reference articles (Calculus, Linear Algebra, stats, transforms)
  - Offline compatibility using deterministic embedding fallback if llama-server is offline.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "moko_core"))

from moko_config import settings
from moko_memory.disk_manager import DiskManager
from moko_agents.llm_engine import engine


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE ARTICLES DATABASE
# ─────────────────────────────────────────────────────────────────────────────

PROGRAMMING_ARTICLES = [
    (
        "cpp_memory_management.md",
        "C++ Memory Management: Stack vs Heap Allocation and RAII",
        "Dalam bahasa pemrograman C++, memori dibagi menjadi Stack dan Heap. Stack digunakan untuk alokasi otomatis yang cepat dengan cakupan scope yang jelas. Heap digunakan untuk alokasi dinamis menggunakan operator 'new'. "
        "Masalah utama heap adalah kerawanan terhadap kebocoran memori (memory leak). Konsep RAII (Resource Acquisition Is Initialization) mengikat siklus hidup resource dengan objek stack. "
        "Smart pointers seperti std::unique_ptr dan std::shared_ptr menerapkan RAII secara otomatis membebaskan memori ketika pointer keluar dari scope."
    ),
    (
        "cpp_virtual_destructors.md",
        "C++ Polymorphism: The Importance of Virtual Destructors",
        "Ketika merancang kelas turunan (polimorfisme) di C++, destructor di kelas induk (base class) wajib dideklarasikan sebagai 'virtual'. "
        "Jika tidak, penghapusan objek Derived melalui pointer Base* akan memicu undefined behavior di mana destructor Derived tidak dipanggil. "
        "Hal ini menyebabkan memory leak pada variabel anggota yang dikelola Derived di heap. Deklarasi virtual memastikan vtable memetakan pemanggilan destructor Derived dengan benar."
    ),
    (
        "python_gil_mechanics.md",
        "Python Global Interpreter Lock (GIL) and Concurrency",
        "Global Interpreter Lock (GIL) adalah mutex di CPython yang memastikan hanya ada satu thread OS yang mengeksekusi bytecode Python dalam satu waktu. "
        "GIL membatasi performa aplikasi multi-threaded pada tugas CPU-bound karena thread tidak bisa berjalan paralel di multi-core CPU. "
        "Untuk tugas CPU-bound, disarankan menggunakan modul 'multiprocessing' untuk mem-spawn proses baru dengan interpreter mandiri, atau menggunakan C-extensions untuk melepas GIL."
    ),
    (
        "python_asyncio_event_loop.md",
        "Python Asyncio: Asynchronous I/O and Event Loop Architecture",
        "Modul asyncio di Python memfasilitasi pemrograman konkuren satu thread menggunakan coroutine, async/await, dan event loop. "
        "Event loop berjalan melingkar memantau status I/O non-blocking (seperti socket jaringan atau pembacaan disk). "
        "Ketika coroutine melakukan operasi I/O dan memanggil 'await', kontrol dikembalikan ke event loop untuk menjalankan tugas lain. "
        "Ini sangat efisien untuk aplikasi I/O-bound dibanding thread-based concurrency yang memakan overhead memori besar."
    ),
    (
        "rust_ownership_borrowing.md",
        "Rust Ownership, Borrowing, and Lifetimes Architecture",
        "Sistem manajemen memori Rust berpusat pada konsep Ownership (kepemilikan) tanpa garbage collector. "
        "Aturan utamanya: setiap nilai memiliki satu owner; ketika owner keluar scope, nilai dibebaskan. "
        "Borrowing (peminjaman) mengizinkan akses ke data melalui reference: bisa banyak immutable reference (&T) sekaligus, ATAU hanya satu mutable reference (&mut T). "
        "Lifetimes (') adalah anotasi statis bagi compiler untuk menjamin tidak ada dangling reference di runtime."
    ),
    (
        "go_concurrency_channels.md",
        "Go Concurrency: Goroutines and CSP Channel Communications",
        "Bahasa pemrograman Go menerapkan teori Communicating Sequential Processes (CSP) menggunakan Goroutine dan Channels. "
        "Goroutine adalah green thread ringan yang dijadwalkan secara kooperatif oleh Go runtime di atas thread OS asli. "
        "Channels menyediakan sinkronisasi aman untuk mengirim dan menerima data antar goroutine tanpa lock manual (mutex). "
        "Buffered channels mengizinkan pengiriman asinkron sampai batas kapasitas terlampaui."
    ),
    (
        "cmake_target_based_design.md",
        "Modern CMake: Target-Based Build System Configurations",
        "Praktik CMake modern meninggalkan penetapan variabel inklusi global seperti include_directories() dan link_libraries(). "
        "Sebagai gantinya, gunakan desain berbasis Target: add_executable() atau add_library() mendefinisikan objek target. "
        "Gunakan target_include_directories() dan target_link_libraries() dengan modifier PRIVATE, PUBLIC, atau INTERFACE "
        "untuk mengontrol propagasi direktori include dan pustaka link ke target dependen lainnya secara bersih."
    ),
    (
        "git_merge_conflict_mechanics.md",
        "Git Version Control: Under the Hood of Merge Conflict Resolution",
        "Merge conflict terjadi saat Git mendeteksi perubahan pada baris yang sama di file yang sama dari dua komit berbeda yang sedang digabungkan. "
        "Git menandai file berkonflik dengan penanda khusus: <<<<<<< HEAD (branch saat ini), ======= (pembatas), dan >>>>>>> branch_name (branch target). "
        "Penyelesaian konflik membutuhkan pengeditan manual file tersebut untuk membuang penanda konflik dan mempertahankan baris kode yang benar sebelum melakukan git commit."
    ),
    (
        "docker_multistage_optimization.md",
        "Docker Containerization: Multi-stage Builds for Size Optimization",
        "Multi-stage builds di Docker memungkinkan penggunaan beberapa stage 'FROM' dalam satu Dockerfile. "
        "Tahap pertama (build stage) memuat SDK lengkap untuk mengompilasi kode program menjadi binary mandiri. "
        "Tahap kedua (runtime stage) hanya memuat base image minimal (seperti alpine atau scratch) dan menyalin binary hasil kompilasi dari stage sebelumnya. "
        "Ini menghasilkan image Docker akhir berukuran sangat kecil dan aman karena tidak menyertakan source code dan perkakas build."
    ),
    (
        "web_security_csrf_xss.md",
        "Web Security: Preventing Cross-Site Scripting (XSS) and CSRF",
        "XSS (Cross-Site Scripting) terjadi ketika penyerang berhasil menginjeksikan script client-side berbahaya ke halaman web yang dikunjungi user. Pencegahannya adalah sanitasi input dan HTML escaping. "
        "CSRF (Cross-Site Request Forgery) memanipulasi browser user untuk mengirimkan request tidak sah ke situs tempat user terotentikasi. "
        "Pencegahannya adalah menggunakan anti-CSRF tokens yang divalidasi di sisi server pada setiap transaksi state-changing."
    ),
    (
        "cpp_structure_alignment.md",
        "C++ Memory Layout: Structure Padding and Member Alignment",
        "Untuk efisiensi pembacaan memori oleh bus data CPU, compiler C++ melakukan alignment alamat memori variabel anggota di dalam struct. "
        "Jika variabel berukuran kecil bersebelahan dengan variabel besar, compiler menyisipkan byte kosong (padding). "
        "Susunan struct yang tidak teratur membuang memori RAM secara sia-sia. Menyusun variabel anggota dari tipe data terbesar (double, pointer) ke terkecil (char) meminimalkan padding."
    ),
    (
        "python_decorators_metaclasses.md",
        "Python Metaprogramming: Decorators vs Metaclasses",
        "Decorator adalah fungsi pembungkus yang menerima fungsi atau kelas lain dan memodifikasi perilakunya di runtime tanpa mengubah definisinya secara langsung. "
        "Metaclass adalah 'kelas dari kelas' yang mengatur bagaimana kelas instansiasi dideklarasikan dan dibangun. "
        "Decorator ideal untuk memodifikasi method atau wrapping cepat (seperti logging, caching). "
        "Metaclass digunakan untuk kontrol arsitektur yang ketat di tingkat pembuatan kelas (seperti ORM models)."
    ),
    (
        "networking_socket_programming.md",
        "Computer Networks: Low-level TCP/IP Socket Programming",
        "Socket programming adalah API untuk komunikasi antar host di jaringan TCP/IP. "
        "Server membuat socket, mengikatnya ke IP/Port menggunakan bind(), mendengarkan koneksi dengan listen(), dan menerima client dengan accept(). "
        "Client membuat socket dan menghubungi server menggunakan connect(). "
        "TCP menyediakan socket bertipe SOCK_STREAM yang berorientasi koneksi dan reliabel, sedangkan UDP bertipe SOCK_DGRAM yang berorientasi pesan tanpa jaminan pengiriman."
    ),
    (
        "software_patterns_observer.md",
        "Software Engineering: The Observer Design Pattern in Action",
        "Observer pattern adalah pola desain perilaku di mana suatu objek (Subject) memelihara daftar dependennya (Observers) "
        "dan memberi tahu mereka secara otomatis tentang setiap perubahan state, biasanya dengan memanggil salah satu metode mereka. "
        "Pola ini memisahkan Subject dari Observers secara longgar (loose coupling). "
        "Di Qt5, pola ini diimplementasikan secara native melalui mekanisme Signals and Slots untuk interaksi antar UI widgets."
    ),
    (
        "rust_memory_safety_unsafe.md",
        "Rust Memory Safety: Safe Code Guarantees and Unsafe Blocks",
        "Rust menjamin keamanan memori secara statis tanpa overhead runtime. Namun, untuk interaksi tingkat rendah dengan hardware atau FFI (Foreign Function Interface), Rust menyediakan keyword 'unsafe'. "
        "Di dalam blok unsafe, developer diizinkan melakukan dereferensi raw pointer, memanggil fungsi unsafe, dan memodifikasi variabel statis mutable. "
        "Developer wajib memastikan secara manual bahwa kode di dalam blok unsafe tetap memenuhi invariants keamanan memori Rust."
    )
]

MATH_ARTICLES = [
    (
        "linear_algebra_transformations.md",
        "Linear Algebra: Coordinate System Transformations and Rotation Matrices",
        "Transformasi linear memetakan vektor dari satu ruang koordinat ke ruang koordinat lain menggunakan perkalian matriks. "
        "Matriks rotasi R dalam ruang 2D memutar vektor sebesar sudut theta berlawanan arah jarum jam melalui rumus: "
        "R = [[cos(t), -sin(t)], [sin(t), cos(t)]]. "
        "Setiap transformasi linear mempertahankan operasi penjumlahan vektor dan perkalian skalar: "
        "T(u + v) = T(u) + T(v) dan T(c*u) = c*T(u). Ini mendasari grafika komputer 3D."
    ),
    (
        "matrix_eigenvalues_eigenvectors.md",
        "Linear Algebra: Eigenvalues and Eigenvectors Mechanics",
        "Untuk matriks persegi A berukuran n x n, vektor non-nol v disebut eigenvector jika memenuhi persamaan: A*v = lambda*v. "
        "Di sini, lambda adalah skalar yang disebut eigenvalue. "
        "Secara geometris, eigenvector tidak berubah arah di bawah transformasi A, hanya mengalami penskalaan sebesar lambda. "
        "Eigenvalue dihitung dengan mencari akar-akar persamaan karakteristik: det(A - lambda*I) = 0."
    ),
    (
        "calculus_integration_by_parts.md",
        "Calculus: Integration by Parts and Product Rule Derivation",
        "Integrasi parsial adalah teknik kalkulus untuk mengintegrasikan hasil perkalian dua fungsi. "
        "Teknik ini diturunkan dari aturan perkalian turunan (product rule): d(uv)/dx = u*(dv/dx) + v*(du/dx). "
        "Rumus dasarnya adalah: integral(u dv) = u*v - integral(v du). "
        "Pemilihan fungsi 'u' biasanya mengikuti aturan prioritas LIATE: Logarithmic, Inverse trigonometric, Algebraic, Trigonometric, Exponential."
    ),
    (
        "calculus_taylor_series.md",
        "Calculus: Taylor and Maclaurin Series Expansions",
        "Deret Taylor adalah representasi fungsi matematika sebagai jumlah tak hingga suku dari turunan-turunan fungsi tersebut di satu titik a. "
        "Rumus deret Taylor untuk fungsi f(x) di sekitar titik a adalah: f(x) = sum_{n=0}^{inf} (f^(n)(a) / n!) * (x - a)^n. "
        "Jika a = 0, deret tersebut dinamakan Deret Maclaurin. Deret ini digunakan komputer untuk menghitung nilai hampiran fungsi transenden seperti sin(x), cos(x), dan e^x."
    ),
    (
        "math_laplace_transform.md",
        "Advanced Mathematics: The Laplace Transform and ODE Solving",
        "Transformasi Laplace memetakan fungsi dari domain waktu t ke domain frekuensi kompleks s. "
        "Rumus integralnya adalah: L{f(t)} = integral_{0}^{inf} e^{-st} * f(t) dt. "
        "Transformasi ini sangat berguna untuk menyederhanakan penyelesaian persamaan diferensial biasa (Ordinary Differential Equations) "
        "karena operasi turunan d/dt di domain waktu berubah menjadi operasi perkalian dengan variabel s di domain frekuensi."
    ),
    (
        "math_fourier_analysis.md",
        "Advanced Mathematics: Fourier Series and Discrete Fourier Transform (DFT)",
        "Analisis Fourier menyatakan fungsi periodik sebagai penjumlahan gelombang sinus dan cosinus harmonik (frekuensi berbeda). "
        "Discrete Fourier Transform (DFT) menganalisis sinyal diskret menjadi komponen frekuensi penyusunnya dengan rumus: "
        "X_k = sum_{n=0}^{N-1} x_n * e^{-i*2*pi*k*n/N}. "
        "Fast Fourier Transform (FFT) adalah algoritma optimasi yang menghitung DFT dalam kompleksitas waktu O(N log N) dibanding O(N^2) standar."
    ),
    (
        "probability_distributions.md",
        "Probability Theory: Probability Density Functions (PDF) and CDF",
        "Fungsi Kepadatan Probabilitas (Probability Density Function - PDF) menjelaskan peluang variabel acak kontinu mengambil nilai tertentu. "
        "Fungsi Distribusi Kumulatif (Cumulative Distribution Function - CDF) menghitung peluang variabel bernilai kurang dari atau sama dengan x: "
        "F(x) = P(X <= x) = integral_{-inf}^{x} f(t) dt. "
        "Distribusi Normal (Gaussian) memiliki bentuk kurva lonceng simetris yang ditentukan oleh rata-rata (mean) dan standar deviasi."
    ),
    (
        "probability_bayes_theorem.md",
        "Probability Theory: Bayes' Theorem and Conditional Probability",
        "Teorema Bayes menghitung probabilitas kondisional suatu kejadian berdasarkan pengetahuan sebelumnya tentang kondisi terkait. "
        "Rumusnya adalah: P(A|B) = (P(B|A) * P(A)) / P(B). "
        "Di sini, P(A|B) adalah posterior probability, P(B|A) adalah likelihood, P(A) adalah prior probability, dan P(B) adalah marginal likelihood. "
        "Teorema ini mendasari algoritma Naive Bayes classifier dalam machine learning."
    ),
    (
        "optimization_gradient_descent.md",
        "Optimization: Gradient Descent Mathematics for Machine Learning",
        "Gradient descent adalah algoritma optimasi numerik orde pertama untuk menemukan nilai minimum lokal dari fungsi terdiferensiasi. "
        "Langkah pembaruan parameter w dihitung berlawanan dengan arah gradien fungsi loss L terhadap w: "
        "w_new = w_old - alpha * gradien(L(w)). "
        "Di sini, skalar alpha mewakili learning rate. Jika alpha terlalu besar, optimasi berosilasi; jika terlalu kecil, konvergensi lambat."
    ),
    (
        "matrix_multiplication_rules.md",
        "Linear Algebra: Matrix Multiplication Requirements and Inner Product",
        "Dua matriks A dan B dapat dikalikan (A * B) hanya jika jumlah kolom matriks A sama dengan jumlah baris matriks B. "
        "Jika A berukuran m x n dan B berukuran n x p, maka matriks hasil C akan berukuran m x p. "
        "Setiap elemen C_{i,j} dihitung sebagai perkalian titik (inner product) dari baris ke-i matriks A dan kolom ke-j matriks B: "
        "C_{i,j} = sum_{k=1}^{n} A_{i,k} * B_{k,j}."
    ),
    (
        "math_multivariable_calculus.md",
        "Calculus: Partial Derivatives and the Jacobian Matrix",
        "Dalam kalkulus multivariabel, turunan parsial mengukur laju perubahan fungsi terhadap satu variabel independen dengan menganggap variabel lain konstan. "
        "Untuk fungsi bernilai vektor F: R^n -> R^m, matriks Jacobian J berisi semua turunan parsial pertama: "
        "J_{i,j} = dF_i / dx_j. Matriks ini mewakili aproksimasi linear terbaik dari fungsi nonlinear di sekitar titik tertentu."
    ),
    (
        "math_differential_equations.md",
        "Differential Equations: First-order Linear ODE Solutions",
        "Persamaan diferensial biasa (Ordinary Differential Equation - ODE) tingkat satu memiliki bentuk umum: dy/dx + P(x)*y = Q(x). "
        "Persamaan ini dapat diselesaikan secara analitis menggunakan Faktor Integrasi I(x) = e^{integral P(x) dx}. "
        "Dengan mengalikan seluruh sisi dengan I(x), persamaan menyederhana menjadi d/dx (I(x)*y) = I(x)*Q(x), "
        "sehingga solusi umum didapat dengan mengintegrasikan kedua sisi terhadap x."
    ),
    (
        "math_number_theory.md",
        "Number Theory: Prime Numbers and Euclid's Greatest Common Divisor (GCD)",
        "Teori bilangan mempelajari sifat-sifat bilangan bulat. Bilangan prima adalah bilangan yang hanya habis dibagi 1 dan dirinya sendiri. "
        "Algoritma Euclid adalah metode efisien untuk menghitung Greatest Common Divisor (GCD) dari dua bilangan bulat a dan b. "
        "Metode ini didasarkan pada prinsip GCD(a, b) = GCD(b, a mod b) yang diulang secara rekursif hingga sisa pembagian (modulo) bernilai nol."
    ),
    (
        "math_combinatorics_permutations.md",
        "Combinatorics: Permutations and Combinations Formulae",
        "Kombinatorika menghitung susunan objek. Permutasi adalah susunan r objek dari n objek di mana urutan susunan diperhatikan: "
        "P(n, r) = n! / (n - r)!. "
        "Kombinasi adalah pilihan r objek dari n objek di mana urutan susunan tidak diperhatikan: "
        "C(n, r) = n! / (r! * (n - r)!). Kombinasi sering dilambangkan dengan koefisien binomial."
    ),
    (
        "math_graph_theory.md",
        "Graph Theory: Vertices, Edges, and Adjacency Matrices",
        "Graf G = (V, E) terdiri dari sekumpulan simpul (Vertices) V dan sekumpulan busur (Edges) E yang menghubungkan pasangan simpul. "
        "Matriks ketetanggaan (Adjacency Matrix) A untuk graf dengan n simpul adalah matriks n x n di mana "
        "A_{i,j} bernilai 1 jika terdapat busur yang menghubungkan simpul i dan j, dan bernilai 0 jika tidak ada. "
        "Graf digunakan untuk memodelkan jaringan komputer, struktur molekul, dan tautan memori kognitif."
    )
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGESTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion():
    print("=" * 60)
    print("MOKO OMNI RAG Ingestion Pipeline — Injecting Math & Programming")
    print("=" * 60)

    # Initialize DiskManager
    workspace = settings.WORKSPACE_DIR
    print(f"Workspace: {workspace}")
    print(f"Omni Directory: {settings.OMNI_DIR}")
    
    disk_mgr = DiskManager(workspace)
    
    # 1. Ingest Programming Articles
    print("\n--- Phase 1: Ingesting Programming & Systems Knowledge ---")
    prog_count = 0
    for filename, title, content in PROGRAMMING_ARTICLES:
        print(f"Ingesting {filename} ('{title}')...")
        try:
            # Generate embedding
            emb = engine.get_embedding(content)
            
            # Save using compatibility save_memory method
            res = disk_mgr.save_memory(
                text=f"{title}\n\n{content}",
                embedding=emb,
                domain="code",
                metadata={
                    "source": filename,
                    "path": f"manual://code/{filename}",
                    "timestamp": time.time(),
                    "source_type": "factual"
                }
            )
            if res and res[0] not in ("DEDUP_SKIP", "CONFIDENCE_LOCKED", "TOO_SHORT"):
                print(f"  -> SUCCESS saved to bucket: {res[0]}")
                prog_count += 1
            else:
                print(f"  -> SKIPPED (reason: {res[0] if res else 'unknown'})")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            
    # 2. Ingest Math Articles
    print("\n--- Phase 2: Ingesting Mathematics & Calculus Knowledge ---")
    math_count = 0
    for filename, title, content in MATH_ARTICLES:
        print(f"Ingesting {filename} ('{title}')...")
        try:
            # Generate embedding
            emb = engine.get_embedding(content)
            
            # Save using compatibility save_memory method
            res = disk_mgr.save_memory(
                text=f"{title}\n\n{content}",
                embedding=emb,
                domain="math",
                metadata={
                    "source": filename,
                    "path": f"manual://math/{filename}",
                    "timestamp": time.time(),
                    "source_type": "factual"
                }
            )
            if res and res[0] not in ("DEDUP_SKIP", "CONFIDENCE_LOCKED", "TOO_SHORT"):
                print(f"  -> SUCCESS saved to bucket: {res[0]}")
                math_count += 1
            else:
                print(f"  -> SKIPPED (reason: {res[0] if res else 'unknown'})")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("=" * 60)
    print(f"INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Programming articles saved: {prog_count}/{len(PROGRAMMING_ARTICLES)}")
    print(f"Math articles saved: {math_count}/{len(MATH_ARTICLES)}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
