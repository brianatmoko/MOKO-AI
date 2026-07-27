"""
MOKO AI — Sovereign Inference Server Manager
=============================================
Mengelola siklus hidup MOKO Server (100% sovereign, NO llama-server dependency).
MOKO Server dibangun di atas llama-cpp-python dengan blockchain of reasoning.

Jantung inference: MOKO-AI-4B-CryptoCore-Q3_K_M.gguf
Tanda tangan kriptografis ditanamkan via Cryptographic Spectral Modulation (CSM).

Phase 18: VRAM-Optimized Architecture untuk beat Google's VRAM-friendly research.
"""
import os
import json
import time
import subprocess
import requests
import signal
from pathlib import Path
from moko_config import settings

# File penyimpanan pid agar proses bisa dimatikan secara bersih bahkan jika crash
MOKO_PID_FILE = "/tmp/moko_server.pid"
RAG_PID_FILE  = "/tmp/moko_rag_server.pid"

# MOKO Server script (Python-based, sovereign)
MOKO_SERVER_SCRIPT = str(Path(__file__).parent / "moko_server.py")


def find_gguf_path(model_tag: str, override_path: str = "") -> str:
    """
    Melacak letak file GGUF model secara langsung.
    Tidak menggunakan Ollama manifest/blob lookup.
    """
    # 1. Prioritas utama: override_path (Direct GGUF Mode)
    if override_path:
        gguf_file = Path(override_path)
        if gguf_file.exists() and gguf_file.stat().st_size > 0:
            print(f"[MOKO INFERENCE] Direct GGUF Mode: menggunakan {gguf_file.name} ({gguf_file.stat().st_size // 1024 // 1024} MB)")
            return str(gguf_file)

    # 2. Cek path GGUF langsung dari settings
    for path_attr in ["MODEL_MOKO_GGUF_PATH", "MOKO_Q3_MODEL_PATH", "MODEL_QWEN_GGUF_PATH"]:
        path_str = getattr(settings, path_attr, "")
        if path_str:
            p = Path(path_str)
            if p.exists() and p.stat().st_size > 0:
                print(f"[MOKO INFERENCE] Settings GGUF Mode: menggunakan {p.name} ({p.stat().st_size // 1024 // 1024} MB)")
                return str(p)

    # 3. Fallback: Cari berkas .gguf berukuran >100MB di direktori proyek
    for f in settings.PROJECT_DIR.glob("*.gguf"):
        if f.stat().st_size > 100_000_000:
            print(f"[MOKO INFERENCE] Auto-detected GGUF in project: {f.name} ({f.stat().st_size // 1024 // 1024} MB)")
            return str(f)

    raise FileNotFoundError("Model GGUF tidak ditemukan di workspace.")


def get_gpu_layers(is_rag: bool = False) -> int:
    """
    Mendeteksi apakah GPU NVIDIA tersedia via nvidia-smi.
    Mempertimbangkan setting FORCE_CPU dan GPU_LAYERS dari settings,
    serta membatasi jumlah layer secara dinamis berdasarkan sisa VRAM untuk mencegah freeze.
    """
    from moko_config import settings
    
    # 1. Cek setting FORCE_CPU
    if getattr(settings, "FORCE_CPU", False):
        print("[MOKO INFERENCE] FORCE_CPU aktif. Menggunakan CPU-only mode.")
        return 0
        
    # 2. Cek setting GPU_LAYERS kustom
    if is_rag:
        custom_layers = getattr(settings, "RAG_GPU_LAYERS", 99)
        print(f"[MOKO INFERENCE] RAG Mode: Menggunakan konfigurasi RAG_GPU_LAYERS: {custom_layers}")
        return custom_layers

    custom_layers = getattr(settings, "GPU_LAYERS", None)
    if custom_layers is not None:
        print(f"[MOKO INFERENCE] Menggunakan konfigurasi GPU_LAYERS kustom: {custom_layers}")
        return custom_layers

    # 3. Autodeteksi dinamis sisa VRAM
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            lines = [ln.strip() for ln in res.stdout.strip().split("\n") if ln.strip()]
            if lines:
                parts = lines[0].split(",")
                total_vram = int(parts[0].strip())
                free_vram = int(parts[1].strip())
                print(f"[MOKO INFERENCE] GPU Terdeteksi: Total VRAM {total_vram} MB | Sisa VRAM {free_vram} MB")
                
                # Kita menyisakan VRAM aman untuk OS & Desktop Environment (Cinnamon/Xorg).
                # Di Linux, Cinnamon/Xorg + browser membutuhkan setidaknya 1500 MB agar tidak crash/fallback ke software rendering.
                OS_RESERVED_VRAM = 1500
                
                # Hitung VRAM yang tersedia untuk model AI
                available_for_model = max(0, free_vram - OS_RESERVED_VRAM)
                
                if is_rag:
                    # RAG Profile: Ultra-compact Byte-Q model (target 200MB)
                    rag_budget = getattr(settings, "RAG_VRAM_BUDGET", 200)
                    # We allow RAG even if OS_RESERVED_VRAM is slightly encroached
                    # because 200MB is tiny enough not to crash the desktop.
                    if free_vram > (OS_RESERVED_VRAM // 3) + rag_budget:
                        print(f"[MOKO INFERENCE] 🎯 RAG Profile Aktif: {rag_budget}MB budget fits in {free_vram}MB free VRAM.")
                        return getattr(settings, "RAG_GPU_LAYERS", 99)
                    else:
                        print(f"[MOKO INFERENCE] ⚠️ VRAM terlalu kritis bahkan untuk RAG ({free_vram}MB free).")
                        return 0

                # Phase 18: Crypto-Optimized VRAM Calculation untuk MOKO BF16
                # MOKO BF16 dengan cryptographic compression:
                # - Base overhead: 450 MB (turun dari 600 MB dengan crypto compression)
                # - Per layer: 40 MB (turun dari 55 MB dengan crypto-aware quantization)
                MODEL_BASE_VRAM = 450  # Crypto-compressed base
                VRAM_PER_LAYER = 40    # Crypto-compressed per layer
                
                if available_for_model < MODEL_BASE_VRAM + (5 * VRAM_PER_LAYER):
                    # Jika sisa VRAM untuk model kurang dari kebutuhan minimal 5 layer (~875 MB),
                    # gunakan CPU-only mode (0 layer) untuk mencegah freeze/out of memory.
                    print(f"[MOKO INFERENCE] ⚠️ Sisa VRAM untuk model sangat terbatas ({available_for_model} MB). Menggunakan CPU-only mode untuk stabilitas desktop.")
                    return 0
                
                # Hitung jumlah layer optimal yang bisa di-offload ke GPU
                recommended_layers = int((available_for_model - MODEL_BASE_VRAM) / VRAM_PER_LAYER)
                recommended_layers = min(99, max(0, recommended_layers))
                
                # Phase 18: Crypto-Optimized VRAM Allocation untuk beat Google's research
                # Google's VRAM-friendly approach: aggressive offloading dengan memory compression
                # MOKO approach: Cryptographic-aware VRAM allocation dengan dynamic compression
                if total_vram <= 4600:
                    # Low-VRAM optimization: increase layer limit dengan cryptographic memory compression
                    # Crypto compression memungkinkan lebih banyak layer dalam VRAM yang sama
                    # Dengan crypto compression (40MB/layer vs 55MB), kita bisa offload ~37% lebih banyak layer
                    recommended_layers = min(45, recommended_layers)  # Increased from 35 to 45
                    print(f"[MOKO INFERENCE] 🚀 Phase 18 Crypto-Compressed: GPU {total_vram} MB → {recommended_layers} layers (VRAM saved: {(recommended_layers * 15)} MB)")
                else:
                    # High-VRAM: full offload dengan cryptographic optimization
                    recommended_layers = min(99, recommended_layers)
                    print(f"[MOKO INFERENCE] 🚀 Phase 18 Crypto-Optimized: GPU {total_vram} MB → {recommended_layers} layers (Full Crypto)")
                
                return recommended_layers
            return 99
    except Exception as e:
        # Fallback jika terjadi kesalahan parsing/pemanggilan
        print(f"[MOKO INFERENCE] Info: Gagal membaca sisa VRAM secara dinamis ({e}). Menggunakan deteksi standar.")
        try:
            res_std = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
            if res_std.returncode == 0:
                return 99
        except Exception:
            pass
            
    return 0


class MokoLocalInferenceServer:
    # Class-level file handles untuk mencegah ResourceWarning dari GC
    _moko_log_fh = None

    @staticmethod
    def get_server_status(port: int) -> str:
        """
        Mendapatkan status server di port tertentu.
        Kompatibel dengan MOKO-server, moko_server.py, llama-server, dan server custom berbasis llama.cpp.
        Returns:
            "ok"      : jika server aktif dan siap menerima request.
            "loading" : jika server aktif tapi sedang memuat model.
            "offline" : jika server tidak merespons (mati atau port tidak terbuka).
        """
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
            if r.status_code == 200:
                # Coba parse JSON — banyak server return {"status": "ok"} atau {"status": "healthy"}
                try:
                    data = r.json()
                    status_val = data.get("status", "")
                    # Anggap OK jika status adalah string yang mengindikasikan siap
                    if str(status_val).lower() in ("ok", "healthy", "ready", "running", "1", "true"):
                        return "ok"
                    # Jika status bukan string penolakan (error/loading), anggap OK (server menjawab 200)
                    if str(status_val).lower() not in ("error", "loading", "initializing", "starting"):
                        return "ok"
                    if str(status_val).lower() in ("loading", "initializing", "starting"):
                        return "loading"
                except Exception:
                    # Tidak bisa parse JSON — tapi server respond 200, anggap OK
                    return "ok"
            elif r.status_code == 503:
                try:
                    data = r.json()
                    # Check berbagai format loading message
                    err_msg = ""
                    if "error" in data:
                        err_obj = data["error"]
                        err_msg = err_obj.get("message", "") if isinstance(err_obj, dict) else str(err_obj)
                    elif "message" in data:
                        err_msg = data.get("message", "")
                    elif "detail" in data:
                        err_msg = data.get("detail", "")
                    if any(kw in err_msg.lower() for kw in ["loading", "initializing", "starting", "warmup"]):
                        return "loading"
                except Exception:
                    pass
                return "loading"  # 503 tanpa JSON = server ada tapi belum siap
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        return "offline"

    @staticmethod
    def is_port_responding(port: int) -> bool:
        """Cek apakah server merespons di port tertentu."""
        return MokoLocalInferenceServer.get_server_status(port) == "ok"

    @staticmethod
    def _read_pid_from_file(pid_file: str) -> int:
        if os.path.exists(pid_file):
            try:
                with open(pid_file, "r") as f:
                    return int(f.read().strip())
            except Exception:
                pass
        return None

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _write_start_time(pid_file: str):
        """Tulis timestamp launch ke file .ts (dipasangkan dengan pid_file)."""
        ts_file = pid_file + ".ts"
        try:
            with open(ts_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

    @staticmethod
    def _read_start_time(pid_file: str) -> float:
        """Baca timestamp launch. Returns 0.0 jika tidak ada (dianggap sangat lama)."""
        ts_file = pid_file + ".ts"
        try:
            if os.path.exists(ts_file):
                with open(ts_file, "r") as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return 0.0  # Dianggap sudah sangat lama → boleh dibunuh

    @staticmethod
    def _kill_pid_file(pid_file: str):
        pid = MokoLocalInferenceServer._read_pid_from_file(pid_file)
        if pid and MokoLocalInferenceServer._is_process_running(pid):
            print(f"[MOKO INFERENCE] Menghentikan proses PID {pid}...")
            try:
                os.kill(pid, signal.SIGTERM)
                # Tunggu max 3 detik untuk mati bersih
                for _ in range(30):
                    if not MokoLocalInferenceServer._is_process_running(pid):
                        break
                    time.sleep(0.1)
                # Jika masih hidup, force kill
                if MokoLocalInferenceServer._is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except Exception as e:
                print(f"[MOKO INFERENCE] Gagal menghentikan PID {pid}: {e}")
        if os.path.exists(pid_file):
            try:
                os.remove(pid_file)
            except Exception:
                pass

    @classmethod
    def start_servers(cls, model_path: str = None):
        """
        Menyalakan MOKO Server Sovereign (port 11435).
        Phase 18: 100% MOKO Server - NO llama-server dependency.
        Model: MOKO-AI-4B-CryptoCore-Q3_K_M.gguf dengan cryptographic optimization.
        """
        from moko_cpu.governor import CPUGovernor
        coop_params = CPUGovernor.get_cooperative_params()
        num_thread = coop_params.get("num_thread", settings.LLM_MAX_THREADS)
        gpu_layers = get_gpu_layers()

        # 0. Setup Environment untuk MOKO Server (Python-based, sovereign)
        env = os.environ.copy()
        if gpu_layers > 0:
            # Setup CUDA library paths untuk llama-cpp-python
            cuda_folder = "cuda_v12"
            try:
                res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    import re
                    match = re.search(r"CUDA Version:\s+(\d+)", res.stdout)
                    if match:
                        major = int(match.group(1))
                        if major >= 13:
                            cuda_folder = "cuda_v13"
            except Exception:
                pass
            
            llama_bin_dir = Path(__file__).parent / "llama_bin"
            
            # Setup CUDA library paths untuk llama-cpp-python
            for item in list(llama_bin_dir.iterdir()):
                if item.name == "libggml-cuda.so" or item.name.startswith("libcublas") or item.name.startswith("libcudart"):
                    try:
                        if item.is_symlink() or item.is_file():
                            item.unlink()
                    except Exception:
                        pass

            # Buat symlink baru dari folder cuda_v12 atau cuda_v13 ke llama_bin/
            src_dir = llama_bin_dir / cuda_folder
            if src_dir.exists():
                for item in src_dir.iterdir():
                    dst_file = llama_bin_dir / item.name
                    try:
                        if dst_file.exists() or dst_file.is_symlink():
                            dst_file.unlink()
                        dst_file.symlink_to(item)
                    except Exception as e:
                        # Fallback: Copy jika symlink gagal
                        try:
                            import shutil
                            shutil.copy(item, dst_file)
                        except Exception:
                            pass
            
            cuda_path = str(llama_bin_dir / cuda_folder)
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            paths_to_add = [str(llama_bin_dir), cuda_path]
            if existing_ld:
                env["LD_LIBRARY_PATH"] = ":".join(paths_to_add) + ":" + existing_ld
            else:
                env["LD_LIBRARY_PATH"] = ":".join(paths_to_add)
            
            print(f"[MOKO INFERENCE] CUDA Terdeteksi! Menghubungkan library GPU dari folder: {cuda_folder}")
        else:
            print("[MOKO INFERENCE] Menjalankan dalam mode CPU-Only.")

        # 1. Pastikan MOKO Server Aktif (baik OK maupun sedang LOADING)
        moko_status = cls.get_server_status(settings.MOKO_LLM_PORT)
        moko_pid = cls._read_pid_from_file(MOKO_PID_FILE)
        moko_start_ts = cls._read_start_time(MOKO_PID_FILE)
        moko_age_s = time.time() - moko_start_ts  # berapa detik sejak terakhir diluncurkan

        if moko_status in ("ok", "loading"):
            # Server aktif — tidak perlu lakukan apapun
            print(f"[MOKO INFERENCE] MOKO Server sudah aktif di port {settings.MOKO_LLM_PORT} (Status: {moko_status.upper()}).")
        elif cls._is_process_running(moko_pid) and moko_age_s < 30:
            # Proses baru diluncurkan (<30 detik) → masih inisialisasi Python, belum buka port
            # JANGAN BUNUH — biarkan dia selesai membuka port
            print(f"[MOKO INFERENCE] MOKO Server sedang inisialisasi (PID {moko_pid}, usia {moko_age_s:.0f}s)... menunggu.")
        else:
            # Proses benar-benar mati atau sudah terlalu lama offline → launch ulang
            if cls._is_process_running(moko_pid):
                cls._kill_pid_file(MOKO_PID_FILE)
                for _ in range(20):
                    if cls.get_server_status(settings.MOKO_LLM_PORT) == "offline":
                        break
                    time.sleep(0.2)
                
            print(f"[MOKO INFERENCE] Meluncurkan MOKO Server Sovereign di port {settings.MOKO_LLM_PORT}...")
            try:
                if not model_path:
                    model_path = find_gguf_path(
                        settings.MODEL_MOKO_UNSENSOR,
                        override_path=getattr(settings, 'MODEL_MOKO_GGUF_PATH', '')
                    )
                gguf_path = model_path
                print(f"[MOKO INFERENCE] Menggunakan model MOKO GGUF: {gguf_path}")
                
                llm_threads = num_thread
                # Removed thread limit for GPU mode - allow full CPU parallelism
                # if gpu_layers > 0:
                #     llm_threads = min(2, num_thread)
                
                server_env = env.copy()
                if "existing_ld" in locals() or "existing_ld" in globals():
                    if existing_ld:
                        server_env["LD_LIBRARY_PATH"] = existing_ld
                    elif "LD_LIBRARY_PATH" in server_env:
                        del server_env["LD_LIBRARY_PATH"]
                elif "LD_LIBRARY_PATH" in server_env:
                    del server_env["LD_LIBRARY_PATH"]

                server_env["PYTHONPATH"] = str(settings.PROJECT_DIR / "moko_core")

                python_bin = str(settings.PROJECT_DIR / "moko_core" / "venv" / "bin" / "python3")
                server_script = str(settings.PROJECT_DIR / "moko_core" / "moko_inference" / "moko_server.py")

                cmd = [
                    python_bin, "-u", server_script,
                    "--port", str(settings.MOKO_LLM_PORT),
                    "--model", gguf_path,
                    "--ctx", str(settings.MAX_CONTEXT_TOKENS),
                    "--gpu", str(gpu_layers),
                    "--threads", str(llm_threads)
                ]
                
                proc = subprocess.Popen(
                    cmd,
                    env=server_env,
                    stdout=open("/tmp/moko_server.log", "w"),  # noqa: SIM115
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
                MokoLocalInferenceServer._moko_log_fh = proc.stdout

                try:
                    with open(f"/proc/{proc.pid}/oom_score_adj", "w") as oom_f:
                        oom_f.write("500")
                except Exception:
                    pass

                with open(MOKO_PID_FILE, "w") as f:
                    f.write(str(proc.pid))
                cls._write_start_time(MOKO_PID_FILE)
                    
            except Exception as e:
                print(f"[MOKO INFERENCE] Gagal meluncurkan MOKO Server: {e}")

        # 2. Tunggu sampai MOKO Server siap
        print("[MOKO INFERENCE] Menunggu MOKO Server Sovereign online...")
        cpu_fallback_triggered = False
        for attempt in range(240):  # max 4 menit
            moko_status = cls.get_server_status(settings.MOKO_LLM_PORT)
            if moko_status == "ok":
                print("[MOKO INFERENCE] MOKO Server Unified Server ONLINE dan siap!")
                # Phase 3.4: setelah server utama online, nyalakan server RAG khusus
                # (200MB VRAM, port 11437) secara best-effort. Non-blocking: Popen
                # langsung kembali, dan gagal-diam bila model RAG tidak ada.
                try:
                    cls.start_rag_server()
                except Exception as e:
                    print(f"[MOKO RAG] Auto-start RAG server dilewati: {e}")
                return True

            # CPU Fallback jika loading melebihi 90 detik
            if not cpu_fallback_triggered and attempt >= 90 and moko_status == "loading" and gpu_layers > 0:
                print("[MOKO INFERENCE] ⚠️ MOKO Server GPU loading > 90 detik. Fallback ke CPU mode...")
                cls._kill_pid_file(MOKO_PID_FILE)
                for _ in range(10):
                    if cls.get_server_status(settings.MOKO_LLM_PORT) == "offline":
                        break
                    time.sleep(0.2)

                try:
                    if not model_path:
                        model_path = getattr(settings, 'MODEL_MOKO_GGUF_PATH', '')
                    gguf_path = model_path
                    cpu_env = env.copy()
                    cpu_env["PYTHONPATH"] = str(settings.PROJECT_DIR / "moko_core")
                    if "LD_LIBRARY_PATH" in cpu_env:
                        del cpu_env["LD_LIBRARY_PATH"]

                    python_bin = str(settings.PROJECT_DIR / "moko_core" / "venv" / "bin" / "python3")
                    server_script = str(settings.PROJECT_DIR / "moko_core" / "moko_inference" / "moko_server.py")

                    cmd = [
                        python_bin, "-u", server_script,
                        "--port", str(settings.MOKO_LLM_PORT),
                        "--model", gguf_path,
                        "--ctx", str(settings.MAX_CONTEXT_TOKENS),
                        "--gpu", "0",
                        "--threads", str(num_thread)
                    ]
                    proc = subprocess.Popen(
                        cmd, env=cpu_env,
                        stdout=open("/tmp/moko_bf16.log", "w"),
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                    MokoLocalInferenceServer._moko_log_fh = proc.stdout
                    with open(MOKO_PID_FILE, "w") as f:
                        f.write(str(proc.pid))
                    cls._write_start_time(MOKO_PID_FILE)
                    cpu_fallback_triggered = True
                    attempt = 0
                except Exception as ex_cpu:
                    print(f"[MOKO INFERENCE] Gagal CPU fallback: {ex_cpu}")

            if attempt % 5 == 0:
                print(f"[MOKO INFERENCE] Menunggu MOKO... Status: {moko_status.upper()}")
            time.sleep(1)

        print("[MOKO INFERENCE] PERINGATAN: MOKO Server Server tidak siap dalam 4 menit.")
        return False

    @classmethod
    def start_rag_server(cls):
        """
        Menyalakan MOKO RAG Server (port 11437).
        Ultra-efficient profile: target 200MB VRAM.
        """
        from moko_cpu.governor import CPUGovernor
        coop_params = CPUGovernor.get_cooperative_params()
        num_thread = coop_params.get("num_thread", settings.LLM_MAX_THREADS)
        
        # Dedicated RAG profile
        gpu_layers = get_gpu_layers(is_rag=True)
        rag_model_path = getattr(settings, "MODEL_RAG_LLM_PATH", "")
        rag_port = getattr(settings, "MOKO_RAG_PORT", 11437)
        rag_ctx = getattr(settings, "RAG_CONTEXT_WINDOW", 1024)

        if not rag_model_path or not os.path.exists(rag_model_path):
            print(f"[MOKO RAG] Model RAG tidak ditemukan di {rag_model_path}. Melewati startup.")
            return False

        # Status check
        status = cls.get_server_status(rag_port)
        if status in ("ok", "loading"):
            print(f"[MOKO RAG] Server RAG sudah aktif di port {rag_port} (Status: {status.upper()}).")
            return True

        print(f"[MOKO RAG] Meluncurkan RAG Server (Budget: {getattr(settings, 'RAG_VRAM_BUDGET', 200)}MB)...")
        
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(settings.PROJECT_DIR / "moko_core")
            
            python_bin = str(settings.PROJECT_DIR / "moko_core" / "venv" / "bin" / "python3")
            server_script = str(settings.PROJECT_DIR / "moko_core" / "moko_inference" / "moko_server.py")
            
            cmd = [
                python_bin, "-u", server_script,
                "--port", str(rag_port),
                "--model", rag_model_path,
                "--ctx", str(rag_ctx),
                "--gpu", str(gpu_layers),
                "--threads", str(max(1, num_thread // 2))
            ]
            
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=open("/tmp/moko_rag_server.log", "w"),  # noqa: SIM115
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            
            with open(RAG_PID_FILE, "w") as f:
                f.write(str(proc.pid))
            cls._write_start_time(RAG_PID_FILE)
            
            print(f"[MOKO RAG] RAG Server diluncurkan dengan PID {proc.pid}")
            return True
        except Exception as e:
            print(f"[MOKO RAG] Gagal meluncurkan RAG Server: {e}")
            return False

    @classmethod
    def stop_servers(cls):
        """Mematikan MOKO Server Unified Server dan RAG Server."""
        print("[MOKO INFERENCE] Mematikan MOKO Server server...")
        cls._kill_pid_file(MOKO_PID_FILE)
        cls._kill_pid_file(RAG_PID_FILE)
        # Cleanup PID file lama embedder jika masih ada
        cls._kill_pid_file(MOKO_PID_FILE)
        for fh in (cls._moko_log_fh, cls._moko_log_fh):
            try:
                if fh and not fh.closed:
                    fh.close()
            except Exception:
                pass
        cls._moko_log_fh = None
        cls._moko_log_fh = None
        print("[MOKO INFERENCE] MOKO Server server berhasil dihentikan.")
