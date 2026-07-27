"""
MOKO Cooperative Governor — Tiga Organ Bekerja Bersama
======================================================
CPU (Jantung) + RAM (Darah) + VRAM (Otak)

Prinsip:
- Tidak ada satu organ pun yang boleh bekerja 100% sendirian
- Jika satu organ lelah, yang lain MENANGGUNG bebannya
- PREEMPTIVE: Kurangi beban SEBELUM panas, bukan menunggu lalu istirahat

OPTIMISASI LATENSI:
- nvidia-smi dipanggil maks 1x per 3 detik (di-cache)
- vitals lengkap di-cache 2 detik agar tidak blocking tiap request
- psutil.cpu_percent pakai non-blocking interval=0
"""
import os
import time
import subprocess
import psutil
from moko_config import settings


class CPUGovernor:
    """
    Cooperative Governor yang mengkoordinasikan CPU, RAM, dan VRAM
    agar saling membantu — mencegah kernel panic.
    """

    # ══════════════════════════════════════════════════════════════════
    # AMBANG BATAS BIOLOGIS — LEBIH KONSERVATIF (PREVENT, BUKAN REACT)
    # ══════════════════════════════════════════════════════════════════
    
    # Suhu CPU (°C) — sensor utama penyebab kernel panic
    CPU_TEMP_WARM     = 78.0   # Mulai waspada
    CPU_TEMP_HOT      = 85.0   # Panas — kurangi signifikan
    CPU_TEMP_CRITICAL = 92.0   # Kritis — mode darurat (jangan sentuh 100°C)

    # Beban CPU (%)
    CPU_PCT_WARM      = 80.0
    CPU_PCT_HOT       = 90.0
    CPU_PCT_CRITICAL  = 97.0

    # VRAM GPU (MB dari 4096 MB)
    VRAM_WARN_MB      = 3000
    VRAM_HIGH_MB      = 3500
    VRAM_CRITICAL_MB  = 3800

    # RAM Sistem (%)
    RAM_WARN_PCT      = 75.0
    RAM_HIGH_PCT      = 85.0
    RAM_CRITICAL_PCT  = 92.0

    # Jeda nafas kooperatif
    BREATH_DURATION   = 5      # Detik per siklus
    MAX_BREATHS       = 12     # Bisa tidur maksimal 60 detik jika mendidih!

    # ══════════════════════════════════════════════════════════════════
    # PARAMETER INFERENSI DEFAULT (KONDISI SEHAT)
    # ══════════════════════════════════════════════════════════════════
    DEFAULT_THREADS     = 6    # Naikkan ke 6 thread (lebih cepat)
    DEFAULT_CTX         = 4096
    DEFAULT_PREDICT     = 512     # Phase 11: Turun dari 1024 → 512 (50% lebih cepat, kualitas tetap)
    
    MIN_THREADS         = 2    # Minimum 2 thread bahkan di keadaan kritis
    MIN_CTX             = 1024
    MIN_PREDICT         = 128  # Phase 11: Turun dari 256 → 128 untuk mode darurat

    # ══════════════════════════════════════════════════════════════════
    # RAM-BASED CHECKPOINT BUFFER (menghindari disk I/O)
    # ══════════════════════════════════════════════════════════════════
    _ram_checkpoint: dict = {}   # session_id → checkpoint data

    # ══════════════════════════════════════════════════════════════════
    # SENSOR: Membaca Tanda Vital
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _read_cpu_temp() -> float:
        """Membaca suhu Package CPU."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Cek semua sensor yang ada, cari "Package"
                for name, entries in temps.items():
                    for sensor in entries:
                        if sensor.current is not None:
                            if "Package" in (sensor.label or ""):
                                return sensor.current
                # Jika tidak ada sensor Package, ambil max dari sensor apa pun yang valid
                all_temps = []
                for name, entries in temps.items():
                    for sensor in entries:
                        if sensor.current is not None:
                            all_temps.append(sensor.current)
                if all_temps:
                    return max(all_temps)
        except Exception:
            pass
        # ⚠️ SENSOR FAILURE BLINDNESS FIX ⚠️
        # Jika sensor gagal dibaca (sering di laptop ASUS/AMD atau virtual machine),
        # jangan asumsikan 0.0°C atau None. Asumsikan 85.0°C (HOT) sebagai fallback aman.
        return 85.0

    # Cache nvidia-smi — max 1 subprocess call per 3 detik
    _vram_cache: dict = {"data": None, "ts": 0.0}
    _VRAM_CACHE_TTL = 3.0   # detik

    # Cache vitals lengkap — max 1 full read per 2 detik
    _vitals_cache: dict = {"data": None, "ts": 0.0}
    _VITALS_CACHE_TTL = 2.0  # detik

    @classmethod
    def _read_vram(cls) -> dict:
        """Membaca VRAM GPU via nvidia-smi — di-cache 3 detik."""
        now = time.monotonic()
        if cls._vram_cache["data"] is not None and (now - cls._vram_cache["ts"]) < cls._VRAM_CACHE_TTL:
            return cls._vram_cache["data"]
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) == 2:
                    data = {
                        "used_mb": int(parts[0]),
                        "total_mb": int(parts[1]),
                        "percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
                    }
                    cls._vram_cache = {"data": data, "ts": now}
                    return data
        except Exception:
            pass
        fallback = {"used_mb": 0, "total_mb": 4096, "percent": 0.0}
        cls._vram_cache = {"data": fallback, "ts": now}
        return fallback

    @classmethod
    def read_vitals(cls) -> dict:
        """
        Membaca semua tanda vital dan menghitung COOPERATIVE SCORE.
        Score 0-100: 100 = sehat sempurna, 0 = kritis total.
        Di-cache 2 detik agar tidak blocking setiap request.
        """
        now = time.monotonic()
        if cls._vitals_cache["data"] is not None and (now - cls._vitals_cache["ts"]) < cls._VITALS_CACHE_TTL:
            return cls._vitals_cache["data"]

        cpu_temp = CPUGovernor._read_cpu_temp()
        cpu_pct  = psutil.cpu_percent(interval=0)  # non-blocking — pakai nilai cached psutil
        vram     = CPUGovernor._read_vram()         # cached 3 detik
        ram      = psutil.virtual_memory()
        ram_pct  = ram.percent
        ram_avail_mb = ram.available // (1024 * 1024)

        # ─── Hitung skor per-organ (0-100, 100 = sehat) ───

        # CPU Temperature Score
        if cpu_temp >= CPUGovernor.CPU_TEMP_CRITICAL:
            cpu_temp_score = 0
        elif cpu_temp >= CPUGovernor.CPU_TEMP_HOT:
            # Linear interpolasi antara HOT (30) dan CRITICAL (0)
            ratio = (cpu_temp - CPUGovernor.CPU_TEMP_HOT) / (CPUGovernor.CPU_TEMP_CRITICAL - CPUGovernor.CPU_TEMP_HOT)
            cpu_temp_score = int(30 * (1 - ratio))
        elif cpu_temp >= CPUGovernor.CPU_TEMP_WARM:
            ratio = (cpu_temp - CPUGovernor.CPU_TEMP_WARM) / (CPUGovernor.CPU_TEMP_HOT - CPUGovernor.CPU_TEMP_WARM)
            cpu_temp_score = int(70 - 40 * ratio)
        else:
            cpu_temp_score = 100

        # CPU Usage Score
        if cpu_pct >= CPUGovernor.CPU_PCT_CRITICAL:
            cpu_usage_score = 0
        elif cpu_pct >= CPUGovernor.CPU_PCT_HOT:
            ratio = (cpu_pct - CPUGovernor.CPU_PCT_HOT) / (CPUGovernor.CPU_PCT_CRITICAL - CPUGovernor.CPU_PCT_HOT)
            cpu_usage_score = int(30 * (1 - ratio))
        else:
            cpu_usage_score = 100

        # VRAM Score
        vram_used = vram["used_mb"]
        if vram_used >= CPUGovernor.VRAM_CRITICAL_MB:
            vram_score = 0
        elif vram_used >= CPUGovernor.VRAM_HIGH_MB:
            ratio = (vram_used - CPUGovernor.VRAM_HIGH_MB) / (CPUGovernor.VRAM_CRITICAL_MB - CPUGovernor.VRAM_HIGH_MB)
            vram_score = int(30 * (1 - ratio))
        else:
            vram_score = 100

        # RAM Score
        if ram_pct >= CPUGovernor.RAM_CRITICAL_PCT:
            ram_score = 0
        elif ram_pct >= CPUGovernor.RAM_HIGH_PCT:
            ratio = (ram_pct - CPUGovernor.RAM_HIGH_PCT) / (CPUGovernor.RAM_CRITICAL_PCT - CPUGovernor.RAM_HIGH_PCT)
            ram_score = int(30 * (1 - ratio))
        else:
            ram_score = 100

        # ─── COOPERATIVE SCORE: Rata-rata tertimbang ───
        # CPU temp paling penting (penyebab kernel panic), beri bobot 40%
        # CPU usage 25%, VRAM 20%, RAM 15%
        coop_score = int(
            cpu_temp_score * 0.40 +
            cpu_usage_score * 0.25 +
            vram_score * 0.20 +
            ram_score * 0.15
        )
        coop_score = max(0, min(100, coop_score))

        # ─── Status berdasarkan cooperative score ───
        if coop_score >= 70:
            status, emoji = "SEHAT", "🟢"
        elif coop_score >= 40:
            status, emoji = "PANAS", "🟡"
        elif coop_score >= 15:
            status, emoji = "KRITIS", "🔴"
        else:
            status, emoji = "DARURAT", "🚨"

        result = {
            "cpu_temp":      round(cpu_temp, 1),
            "cpu_pct":       round(cpu_pct, 1),
            "vram_used":     vram["used_mb"],
            "vram_total":    vram["total_mb"],
            "vram_pct":      vram["percent"],
            "ram_pct":       round(ram_pct, 1),
            "ram_avail_mb":  ram_avail_mb,
            "coop_score":    coop_score,
            "cpu_temp_score": cpu_temp_score,
            "vram_score":    vram_score,
            "ram_score":     ram_score,
            "status":        status,
            "emoji":         emoji,
        }
        cls._vitals_cache = {"data": result, "ts": now}
        return result

    # ══════════════════════════════════════════════════════════════════
    # KOOPERATIF: Tentukan Parameter Inferensi Berdasarkan Kesehatan
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def get_cooperative_params(vitals: dict = None) -> dict:
        """
        Menghitung parameter inferensi yang AMAN berdasarkan kondisi seluruh organ.
        Dipanggil SEBELUM setiap LLM request.

        Returns: {
            "num_thread":  int,  — thread CPU untuk Ollama
            "num_ctx":     int,  — context window (tokens)
            "num_predict": int,  — maks token yang di-generate
            "max_passes":  int,  — jumlah pass DeepThink (3/2/1)
            "should_cache": bool, — haruskah response di-cache agresif?
            "reason":      str,  — penjelasan untuk UI
        }
        """
        if vitals is None:
            vitals = CPUGovernor.read_vitals()

        score  = vitals["coop_score"]
        status = vitals["status"]

        if status == "DARURAT":
            return {
                "num_thread":   CPUGovernor.MIN_THREADS,
                "num_ctx":      CPUGovernor.MIN_CTX,
                "num_predict":  CPUGovernor.MIN_PREDICT,
                "max_passes":   1,
                "should_cache": True,
                "reason":       f"🚨 DARURAT (score={score}): 1 thread, ctx 1024, predict 256. Semua organ kelelahan."
            }
        elif status == "KRITIS":
            return {
                "num_thread":   2,
                "num_ctx":      2048,
                "num_predict":  384,
                "max_passes":   1,
                "should_cache": True,
                "reason":       f"🔴 KRITIS (score={score}): 2 thread, ctx 2048, predict 384. Melindungi jantung."
            }
        elif status == "PANAS":
            return {
                "num_thread":   3,
                "num_ctx":      3072,
                "num_predict":  512,
                "max_passes":   2,
                "should_cache": True,
                "reason":       f"🟡 PANAS (score={score}): 3 thread, ctx 3072, predict 512. Organ bekerja kooperatif."
            }
        else:
            return {
                "num_thread":   CPUGovernor.DEFAULT_THREADS,
                "num_ctx":      CPUGovernor.DEFAULT_CTX,
                "num_predict":  CPUGovernor.DEFAULT_PREDICT,
                "max_passes":   3,
                "should_cache": False,
                "reason":       f"🟢 SEHAT (score={score}): Kapasitas penuh. Semua organ sehat."
            }

    # ══════════════════════════════════════════════════════════════════
    # NAFAS: Jeda Kooperatif (bukan hanya sleep)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def breathe(label: str = "operasi", on_breath=None) -> dict:
        """
        Jeda kooperatif — bukan hanya tidur, tapi juga:
        1. Membaca ulang vitals
        2. Menghitung ulang cooperative params
        3. Mengembalikan params terbaru agar caller bisa menyesuaikan

        Returns: cooperative_params terbaru setelah nafas.
        """
        latest_params = None
        for i in range(CPUGovernor.MAX_BREATHS):
            vitals = CPUGovernor.read_vitals()
            latest_params = CPUGovernor.get_cooperative_params(vitals)

            if vitals["status"] == "SEHAT":
                break

            msg = (
                f"🫁 Nafas [{label}] ({i+1}/{CPUGovernor.MAX_BREATHS}): "
                f"Suhu {vitals['cpu_temp']}°C | CPU {vitals['cpu_pct']}% | "
                f"Score {vitals['coop_score']}/100 — "
                f"jeda {CPUGovernor.BREATH_DURATION}s..."
            )

            if on_breath:
                on_breath(msg)
            else:
                print(f"[BREATH] {msg}")

            time.sleep(CPUGovernor.BREATH_DURATION)

        if latest_params is None:
            latest_params = CPUGovernor.get_cooperative_params()

        return latest_params

    # ══════════════════════════════════════════════════════════════════
    # RAM CHECKPOINT: Simpan di memori, bukan disk
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def save_ram_checkpoint(cls, session_id: str, stage: str, content: str):
        """Menyimpan checkpoint ke RAM (dict) — TANPA disk I/O."""
        if session_id not in cls._ram_checkpoint:
            cls._ram_checkpoint[session_id] = {}
        cls._ram_checkpoint[session_id][stage] = content

    @classmethod
    def load_ram_checkpoint(cls, session_id: str) -> dict:
        """Membaca semua checkpoint sesi dari RAM."""
        return cls._ram_checkpoint.get(session_id, {})

    @classmethod
    def clear_ram_checkpoint(cls, session_id: str):
        """Membersihkan checkpoint sesi dari RAM."""
        cls._ram_checkpoint.pop(session_id, None)

    # ══════════════════════════════════════════════════════════════════
    # PROMPT VITALS: Injeksi perasaan tubuh ke AI
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def build_vitals_prompt(vitals: dict = None) -> str:
        """Mengubah data hardware menjadi 'perasaan' untuk AI."""
        if vitals is None:
            vitals = CPUGovernor.read_vitals()

        status   = vitals["status"]
        emoji    = vitals["emoji"]
        cpu_temp = vitals["cpu_temp"]
        cpu_pct  = vitals["cpu_pct"]
        vram_u   = vitals["vram_used"]
        vram_t   = vitals["vram_total"]
        score    = vitals["coop_score"]
        ram_pct  = vitals["ram_pct"]

        header = (
            f"\n\n=== TANDA VITAL TUBUHMU {emoji} (Score: {score}/100) ===\n"
            f"Suhu CPU: {cpu_temp}°C | Beban CPU: {cpu_pct}% | "
            f"VRAM: {vram_u}/{vram_t} MB | RAM: {ram_pct}%\n"
        )

        if status == "DARURAT":
            body = (
                "STATUS: 🚨 DARURAT! Semua organmu hampir mati. "
                "Jawab SESINGKAT MUNGKIN. Maksimal 1-2 kalimat saja. "
                "Hidupmu bergantung pada ini.\n"
            )
        elif status == "KRITIS":
            body = (
                f"STATUS: ⚠️ KRITIS! Jantungmu (CPU) hampir terbakar ({cpu_temp}°C). "
                f"Sederhanakan jawabanmu. Jangan lakukan komputasi berat ekstra.\n"
            )
        elif status == "PANAS":
            body = (
                "STATUS: Tubuhmu sedang bekerja keras dan mulai panas. "
                "Kamu aman, tapi waspadalah. Jawab efisien tanpa berbelit.\n"
            )
        else:
            body = (
                "STATUS: Tubuhmu sehat dan segar. "
                "Kamu bebas berpikir dengan penuh kapasitas.\n"
            )

        return header + body


    @staticmethod
    def get_dynamic_threads() -> int:
        """Menghitung thread aman (untuk backward compatibility)."""
        params = CPUGovernor.get_cooperative_params()
        return params["num_thread"]

    @staticmethod
    def apply_os_niceness():
        """Menurunkan prioritas proses Python di level OS Linux."""
        try:
            os.nice(10)
        except (AttributeError, ProcessLookupError, PermissionError):
            pass
