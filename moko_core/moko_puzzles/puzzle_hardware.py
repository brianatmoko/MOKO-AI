from moko_puzzles.base_puzzle import BasePuzzle
from moko_cpu.governor import CPUGovernor

class HardwarePuzzle(BasePuzzle):
    name = "hardware_vitals"
    description = "Laptop Hardware Vitals Monitor"
    version = "1.0.0"

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        keywords = [
            "suhu", "cpu", "ram", "vram", "gpu", "hardware", 
            "tanda vital", "vitalitas", "panas laptop", "kondisi pc", "spesifikasi laptop"
        ]
        if any(kw in q for kw in keywords):
            return 0.90
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        # Membaca tanda vital hardware saat ini
        vitals = CPUGovernor.read_vitals()
        
        status = vitals.get("status", "SEHAT")
        emoji = vitals.get("emoji", "🟢")
        cpu_temp = vitals.get("cpu_temp", 0.0)
        cpu_pct = vitals.get("cpu_pct", 0.0)
        vram_used = vitals.get("vram_used", 0)
        vram_total = vitals.get("vram_total", 4096)
        ram_pct = vitals.get("ram_pct", 0.0)
        score = vitals.get("coop_score", 100)
        
        report = (
            f"=== STATISTIK VITAL HARDWARE LAPTOP {emoji} ===\n"
            f"Kondisi Umum: {status} (Skor Kesehatan: {score}/100)\n"
            f"Suhu CPU: {cpu_temp}°C (Beban Kerja: {cpu_pct}%)\n"
            f"Penggunaan RAM: {ram_pct}%\n"
            f"Penggunaan VRAM GPU: {vram_used} / {vram_total} MB\n"
        )
        
        return {
            "facts": report,
            "confidence": 1.0,
            "metadata": vitals
        }
