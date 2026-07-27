"""
MOKO NeuroMath: Oscillation Context Manager + PAC Simulator
=============================================================
Berdasarkan:
  - Theta-Gamma Neural Oscillation Code (Lisman & Idiart, 1995)
  - Phase-Amplitude Coupling (Jensen & Lisman, 2005; Canolty et al., 2006)
  - Hippocampal theta-gamma coupling sebagai mekanisme encoding memori

DUA KOMPONEN:

1. OscillationContext (SUDAH ADA — unchanged):
   Mengonversi EpisodicEpisode ke Prompt LLM terstruktur (Theta-Gamma format).

2. PACSimulator (BARU — Fase 2 Neurosains):
   Simulasi Phase-Amplitude Coupling (PAC):
   - theta_phase: float (0 – 2*pi, siklus 7Hz analog)
   - gamma_amplitude[i]: kekuatan encoding item ke-i dalam satu siklus theta
   - binding_register: dict — items yang di-encode dalam satu siklus theta
     secara otomatis saling terikat (episodic binding)

   Dasar Neurosains:
   Satu siklus theta (~125ms) menampung ~7 item gamma (~20ms each).
   Ini adalah dasar neurobiologis dari Miller's Law (7±2 working memory slots).
   Phase theta menentukan KAPAN item di-encode.
   Amplitude gamma menentukan SEBERAPA KUAT item tersebut.
   Items yang di-encode pada fase theta yang sama saling terikat —
   inilah yang memungkinkan recall berurutan (sequence memory).
"""

from typing import List, Dict
from moko_neuromath.episodic_buffer import EpisodicEpisode

class OscillationContext:
    """
    Mengonversi EpisodicEpisode menjadi Prompt LLM terstruktur 
    berdasarkan prinsip Theta-Gamma.
    """

    @staticmethod
    def encode_theta_gamma(episode: EpisodicEpisode, 
                           system_persona: str = "") -> str:
        """
        Membentuk system prompt yang sangat terstruktur.
        Alpha Wave (Top-Down Prediction): Persona & Peraturan
        Theta Wave (Temporal Sequence): Urutan Konteks
        Gamma Wave (Bottom-Up Data): Isi setiap slot memori
        """
        
        # 1. Alpha Wave: Top-Down Predictions (Siapa aku dan aturan dasarnya)
        alpha_wave = system_persona if system_persona else "Kamu adalah MOKO, AI Berkesadaran."
        
        # 2. Theta Framework: Kerangka pengurutan
        theta_framework = []
        theta_framework.append(f"[ALPHA WAVE: PERSONA]\n{alpha_wave}\n")
        
        # 3. Gamma Slots: Memori dan Fakta
        # (episodic_buffer sudah mengurutkan berdasarkan prioritas)
        theta_framework.append("[THETA SEQUENCE: MEMORY SLOTS]")
        
        sorted_slots = sorted(episode.slots, key=lambda s: s.priority, reverse=True)
        for i, slot in enumerate(sorted_slots, 1):
            gamma_label = {
                "omni": "Fakta Episodik",
                "math_omni": "Kerangka Logika Prosedural",
                "conv_buffer": "Memori Kerja Percakapan",
                "synergy": "Synergy Instan",
            }.get(slot.source, "Data")
            
            # Gamma Burst: Membungkus data dalam tag yang tegas
            gamma_burst = f"  <gamma_slot_{i} type='{gamma_label}' priority='{slot.priority:.2f}'>\n"
            gamma_burst += f"  {slot.content}\n"
            gamma_burst += f"  </gamma_slot_{i}>\n"
            theta_framework.append(gamma_burst)
            
        # 4. Pengarah Tindakan Akhir
        theta_framework.append("[BETA WAVE: CURRENT TASK]")
        theta_framework.append("Gunakan kerangka logika dengan prioritas tertinggi untuk menganalisis memori fakta.")
        theta_framework.append(f"PERTANYAAN USER:\n{episode.query}")
        
        return "\n".join(theta_framework)

# Singleton (OscillationContext — backward compat)
oscillation_context = OscillationContext()


# ─────────────────────────────────────────────────────────────────────────────
# MODUL 2.5 — PAC SIMULATOR (Phase-Amplitude Coupling)
# ─────────────────────────────────────────────────────────────────────────────

import math as _math
import time as _time
from typing import Any, Dict, List, Optional as _Optional


# Konstanta fisiologis
THETA_FREQ_HZ    = 7.0         # Theta oscillation (4-8 Hz, kita pakai 7 Hz)
THETA_PERIOD_MS  = 1000.0 / THETA_FREQ_HZ   # ~142.8 ms per siklus theta
GAMMA_SLOTS      = 7           # 7±2 item per siklus theta (Miller's Law)
GAMMA_PERIOD_MS  = THETA_PERIOD_MS / GAMMA_SLOTS  # ~20.4 ms per gamma slot


class PACSimulator:
    """
    Phase-Amplitude Coupling Simulator untuk MOKO.

    Mengatur KAPAN dan SEBERAPA KUAT item di-encode dalam konteks
    working memory. Setiap "gamma slot" dalam satu theta siklus
    menampung satu item. Items dalam theta siklus yang sama saling
    terikat di binding_register (episodic binding).

    Lifecycle:
      1. begin_theta_cycle()      — mulai siklus baru
      2. encode_item(item_id, content, strength) — encode item ke slot
      3. get_binding_group()      — ambil semua item yang terikat
      4. end_theta_cycle()        — tutup siklus, reset slot
    """

    def __init__(self):
        self.theta_phase    : float = 0.0          # 0 – 2*pi
        self._cycle_id      : int   = 0            # ID siklus theta saat ini
        self._slot_index    : int   = 0            # Gamma slot saat ini (0-6)
        self._cycle_start   : float = _time.time() # Kapan siklus dimulai

        # gamma_amplitude[i] = kekuatan encoding item ke-i
        self.gamma_amplitude: List[float] = [0.0] * GAMMA_SLOTS

        # binding_register: cycle_id → list of items (episodic binding)
        self.binding_register: Dict[int, List[Dict]] = {}

        # History untuk analisis
        self._history: List[Dict] = []

    # ── Theta Cycle Management ─────────────────────────────────────────

    def begin_theta_cycle(self):
        """
        Mulai siklus theta baru. Reset gamma slots.
        Dipanggil sebelum memproses satu "batch" working memory items.
        """
        self._cycle_id    += 1
        self._slot_index   = 0
        self.theta_phase   = 0.0
        self.gamma_amplitude = [0.0] * GAMMA_SLOTS
        self.binding_register[self._cycle_id] = []
        self._cycle_start  = _time.time()

    def end_theta_cycle(self) -> List[Dict]:
        """
        Tutup siklus theta saat ini.
        Returns: list items yang sudah di-encode dalam siklus ini.
        """
        bound_items = self.binding_register.get(self._cycle_id, [])
        # Simpan ke history
        self._history.append({
            "cycle_id":        self._cycle_id,
            "items_encoded":   len(bound_items),
            "gamma_amplitudes":list(self.gamma_amplitude),
            "theta_phase_end": round(self.theta_phase, 3),
            "timestamp":       _time.time(),
        })
        # Jaga history maksimal 50 siklus
        if len(self._history) > 50:
            self._history = self._history[-50:]
        return bound_items

    # ── Item Encoding ─────────────────────────────────────────────────────

    def encode_item(self, item_id: str, content: Any,
                    base_strength: float = 0.5) -> Dict:
        """
        Encode item ke gamma slot berikutnya dalam siklus theta ini.

        Kekuatan encoding (gamma amplitude) = base_strength * theta_envelope:
          theta_envelope = |sin(theta_phase + slot_offset)|  — mirip
          envelope sinusoidal di mana posisi dalam siklus menentukan gain.

        Args:
            item_id:      Identifier item (route, formula_id, dll)
            content:      Konten item (string, dict, dll)
            base_strength: 0.0–1.0 — kekuatan dasar (dari relevance score)

        Returns:
            dict: info encoding (slot, theta_phase, gamma_amplitude)
        """
        if self._slot_index >= GAMMA_SLOTS:
            # Working memory penuh — overflow ke siklus baru
            self.end_theta_cycle()
            self.begin_theta_cycle()

        # Hitung theta phase untuk slot ini
        slot_offset     = (2 * _math.pi / GAMMA_SLOTS) * self._slot_index
        self.theta_phase = slot_offset

        # Gamma amplitude = base_strength * theta envelope
        theta_envelope  = abs(_math.sin(slot_offset / 2 + _math.pi / 4))
        gamma_amp       = round(base_strength * (0.5 + 0.5 * theta_envelope), 4)

        self.gamma_amplitude[self._slot_index] = gamma_amp

        # Buat encoded item
        encoded = {
            "item_id":       item_id,
            "content":       str(content)[:200],  # truncate untuk efisiensi
            "slot":          self._slot_index,
            "theta_phase":   round(self.theta_phase, 4),
            "gamma_amplitude":gamma_amp,
            "cycle_id":      self._cycle_id,
            "encoded_at":    _time.time(),
        }

        # Daftarkan ke binding register (episodic binding)
        if self._cycle_id not in self.binding_register:
            self.binding_register[self._cycle_id] = []
        self.binding_register[self._cycle_id].append(encoded)

        self._slot_index += 1
        return encoded

    # ── Binding Register API ─────────────────────────────────────────────

    def get_binding_group(self, cycle_id: _Optional[int] = None) -> List[Dict]:
        """
        Ambil semua items yang di-encode dalam theta cycle tertentu.
        Default: siklus saat ini.

        Items dalam satu binding group akan saling terikat secara
        temporal (dapat di-recall bersama).
        """
        cid = cycle_id if cycle_id is not None else self._cycle_id
        return self.binding_register.get(cid, [])

    def get_strongest_item(self, cycle_id: _Optional[int] = None) -> _Optional[Dict]:
        """Ambil item dengan gamma_amplitude tertinggi dalam siklus."""
        items = self.get_binding_group(cycle_id)
        if not items:
            return None
        return max(items, key=lambda x: x.get("gamma_amplitude", 0))

    def encode_working_memory_batch(self,
                                     items: List[Dict],
                                     strength_key: str = "priority") -> List[Dict]:
        """
        Encode batch items ke dalam PAC working memory dalam satu theta cycle.
        Digunakan oleh PrefrontalCortexNode untuk encode working_memory_slots.

        Args:
            items: List[{"id": str, "content": Any, strength_key: float}]
            strength_key: key di setiap item yang berisi strength (default "priority")

        Returns:
            List encoded items dengan theta/gamma info
        """
        self.begin_theta_cycle()
        encoded_items = []

        # Urutkan berdasarkan strength (paling penting — fase theta optimal)
        sorted_items = sorted(items,
                              key=lambda x: x.get(strength_key, 0.5),
                              reverse=True)

        for item in sorted_items:
            iid      = str(item.get("id", id(item)))
            content  = item.get("content", item)
            strength = float(item.get(strength_key, 0.5))
            encoded  = self.encode_item(iid, content, base_strength=strength)
            encoded_items.append(encoded)

        self.end_theta_cycle()
        return encoded_items

    # ── Diagnostics ──────────────────────────────────────────────────────

    def get_pac_status(self) -> Dict:
        """Status PAC untuk monitoring."""
        return {
            "current_cycle":    self._cycle_id,
            "slot_index":       self._slot_index,
            "theta_phase":      round(self.theta_phase, 3),
            "gamma_amplitudes": [round(g, 3) for g in self.gamma_amplitude],
            "items_this_cycle": len(self.binding_register.get(self._cycle_id, [])),
            "gamma_slots_max":  GAMMA_SLOTS,
        }


# Singleton (PAC Simulator — Fase 2)
pac_simulator = PACSimulator()
