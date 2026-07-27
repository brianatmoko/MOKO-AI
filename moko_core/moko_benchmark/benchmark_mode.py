#!/usr/bin/env python3
"""
MOKO OS: Comprehensive Deficit Benchmark — 42-Category Analysis
================================================================
Fokus utama: menampilkan semua sektor di mana Moko OS kalah
secara signifikan dari ChatGPT (GPT-4o) dan Claude (3.5 Sonnet).

Dasar data: kombinasi pengukuran langsung (latency, akurasi)
dan analisis struktural (arsitektur, feature set).

Jalankan:
  ./bin/python moko_core/moko_benchmark/benchmark_mode.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moko_config import settings


def run_quick_health():
    """Quick system health snapshot."""
    print("⏳ Running health check...")
    try:
        from moko_agents.llm_engine import engine
        from moko_inference.server_manager import MokoLocalInferenceServer

        q_ok = MokoLocalInferenceServer.is_port_responding(settings.MOKO_LLM_PORT)
        if not q_ok:
            return {"status": "Offline", "tps": 0.0, "latency_ms": 0.0}

        t0 = time.perf_counter()
        resp = engine.generate_text(
            prompt="Hai, jawab singkat.",
            system_prompt="",
            coop_params={"num_predict": 8, "enable_thinking": False}
        )
        elapsed = time.perf_counter() - t0
        tps = (len(resp.split()) * 1.3) / max(elapsed, 0.01)

        return {"status": "Online", "tps": round(tps, 2), "latency_ms": round(elapsed * 1000, 0)}
    except Exception as e:
        return {"status": f"Error: {e}", "tps": 0.0, "latency_ms": 0.0}


def build_scorecard():
    """Return 42 benchmark categories with (name, gpt4, claude, moko, group, note)."""
    return [
        # ── GRUP A: Kemampuan Dasar ───────────────────────────────────────────
        ("A", "Raw throughput (tok/s)",            10, 10, 1,  "80-120 tok/s cloud vs 2-3 tok/s Moko (diukur)"),
        ("A", "Time-to-first-token",               10, 10, 1,  "Cloud <0.5s; Moko 5-10s (diukur: 9.9s)"),
        ("A", "Context window (effective)",        10, 10, 3,  "128K/200K cloud vs 32K Moko; KV cache 4GB overhead"),
        ("A", "Factual accuracy (simple)",         10, 10, 8,  "Moko 75.0% akurasi faktual dasar (diukur)"),
        ("A", "Factual accuracy (niche/deep)",     9,  10, 3,  "Estimasi: pengetahuan spesifik sangat terbatas"),
        ("A", "Instruction following precision",   10, 10, 8,  "Moko 80% (4/5 test lulus), gagal di word count"),
        ("A", "Numeric format consistency",        10, 10, 8,  "Moko menggunakan format titik desimal & non-dot"),
        ("A", "JSON structured output",            9,  10, 7,  "SCoT template membatasi markdown tak diminta"),
        ("A", "Self-correction capability",        9,  10, 3,  "Moko tidak bisa deteksi & koreksi error sendiri"),
        ("A", "Consistency across re-runs",        9,   9, 8,  "Temperature 0.0 (precise tasks) meningkatkan konsistensi"),

        # ── GRUP B: Pengetahuan & Reasoning ──────────────────────────────────
        ("B", "Model parameter scale",             10, 10, 1,  "200B+ vs 4B — perbedaan 50x, kalah telak"),
        ("B", "Pretraining budget & data",         10, 10, 1,  "$100M+ vs $0 — entire web vs curated set"),
        ("B", "Knowledge recency (cutoff)",        9,   9, 4,  "GPT-4o April 2024; Moko: tergantung MOKO cutoff"),
        ("B", "Real-time web knowledge",           9,   8, 0,  "Moko tidak ada search; tidak tahu berita hari ini"),
        ("B", "Cross-domain transfer",             10, 10, 4,  "4B param tidak bisa generalisasi domain baru"),
        ("B", "Medical domain depth",              9,  10, 3,  "Tidak ada medical training data spesifik"),
        ("B", "Legal / regulatory knowledge",      8,   9, 2,  "Tidak ada legal corpus dalam training data"),
        ("B", "Scientific literature depth",       10, 10, 3,  "Arxiv / PubMed knowledge sangat terbatas"),
        ("B", "Multi-step logical reasoning",      9,  10, 5,  "CES 4-phase membantu tapi model kecil limitasi"),
        ("B", "MMLU benchmark (est.)",             10, 10, 6,  "GPT-4o ~87%, Claude ~89%; MOKO 4B ~55-60%"),

        # ── GRUP C: Kemampuan Bahasa ──────────────────────────────────────────
        ("C", "Cross-lingual scope",               10, 10, 2,  "ChatGPT 100+ bahasa; Moko fokus ID+EN+teknis"),
        ("C", "Indonesian language fluency",        9,  8, 7,  "Terbaik Moko OS, tapi masih kalah cloud"),
        ("C", "English language fluency",          10, 10, 5,  "Model kecil → coherensi teks panjang buruk"),
        ("C", "Creative writing quality",          10, 10, 3,  "Prosa kreatif dan naratif sangat terbatas"),
        ("C", "Poetry & literary analysis",         9, 10, 2,  "Hampir tidak ada kemampuan puisi/sastra"),
        ("C", "Humor & cultural nuance",            9,  9, 2,  "Humor sering flat, nuansa budaya hilang"),
        ("C", "Long doc summarization",            10, 10, 4,  "Context limit 32K tidak cukup untuk dokumen panjang"),
        ("C", "Translation accuracy",              10, 10, 2,  "Terjemahan seringkali kaku/tidak natural"),

        # ── GRUP D: Kemampuan Teknis ──────────────────────────────────────────
        ("D", "Code generation (novel tasks)",     10, 10, 7,  "HumanEval est. ~70% setelah temperature 0.0 & SCoT"),
        ("D", "Code debugging precision",          10, 10, 6,  "Lebih teliti memeriksa edge case & return types"),
        ("D", "Native function calling",           10, 10, 2,  "Tidak ada tool_calls JSON schema; hanya RUN_BASH"),
        ("D", "Streaming token delivery",          10, 10, 2,  "llm_engine.py blocking POST, tidak stream"),
        ("D", "Concurrent request handling",       10, 10, 1,  "-np 1 → hanya 1 user sekaligus, tidak scalable"),
        ("D", "Structured output (strict)",         9, 10, 7,  "Strict templates meminimalkan penjelasan ekstra"),
        ("D", "Long document processing",          10, 10, 3,  "Dokumen >32K token tidak bisa diproses"),
        ("D", "API reliability & SLA",             10, 10, 0,  "Tidak ada SLA, tidak ada uptime guarantee"),

        # ── GRUP E: Infrastruktur & Ekosistem ────────────────────────────────
        ("E", "Multimodal vision input",           10,  9, 0,  "Belum ada mmproj model; saat ini 0/10"),
        ("E", "Voice / audio input",               10,  0, 0,  "Tidak ada audio processing sama sekali"),
        ("E", "File attachment (PDF/xlsx)",        10, 10, 0,  "Tidak bisa baca file apapun secara langsung"),
        ("E", "Plugin / tool ecosystem",           10,  7, 0,  "Tidak ada marketplace plugin"),
        ("E", "Mobile app availability",           10,  8, 0,  "Tidak ada mobile app"),
        ("E", "Enterprise features & SSO",         10, 10, 0,  "Tidak ada enterprise management"),
        ("E", "RAM pressure @ max context",        10, 10, 2,  "12.4GB/16GB dipakai → 3.6GB headroom saja"),
    ]


def print_full_dashboard(health):
    cats = build_scorecard()

    W = 80
    print("\n" + "═" * W)
    print(" 📉 MOKO OS — AUDIT KELEMAHAN KOMPREHENSIF vs ChatGPT & Claude")
    print("═" * W)
    print(f" Status: {health['status']}  |  Speed: {health['tps']} tok/s  |  TTFT: {health['latency_ms']} ms")
    print(f" Model: MOKO 3.5 4B BF16  |  Threads: {settings.LLM_MAX_THREADS}  |  Context: {settings.MAX_CONTEXT_TOKENS} tokens")
    print("─" * W)

    groups = {"A": "KEMAMPUAN DASAR", "B": "PENGETAHUAN & REASONING",
              "C": "KEMAMPUAN BAHASA", "D": "TEKNIS", "E": "INFRASTRUKTUR"}
    cur_grp = None
    deficits = []

    for grp, name, gpt, cld, moko, note in cats:
        if grp != cur_grp:
            cur_grp = grp
            print(f"\n {'─'*3} GRUP {grp}: {groups[grp]} {'─'*3}")
            print(f"  {'Kategori':<32} | {'GPT-4o':^7} | {'Claude':^7} | {'Moko':^7} | {'Gap':^6} | Catatan")
            print(f"  {'─'*32}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*6}-+{'─'*22}")

        gap = moko - ((gpt + cld) // 2)
        deficits.append(gap)
        gap_str = f"{gap:+d}"
        bar = "█" * moko + "░" * (10 - moko)
        warn = " ⚠️" if moko <= 2 else (" 🔴" if moko <= 4 else "")
        print(f"  {name:<32} | {gpt:^7} | {cld:^7} | {moko:^7} | {gap_str:^6} |{warn} {note[:35]}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n" + "═" * W)
    avg_gpt  = sum(c[2] for c in cats) / len(cats)
    avg_cld  = sum(c[3] for c in cats) / len(cats)
    avg_moko = sum(c[4] for c in cats) / len(cats)
    lose_count = sum(1 for c in cats if c[4] < 6)
    zero_count = sum(1 for c in cats if c[4] == 0)
    avg_gap  = avg_moko - ((avg_gpt + avg_cld) / 2)

    print(f"\n STATISTIK RINGKASAN (dari {len(cats)} kategori):")
    print(f"  ChatGPT (GPT-4o) avg score : {avg_gpt:.1f}/10")
    print(f"  Claude 3.5 Sonnet avg score: {avg_cld:.1f}/10")
    print(f"  Moko OS avg score          : {avg_moko:.1f}/10  ← {avg_gap:+.1f} vs cloud average")
    print(f"  Kategori Moko kalah (<6/10): {lose_count}/{len(cats)} = {100*lose_count//len(cats)}%")
    print(f"  Kategori Moko = 0/10       : {zero_count}/{len(cats)} = {100*zero_count//len(cats)}%")
    print(f"  Pemenang keseluruhan       : ChatGPT (GPT-4o) & Claude 3.5 Sonnet")

    print("\n 🔴 KELEMAHAN PALING KRUSIAL (skor ≤ 2/10):")
    critical = [(name, moko, note) for grp, name, gpt, cld, moko, note in cats if moko <= 2]
    for i, (name, score, note) in enumerate(critical, 1):
        print(f"  {i:2}. [{score}/10] {name:<35} — {note[:45]}")

    print("═" * W + "\n")


if __name__ == "__main__":
    health = run_quick_health()
    print_full_dashboard(health)
