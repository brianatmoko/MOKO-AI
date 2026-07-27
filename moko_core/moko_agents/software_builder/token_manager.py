"""
token_manager.py — Token Marathon System untuk Software Builder
===============================================================
Sistem yang memastikan kode tidak terpotong di tengah saat LLM mencapai token limit.

Komponen utama:
  1. TokenBudgetManager  — estimasi jumlah token, prune prompt cerdas
  2. MarathonCodeGenerator — generate kode panjang dalam beberapa iterasi, sambungkan
  3. marathon_call_llm()  — wrapper drop-in pengganti _call_llm() dengan auto-continue

Prinsip desain (berdasarkan riset auto-continue dari MOKO AutoContinueEngine):
  - Estimasi 1 token ≈ 4 karakter (konservatif untuk model berbasis UTF-8)
  - Prompt budget: total_ctx - output_reserve
  - Jika prompt melebihi budget → prune dengan urutan prioritas (least-important first):
      1. Potong few-shot example (hanya sisakan header)
      2. Potong repo map (hanya sisakan file tree, buang definisi)
      3. Potong existing_code snippet (sisakan 2 langkah terakhir)
      4. Potong RAG context (sisakan 500 chars)
  - Jika output terpotong (finish_reason == "length") → auto-continue maks 8 iterasi
  - Untuk kode yang diprediksi sangat panjang → MarathonCodeGenerator split jadi bagian
"""
from __future__ import annotations

import re
import time
from typing import Callable, List, Optional, Tuple


# ─── Konstanta ─────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4          # Estimasi konservatif: 1 token ≈ 4 karakter

# Context window model kecil (1.5B–4B) biasanya 4096–8192 token
# Kita pakai 4096 sebagai safe default (override lewat env var)
DEFAULT_CTX_TOKENS = 4096

# Token yang disisihkan untuk output
OUTPUT_RESERVE_TOKENS = 1500  # ~6000 karakter output

# Max token untuk prompt sebelum prune
MAX_PROMPT_TOKENS = DEFAULT_CTX_TOKENS - OUTPUT_RESERVE_TOKENS  # 2596 token ≈ ~10384 chars

# Max iterasi auto-continue untuk kode (lebih banyak dari konversasi biasa)
MAX_MARATHON_CONTINUES = 8


# ─── TokenBudgetManager ────────────────────────────────────────────────────────

class TokenBudgetManager:
    """
    Mengelola token budget untuk prompt-prompt Software Builder.
    
    Mencegah prompt meledak dengan melakukan pruning bertahap pada bagian yang
    paling tidak krusial, sambil mempertahankan konteks yang paling penting.
    
    Urutan pruning (dari yang paling tidak penting ke paling penting):
      1. Few-shot examples (besar, bisa dikompres)
      2. Repo map defs section (bisa diringkas menjadi file tree saja)
      3. Existing code snippets (sisakan hanya 2 langkah terakhir)
      4. RAG context (sisakan 500 chars)
      5. (tidak pernah prune): step description, project context, core instructions
    """

    def __init__(self, ctx_tokens: int = DEFAULT_CTX_TOKENS):
        self.ctx_tokens = ctx_tokens
        self.output_reserve = OUTPUT_RESERVE_TOKENS
        self.prompt_budget_chars = (ctx_tokens - self.output_reserve) * CHARS_PER_TOKEN

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimasi jumlah token dari string (chars / 4, konservatif)."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    def fits_in_budget(self, text: str) -> bool:
        """Return True jika teks muat dalam prompt budget."""
        return len(text) <= self.prompt_budget_chars

    def prune_prompt(self, prompt: str) -> str:
        """
        Prune prompt secara bertahap sampai muat dalam budget.
        
        Strategi: identifikasi section markers dan hapus dari yang paling tidak penting.
        Menggunakan marker yang ada di template prompt_enrichment.py dan step_executor.py.
        
        Returns:
            Prompt yang sudah diperkecil (atau original jika sudah muat)
        """
        if self.fits_in_budget(prompt):
            return prompt

        current = prompt
        
        # Tahap 1: Kompres few-shot example (hanya sisakan 3 baris pertama + ringkasan)
        current = self._prune_few_shot(current)
        if self.fits_in_budget(current):
            return current

        # Tahap 2: Kompres repo map (hanya sisakan file tree, buang KEY DEFINITIONS)
        current = self._prune_repo_map_defs(current)
        if self.fits_in_budget(current):
            return current

        # Tahap 3: Kompres existing code (kurangi dari awal)
        current = self._prune_existing_code(current)
        if self.fits_in_budget(current):
            return current

        # Tahap 4: Kompres RAG context (sisakan 500 chars)
        current = self._prune_rag_context(current)
        if self.fits_in_budget(current):
            return current

        # Tahap 5: Hard truncate — potong bagian awal yang bukan instruksi inti
        budget = self.prompt_budget_chars
        if len(current) > budget:
            # Cari posisi instruksi inti dan pertahankan dari sana
            core_markers = [
                "## Step", "## YOUR TASK", "## STEP GOAL", 
                "STEP GOAL:", "Generate COMPLETE", "You are MOKO"
            ]
            for marker in core_markers:
                idx = current.find(marker)
                if idx > 0 and len(current) - idx <= budget:
                    # Potong bagian sebelum marker
                    current = current[idx:]
                    break
            # Final hard truncate jika masih terlalu panjang
            if len(current) > budget:
                # Simpan akhir (lebih penting) bukan awal
                current = current[-budget:]

        return current

    def _prune_few_shot(self, text: str) -> str:
        """Kompres few-shot example menjadi hanya header + 3 baris pertama."""
        # Cari section EXAMPLE OUTPUT
        pattern = re.compile(
            r"(## EXAMPLE OUTPUT[^\n]*\n)(.*?)(\n## |\Z)",
            re.DOTALL
        )
        def replace_example(m):
            header = m.group(1)
            content = m.group(2)
            suffix = m.group(3)
            # Ambil max 3 baris pertama dari contoh
            lines = content.strip().split("\n")
            short = "\n".join(lines[:4])
            return f"{header}{short}\n[...example truncated for token budget...]{suffix}"
        
        result = pattern.sub(replace_example, text, count=1)
        return result

    def _prune_repo_map_defs(self, text: str) -> str:
        """Hapus section KEY DEFINITIONS dari repo map (sisakan file tree saja)."""
        pattern = re.compile(
            r"## KEY DEFINITIONS \(classes & functions\).*?(?=\n##|\Z)",
            re.DOTALL
        )
        result = pattern.sub(
            "## KEY DEFINITIONS\n[...definitions removed for token budget...]\n",
            text,
            count=1
        )
        return result

    def _prune_existing_code(self, text: str) -> str:
        """Kompres existing code section — sisakan hanya 1200 chars dari bagian akhir."""
        # Cari marker existing code
        patterns = [
            r"(## EXISTING CODE[^\n]*\n)(.*?)(\n## |\Z)",
            r"(# --- Step \d+:.*?---\n)(.*?)(\n# --- Step|\Z)",
        ]
        for pattern_str in patterns:
            pattern = re.compile(pattern_str, re.DOTALL)
            match = pattern.search(text)
            if match:
                full_section = match.group(0)
                if len(full_section) > 1500:
                    # Sisakan hanya 1200 chars terakhir dari kode
                    truncated = full_section[-1200:]
                    # Pastikan mulai dari baris yang bersih
                    nl_idx = truncated.find("\n")
                    if nl_idx > 0:
                        truncated = truncated[nl_idx + 1:]
                    replacement = (
                        match.group(1) +
                        "[...older code truncated for token budget...]\n" +
                        truncated
                    )
                    text = text[:match.start()] + replacement + text[match.end():]
                    break
        return text

    def _prune_rag_context(self, text: str) -> str:
        """Kompres RAG context menjadi max 500 chars."""
        pattern = re.compile(
            r"(## TECHNICAL CONTEXT[^\n]*\n)(.*?)(\n## |\Z)",
            re.DOTALL
        )
        def replace_rag(m):
            header = m.group(1)
            content = m.group(2)
            suffix = m.group(3)
            if len(content) > 500:
                return f"{header}{content[:500]}\n[...context truncated...]{suffix}"
            return m.group(0)
        
        result = pattern.sub(replace_rag, text, count=1)
        return result


# ─── MarathonCodeGenerator ─────────────────────────────────────────────────────

class MarathonCodeGenerator:
    """
    Generator kode yang bisa menangani output sangat panjang dengan cara:
    1. Panggil LLM dengan auto-continue hingga finish_reason == "stop"
    2. Jika masih belum lengkap setelah MAX_MARATHON_CONTINUES, lakukan
       "completion pass": kirim kode parsial + instruksi untuk menyelesaikan
    3. Sambungkan semua chunk dengan overlap detection

    Didesain sebagai drop-in replacement untuk _call_llm() di StepExecutor.
    """

    def __init__(self, llm_engine=None, log_fn: Optional[Callable] = None):
        """
        Args:
            llm_engine: Instance MokoEngine (dari moko_agents.llm_engine)
            log_fn: Optional callable(message, color) untuk logging ke UI
        """
        self._engine = llm_engine
        self._log = log_fn or (lambda msg, color: None)

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from moko_agents.llm_engine import engine
            return engine
        except Exception:
            return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.15,
        system_prompt: str = "",
        max_continues: int = MAX_MARATHON_CONTINUES,
    ) -> str:
        """
        Generate teks dengan marathon auto-continue.
        
        Otomatis lanjutkan generasi jika finish_reason == "length", sampai:
          - finish_reason == "stop" (selesai natural), atau
          - max_continues iterasi tercapai
        
        Args:
            prompt: Prompt lengkap untuk LLM
            max_tokens: Jumlah token output per panggilan
            temperature: Temperatur generasi
            system_prompt: Optional system prompt
            max_continues: Maks iterasi lanjutan
        
        Returns:
            String output lengkap (gabungan semua iterasi)
        """
        llm = self._get_engine()
        if llm is None:
            self._log("  ⚠️ [Marathon] LLM engine tidak tersedia", "#ff8800")
            return ""

        # Budget manager untuk prune prompt jika perlu
        budget_mgr = TokenBudgetManager()
        pruned_prompt = budget_mgr.prune_prompt(prompt)
        
        if len(pruned_prompt) < len(prompt):
            reduction = len(prompt) - len(pruned_prompt)
            self._log(
                f"  ✂️ [Marathon] Prompt diperkecil: -{reduction} chars "
                f"(dari {len(prompt)} → {len(pruned_prompt)})",
                "#ffaa00"
            )

        # Build coop_params
        coop_params = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "enable_thinking": False,  # Untuk kode: matikan thinking agar lebih cepat
        }

        # Panggilan pertama
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": pruned_prompt})

        content, finish_reason = llm.generate_text_raw(
            prompt=pruned_prompt,
            system_prompt=system_prompt,
            coop_params=coop_params
        )

        if finish_reason in ("stop", "error") or not content:
            return content.strip() if content else ""

        # Terpotong → masuk marathon loop
        accumulated = content
        self._log(
            f"  🏃 [Marathon] Token limit tercapai, melanjutkan... "
            f"({len(accumulated)} chars sejauh ini)",
            "#00e6ff"
        )

        for iteration in range(1, max_continues + 1):
            # Susun continue messages (riwayat + partial)
            CONTEXT_SAFE_LIMIT = 6000
            if len(accumulated) > CONTEXT_SAFE_LIMIT:
                partial_to_send = (
                    accumulated[:1500] +
                    "\n\n...[MARATHON CONTEXT: MIDDLE COMPRESSED]...\n\n" +
                    accumulated[-4000:]
                )
            else:
                partial_to_send = accumulated

            continuation, new_reason = llm.continue_generation(
                messages=messages,
                partial_response=partial_to_send,
                coop_params=coop_params
            )

            if not continuation:
                self._log(
                    f"  ⚠️ [Marathon] LLM mengembalikan kosong di iterasi {iteration}",
                    "#ffaa00"
                )
                break

            # Gabungkan dengan overlap detection
            accumulated = self._smart_join(accumulated, continuation)

            self._log(
                f"  🏃 [Marathon] Iterasi {iteration}: +{len(continuation)} chars "
                f"(total {len(accumulated)})",
                "rgba(0,230,255,0.6)"
            )

            if new_reason == "stop":
                self._log(
                    f"  ✅ [Marathon] Selesai setelah {iteration} iterasi. "
                    f"Total: {len(accumulated)} chars",
                    "#00ff88"
                )
                break
            elif new_reason == "error":
                self._log(f"  ❌ [Marathon] Error di iterasi {iteration}", "#ff4444")
                break
        else:
            # Batas iterasi tercapai — lakukan completion pass
            self._log(
                f"  ⚠️ [Marathon] Batas {max_continues} iterasi. "
                f"Mencoba completion pass...",
                "#ffaa00"
            )
            accumulated = self._completion_pass(
                accumulated, messages, coop_params, max_tokens
            )

        return accumulated.strip()

    def _smart_join(self, accumulated: str, continuation: str) -> str:
        """
        Gabungkan accumulated + continuation dengan overlap detection.
        Menghindari duplikasi teks saat LLM mengulang bagian terakhir.
        """
        if not continuation:
            return accumulated
        
        # Cari overlap: cek 100 karakter terakhir dari accumulated
        overlap_window = min(120, len(accumulated), len(continuation))
        for overlap_len in range(overlap_window, 10, -1):
            tail = accumulated[-overlap_len:]
            if continuation.startswith(tail):
                # Overlap ditemukan → sambungkan tanpa duplikasi
                return accumulated + continuation[overlap_len:]
        
        # Tidak ada overlap — langsung sambungkan
        return accumulated + continuation

    def _completion_pass(
        self,
        partial_code: str,
        base_messages: list,
        coop_params: dict,
        max_tokens: int
    ) -> str:
        """
        Lakukan "completion pass": kirim kode parsial + instruksi untuk menyelesaikan.
        Digunakan jika marathon loop sudah mencapai batas iterasi.
        """
        llm = self._get_engine()
        if llm is None:
            return partial_code

        # Ambil bagian akhir kode parsial (paling relevan untuk dilanjutkan)
        code_tail = partial_code[-2000:] if len(partial_code) > 2000 else partial_code
        
        completion_instruction = (
            f"The following code was cut off. Complete it from where it stopped:\n\n"
            f"```\n{code_tail}\n```\n\n"
            f"Continue from the exact stopping point. Do NOT repeat what's already written."
        )
        
        completion_messages = list(base_messages) + [
            {"role": "assistant", "content": partial_code},
            {"role": "user", "content": completion_instruction}
        ]

        try:
            completion_coop = dict(coop_params)
            completion_coop["num_predict"] = max_tokens
            
            cont, _ = llm.continue_generation(
                messages=base_messages,
                partial_response=partial_code[-3000:],
                coop_params=completion_coop
            )
            if cont:
                return self._smart_join(partial_code, cont)
        except Exception as e:
            self._log(f"  ⚠️ [Marathon] Completion pass gagal: {e}", "#ff8800")
        
        return partial_code


# ─── marathon_call_llm() — Drop-in Wrapper ─────────────────────────────────────

def marathon_call_llm(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.15,
    system_prompt: str = "",
    log_fn: Optional[Callable] = None,
    max_continues: int = MAX_MARATHON_CONTINUES,
) -> str:
    """
    Drop-in replacement untuk _call_llm() di StepExecutorWorker dan PlanGenerationWorker.
    
    Otomatis:
    1. Prune prompt jika melebihi token budget
    2. Auto-continue jika output terpotong
    3. Completion pass jika batas iterasi tercapai
    
    Args:
        prompt:        Prompt lengkap
        max_tokens:    Max token output per panggilan
        temperature:   Temperatur generasi (default: 0.15 untuk kode)
        system_prompt: Optional system prompt
        log_fn:        Optional callable(message, color) untuk UI logging
        max_continues: Max iterasi marathon
    
    Returns:
        Output LLM yang lengkap (atau sebisa mungkin lengkap)
    """
    generator = MarathonCodeGenerator(log_fn=log_fn)
    return generator.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        max_continues=max_continues,
    )


# ─── Unit Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Unit Test: token_manager.py ===\n")

    mgr = TokenBudgetManager()

    # Test 1: estimate_tokens
    assert mgr.estimate_tokens("hello world") == 2, f"Expected 2, got {mgr.estimate_tokens('hello world')}"
    assert mgr.estimate_tokens("") == 1
    print("  ✅ estimate_tokens → OK")

    # Test 2: fits_in_budget
    short_text = "x" * 100
    long_text = "x" * (mgr.prompt_budget_chars + 1)
    assert mgr.fits_in_budget(short_text)
    assert not mgr.fits_in_budget(long_text)
    print("  ✅ fits_in_budget → OK")

    # Test 3: prune_few_shot
    prompt_with_example = (
        "You are MOKO. Here is your task.\n\n"
        "## EXAMPLE OUTPUT (for a simple game project):\n"
        "## Step 1: Project Setup\nInstall pygame, create project structure.\nFiles: main.py\n"
        "## Step 2: Game Window\nCreate 800x600 window.\nFiles: game.py\n"
        "## Step 3: Player\nBuild player class.\nFiles: player.py\n"
        "## Step 4: World\nBuild world.\nFiles: world.py\n"
        "## Step 5: HUD\nBuild HUD.\nFiles: hud.py\n"
        "\n## YOUR PLAN FOR THIS PROJECT\nNow create:"
    )
    pruned = mgr._prune_few_shot(prompt_with_example)
    assert "EXAMPLE OUTPUT" in pruned
    assert "[...example truncated" in pruned
    print("  ✅ _prune_few_shot → OK")

    # Test 4: prune_repo_map_defs
    prompt_with_defs = (
        "## EXISTING PROJECT FILES (Repo Map)\n  main.py\n  player.py\n"
        "\n## KEY DEFINITIONS (classes & functions)\n"
        "  main.py: class Game:\n    def run(self, ...)\n"
        "  player.py: class Player:\n    def move(self, ...)\n"
        "\n## STEP GOAL\nCreate the player module."
    )
    pruned2 = mgr._prune_repo_map_defs(prompt_with_defs)
    assert "KEY DEFINITIONS" in pruned2
    assert "definitions removed for token budget" in pruned2
    print("  ✅ _prune_repo_map_defs → OK")

    # Test 5: prune_rag_context
    long_rag = "X" * 1000
    prompt_with_rag = f"## TECHNICAL CONTEXT (from MOKO Knowledge Base)\n{long_rag}\n## YOUR PLAN\nBuild it."
    pruned3 = mgr._prune_rag_context(prompt_with_rag)
    assert len(pruned3) < len(prompt_with_rag)
    assert "context truncated" in pruned3
    print("  ✅ _prune_rag_context → OK")

    # Test 6: full prune_prompt dengan prompt yang terlalu besar
    # Buat prompt sintetis yang melebihi budget
    fat_prompt = (
        "You are MOKO coder. CRITICAL RULES: never lazy.\n\n"
        "## EXAMPLE OUTPUT (for a simple game project):\n" + ("Step N: Some step\nFiles: x.py\n" * 30) + "\n"
        "## TECHNICAL CONTEXT (from MOKO Knowledge Base)\n" + ("RAG context " * 200) + "\n"
        "## EXISTING PROJECT FILES (Repo Map)\n  main.py\n  player.py\n"
        "## KEY DEFINITIONS (classes & functions)\n" + ("  file.py: def func():\n" * 50) + "\n"
        "## STEP GOAL\nCreate the player module with full implementation.\n"
        "Files: player.py\n"
    )
    pruned_full = mgr.prune_prompt(fat_prompt)
    assert len(pruned_full) <= mgr.prompt_budget_chars, (
        f"Prune gagal: {len(pruned_full)} > {mgr.prompt_budget_chars}"
    )
    print(f"  ✅ prune_prompt: {len(fat_prompt)} → {len(pruned_full)} chars (dalam budget)")

    # Test 7: _smart_join overlap detection
    gen = MarathonCodeGenerator()
    a = "def hello():\n    print('world')\n\ndef main():"
    b = "def main():\n    hello()\n\nif __name__ == '__main__':\n    main()"
    joined = gen._smart_join(a, b)
    assert "def hello" in joined
    assert "def main" in joined
    # Pastikan tidak ada duplikasi "def main"
    assert joined.count("def main():") == 1, f"Duplikasi terdeteksi: {joined.count('def main():')}"
    print("  ✅ _smart_join overlap detection → OK")

    # Test 8: _smart_join tanpa overlap
    a2 = "import pygame\n\npygame.init()"
    b2 = "\n\nscreen = pygame.display.set_mode((800, 600))"
    joined2 = gen._smart_join(a2, b2)
    assert "import pygame" in joined2
    assert "set_mode" in joined2
    print("  ✅ _smart_join tanpa overlap → OK")

    print("\n🏃 ALL MARATHON SYSTEM TESTS PASSED!")
