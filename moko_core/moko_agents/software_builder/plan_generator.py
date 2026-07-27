"""
plan_generator.py — Plan Generator (LLM Output Parser)
=======================================================
Meng-parse output LLM (string) menjadi List[PlanStep] yang terstruktur.

Format yang di-parse:
  ## Step N: <Title>
  <Description>
  Files: file1.py, file2.py

Jika parsing gagal, fallback ke template plan generik berdasarkan tipe software.
"""
from __future__ import annotations

import re
from typing import List, Optional

from moko_agents.software_builder.models import InterviewData, PlanStep


# Regex untuk parsing format ## Step N: Title
_STEP_PATTERN = re.compile(
    r"##\s*Step\s+(\d+)\s*:\s*(.+?)\n(.*?)(?=##\s*Step\s+\d+\s*:|$)",
    re.DOTALL | re.IGNORECASE
)

# Regex untuk parsing baris Files:
_FILES_PATTERN = re.compile(
    r"Files?\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE
)


def parse_plan(llm_response: str, interview_data: Optional[InterviewData] = None) -> List[PlanStep]:
    """
    Parse output LLM menjadi List[PlanStep].
    
    Coba berbagai strategi parsing:
    1. Parse format ## Step N: Title
    2. Jika gagal, coba format alternatif (numbered list, dll.)
    3. Jika semua gagal, gunakan template generik
    
    Args:
        llm_response: Raw string dari LLM
        interview_data: Data interview untuk fallback template
    
    Returns:
        List[PlanStep] dengan minimal 5 langkah
    """
    if not llm_response or not llm_response.strip():
        print("  ⚠️ [PlanGenerator] LLM response kosong, menggunakan template")
        return _generate_fallback_plan(interview_data)

    # Strategi 1: Parse format ## Step N: Title
    steps = _parse_step_format(llm_response)

    if len(steps) >= 3:
        print(f"  ✅ [PlanGenerator] Berhasil parse {len(steps)} langkah dari format ## Step")
        return steps

    # Strategi 2: Parse numbered list (1. Title / 1) Title)
    steps = _parse_numbered_list(llm_response)
    if len(steps) >= 3:
        print(f"  ✅ [PlanGenerator] Berhasil parse {len(steps)} langkah dari numbered list")
        return steps

    # Strategi 3: Fallback ke template generik
    print(f"  ⚠️ [PlanGenerator] Parsing gagal, menggunakan template generik")
    return _generate_fallback_plan(interview_data)


def _parse_step_format(text: str) -> List[PlanStep]:
    """Parse format ## Step N: Title\nDescription\nFiles: ..."""
    steps = []
    matches = list(_STEP_PATTERN.finditer(text))

    for match in matches:
        step_num = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()

        # Pisahkan deskripsi dan files
        files_match = _FILES_PATTERN.search(body)
        files_list = []
        description = body

        if files_match:
            files_raw = files_match.group(1).strip()
            files_list = [f.strip() for f in re.split(r"[,\s]+", files_raw) if f.strip() and "." in f]
            # Hapus baris Files: dari deskripsi
            description = body[:files_match.start()].strip()

        # Bersihkan deskripsi dari karakter ekstra
        description = re.sub(r"\n+", " ", description).strip()
        if len(description) > 300:
            description = description[:300] + "..."

        steps.append(PlanStep(
            step_number=step_num,
            title=title[:80],
            description=description or f"Implementasi {title}",
            files_to_create=files_list[:6],
            status="PENDING"
        ))

    return sorted(steps, key=lambda s: s.step_number)


def _parse_numbered_list(text: str) -> List[PlanStep]:
    """
    Parse format numbered list:
    1. Title / 1) Title
    Description...
    """
    pattern = re.compile(
        r"^(\d+)[.)]\s+(.+?)(?:\n((?:(?!\d+[.)]).+\n?)*))?",
        re.MULTILINE
    )
    steps = []
    for match in pattern.finditer(text):
        step_num = int(match.group(1))
        title = match.group(2).strip()
        description = (match.group(3) or "").strip()
        description = re.sub(r"\n+", " ", description)[:200]

        # Cari files dalam deskripsi
        files_match = _FILES_PATTERN.search(description)
        files_list = []
        if files_match:
            files_raw = files_match.group(1).strip()
            files_list = [f.strip() for f in re.split(r"[,\s]+", files_raw) if f.strip() and "." in f]

        steps.append(PlanStep(
            step_number=step_num,
            title=title[:80],
            description=description or f"Implementasi {title}",
            files_to_create=files_list[:6],
            status="PENDING"
        ))

    return sorted(steps, key=lambda s: s.step_number)


# ─── Template Plans (Fallback) ───────────────────────────────────────────────

def _generate_fallback_plan(interview_data: Optional[InterviewData] = None) -> List[PlanStep]:
    """
    Generate template plan generik berdasarkan tipe software dari InterviewData.
    Digunakan saat LLM gagal menghasilkan plan yang parseable.
    """
    if interview_data is None:
        return _generic_fallback()

    stype = interview_data.software_type.lower()

    if "game" in stype:
        return _game_template(interview_data)
    elif "web" in stype:
        return _web_template(interview_data)
    else:
        return _tool_template(interview_data)


def _game_template(data: InterviewData) -> List[PlanStep]:
    """Template plan untuk game."""
    lang = data.language
    ext = "py" if lang == "python" else "js"
    game_name = data.sub_type.lower().replace(" ", "_") or "game"
    lib = "pygame" if lang == "python" else "phaser"

    return [
        PlanStep(1, "Setup Project & Dependencies",
                 f"Buat direktori proyek, install {lib}, setup file utama",
                 [f"main.{ext}", f"requirements.txt"],
                 "PENDING"),
        PlanStep(2, "Game Window & Main Loop",
                 f"Inisialisasi {lib}, buat window game, implementasi game loop utama",
                 [f"main.{ext}", f"{game_name}.{ext}"],
                 "PENDING"),
        PlanStep(3, "Player Character & Movement",
                 "Buat class Player dengan sprite, input keyboard, physics movement",
                 [f"player.{ext}"],
                 "PENDING"),
        PlanStep(4, "Game World & Collisions",
                 "Buat environment, platform/terrain, collision detection system",
                 [f"world.{ext}", f"collision.{ext}"],
                 "PENDING"),
        PlanStep(5, "Game Mechanics & Scoring",
                 f"Implementasi mechanic utama: {', '.join(data.mechanics[:3])}, sistem skor",
                 [f"mechanics.{ext}", f"score.{ext}"],
                 "PENDING"),
        PlanStep(6, "UI & Game States",
                 "Menu utama, HUD in-game, game over screen, pause menu",
                 [f"ui.{ext}", f"states.{ext}"],
                 "PENDING"),
        PlanStep(7, "Polish & Finalize",
                 "Sound effects, animasi, balancing, testing akhir, dokumentasi",
                 [f"assets.{ext}", f"config.{ext}"],
                 "PENDING"),
    ]


def _web_template(data: InterviewData) -> List[PlanStep]:
    """Template plan untuk web app."""
    lang = data.language
    ext = "py" if lang == "python" else "js"
    fw = "Flask" if lang == "python" else "Express"
    app_name = data.sub_type.lower().replace(" ", "_") or "app"

    return [
        PlanStep(1, "Setup Project & Framework",
                 f"Inisialisasi proyek {fw}, setup dependencies, struktur folder",
                 [f"app.{ext}", "requirements.txt"],
                 "PENDING"),
        PlanStep(2, "Database Models",
                 "Definisi model data, setup database (SQLite/PostgreSQL)",
                 [f"models.{ext}", "database.py"],
                 "PENDING"),
        PlanStep(3, "API Routes & Controllers",
                 f"Implementasi REST API endpoints untuk {', '.join(data.mechanics[:2])}",
                 [f"routes.{ext}", f"controllers.{ext}"],
                 "PENDING"),
        PlanStep(4, "Authentication & Security",
                 "JWT/Session auth, password hashing, middleware security",
                 [f"auth.{ext}", f"middleware.{ext}"],
                 "PENDING"),
        PlanStep(5, "Frontend UI",
                 "HTML templates, CSS styling, JavaScript interactivity",
                 ["templates/index.html", "static/style.css", "static/app.js"],
                 "PENDING"),
        PlanStep(6, "Testing & Deployment",
                 "Unit tests, integration tests, dokumentasi API",
                 [f"tests.{ext}", "README.md"],
                 "PENDING"),
    ]


def _tool_template(data: InterviewData) -> List[PlanStep]:
    """Template plan untuk tool/automation."""
    lang = data.language
    ext = "py" if lang == "python" else "js"

    return [
        PlanStep(1, "Setup & Configuration",
                 "Inisialisasi proyek, setup dependencies, config file",
                 [f"main.{ext}", "config.yaml"],
                 "PENDING"),
        PlanStep(2, "Core Logic",
                 f"Implementasi logika utama untuk {data.sub_type}",
                 [f"core.{ext}"],
                 "PENDING"),
        PlanStep(3, "Input/Output Handler",
                 "Parsing input, validasi data, format output",
                 [f"io_handler.{ext}"],
                 "PENDING"),
        PlanStep(4, "Feature Implementation",
                 f"Implementasi fitur: {', '.join(data.mechanics[:3])}",
                 [f"features.{ext}"],
                 "PENDING"),
        PlanStep(5, "Testing & Documentation",
                 "Unit tests, usage examples, README",
                 [f"test_{lang}.{ext}", "README.md"],
                 "PENDING"),
    ]


def _generic_fallback() -> List[PlanStep]:
    """Template paling generik jika tidak ada info apapun."""
    return [
        PlanStep(1, "Project Setup", "Setup struktur proyek dan dependencies", ["main.py"], "PENDING"),
        PlanStep(2, "Core Module", "Implementasi modul utama", ["core.py"], "PENDING"),
        PlanStep(3, "Features", "Tambahkan fitur utama", ["features.py"], "PENDING"),
        PlanStep(4, "Testing", "Tulis dan jalankan tests", ["test_main.py"], "PENDING"),
        PlanStep(5, "Documentation", "Buat README dan dokumentasi", ["README.md"], "PENDING"),
    ]


if __name__ == "__main__":
    print("=== Unit Test: plan_generator.py ===\n")

    # Test parsing format ## Step N:
    sample_llm_output = """
## Step 1: Project Setup
Create the project directory and install pygame library.
Files: main.py, requirements.txt

## Step 2: Game Window
Initialize pygame and create the game window with main loop.
Files: main.py, game.py

## Step 3: Player Character
Implement the player class with movement and collision.
Files: player.py

## Step 4: Enemy System
Create enemy class with basic AI pathfinding.
Files: enemy.py, ai.py

## Step 5: Scoring & UI
Add score counter, health bar, and game over screen.
Files: ui.py, score.py
"""

    steps = parse_plan(sample_llm_output)
    assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}"
    assert steps[0].step_number == 1
    assert steps[0].title == "Project Setup"
    assert "main.py" in steps[0].files_to_create
    assert steps[4].step_number == 5
    print(f"✅ parse_plan (## Step format): {len(steps)} steps → OK")
    for s in steps:
        print(f"   Step {s.step_number}: {s.title} | Files: {s.files_to_create}")

    # Test parsing gagal → fallback
    steps_fallback = parse_plan("ini bukan format yang benar sama sekali", None)
    assert len(steps_fallback) >= 5
    print(f"\n✅ parse_plan (fallback generic): {len(steps_fallback)} steps → OK")

    # Test fallback dengan InterviewData game
    data = InterviewData(
        software_type="game", sub_type="platformer",
        mechanics=["movement", "collision"], language="python",
        platform="desktop", complexity="medium"
    )
    steps_game = parse_plan("bukan format valid", data)
    assert len(steps_game) >= 5
    assert any("main.py" in s.files_to_create or "game.py" in s.files_to_create
               for s in steps_game)
    print(f"✅ fallback game template: {len(steps_game)} steps → OK")

    print("\n✅ Semua unit test plan_generator berhasil!")
