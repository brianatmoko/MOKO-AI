"""
prompt_enrichment.py — Prompt Enrichment Engine (UPGRADED v2)
=============================================================
Mengubah InterviewData + konteks RAG menjadi super-prompt yang
kaya konteks untuk dikirim ke LLM.

Teknik yang diimplementasikan (berdasarkan riset Aider, Continue.dev, Cursor):
  - Anti-laziness instructions (dari Aider unified diffs research)
  - Few-shot examples per tipe software
  - Chain-of-thought planning section
  - Strict output format enforcement
  - SEARCH/REPLACE format hints untuk code steps

Alur:
  1. Terima InterviewData dari InterviewManager
  2. Query RAGAgent untuk konteks relevan (struktur proyek, best practice)
  3. Pilih few-shot example yang relevan berdasarkan software_type
  4. Gabungkan menjadi super-prompt terstruktur dengan anti-laziness rules
  5. Return super-prompt siap kirim ke CoreNode/LLM
"""
from __future__ import annotations

from typing import Optional

from moko_agents.software_builder.models import InterviewData


# ─── Few-Shot Examples per Software Type ─────────────────────────────────────
# Teknik dari Aider & Continue.dev: show 1 concrete example → LLM mimics format

_FEW_SHOT_GAME = """## EXAMPLE OUTPUT (for a simple game project):

## Step 1: Project Setup & Dependencies
Install pygame, create project structure with main entry point and requirements file.
Files: main.py, requirements.txt

## Step 2: Game Window & Main Loop
Create a 800x600 pygame window with a proper game loop, clock, and event handling.
Files: game.py

## Step 3: Player Class
Implement a Player sprite class with movement, animation states, and physics.
Files: player.py

## Step 4: World & Collision System
Build the game world with tile map loading and AABB collision detection.
Files: world.py, tilemap.py

## Step 5: Game State Manager
Implement game states (menu, playing, game_over) with transitions.
Files: states.py

## Step 6: Score & HUD
Create a HUD class showing score, lives, and level progress.
Files: hud.py

## Step 7: Main Entry & Polish
Wire everything together in main.py with error handling and a proper game loop.
Files: main.py"""


_FEW_SHOT_WEBAPP = """## EXAMPLE OUTPUT (for a web app project):

## Step 1: Project Setup & Dependencies
Initialize project with requirements.txt, folder structure (app/, static/, templates/).
Files: requirements.txt, app/__init__.py

## Step 2: Database Models
Define SQLAlchemy/SQLite models for all data entities with relationships.
Files: app/models.py

## Step 3: Core Routes & Controllers
Implement Flask/FastAPI routes for CRUD operations with proper HTTP methods.
Files: app/routes.py

## Step 4: HTML Templates & UI
Create Jinja2/HTML templates with Bootstrap for a clean responsive layout.
Files: templates/base.html, templates/index.html

## Step 5: Authentication System
Implement user login, registration, and session management.
Files: app/auth.py, templates/login.html

## Step 6: Static Assets & Styling
Add CSS/JS assets and configure static file serving.
Files: static/style.css, static/app.js

## Step 7: Run Configuration
Create run.py entry point with environment config and database initialization.
Files: run.py, app/config.py"""


_FEW_SHOT_TOOL = """## EXAMPLE OUTPUT (for a tool/utility project):

## Step 1: Project Setup
Create project structure with entry point, requirements, and argument parsing.
Files: main.py, requirements.txt

## Step 2: Core Logic Module
Implement the main processing logic with proper class structure and error handling.
Files: core.py

## Step 3: Input/Output Handlers
Build functions for reading input (file/stdin/args) and writing output (file/stdout).
Files: io_handler.py

## Step 4: Configuration & Settings
Add config file support (JSON/YAML/INI) with sensible defaults.
Files: config.py, settings.json

## Step 5: CLI Interface
Build a polished CLI with click/argparse, help text, and progress indicators.
Files: cli.py

## Step 6: Tests & Validation
Write unit tests covering core logic, edge cases, and error conditions.
Files: test_core.py

## Step 7: Entry Point & Packaging
Wire up entry points, add README, and make it distributable.
Files: main.py, README.md"""


# ─── Anti-Laziness Rules (dari riset Aider unified diffs) ─────────────────────
# Aider research: prompts telling LLM to write COMPLETE code (tidak lazy comments)
# meningkatkan benchmark score 3x

_ANTI_LAZINESS_RULES = """## CRITICAL RULES — READ CAREFULLY
You are writing REAL CODE that will be executed directly. Follow these rules EXACTLY:

1. NEVER write lazy placeholder comments like:
   - "# ... rest of implementation here"
   - "# TODO: implement this"
   - "# [Add your code here]"
   - "# ... (similar for other cases)"
   
2. ALWAYS write COMPLETE, WORKING code — every function must have a full body.

3. EVERY import must be valid and available (use only stdlib + the specific lib mentioned).

4. Code must run without modification — no placeholder variables, no missing pieces.

5. If a step creates multiple files, separate them with:
   # === FILE: filename.py ===
   [complete file content here]
   # === END FILE ==="""


# ─── Plan Generation Template (v2 — enhanced) ────────────────────────────────
_PLAN_TEMPLATE_V2 = """You are MOKO, an expert software engineer and senior developer.
Your task: create a detailed, executable implementation plan for a software project.

{anti_laziness}

## PROJECT REQUIREMENTS
{requirements}

## TECHNICAL CONTEXT (from MOKO Knowledge Base)
{rag_context}

## YOUR THINKING PROCESS (Chain of Thought)
Before writing the plan, think about:
1. What is the minimal viable architecture for this {software_type}?
2. What dependencies are needed (list specific libraries)?
3. In what logical order should the components be built?
4. What are the potential failure points and how to avoid them?

## FORMAT — FOLLOW EXACTLY (machine-parsed)
Each step MUST be in this exact format:

## Step N: <Specific, Action-Oriented Title>
<2-3 sentence description of exactly what gets implemented>
Files: <filename1.ext>, <filename2.ext>

{few_shot_example}

## YOUR PLAN FOR THIS PROJECT
Now create the complete implementation plan for: {project_title}
Language: {language} | Complexity: {complexity} | Min steps: 5, Max steps: 12

Start with "## Step 1:" and continue until all necessary steps are defined:"""


def _get_few_shot_example(software_type: str) -> str:
    """Pilih few-shot example yang relevan berdasarkan tipe software."""
    st = software_type.lower()
    if "game" in st:
        return _FEW_SHOT_GAME
    elif "web" in st:
        return _FEW_SHOT_WEBAPP
    else:
        return _FEW_SHOT_TOOL


def build_super_prompt(
    interview_data: InterviewData,
    rag_context: str = ""
) -> str:
    """
    Bangun super-prompt v2 dari InterviewData + konteks RAG.
    
    Menggunakan teknik terbaik dari Aider, Continue.dev, dan Cursor:
    - Anti-laziness rules
    - Few-shot examples per tipe software
    - Chain-of-thought section
    - Strict format enforcement
    
    Args:
        interview_data: Data hasil interview dari InterviewManager
        rag_context: Konteks dari RAGAgent (optional, bisa kosong)
    
    Returns:
        Super-prompt string siap kirim ke LLM (upgrade v2)
    """
    mechanics_str = ", ".join(interview_data.mechanics) if interview_data.mechanics else "basic features"
    project_title = f"{interview_data.sub_type} {interview_data.software_type}"

    requirements = (
        f"Software Type: {interview_data.software_type.upper()} ({interview_data.sub_type})\n"
        f"Key Features/Mechanics: {mechanics_str}\n"
        f"Programming Language: {interview_data.language}\n"
        f"Target Platform: {interview_data.platform or 'desktop'}\n"
        f"Complexity Level: {interview_data.complexity}\n"
        + (f"Additional Notes: {interview_data.extra_notes}\n" if interview_data.extra_notes else "")
    )

    context_section = rag_context.strip() if rag_context else (
        f"Use standard {interview_data.language} best practices.\n"
        f"Recommend well-known libraries: "
        + _get_recommended_libs(interview_data)
    )

    few_shot = _get_few_shot_example(interview_data.software_type)

    prompt = _PLAN_TEMPLATE_V2.format(
        anti_laziness=_ANTI_LAZINESS_RULES,
        requirements=requirements,
        rag_context=context_section,
        software_type=interview_data.software_type,
        few_shot_example=few_shot,
        project_title=project_title,
        language=interview_data.language,
        complexity=interview_data.complexity,
    )

    return prompt


def _get_recommended_libs(interview_data: InterviewData) -> str:
    """Return recommended library string berdasarkan language dan software type."""
    lang = interview_data.language.lower()
    stype = interview_data.software_type.lower()

    lib_map = {
        ("python", "game"): "pygame>=2.0, numpy",
        ("python", "web app"): "flask>=2.0 or fastapi>=0.100, sqlalchemy, jinja2",
        ("python", "tool"): "click>=8.0, rich, pathlib (stdlib)",
        ("python", "automation"): "requests, selenium or playwright, schedule",
        ("javascript", "game"): "Phaser 3, or plain HTML5 Canvas API",
        ("javascript", "web app"): "Express.js, mongoose (MongoDB) or sqlite3",
        ("javascript", "tool"): "commander.js, chalk, fs-extra",
    }
    key = (lang, stype)
    return lib_map.get(key, f"standard {lang} ecosystem libraries")


def build_rag_query(interview_data: InterviewData) -> str:
    """
    Bangun query untuk RAGAgent berdasarkan InterviewData.
    Query yang baik meningkatkan relevansi konteks yang didapat.
    """
    parts = []

    # Query utama: struktur proyek
    if interview_data.software_type == "game":
        lang_lib = {
            "python": "pygame",
            "javascript": "Phaser",
            "c++": "SDL2 SFML",
            "c#": "Unity MonoBehaviour",
        }.get(interview_data.language, interview_data.language)
        parts.append(f"{lang_lib} {interview_data.sub_type} game structure")
        parts.append(f"game loop game state machine {interview_data.language}")
    elif interview_data.software_type == "web app":
        lang_fw = {
            "python": "Flask FastAPI",
            "javascript": "Express Node.js",
            "typescript": "TypeScript Express",
        }.get(interview_data.language, interview_data.language)
        parts.append(f"{lang_fw} web application structure")
        parts.append(f"REST API {interview_data.sub_type} {interview_data.language}")
    else:
        parts.append(f"{interview_data.language} {interview_data.software_type} project structure")

    # Query untuk mechanics
    if interview_data.mechanics:
        top_mechanics = interview_data.mechanics[:3]
        parts.append(f"{' '.join(top_mechanics)} implementation {interview_data.language}")

    return " | ".join(parts)


def get_rag_context(interview_data: InterviewData) -> str:
    """
    Ambil konteks relevan dari RAGAgent berdasarkan InterviewData.
    Fallback ke string kosong jika RAG tidak tersedia.
    """
    try:
        from moko_agents.rag_agent import get_rag_agent
        agent = get_rag_agent()
        query = build_rag_query(interview_data)
        context = agent.search_context(query, top_k=4)
        return context or ""
    except Exception as e:
        print(f"  ⚠️ [PromptEnrichment] RAG context error: {e}")
        return ""


def enrich_prompt(interview_data: InterviewData) -> str:
    """
    Entry point utama: ambil RAG context dan bangun super-prompt.
    
    Args:
        interview_data: Data lengkap dari interview
    
    Returns:
        Super-prompt string yang sudah diperkaya dengan konteks RAG
    """
    print(f"  🔍 [PromptEnrichment] Fetching RAG context for: {interview_data.software_type} ({interview_data.sub_type})")
    rag_context = get_rag_context(interview_data)

    if rag_context:
        print(f"  ✅ [PromptEnrichment] RAG context found ({len(rag_context)} chars)")
    else:
        print(f"  ⚠️ [PromptEnrichment] No RAG context, using defaults")

    super_prompt = build_super_prompt(interview_data, rag_context)
    print(f"  ✅ [PromptEnrichment] Super-prompt built ({len(super_prompt)} chars)")
    return super_prompt


if __name__ == "__main__":
    # Unit test sederhana
    print("=== Unit Test: prompt_enrichment.py ===\n")

    data = InterviewData(
        software_type="game",
        sub_type="platformer",
        mechanics=["player movement", "collision", "scoring"],
        language="python",
        platform="desktop",
        complexity="medium"
    )

    # Test build_rag_query
    query = build_rag_query(data)
    assert "pygame" in query.lower() or "python" in query.lower()
    print(f"✅ build_rag_query: '{query[:60]}...' → OK")

    # Test build_super_prompt (tanpa RAG)
    prompt = build_super_prompt(data, "Test RAG context here")
    assert "## Step" in prompt
    assert "python" in prompt.lower()
    assert "platformer" in prompt.lower()
    print(f"✅ build_super_prompt: {len(prompt)} chars, berisi format ## Step → OK")

    # Test tanpa RAG context
    prompt_no_rag = build_super_prompt(data, "")
    assert "No specific context" in prompt_no_rag
    print(f"✅ build_super_prompt tanpa RAG: fallback text ada → OK")

    print("\n✅ Semua unit test prompt_enrichment berhasil!")
