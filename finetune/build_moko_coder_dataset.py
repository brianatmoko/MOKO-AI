import json
import os
import re
from pathlib import Path

# Config
PROJECT_DIR = Path(__file__).resolve().parent.parent
HACKING_DIR = PROJECT_DIR / "Ethical-Hacking-Course-Bank-main"
MOKO_CORE_DIR = PROJECT_DIR / "moko_core"
OUTPUT_FILE = PROJECT_DIR / "finetune" / "moko_datasets" / "hacking_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are MOKO Coder, a specialized AI assistant for programming, cybersecurity, "
    "and ethical hacking. You provide precise, secure, and high-quality code. "
    "You follow MOKO OS principles and answer in Indonesian by default. "
    "Selalu berpikir dulu di dalam blok <thought>...</thought> (penalaran CoT ala "
    "Sistem 2/DeepSeek), lalu tuliskan tindakan/kode di dalam blok "
    "<action>...</action> (eksekusi ala Sistem 1/Kimi)."
)

# ── Agentic SFT Format (DeepSeek CoT <thought> + Kimi action <action>) ─────────
THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"
ACTION_OPEN, ACTION_CLOSE = "<action>", "</action>"


def build_thought(user_msg: str) -> str:
    """Hasilkan blok penalaran CoT ringkas (Sistem 2) untuk sebuah instruksi."""
    return (
        f"Menganalisis instruksi: {user_msg.strip()} "
        "Menyusun langkah solusi yang tepat, aman, dan konsisten berbahasa Indonesia."
    )


def make_agentic_content(thought: str, action_body: str) -> str:
    """Bungkus jawaban ke format agentic berpasangan <thought>/<action>."""
    return (
        f"{THOUGHT_OPEN}\n{thought.strip()}\n{THOUGHT_CLOSE}\n"
        f"{ACTION_OPEN}\n{action_body.strip()}\n{ACTION_CLOSE}"
    )


def _extract_python_blocks(text: str):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return [b for b in blocks if b.strip()]


def verify_python_syntax(code: str) -> bool:
    """Verifiable Reward: kode Python harus lolos kompilasi (syntax valid)."""
    try:
        compile(code, "<sft_sample>", "exec")
        return True
    except Exception:
        return False


def compute_rewards(assistant_content: str, expect_code: bool = False) -> dict:
    """Metrik reward multi-objektif: Verifiable, Format, dan Konsistensi Bahasa."""
    has_format = all(
        tag in assistant_content
        for tag in (THOUGHT_OPEN, THOUGHT_CLOSE, ACTION_OPEN, ACTION_CLOSE)
    )
    # Verifiable Reward: kompilasi blok kode python bila ada.
    py_blocks = _extract_python_blocks(assistant_content)
    if py_blocks:
        verifiable = 1.0 if all(verify_python_syntax(b) for b in py_blocks) else 0.0
    else:
        verifiable = 0.0 if expect_code else None
    # Konsistensi bahasa (heuristik penanda bahasa Indonesia).
    id_markers = ("yang", "adalah", "untuk", "berikut", "dengan", "ini")
    language = 1.0 if any(m in assistant_content.lower() for m in id_markers) else 0.0
    scores = {"format": 1.0 if has_format else 0.0, "verifiable": verifiable,
              "language": language}
    present = [v for v in scores.values() if v is not None]
    scores["reward_total"] = round(sum(present) / len(present), 4) if present else 0.0
    return scores


def make_sample(user_msg: str, thought: str, action_body: str, expect_code: bool = False) -> dict:
    """Bangun satu sampel SFT agentic lengkap dengan metrik reward."""
    content = make_agentic_content(thought, action_body)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": content},
        ],
        "reward": compute_rewards(content, expect_code=expect_code),
    }


def clean_text(text):
    return text.strip()

def process_hacking_docs():
    samples = []
    if not HACKING_DIR.exists():
        print(f"Warning: {HACKING_DIR} not found")
        return []
    
    for md_file in HACKING_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Split by H2 or H1
        sections = re.split(r'\n#+\s+', content)
        for section in sections:
            lines = section.strip().split('\n')
            if not lines: continue
            
            title = lines[0]
            body = '\n'.join(lines[1:])
            
            if len(body) < 50: continue
            
            # Create a sample (format agentic <thought>/<action>)
            user_msg = f"Jelaskan tentang {title} dalam konteks ethical hacking."
            samples.append(make_sample(user_msg, build_thought(user_msg), body))
            
            # Create another variant for code blocks if present
            code_blocks = re.findall(r'```(?:bash|python|php|sql|js)?\n(.*?)```', body, re.DOTALL)
            for block in code_blocks:
                if len(block) < 20: continue
                user_code = f"Berikan contoh perintah atau kode untuk: {title}"
                samples.append(make_sample(
                    user_code, build_thought(user_code),
                    f"```\n{block.strip()}\n```", expect_code=True))
    return samples

def process_moko_core():
    samples = []
    # Index more files to teach Moko about itself
    print("Scanning moko_core for Python files...")
    
    count = 0
    max_core_samples = 1000 # Limit to avoid bloat
    
    for py_file in MOKO_CORE_DIR.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            content = py_file.read_text(encoding="utf-8")
            if len(content) < 200: continue
            
            rel_path = py_file.relative_to(PROJECT_DIR)
            
            user_msg = f"Berikan contoh kode dari module `{rel_path}` di MOKO OS."
            action_body = f"Berikut adalah potongan kode dari `{rel_path}`:\n\n```python\n{content[:2000]}\n```"
            samples.append(make_sample(user_msg, build_thought(user_msg), action_body))
            count += 1
            if count >= max_core_samples:
                break
        except Exception:
            continue
            
    print(f"Extracted {count} samples from moko_core.")
    return samples

def generate_synthetic_coding():
    # Simple synthetic data to boost coding skills
    coding_tasks = [
        ("Python binary search", "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"),
        ("FastAPI hello world", "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}"),
        ("SQL injection prevention", "To prevent SQL injection in Python, use parameterized queries:\n\n```python\n# BAD\ncursor.execute(f'SELECT * FROM users WHERE name = \"{name}\"')\n\n# GOOD\ncursor.execute('SELECT * FROM users WHERE name = %s', (name,))\n```"),
        ("C++ reverse string", "#include <iostream>\n#include <string>\n#include <algorithm>\n\nstd::string reverse(std::string s) {\n    std::reverse(s.begin(), s.end());\n    return s;\n}"),
    ]
    
    samples = []
    for task, code in coding_tasks:
        user_msg = f"Buatlah kode untuk {task}."
        thought = (
            f"Permintaan: {task}. Merancang solusi paling ringkas & benar, "
            "memverifikasi sintaksis sebelum dikirim (Verifiable Reward)."
        )
        action_body = f"Tentu, berikut adalah implementasi {task}:\n\n```\n{code}\n```"
        samples.append(make_sample(user_msg, thought, action_body, expect_code=True))
    return samples

def main():
    all_samples = []
    print("Processing hacking docs...")
    all_samples.extend(process_hacking_docs())
    print("Processing MOKO core...")
    all_samples.extend(process_moko_core())
    print("Generating synthetic coding data...")
    all_samples.extend(generate_synthetic_coding())
    
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Ringkasan metrik reward (Verifiable / Format / Language).
    total = len(all_samples) or 1
    fmt_ok = sum(1 for s in all_samples if s.get("reward", {}).get("format") == 1.0)
    verified = sum(1 for s in all_samples if s.get("reward", {}).get("verifiable") == 1.0)
    avg_reward = round(
        sum(s.get("reward", {}).get("reward_total", 0.0) for s in all_samples) / total, 4
    )
    print(f"Success! Generated {len(all_samples)} samples in {OUTPUT_FILE}")
    print(f"  Agentic format OK : {fmt_ok}/{len(all_samples)}")
    print(f"  Verifiable (code) : {verified} sampel lolos kompilasi")
    print(f"  Rata-rata reward  : {avg_reward}")

if __name__ == "__main__":
    main()
