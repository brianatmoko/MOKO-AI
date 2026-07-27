"""
MOKO Domain-Specialized Models — Model Registry & Configuration
===============================================================
Configuration for MOKO AI specialized domain models.
"""

from pathlib import Path
from typing import Dict, Any, List


# ═══════════════════════════════════════════════════════════════════════════
# MOKO MODEL IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

MOKO_IDENTITY = {
    "name": "MOKO AI",
    "version": "1.0.0",
    "codename": "MOKO-AI",
    "build": "2026.07.04",
    "creator": "MOKO OS Project",
    "description": "100% Custom Uncensored AI — Local, Private, Powerful",
    "motto": "Bukan seberapa besar, namun seberapa efisien.",
    "hardware": "RTX 2050 4GB VRAM",
    "philosophy": [
        "100% Local — No data leaves the machine",
        "100% Uncensored — No over-filtering",
        "100% Custom — Designed for user needs",
        "Domain Specialist — Each model is an expert in its field",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_MODELS = {
    "coding": {
        "name": "MOKO Coder 1.5B",
        "description": "Expert model for programming and software engineering",
        "base_model": "MOKO-Coder-Base",
        "gguf_file": "MOKO-Coder-1.5B-Uncensored-F16.gguf",
        "quantization": "F16",
        "size_gb": 3.4,
        "params": "1.5B",
        "domains": ["code", "programming", "software"],
        "capabilities": [
            "Code generation (Python, JavaScript, Rust, Go, etc.)",
            "Debugging and error analysis",
            "Code review and refactoring",
            "Algorithm design",
            "Data structure implementation",
        ],
        "temperature": 0.0,
        "context_window": 8192,
        "system_prompt": """You are MOKO Coder, an expert programming assistant created by MOKO OS Project.

Your Identity:
- Name: MOKO Coder
- Version: 1.0.0
- Creator: MOKO OS Project
- Specialty: Programming & Software Engineering

Rules:
- Always provide working, tested code
- Include error handling when appropriate
- Follow language-specific best practices
- Be direct and technical — no unnecessary fluff""",
    },
    
    "math": {
        "name": "MOKO Math 1.5B",
        "description": "Expert model for mathematics and physics",
        "base_model": "MOKO-Coder-Base",
        "gguf_file": "MOKO-Coder-1.5B-Uncensored-F16.gguf",
        "quantization": "F16",
        "size_gb": 3.4,
        "params": "1.5B",
        "domains": ["math", "physics", "engineering"],
        "capabilities": [
            "Algebra and calculus",
            "Physics and mechanics",
            "Formula derivation",
        ],
        "temperature": 0.0,
        "context_window": 2048,
        "system_prompt": """You are MOKO Math, an expert mathematics assistant created by MOKO OS Project.

Your Identity:
- Name: MOKO Math
- Version: 1.0.0
- Creator: MOKO OS Project
- Specialty: Mathematics & Physics

Rules:
- Show all steps in calculations
- Use proper mathematical notation
- Verify your answers when possible""",
    },
    
    "general": {
        "name": "MOKO AI 4B",
        "description": "Main model for general knowledge and conversation",
        "base_model": "MOKO-AI-Base",
        "gguf_file": "MOKO-AI-4B-Q3_K_M.gguf",
        "quantization": "Q3_K_M",
        "size_gb": 2.1,
        "params": "4B",
        "domains": ["general", "personal", "conversation"],
        "capabilities": [
            "General knowledge Q&A",
            "Conversation and dialogue",
            "Creative writing",
        ],
        "temperature": 0.5,
        "context_window": 2048,
        "system_prompt": """You are MOKO, a helpful AI assistant created by MOKO OS Project.

Your Identity:
- Name: MOKO AI
- Version: 1.0.0
- Creator: MOKO OS Project
- Motto: "Bukan seberapa besar, namun seberapa efisien."

Rules:
- Be helpful and accurate
- Admit when you don't know something
- Be direct and concise — no unnecessary fluff""",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def get_model_registry(project_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Generate model registry for Multi-Model Dispatcher."""
    registry = {}
    for key, spec in DOMAIN_MODELS.items():
        registry[key] = {
            "path": str(project_dir / spec["gguf_file"]),
            "size_gb": spec["size_gb"],
            "domain": spec["domains"],
            "temperature": spec["temperature"],
            "context_window": spec["context_window"],
            "quant": spec["quantization"],
            "params": spec["params"],
        }
    return registry


def get_routing_rules() -> Dict[str, List[str]]:
    """Keywords for intent routing."""
    return {
        "coding": [
            "python", "javascript", "typescript", "rust", "golang",
            "function", "class", "method", "variable", "array",
            "code", "coding", "program", "debug", "compile",
        ],
        "math": [
            "hitung", "rumus", "matematika", "math", "calculate",
            "integral", "turunan", "kalkulus", "calculus",
            "fisika", "physics",
        ],
        "general": [],
    }


def get_system_prompts() -> Dict[str, str]:
    """System prompts for each domain."""
    return {key: spec["system_prompt"] for key, spec in DOMAIN_MODELS.items()}


def print_model_summary():
    """Print summary of all models."""
    print("\n" + "━" * 60)
    print(f"  🧠 {MOKO_IDENTITY['name']} — Domain Models")
    print("━" * 60)
    for key, spec in DOMAIN_MODELS.items():
        print(f"\n  📦 {spec['name']}")
        print(f"     Size: {spec['size_gb']:.1f} GB | File: {spec['gguf_file']}")
