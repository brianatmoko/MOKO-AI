import json
from pathlib import Path
def is_local_llm_enabled():
    try:
        p = Path(__file__).parent.parent.parent / "moko_config" / "moko_settings.json"
        if p.exists():
            with open(p, "r") as f:
                cfg = json.load(f)
                return cfg.get("local_llm_enabled", False)
    except Exception:
        pass
    return True
print(is_local_llm_enabled())
