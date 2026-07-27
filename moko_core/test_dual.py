import sys
from moko_ui.workers.cognitive_worker import set_mode, get_current_mode, core, analyst, _cfg

set_mode("DUAL")
print("AI_MODE:", get_current_mode())
print("MODEL_LLM default:", _cfg.MODEL_LLM)
print("MODEL_DOLPHIN:", _cfg.MODEL_DOLPHIN)

question = "jalankan perintah pwd dan beri tahu saya posisiku"
print(f"Mengirim pertanyaan: '{question}' ke Analyst (Dolphin)...")

# Kita tes Analyst
thoughts = analyst.deep_think_loop(question, model_override=_cfg.MODEL_DOLPHIN)
print("="*40)
print("THOUGHTS:")
print(thoughts)
print("="*40)

# Kita tes Core (MOKO)
final = core.amplify_response(question, thoughts)
print("FINAL:")
print(final)
