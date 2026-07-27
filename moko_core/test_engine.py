import sys
from moko_agents.llm_engine import engine

print("Memulai...")
ans = engine.generate_text("hai", "Kamu adalah MOKO.")
print(f"Jawaban: '{ans}'")
