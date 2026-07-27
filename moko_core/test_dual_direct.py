import sys
import moko_config.settings as _cfg
from moko_agents.core_node import CoreNode
from moko_agents.analyst_node import AnalystNode
from moko_memory.disk_manager import DiskManager

disk_mgr = DiskManager(_cfg.WORKSPACE_DIR)
core = CoreNode(disk_mgr)
analyst = AnalystNode(disk_mgr)

question = "Tolong periksa kernel OS kita menggunakan perintah uname -a"
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
