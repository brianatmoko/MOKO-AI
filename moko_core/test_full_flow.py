from moko_memory.disk_manager import DiskManager
from moko_agents.llm_engine import engine
from moko_agents.core_node import CoreNode
from moko_config import settings

disk_mgr = DiskManager(settings.WORKSPACE_DIR)
core = CoreNode(disk_mgr)

# Print system prompt yang sebenarnya dikirim ke model
sys_prompt = core.generate_system_prompt("hai")
print("=" * 60)
print("SYSTEM PROMPT YANG DIKIRIM KE MODEL:")
print(sys_prompt)
print("=" * 60)

# Sekarang test jawaban aktual
print("\nMenunggu jawaban dari model...")
ans = core.quick_reply("hai")
print(f"\nJAWABAN MODEL: '{ans}'")
