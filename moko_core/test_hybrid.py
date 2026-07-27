import sys
sys.path.insert(0, '.')
from unittest.mock import patch, MagicMock
from moko_agents.hybrid_chat_engine import HybridChatEngine
from moko_agents.core_node import CoreNode
from moko_memory.disk_manager import DiskManager

@patch("moko_agents.dual_system.api_client.requests.post")
@patch("moko_agents.dual_system.worker_pool.WorkerPool._fetch_gateway_models")
def run_test(mock_fetch, mock_post):
    mock_fetch.return_value = ["omniroute-fast"]
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Halo dari OmniRoute!"}}]}
    mock_post.return_value = mock_resp
    
    node = CoreNode(DiskManager("/tmp"))
    
    pool = HybridChatEngine._get_pool()
    pool.active_clients = ["omniroute-free"]
    
    result = node.quick_reply("Halo Moko")
    print("Result:", result)

run_test()
