#!/usr/bin/env python3
import sys
from pathlib import Path

# Impor Modul-modul Sistem Operasi MOKO
from moko_utils.terminal_ui import UI, C
from moko_config import settings
from moko_memory.disk_manager import DiskManager
from moko_agents.analyst_node import AnalystNode
from moko_agents.core_node import CoreNode
from moko_agents.router import CognitiveRouter

def main():
    UI.print_banner()
    
    workspace = settings.WORKSPACE_DIR
    
    print(f"{C.DIM}Mem-boot C++ Kernel dan Manajer Memori...{C.RESET}")
    disk_mgr = DiskManager(workspace)
    
    print(f"{C.DIM}Menghubungkan Jaringan Saraf Agen...{C.RESET}")
    analyst = AnalystNode(disk_mgr)
    core = CoreNode(disk_mgr)

    print(f"{C.GREEN}✓ Boot Sequence Selesai.{C.RESET}")

    while True:
        question = UI.get_input()
        
        if not question: continue
        if question.lower() in ("/exit", "exit"):
            print(f"\n{C.YELLOW}MOKO OS Shutdown.{C.RESET}")
            break
            
        if question.lower() == "/ai off":
            import json
            p = settings.PROJECT_DIR / "moko_config" / "moko_settings.json"
            try:
                with open(p, "r") as f: cfg = json.load(f)
                cfg["local_llm_enabled"] = False
                with open(p, "w") as f: json.dump(cfg, f, indent=4)
                print(f"{C.YELLOW}Local AI dinonaktifkan!{C.RESET}")
            except Exception as e:
                print(f"{C.RED}Gagal: {e}{C.RESET}")
            continue

        if question.lower() == "/ai on":
            import json
            p = settings.PROJECT_DIR / "moko_config" / "moko_settings.json"
            try:
                with open(p, "r") as f: cfg = json.load(f)
                cfg["local_llm_enabled"] = True
                with open(p, "w") as f: json.dump(cfg, f, indent=4)
                print(f"{C.GREEN}Local AI diaktifkan!{C.RESET}")
            except Exception as e:
                print(f"{C.RED}Gagal: {e}{C.RESET}")
            continue
        
        # 0. Route classification — tentukan path (FAST_PATH, DEEP_PATH, BROWSING_PATH, etc)
        route_path, route_reason, route_meta = CognitiveRouter.classify_intent(question)
        
        # Log routing decision
        domain = route_meta.get("domain", "unknown")
        require_web = route_meta.get("require_web_search", False)
        print(f"\n{C.DIM}[Router] Path: {route_path} | Domain: {domain} | Reason: {route_reason}{C.RESET}")
        if require_web:
            print(f"{C.DIM}[Router] 🌐 Web search enabled for this query{C.RESET}")
            
        # 1. Analyst Phase (Iterative DeepThink)
        print(f"{C.CYAN}  [Analyst] Sedang merenung...{C.RESET}")
        analyst_thoughts = analyst.deep_think_loop(question, route_path=route_path, route_meta=route_meta)
        print(f"{C.CYAN}  ✓ {analyst_thoughts}{C.RESET}")
        
        # 2. Core Phase (Uncensored Output)
        print(f"\n{C.MAGENTA}  [MOKO CORE]{C.RESET}")
        final_answer = core.amplify_response(question, analyst_thoughts, route_path=route_path, route_meta=route_meta)
        print(f"{C.WHITE}{final_answer}{C.RESET}\n")

if __name__ == "__main__":
    main()
