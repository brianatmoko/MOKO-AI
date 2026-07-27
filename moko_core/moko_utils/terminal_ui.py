import os
import psutil
from datetime import datetime

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[1;37m"
    GREEN   = "\033[1;32m"
    RED     = "\033[1;31m"
    CYAN    = "\033[1;36m"
    YELLOW  = "\033[1;33m"
    MAGENTA = "\033[1;35m"

MOKO_LOGO = r"""
     ___  ___     ________  ___  ___  ________
    |\  \|\  \   |\   __  \|\  \|\  \|\   __  \
    \ \  \\  \  \ \  \|\  \ \  \\  \ \  \|\  \
     \ \  \\  \  \ \  \\\  \ \  \\  \ \  \\\  \
      \ \  \\  \  \ \  \\\  \ \  \\  \ \  \\\  \
       \ \_______\ \ \_______\ \_______\ \_______\
        \|_______| \|_______|\|_______|\|_______|
"""

class UI:
    @staticmethod
    def clear():
        os.system('clear' if os.name == 'posix' else 'cls')
        
    @staticmethod
    def print_banner():
        UI.clear()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        for line in MOKO_LOGO.split('\n'):
            colored = ""
            for ch in line:
                if ch in ('|', '_', '-', '\\', '/'): colored += f"{C.MAGENTA}{ch}{C.RESET}"
                else: colored += ch
            print(colored)
            
        print(f"{C.MAGENTA}{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.WHITE}{C.BOLD}  🌌 MOKO OS KERNEL v1.0 [{ts}]{C.RESET}")
        print(f"{C.DIM}  C++ Super-Kernel : {C.GREEN}ONLINE (AVX/MMAP){C.RESET}")
        print(f"{C.DIM}  Python Swarm     : {C.GREEN}ACTIVE{C.RESET}")
        print(f"{C.MAGENTA}{C.BOLD}{'═' * 60}{C.RESET}")

    @staticmethod
    def get_input() -> str:
        try:
            return input(f"\n{C.MAGENTA}┌─[MOKO OS]─{C.RESET}\n{C.MAGENTA}└─► {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return "/exit"
