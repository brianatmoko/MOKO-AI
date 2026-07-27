import os
import json
from pathlib import Path
from moko_config import settings

# Status konstanta
STATUS_LOCKED = "locked"      # Terkunci karena prasyarat belum terpenuhi 🔒
STATUS_AVAILABLE = "available"  # Tersedia untuk dipelajari 🔓
STATUS_ACTIVE = "active"      # Sedang dalam proses belajar 🌀
STATUS_MASTERED = "mastered"  # Sudah dikuasai/selesai ✅

CURRICULUM_DATA = {
    # ── Subject 1: Sovereign AI Development ─────────────────────────────────
    "ai_foundations": {
        "subject": "Sovereign AI",
        "title": "AI Foundations & Embeddings",
        "description": "Prinsip dasar representasi vektor, cosine similarity, dan struktur database kognitif Omni-Index.",
        "keywords": ["vector embeddings", "cosine similarity", "dense vector search", "cognitive database structure"],
        "prerequisites": [],
        "min_mastery_satisfied": 85.0
    },
    "free_energy_principle": {
        "subject": "Sovereign AI",
        "title": "Free Energy Principle (FEP)",
        "description": "Koreksi prediktif Karl Friston, tingkat keheranan (Surprisal), dan error cascade prediktif.",
        "keywords": ["Karl Friston Free Energy", "predictive coding hierarchy", "sensory surprisal error cascade", "active inference theory"],
        "prerequisites": ["ai_foundations"],
        "min_mastery_satisfied": 85.0
    },
    "active_inference": {
        "subject": "Sovereign AI",
        "title": "Active Inference & Adaptation",
        "description": "Pemberian tindakan berdasarkan koreksi persepsi dan proses mutasi kognitif AI.",
        "keywords": ["active inference action selection", "formula mutation cognitive AI", "sensory prediction error", "generative model adaptation"],
        "prerequisites": ["free_energy_principle"],
        "min_mastery_satisfied": 85.0
    },
    "synaptic_plasticity": {
        "subject": "Sovereign AI",
        "title": "BCM Synaptic Plasticity & LTP/LTD",
        "description": "Model Biersack-Cooper-Munro, Long-Term Potentiation, dan Oja's Rule untuk penguatan memori.",
        "keywords": ["Bienenstock Cooper Munro BCM", "synaptic plasticity LTP LTD", "Ojas rule synaptic weight", "Hebbian association network"],
        "prerequisites": ["active_inference"],
        "min_mastery_satisfied": 85.0
    },

    # ── Subject 2: Offensive Cyber & Onion Network ──────────────────────────
    "network_security": {
        "subject": "Offensive Cyber",
        "title": "Network Security Basics",
        "description": "Dasar-dasar routing, analisis paket, port scanning, dan protokol jaringan aman.",
        "keywords": ["network packet analysis", "port scanning techniques", "nmap command tutorial", "tcp handshakes security"],
        "prerequisites": [],
        "min_mastery_satisfied": 85.0
    },
    "onion_anonymity": {
        "subject": "Offensive Cyber",
        "title": "Onion Routing & Tor Anonymity",
        "description": "Sistem Tor, relay jaringan, proxy SOCKS5h, Ahmia indexing, dan privasi dark web.",
        "keywords": ["onion routing socks5 proxy", "tor service networking proxy", "ahmia hidden services indexing", "dark web anonymity proxy"],
        "prerequisites": ["network_security"],
        "min_mastery_satisfied": 85.0
    },
    "penetration_testing": {
        "subject": "Offensive Cyber",
        "title": "Penetration Testing Methodology",
        "description": "Footprinting, vulnerability scanning, dan teknik penetrasi terorganisir.",
        "keywords": ["penetration testing methodology", "vulnerability scanning exploit", "owasp top 10 security audit", "metasploit framework cheat sheet"],
        "prerequisites": ["network_security"],
        "min_mastery_satisfied": 85.0
    },
    "buffer_overflow": {
        "subject": "Offensive Cyber",
        "title": "Buffer Overflow Exploits",
        "description": "Kerentanan memory buffer, shellcode execution, stack smashing, dan perlindungan ASLR/DEP.",
        "keywords": ["buffer overflow stack smashing", "shellcode assembly payload", "return oriented programming rop", "aslr dep bypass security"],
        "prerequisites": ["penetration_testing"],
        "min_mastery_satisfied": 85.0
    },

    # ── Subject 3: Low-Level System Engineering ─────────────────────────────
    "c_programming": {
        "subject": "Low-Level Systems",
        "title": "System C Programming",
        "description": "Pointers, alokasi memori manual, malloc/free, dynamic libraries, dan interaksi syscall.",
        "keywords": ["C pointers memory allocation", "dynamic memory leaks C", "linux syscall interface C", "gcc compile compilation process"],
        "prerequisites": [],
        "min_mastery_satisfied": 85.0
    },
    "assembly_x86": {
        "subject": "Low-Level Systems",
        "title": "x86/x64 Assembly & RE",
        "description": "CPU registers, call stack, instruction set assembly, debugging GDB, dan reverse engineering.",
        "keywords": ["x86 64 assembly registers", "calling conventions stack", "gdb debugging commands asm", "disassembly reverse engineering tutorial"],
        "prerequisites": ["c_programming"],
        "min_mastery_satisfied": 85.0
    },
    "kernel_architecture": {
        "subject": "Low-Level Systems",
        "title": "Kernel Architecture & OS Dev",
        "description": "Proses boot, manajemen virtual memory, scheduler CPU, interupsi hardware, dan modul kernel.",
        "keywords": ["operating system kernel scheduling", "virtual memory page tables", "hardware interrupt handlers OS", "linux kernel module tutorial"],
        "prerequisites": ["assembly_x86"],
        "min_mastery_satisfied": 85.0
    }
}


class LearningNode:
    def __init__(self, node_id: str, data: dict):
        self.node_id = node_id
        self.subject = data["subject"]
        self.title = data["title"]
        self.description = data["description"]
        self.keywords = data["keywords"]
        self.prerequisites = data["prerequisites"]
        self.min_mastery_satisfied = data["min_mastery_satisfied"]
        
        # State (load/save)
        self.mastery = 0.0
        self.status = STATUS_LOCKED
        self.synaptic_weight = 1.0
        self.last_surprisal = 1.0


class CurriculumManager:
    def __init__(self):
        self.state_file = Path(settings.WORKSPACE_DIR) / ".moko_super_learning_state.json"
        self.nodes = {}
        for nid, data in CURRICULUM_DATA.items():
            self.nodes[nid] = LearningNode(nid, data)
        self.load_state()

    def get_subjects(self) -> list:
        """Mengembalikan daftar nama subjek yang unik."""
        subjects = []
        for n in self.nodes.values():
            if n.subject not in subjects:
                subjects.append(n.subject)
        return subjects

    def get_nodes_by_subject(self, subject: str) -> list:
        """Mengembalikan daftar node yang tergabung dalam subjek tertentu."""
        return [n for n in self.nodes.values() if n.subject == subject]

    def load_state(self):
        """Memuat perkembangan kurikulum dari file JSON."""
        # Setup status default
        for node in self.nodes.values():
            node.mastery = 0.0
            node.status = STATUS_AVAILABLE if not node.prerequisites else STATUS_LOCKED
            node.synaptic_weight = 1.0
            node.last_surprisal = 1.0

        if not self.state_file.exists():
            self.save_state()
            return

        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            for nid, val in state.items():
                if nid in self.nodes:
                    node = self.nodes[nid]
                    node.mastery = float(val.get("mastery", 0.0))
                    node.status = val.get("status", STATUS_LOCKED)
                    node.synaptic_weight = float(val.get("synaptic_weight", 1.0))
                    node.last_surprisal = float(val.get("last_surprisal", 1.0))
            
            # Re-evaluasi prasyarat (safety guard jika file dirusak)
            self._update_prerequisite_states()
        except Exception as e:
            print(f"[Curriculum] Gagal memuat state: {e}")

    def save_state(self):
        """Menyimpan perkembangan kurikulum ke file JSON."""
        try:
            state = {}
            for nid, node in self.nodes.items():
                state[nid] = {
                    "mastery": round(node.mastery, 2),
                    "status": node.status,
                    "synaptic_weight": round(node.synaptic_weight, 4),
                    "last_surprisal": round(node.last_surprisal, 4)
                }
            self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Curriculum] Gagal menyimpan state: {e}")

    def _update_prerequisite_states(self):
        """Membuka gembok node yang prasyaratnya sudah terpenuhi (mastered)."""
        changed = False
        for node in self.nodes.values():
            if node.status == STATUS_MASTERED:
                continue
                
            # Cek apakah semua prasyaratnya sudah mastered
            all_prereqs_met = True
            for prereq_id in node.prerequisites:
                prereq_node = self.nodes.get(prereq_id)
                if not prereq_node or prereq_node.status != STATUS_MASTERED:
                    all_prereqs_met = False
                    break
            
            if all_prereqs_met:
                if node.status == STATUS_LOCKED:
                    node.status = STATUS_AVAILABLE
                    changed = True
            else:
                if node.status in (STATUS_AVAILABLE, STATUS_ACTIVE):
                    node.status = STATUS_LOCKED
                    changed = True
        return changed

    def start_learning(self, node_id: str) -> bool:
        """Mengubah status node menjadi aktif belajar."""
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status not in (STATUS_AVAILABLE, STATUS_ACTIVE, STATUS_MASTERED):
            return False  # Locked node cannot be studied
        
        # Kembalikan node aktif lain ke status available
        for other in self.nodes.values():
            if other.status == STATUS_ACTIVE and other.node_id != node_id:
                other.status = STATUS_AVAILABLE
        
        node.status = STATUS_ACTIVE
        self.save_state()
        return True

    def update_progress(self, node_id: str, mastery: float, synaptic_weight: float, surprisal: float):
        """Memperbarui nilai perkembangan node."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.mastery = max(0.0, min(100.0, mastery))
        node.synaptic_weight = synaptic_weight
        node.last_surprisal = surprisal
        
        if node.mastery >= node.min_mastery_satisfied and node.status == STATUS_ACTIVE:
            node.status = STATUS_MASTERED
            # Update downstream nodes
            self._update_prerequisite_states()
            
        self.save_state()

    def reset_node(self, node_id: str):
        """Me-reset perkembangan materi tertentu kembali ke 0."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.mastery = 0.0
        node.status = STATUS_AVAILABLE if not node.prerequisites else STATUS_LOCKED
        node.synaptic_weight = 1.0
        node.last_surprisal = 1.0
        self._update_prerequisite_states()
        self.save_state()

    def reset_all(self):
        """Me-reset seluruh kurikulum."""
        for node in self.nodes.values():
            node.mastery = 0.0
            node.status = STATUS_AVAILABLE if not node.prerequisites else STATUS_LOCKED
            node.synaptic_weight = 1.0
            node.last_surprisal = 1.0
        self.save_state()
