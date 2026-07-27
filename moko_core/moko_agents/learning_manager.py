from moko_config import settings
from moko_memory import user_state_manager
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import heapq
from collections import defaultdict, deque
import logging

class LearningStatus(Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    ACTIVE = "active"
    MASTERED = "mastered"

class SkillComplexity(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class LearningNode:
    def __init__(self, skill_id: str, subject: str, title: str, 
                 complexity: SkillComplexity = SkillComplexity.MEDIUM,
                 description: str = ""):
        self.skill_id = skill_id
        self.subject = subject
        self.title = title
        self.complexity = complexity
        self.description = description
        
        # Mastery tracking
        self.mastery_level: float = 0.0
        self.status: LearningStatus = LearningStatus.LOCKED
        self.prerequisites: List[str] = []
        self.next_skills: List[str] = []
        
        # Cognitive development metrics
        self.cognitive_weight: float = 1.0
        self.last_surprisal: float = 1.0
        self.neural_connection_strength: float = 1.0
        self.synapse_efficiency: float = 1.0
        
        # Progress tracking
        self.times_attempted: int = 0
        self.times_mastered: int = 0
        self.current_streak: int = 0
        self.best_performance: float = 0.0
        self.last_attempted: datetime = None
        self.first_mastered: datetime = None
        
        # Learning path metrics
        self.learning_paths: List[List[str]] = []
        self.alternative_paths: List[List[str]] = []
        self.efficiency_score: float = 1.0

class CurriculumManager:
    def __init__(self):
        self.skills: Dict[str, LearningNode] = {}
        self.subjects: Dict[str, List[str]] = defaultdict(list)
        self.learning_paths: List[List[str]] = []
        self.efficiency_cache: Dict[str, float] = {}
        self.prerequisite_graph: Dict[str, List[str]] = defaultdict(list)
        
    def load_curriculum(self):
        """Load comprehensive curriculum from research specification"""
        # Cognitive Development Subject
        cognitive_skills = [
            ("ai_foundations", "Sovereign AI", "AI Foundations & Embeddings", 
             SkillComplexity.HARD, "Prinsip dasar representasi vektor, cosine similarity, dan struktur database kognitif Omni-Index."),
            ("free_energy_principle", "Sovereign AI", "Free Energy Principle (FEP)",
             SkillComplexity.EXPERT, "Koreksi prediktif Karl Friston, tingkat keheranan (Surprisal), dan error cascade prediktif."),
            ("active_inference", "Sovereign AI", "Active Inference & Adaptation",
             SkillComplexity.EXPERT, "Pemberian tindakan berdasarkan koreksi persepsi dan proses mutasi kognitif AI."),
            ("synaptic_plasticity", "Sovereign AI", "BCM Synaptic Plasticity & LTP/LTD",
             SkillComplexity.EXPERT, "Model Biersack-Cooper-Munro, Long-Term Potentiation, dan Oja's Rule untuk penguatan memori."),
        ]
        
        # Offensive Cyber Subject  
        offensive_skills = [
            ("network_security", "Offensive Cyber", "Network Security Basics",
             SkillComplexity.MEDIUM, "Dasar-dasar routing, analisis paket, port scanning, dan protokol jaringan aman."),
            ("onion_anonymity", "Offensive Cyber", "Onion Routing & Tor Anonymity", 
             SkillComplexity.MEDIUM, "Sistem Tor, relay jaringan, proxy SOCKS5h, Ahmia indexing, dan privasi dark web."),
            ("penetration_testing", "Offensive Cyber", "Penetration Testing Methodology",
             SkillComplexity.HARD, "Footprinting, vulnerability scanning, dan teknik penetrasi terorganisir."),
            ("buffer_overflow", "Offensive Cyber", "Buffer Overflow Exploits",
             SkillComplexity.EXPERT, "Kerentanan memory buffer, shellcode execution, stack smashing, dan perlindungan ASLR/DEP."),
        ]
        
        # Low-Level Systems Subject
        system_skills = [
            ("c_programming", "Low-Level Systems", "System C Programming",
             SkillComplexity.MEDIUM, "Pointers, alokasi memori manual, malloc/free, dynamic libraries, dan interaksi syscall."),
            ("assembly_x86", "Low-Level Systems", "x86/x64 Assembly & RE",
             SkillComplexity.EXPERT, "CPU registers, call stack, instruction set assembly, debugging GDB, dan reverse engineering."),
            ("kernel_architecture", "Low-Level Systems", "Kernel Architecture & OS Dev",
             SkillComplexity.EXPERT, "Proses boot, manajemen virtual memory, scheduler CPU, interupsi hardware, dan modul kernel."),
        ]
        
        # Add all skills to curriculum
        all_skills = cognitive_skills + offensive_skills + system_skills
        
        for skill_id, subject, title, complexity, description in all_skills:
            skill = LearningNode(skill_id, subject, title, complexity, description)
            self.skills[skill_id] = skill
            self.subjects[subject].append(skill_id)
            
            # Setup prerequisite relationships based on curriculum logic
            self._setup_prerequisites(skill_id, subject)
        
        # Build learning paths using BFS for optimal progression
        self._generate_learning_paths()
        
    def _setup_prerequisites(self, skill_id: str, subject: str):
        """Setup prerequisite relationships based on subject structure"""
        if subject == "Sovereign AI":
            if skill_id == "free_energy_principle":
                self.skills[skill_id].prerequisites = ["ai_foundations"]
            elif skill_id == "active_inference":
                self.skills[skill_id].prerequisites = ["free_energy_principle"]
            elif skill_id == "synaptic_plasticity":
                self.skills[skill_id].prerequisites = ["active_inference"]
        
        elif subject == "Offensive Cyber":
            if skill_id == "onion_anonymity":
                self.skills[skill_id].prerequisites = ["network_security"]
            elif skill_id == "penetration_testing":
                self.skills[skill_id].prerequisites = ["network_security"]
            elif skill_id == "buffer_overflow":
                self.skills[skill_id].prerequisites = ["penetration_testing"]
        
        elif subject == "Low-Level Systems":
            if skill_id == "assembly_x86":
                self.skills[skill_id].prerequisites = ["c_programming"]
            elif skill_id == "kernel_architecture":
                self.skills[skill_id].prerequisites = ["assembly_x86"]
    
    def _generate_learning_paths(self):
        """Generate all possible learning paths using BFS"""
        from collections import deque
        
        for start_skill in self.skills:
            queue = deque([(start_skill, [start_skill])])
            visited = set()
            
            while queue:
                current_skill, path = queue.popleft()
                
                if tuple(path) in visited:
                    continue
                visited.add(tuple(path))
                self.learning_paths.append(path)
                
                # Add next skills to path
                next_skills = self._get_next_skills(current_skill)
                for next_skill in next_skills:
                    queue.append((next_skill, path + [next_skill]))
    
    def _get_next_skills(self, skill_id: str) -> List[str]:
        """Get skills that depend on the given skill (reverse prerequisites)"""
        next_skills = []
        for sid, skill in self.skills.items():
            if skill_id in skill.prerequisites:
                next_skills.append(sid)
        return next_skills
    
    def optimize_learning_paths(self, user_profile: Dict) -> List[List[str]]:
        """Optimize learning paths based on user profile and performance"""
        base_paths = self.learning_paths.copy()
        
        # Filter by user's subject interests
        if 'preferred_subjects' in user_profile:
            preferred = set(user_profile['preferred_subjects'])
            base_paths = [path for path in base_paths 
                         if any(skill in [s.split('_')[0] for s in path] 
                               for skill in preferred)]
        
        # Optimize based on user's learning pace
        if 'learning_pace' in user_profile:
            pace = user_profile['learning_pace']
            if pace == 'fast':
                # Take shorter paths first
                base_paths.sort(key=len)
            elif pace == 'slow':
                # Take longer paths for deeper learning
                base_paths.sort(key=len, reverse=True)
        
        # Filter by difficulty appropriateness
        if 'difficulty_level' in user_profile:
            user_level = user_profile['difficulty_level']
            base_paths = [path for path in base_paths 
                         if self._path_difficulty_match(path, user_level)]
        
        return base_paths[:10]  # Return top 10 paths
    
    def _path_difficulty_match(self, path: List[str], user_level: str) -> bool:
        """Check if path difficulty matches user level"""
        path_complexity = 0
        for skill_id in path:
            if skill_id in self.skills:
                skill = self.skills[skill_id]
                if skill.complexity == SkillComplexity.EXPERT:
                    path_complexity += 3
                elif skill.complexity == SkillComplexity.HARD:
                    path_complexity += 2
                elif skill.complexity == SkillComplexity.MEDIUM:
                    path_complexity += 1
        
        if user_level == 'beginner' and path_complexity <= 5:
            return True
        elif user_level == 'intermediate' and 5 < path_complexity <= 10:
            return True
        elif user_level == 'advanced' and path_complexity > 10:
            return True
        return False

    def get_skill_by_id(self, skill_id: str) -> Optional[LearningNode]:
        return self.skills.get(skill_id)
    
    def get_subjects(self) -> List[str]:
        return list(self.subjects.keys())
    
    def get_skills_by_subject(self, subject: str) -> List[LearningNode]:
        return [self.skills[skill_id] for skill_id in self.subjects.get(subject, [])]


class SkillTracker:
    def __init__(self, curriculum_manager: CurriculumManager):
        self.curriculum = curriculum_manager
        self.user_progress: Dict[str, Dict[str, LearningNode]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.learning_analytics: Dict[str, Dict] = {}
        
    def initialize_user_skills(self, user_id: str):
        """Initialize all skills for a new user"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        for skill_id, skill in self.curriculum.skills.items():
            if skill_id not in self.user_progress[user_id]:
                user_skill = LearningNode(
                    skill_id, skill.subject, skill.title, 
                    skill.complexity, skill.description
                )
                self.user_progress[user_id][skill_id] = user_skill
                
    def update_skill_mastery(self, user_id: str, skill_id: str, 
                           new_mastery: float, surprisal: float = 1.0,
                           time_spent: float = 0.0) -> Dict:
        """
        Update user skill mastery with cognitive development model
        
        Implements cognitive learning principles:
        - Mastery updates trigger synaptic plasticity
        - High surprisal triggers active inference
        - Requires prerequisite satisfaction
        - Affects dependent skill availability
        """
        if user_id not in self.user_progress or skill_id not in self.user_progress[user_id]:
            return {"success": False, "error": "Skill not found for user"}
        
        user_skill = self.user_progress[user_id][skill_id]
        old_mastery = user_skill.mastery_level
        
        # Validate mastery range
        new_mastery = max(0.0, min(100.0, new_mastery))
        
        # Check prerequisite satisfaction for mastery
        can_learn = self._check_prerequisites_satisfied(user_id, skill_id)
        if new_mastery > 85 and not can_learn:
            return {"success": False, "error": "Prerequisites not satisfied for mastery"}
        
        # Record previous state for learning analytics
        self._record_learning_state_change(user_id, skill_id, user_skill, old_mastery)
        
        # Update mastery level
        user_skill.mastery_level = new_mastery
        user_skill.last_surprisal = surprisal
        user_skill.last_attempted = datetime.now()
        
        if new_mastery >= 85 and old_mastery < 85 and user_skill.times_mastered == 0:
            user_skill.first_mastered = datetime.now()
            user_skill.times_mastered += 1
        
        # Cognitive development updates
        self._update_cognitive_metrics(user_id, skill_id, user_skill, old_mastery)
        
        # Update dependent skills based on mastery
        self._update_dependent_skills(user_id, skill_id)
        
        # Calculate performance improvements
        performance_improvement = self._calculate_performance_improvement(
            user_id, skill_id, old_mastery, new_mastery
        )
        
        # Update analytics
        self._update_learning_analytics(user_id, skill_id, performance_improvement)
        
        return {
            "success": True,
            "old_mastery": old_mastery,
            "new_mastery": new_mastery,
            "performance_improvement": performance_improvement,
            "cognitive_weight": user_skill.cognitive_weight,
            "status": user_skill.status.name
        }
    
    def _check_prerequisites_satisfied(self, user_id: str, skill_id: str) -> bool:
        """Check if all prerequisites are mastered"""
        user_skill = self.user_progress[user_id][skill_id]
        
        for prereq_id in user_skill.prerequisites:
            prereq_skill = self.user_progress[user_id][prereq_id]
            if prereq_skill.mastery_level < prereq_skill.min_mastery_satisfied:
                return False
        
        return True
    
    def _update_cognitive_metrics(self, user_id: str, skill_id: str, 
                                  user_skill: LearningNode, old_mastery: float):
        """Update cognitive development metrics based on mastery changes"""
        
        # Synaptic weight adjustment based on learning success
        if old_mastery < 85 and user_skill.mastery_level >= 85:
            # Mastery achieved - strengthen synaptic connections
            user_skill.synaptic_weight = min(2.0, user_skill.synaptic_weight * 1.5)
            user_skill.neural_connection_strength = min(2.0, user_skill.neural_connection_strength * 1.3)
        
        elif old_mastery >= 85 and user_skill.mastery_level < 85:
            # Regression - weaken connections but maintain baseline
            user_skill.synaptic_weight = max(0.5, user_skill.synaptic_weight * 0.8)
        
        # Neural efficiency improvements
        if user_skill.mastery_level > old_mastery:
            efficiency_gain = (user_skill.mastery_level - old_mastery) / 100 * 0.2
            user_skill.efficiency_score = min(2.0, user_skill.efficiency_score + efficiency_gain)
        
        # Active inference from high surprisal
        if user_skill.last_surprisal > 0.8:
            # High surprise triggers cognitive adaptation
            user_skill.cognitive_weight = min(1.5, user_skill.cognitive_weight * (1 + user_skill.last_surprisal * 0.3))
    
    def _update_dependent_skills(self, user_id: str, mastered_skill_id: str):
        """Update dependent skills when prerequisite is mastered"""
        for skill_id, skill in self.user_progress[user_id].items():
            if mastered_skill_id in skill.prerequisites:
                # Check if all prerequisites are now satisfied
                all_prereqs_met = True
                for prereq_id in skill.prerequisites:
                    if self.user_progress[user_id][prereq_id].mastery_level < 85:
                        all_prereqs_met = False
                        break
                
                if all_prereqs_met and skill.status.name in ["LOCKED", "AVAILABLE"]:
                    skill.status = LearningStatus.AVAILABLE
    
    def _calculate_performance_improvement(self, user_id: str, skill_id: str,
                                           old_mastery: float, new_mastery: float) -> float:
        """Calculate performance improvement metrics"""
        improvement = new_mastery - old_mastery
        
        # Factor in learning efficiency based on time
        user_skill = self.user_progress[user_id][skill_id]
        if user_skill.last_attempted:
            hours_studied = (datetime.now() - user_skill.last_attempted).total_seconds() / 3600
            if hours_studied > 0:
                efficiency = improvement / hours_studied
                user_skill.efficiency_score = min(2.0, user_skill.efficiency_score + efficiency * 0.1)
        
        # Update best performance
        user_skill.best_performance = max(user_skill.best_performance, new_mastery)
        
        return improvement
    
    def _record_learning_state_change(self, user_id: str, skill_id: str,
                                     user_skill: LearningNode, old_mastery: float):
        """Record learning state changes for analytics"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "old_mastery": old_mastery,
            "new_mastery": user_skill.mastery_level,
            "change_amount": user_skill.mastery_level - old_mastery,
            "surprisal": user_skill.last_surprisal,
            "cognitive_weight": user_skill.cognitive_weight
        }
        
        if user_id not in self.learning_analytics:
            self.learning_analytics[user_id] = {}
        if skill_id not in self.learning_analytics[user_id]:
            self.learning_analytics[user_id][skill_id] = []
        
        self.learning_analytics[user_id][skill_id].append(change_record)
        
        # Keep only last 100 records per skill for memory efficiency
        if len(self.learning_analytics[user_id][skill_id]) > 100:
            self.learning_analytics[user_id][skill_id] = self.learning_analytics[user_id][skill_id][-100:]
    
    def _update_learning_analytics(self, user_id: str, skill_id: str,
                                   improvement: float):
        """Update comprehensive learning analytics"""
        if user_id not in self.learning_analytics:
            self.learning_analytics[user_id] = {}
        
        if skill_id not in self.learning_analytics[user_id]:
            self.learning_analytics[user_id][skill_id] = {}
        
        skill_analytics = self.learning_analytics[user_id][skill_id]
        
        # Update analytics
        skill_analytics["total_improvement"] = skill_analytics.get("total_improvement", 0) + improvement
        skill_analytics["sessions"] = skill_analytics.get("sessions", 0) + 1
        skill_analytics["average_improvement"] = skill_analytics["total_improvement"] / skill_analytics["sessions"]
        
        # Calculate learning velocity (mastery gain per unit time)
        if skill_id in self.user_progress[user_id]:
            user_skill = self.user_progress[user_id][skill_id]
            if user_skill.last_attempted and skill_analytics.get("last_session_time"):
                last_time = datetime.fromisoformat(skill_analytics["last_session_time"])
                time_diff = (datetime.now() - last_time).total_seconds() / 3600
                if time_diff > 0:
                    skill_analytics["learning_velocity"] = improvement / time_diff
                skill_analytics["last_session_time"] = datetime.now().isoformat()
    
    def get_user_skill_status(self, user_id: str) -> Dict:
        """Get comprehensive status for all user skills"""
        status = {}
        total_skills = len(self.user_progress.get(user_id, {}))
        mastered_skills = 0
        available_skills = 0
        locked_skills = 0
        active_skills = 0
        
        for skill_id, skill in self.user_progress.get(user_id, {}).items():
            status[skill_id] = {
                "title": skill.title,
                "subject": skill.subject,
                "complexity": skill.complexity.name,
                "mastery": skill.mastery_level,
                "status": skill.status.name,
                "prerequisites": skill.prerequisites,
                "times_attempted": skill.times_attempted,
                "times_mastered": skill.times_mastered,
                "efficiency_score": skill.efficiency_score,
                "cognitive_weight": skill.cognitive_weight
            }
            
            if skill.status == LearningStatus.MASTERED:
                mastered_skills += 1
            elif skill.status == LearningStatus.AVAILABLE:
                available_skills += 1
            elif skill.status == LearningStatus.ACTIVE:
                active_skills += 1
            elif skill.status == LearningStatus.LOCKED:
                locked_skills += 1
        
        return {
            "user_id": user_id,
            "total_skills": total_skills,
            "mastered_skills": mastered_skills,
            "available_skills": available_skills,
            "active_skills": active_skills,
            "locked_skills": locked_skills,
            "mastery_percentage": (mastered_skills / total_skills * 100) if total_skills > 0 else 0,
            "skills": status
        }
    
    def get_recommended_learning_path(self, user_id: str, target_skill: str = None) -> List[str]:
        """Get recommended learning path based on user progress and goals"""
        
        if target_skill:
            # Get specific path for target skill
            skill = self.curriculum.get_skill_by_id(target_skill)
            if not skill:
                return []
            
            # Find paths that include the target skill
            recommended_paths = []
            for path in self.curriculum.learning_paths:
                if target_skill in path:
                    # Check if user has prerequisites for this path
                    if self._user_can_follow_path(user_id, path):
                        recommended_paths.append(path)
            
            return recommended_paths[:3]
        
        # Default: get next recommended skills for user
        user_skills = self.user_progress.get(user_id, {})
        next_skills = []
        
        for skill_id, skill in user_skills.items():
            if skill.status == LearningStatus.MASTERED and skill.mastery_level >= 85:
                # User has mastered this skill, check next skills
                for next_skill_id in skill.next_skills:
                    if self._user_can_access_skill(user_id, next_skill_id):
                        next_skills.append(next_skill_id)
        
        return next_skills[:5]
    
    def _user_can_follow_path(self, user_id: str, path: List[str]) -> bool:
        """Check if user can follow a specific learning path"""
        user_skills = self.user_progress.get(user_id, {})
        
        for skill_id in path:
            skill = user_skills.get(skill_id)
            if not skill:
                return False
            
            # Check prerequisites
            if not self._check_prerequisites_satisfied(user_id, skill_id):
                return False
        
        return True
    
    def _user_can_access_skill(self, user_id: str, skill_id: str) -> bool:
        """Check if user can access a specific skill"""
        user_skill = self.user_progress.get(user_id, {}).get(skill_id)
        if not user_skill:
            return False
        
        return self._check_prerequisites_satisfied(user_id, skill_id)