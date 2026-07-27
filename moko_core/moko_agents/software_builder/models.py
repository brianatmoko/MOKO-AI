"""
models.py — Data Models untuk MOKO Software Builder
=====================================================
Dataclasses yang mendefinisikan kontrak data antar komponen:
  - InterviewData  : Hasil dari proses multi-turn interview dengan user
  - PlanStep       : Satu langkah dalam implementation plan
  - PlanSession    : Sesi lengkap dari interview hingga eksekusi
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional


@dataclass
class InterviewData:
    """
    Data yang dikumpulkan dari proses interview multi-turn.
    Diisi secara bertahap seiring jawaban user masuk.
    """
    software_type: str = ""          # "game", "web app", "tool", "automation"
    sub_type: str = ""               # "RPG", "platformer", "puzzle", "2D action"
    mechanics: List[str] = field(default_factory=list)  # ["movement", "collision", "scoring"]
    language: str = ""               # "python", "javascript", "c++"
    platform: str = ""               # "desktop", "web", "mobile"
    complexity: str = ""             # "simple", "medium", "advanced"
    extra_notes: str = ""            # catatan bebas dari user

    def is_complete(self) -> bool:
        """Cek apakah semua field wajib sudah terisi."""
        return bool(
            self.software_type and
            self.sub_type and
            self.mechanics and
            self.language and
            self.complexity
        )

    def to_summary(self) -> str:
        """Ringkasan data interview untuk ditampilkan ke user."""
        mechanics_str = ", ".join(self.mechanics) if self.mechanics else "-"
        return (
            f"**Ringkasan Kebutuhan Software:**\n"
            f"- Tipe: {self.software_type.upper()} ({self.sub_type})\n"
            f"- Mechanic/Fitur: {mechanics_str}\n"
            f"- Bahasa: {self.language}\n"
            f"- Platform: {self.platform or 'desktop'}\n"
            f"- Kompleksitas: {self.complexity}\n"
            + (f"- Catatan: {self.extra_notes}\n" if self.extra_notes else "")
        )


@dataclass
class PlanStep:
    """
    Satu langkah dalam implementation plan.
    Setiap langkah memiliki kode yang akan di-generate dan dieksekusi.
    """
    step_number: int
    title: str
    description: str
    files_to_create: List[str] = field(default_factory=list)
    status: Literal["PENDING", "IN_PROGRESS", "DONE", "ERROR"] = "PENDING"
    generated_code: Optional[str] = None
    terminal_output: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def status_badge(self) -> str:
        """Badge status untuk ditampilkan di UI."""
        badges = {
            "PENDING":     "⬜ PENDING",
            "IN_PROGRESS": "🔄 IN PROGRESS",
            "DONE":        "✅ DONE",
            "ERROR":       "❌ ERROR",
        }
        return badges.get(self.status, self.status)

    def to_card_html(self) -> str:
        """Representasi HTML untuk PlanStepCard di Chat Panel."""
        files_str = ", ".join(self.files_to_create) if self.files_to_create else "—"
        badge = self.status_badge()
        color_map = {
            "PENDING":     "rgba(150,150,150,0.6)",
            "IN_PROGRESS": "rgba(0,230,255,0.8)",
            "DONE":        "rgba(0,255,136,0.8)",
            "ERROR":       "rgba(255,60,60,0.8)",
        }
        border_color = color_map.get(self.status, "#666")
        return (
            f"<div style='margin:6px 0; padding:10px 14px; "
            f"background:rgba(10,12,25,0.8); "
            f"border-left:3px solid {border_color}; border-radius:6px;'>"
            f"<span style='color:{border_color}; font-size:10px; "
            f"font-family:Fira Code; font-weight:700; letter-spacing:1px;'>"
            f"STEP {self.step_number} · {badge}</span><br>"
            f"<span style='color:#e8d8ff; font-size:13px;'><b>{self.title}</b></span><br>"
            f"<span style='color:rgba(200,210,240,0.7); font-size:11px;'>{self.description}</span><br>"
            f"<span style='color:rgba(0,230,255,0.5); font-size:10px; font-family:Fira Code;'>"
            f"Files: {files_str}</span>"
            f"</div>"
        )


@dataclass
class PlanSession:
    """
    Sesi lengkap Software Builder dari interview hingga eksekusi.
    Disimpan in-memory selama sesi IDE berlangsung.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    interview_data: InterviewData = field(default_factory=InterviewData)
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    workspace_dir: str = ""          # direktori kerja proyek yang dihasilkan
    is_active: bool = True

    def get_current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance_step(self) -> bool:
        """Maju ke langkah berikutnya. Return True jika ada langkah berikutnya."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False

    def completion_percentage(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == "DONE")
        return (done / len(self.steps)) * 100


if __name__ == "__main__":
    # Unit test sederhana untuk validasi dataclasses
    print("=== Unit Test: models.py ===\n")

    # Test InterviewData
    data = InterviewData(
        software_type="game",
        sub_type="platformer",
        mechanics=["movement", "collision", "scoring"],
        language="python",
        platform="desktop",
        complexity="medium"
    )
    assert data.is_complete(), "InterviewData.is_complete() harus True jika semua field terisi"
    print("✅ InterviewData.is_complete() → OK")
    print(data.to_summary())

    # Test InterviewData kosong
    empty_data = InterviewData()
    assert not empty_data.is_complete(), "InterviewData kosong tidak boleh complete"
    print("✅ InterviewData kosong tidak complete → OK")

    # Test PlanStep
    step = PlanStep(
        step_number=1,
        title="Setup Project Structure",
        description="Buat direktori proyek dan file utama",
        files_to_create=["main.py", "game.py"],
        status="PENDING"
    )
    assert step.status_badge() == "⬜ PENDING"
    print("✅ PlanStep.status_badge() → OK")

    step.status = "DONE"
    assert step.status_badge() == "✅ DONE"
    print("✅ PlanStep.status_badge() DONE → OK")

    # Test PlanSession
    session = PlanSession(interview_data=data, steps=[step])
    assert session.completion_percentage() == 100.0
    print("✅ PlanSession.completion_percentage() → OK")

    print("\n✅ Semua unit test berhasil!")
