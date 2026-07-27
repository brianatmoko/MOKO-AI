import json
import re
from typing import Dict
from moko_agents.llm_engine import engine
from moko_config import settings

class ActiveInference:
    """
    Mesin Mutasi Genetik (Rescorla-Wagner & Dopaminergic Prediction Error).
    Melakukan revisi "Active Inference" pada sebuah rumus jika Free Energy tinggi (tidak puas).
    """

    MUTATION_PROMPT = """
Kamu adalah Mesin Mutasi Genetik MOKO (Active Inference Engine).
Sebuah rumus logika telah gagal diterapkan pada sebuah teks karena memiliki tingkat Surprisal (Error) yang tinggi.

Teks Asli:
"{text}"

Rumus Logika Lama (yang gagal):
"{old_instruction}"

Alasan Kegagalan (FEP Feedback):
"{fep_reason}"

Tugasmu: Perbaiki dan modifikasi rumus logika lama tersebut. Lakukan mutasi kata-kata instruksinya agar menjadi kerangka pemikiran baru yang lebih presisi dan dapat menangani teks ini di masa depan tanpa error.
Pertahankan esensi kategori logikanya, namun ubah cara memprosesnya.

KEMBALIKAN HANYA OBJEK JSON (tanpa markdown tambahan):
{{
  "new_instruction": "Instruksi yang sudah diperbarui secara mutasi...",
  "mutation_logic": "Alasan singkat mengapa instruksi ini diubah..."
}}
"""

    @classmethod
    def mutate_formula(cls, text: str, old_instruction: str, fep_reason: str, coop_params: dict = None) -> str:
        """
        Memutasi instruksi berdasarkan umpan balik FEP.
        Mengembalikan new_instruction string.
        """
        prompt = cls.MUTATION_PROMPT.format(
            text=text[:1000],
            old_instruction=old_instruction,
            fep_reason=fep_reason
        )
        
        try:
            response = engine.generate_text(prompt, "Return JSON only.", model_override=settings.MODEL_ANALYST, coop_params=coop_params)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("new_instruction", old_instruction)
            return old_instruction
        except Exception:
            return old_instruction

    @classmethod
    def process_mutation_queue(cls, queue: list[dict], coop_params: dict = None) -> int:
        """
        Memproses seluruh antrian mutasi dalam satu batch LLM call.
        Return: Jumlah formula yang berhasil dimutasi dan diperbarui.
        """
        if not queue:
            return 0
            
        from moko_memory.math_omni import math_omni
        
        batch_items = []
        for entry in queue:
            fid = entry.get("formula_id")
            formula = math_omni._find_entry(fid)
            old_inst = formula.get("instruction", "") if formula else ""
            batch_items.append({
                "formula_id": fid,
                "text": entry.get("chunk_text"),
                "old_instruction": old_inst,
                "fep_score": entry.get("fep_score")
            })
            
        batch_prompt = """
Kamu adalah Mesin Mutasi Genetik MOKO (Active Inference Engine).
Beberapa rumus logika telah gagal diterapkan pada teks tertentu karena memiliki tingkat Surprisal (Error) yang tinggi.
Tugasmu adalah memperbaiki dan memutasi rumus-rumus tersebut dalam satu batch.

Daftar formula yang butuh mutasi:
{batch_data}

Untuk masing-masing item di atas, lakukan mutasi pada 'old_instruction' menggunakan konteks 'text'.
Ubah kata-kata instruksinya agar menjadi kerangka pemikiran baru yang lebih presisi dan dapat menangani jenis teks tersebut di masa depan tanpa error. Pertahankan tipe logikanya.

KEMBALIKAN HANYA ARRAY JSON (tanpa markdown atau teks lainnya):
[
  {{
    "formula_id": "id_rumus_yang_diberikan",
    "new_instruction": "instruksi baru hasil mutasi..."
  }},
  ...
]
""".format(batch_data=json.dumps(batch_items, indent=2, ensure_ascii=False))

        try:
            response = engine.generate_text(
                batch_prompt,
                "Return JSON array only.",
                model_override=settings.MODEL_ANALYST,
                coop_params=coop_params
            )
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return 0
                
            mutated_items = json.loads(json_match.group(0))
            updated_count = 0
            for item in mutated_items:
                fid = item.get("formula_id")
                new_inst = item.get("new_instruction")
                if fid and new_inst:
                    formula = math_omni._find_entry(fid)
                    if formula:
                        formula["instruction"] = new_inst
                        formula["synaptic_weight"] = 1.0
                        updated_count += 1
            
            if updated_count > 0:
                with open(math_omni.sidecar, "w", encoding="utf-8") as f:
                    for e in math_omni._index:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                        
            return updated_count
        except Exception as e:
            print(f"Error in batch process_mutation_queue: {e}")
            return 0

