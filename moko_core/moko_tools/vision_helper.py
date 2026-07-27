"""
MOKO Vision Helper — Local Image Input Processor
================================================
Helper to load local image files, encode them in base64 data URLs,
and format them into OpenAI-compatible multimodal payloads.
"""

import base64
import os
from pathlib import Path
from moko_agents.llm_engine import engine

class MokoVision:
    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Encode a local image file to base64 string."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        ext = path.suffix.lower().replace(".", "")
        if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
            ext = "jpeg"  # default fallback
            
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        return f"data:image/{ext};base64,{encoded_string}"

    @classmethod
    def analyze_image(
        cls,
        image_path: str,
        question: str = "Analyze this image in detail.",
        system_prompt: str = "Kamu adalah Moko, AI vision assistant. Analisis gambar yang dikirimkan user.",
        coop_params: dict = None
    ) -> str:
        """
        Sends an image along with a text query to the local sovereign engine.
        Returns the text explanation.
        """
        try:
            image_url = cls.encode_image_to_base64(image_path)
        except Exception as e:
            return f"Error loading image: {e}"

        # Build multimodal prompt payload
        multimodal_prompt = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

        # Call local inference engine
        if coop_params is None:
            coop_params = {"num_predict": 1024, "enable_thinking": False}

        # Disable MOKO3 thinking since standard thinking models (CausalLM) 
        # may not support vision directly unless it is a VL (Vision-Language) model
        coop_params["enable_thinking"] = False

        return engine.generate_text(
            prompt=multimodal_prompt,
            system_prompt=system_prompt,
            coop_params=coop_params
        )
