"""
MOKO Identity — Phase 20: Digital Identity Layer (Non-Crypto)
=============================================================
MOKO identifies itself via mathematical fingerprints of its neural weights.
No blockchain, no crypto, just pure neural identity.
"""

import hashlib
import os
from typing import Optional


class MokoIdentity:
    """
    Stateless class — all methods are classmethod/staticmethod.
    Called by generate_system_prompt() in CoreNode.
    """

    _gguf_fp_cache: Optional[str] = None   # Cache once — doesn't change during runtime

    @classmethod
    def get_identity_context(cls) -> str:
        """
        Build MOKO's identity context based on neural fingerprints.
        
        Returns string like:
          [IDENTITY:MOKO-AI] [GGUF:b194f1dc5b0e]
        """
        parts = []

        # 1. Identity Name
        parts.append("[IDENTITY:MOKO-AI]")

        # 2. GGUF Fingerprint
        fp = cls._get_gguf_fingerprint()
        if fp:
            parts.append(f"[GGUF:{fp[:14]}]")

        if not parts:
            return ""

        return " ".join(parts)

    @classmethod
    def _get_gguf_fingerprint(cls) -> str:
        """
        SHA-256 of the first 4KB of the active GGUF file.
        """
        if cls._gguf_fp_cache:
            return cls._gguf_fp_cache

        try:
            from moko_config import settings
            # Try main model first
            gguf_path = getattr(settings, "MODEL_MOKO_GGUF_PATH", None)

            if gguf_path and os.path.exists(gguf_path):
                with open(gguf_path, "rb") as f:
                    header = f.read(4096)
                fp = hashlib.sha256(header).hexdigest()
                cls._gguf_fp_cache = fp
                return fp
        except Exception:
            pass

        # Fallback: try identity path
        try:
            from moko_config import settings
            id_path = getattr(settings, "MODEL_IDENTITY_PATH", None)
            if not id_path:
                id_path = getattr(settings, "MODEL_BF16_IDENTITY_PATH", None)

            if id_path and os.path.exists(id_path):
                with open(id_path, "rb") as f:
                    header = f.read(4096)
                fp = hashlib.sha256(header).hexdigest()
                cls._gguf_fp_cache = fp
                return fp
        except Exception:
            pass

        return ""
