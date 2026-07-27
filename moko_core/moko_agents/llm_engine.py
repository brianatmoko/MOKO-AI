"""
MOKO LLM Engine — Mesin Inferensi dengan Throttle Kooperatif
=============================================================
Menggunakan server inferensi LLaMA.cpp lokal (llama-server) yang dimodifikasi.
"""
import time
import requests
from moko_config import settings
from moko_cpu.governor import CPUGovernor


class ThinkFilter:
    def __init__(self):
        self.in_thinking = False
        self.buffer = ""

    def feed(self, token: str) -> str:
        self.buffer += token
        out = ""
        while self.buffer:
            if self.in_thinking:
                idx = self.buffer.find("</think>")
                if idx != -1:
                    self.buffer = self.buffer[idx + 8:]
                    self.in_thinking = False
                else:
                    has_partial = False
                    for i in range(7, 0, -1):
                        if "</think>".startswith(self.buffer[-i:]):
                            self.buffer = self.buffer[-i:]
                            has_partial = True
                            break
                    if not has_partial:
                        self.buffer = ""
                    break
            else:
                idx = self.buffer.find("<think>")
                if idx != -1:
                    out += self.buffer[:idx]
                    self.buffer = self.buffer[idx + 7:]
                    self.in_thinking = True
                else:
                    has_partial = False
                    for i in range(6, 0, -1):
                        if "<think>".startswith(self.buffer[-i:]):
                            out += self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            has_partial = True
                            break
                    if not has_partial:
                        out += self.buffer
                        self.buffer = ""
                    break
        return out


class MokoEngine:
    def __init__(self):
        self.sovereign_url = settings.MOKO_LLM_API_URL
        self.embed_url = settings.MOKO_EMBED_API_URL
        self.model_name = settings.MODEL_LLM
        self._is_ready = False
        self._last_health_check = 0.0  # Timestamp cek terakhir (rate limit 30 detik)

    def _is_local_llm_enabled(self) -> bool:
        import json
        from pathlib import Path
        try:
            p = Path(__file__).parent.parent.parent / "moko_config" / "moko_settings.json"
            if p.exists():
                with open(p, "r") as f:
                    cfg = json.load(f)
                    return cfg.get("local_llm_enabled", True)
        except Exception:
            pass
        return True
        self.latest_crypto_proof = {}

    @staticmethod
    def _fallback_embedding(text: str) -> list:
        """
        Embedding deterministik ringan saat server embedder lokal mati.
        Ini menjaga router/memori tetap berjalan, meski kualitas semantiknya
        lebih rendah daripada model embedding asli.
        """
        import hashlib
        import math
        import re

        vec = [0.0] * 768
        tokens = re.findall(r"[\w]+", (text or "").lower())
        if not tokens:
            return vec

        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "little") % 768
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _wait_until_ready(self):
        if not self._is_local_llm_enabled():
            return
        """
        Memastikan server inferensi sudah aktif atau sedang loading.
        Jika status LOADING → server sedang memuat model, jangan start ulang.
        Jika status OFFLINE → start server.
        Rate-limited: cek health maksimal sekali per 30 detik.
        """
        from moko_inference.server_manager import MokoLocalInferenceServer

        # Rate-limit: jika sudah ready dan belum 30 detik, langsung return
        if self._is_ready:
            now = time.time()
            if now - self._last_health_check < 30.0:
                return
            self._last_health_check = now
            q_status = MokoLocalInferenceServer.get_server_status(settings.MOKO_LLM_PORT)
            if q_status in ("ok", "loading"):
                return
            # Server crash terdeteksi
            print("[MOKO ENGINE] ⚠️ Server crash terdeteksi! Menyalakan ulang...")
            self._is_ready = False

        # Cek status aktual
        q_status = MokoLocalInferenceServer.get_server_status(settings.MOKO_LLM_PORT)

        if q_status == "ok":
            self._is_ready = True
            self._last_health_check = time.time()
            print("[MOKO ENGINE] Sovereign Engine: ONLINE.")
            return

        if q_status in ("ok", "loading"):
            # Server aktif (mungkin masih loading) — jangan start ulang, cukup tunggu
            self._is_ready = True
            self._last_health_check = time.time()
            print("[MOKO ENGINE] Sovereign Engine: ONLINE.")
            return

        # Benar-benar offline — perlu start
        print("[MOKO ENGINE] Server inferensi belum aktif. Menyalakan otomatis...")
        success = MokoLocalInferenceServer.start_servers()
        if success:
            self._is_ready = True
            self._last_health_check = time.time()
            print("[MOKO ENGINE] Sovereign Engine: ONLINE.")
        else:
            print("[MOKO ENGINE] PERINGATAN: Gagal menyalakan server inferensi.")
            self._is_ready = False
            self._last_health_check = time.time()


    def load_model(self, model_path: str):
        """
        Secara paksa memuat model tertentu.
        """
        from moko_inference.server_manager import MokoLocalInferenceServer
        print(f"[MOKO ENGINE] Loading model: {model_path}")
        success = MokoLocalInferenceServer.start_servers(model_path=model_path)
        if success:
            self._is_ready = True
            self._last_health_check = time.time()
            self.model_name = model_path
        return success

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: str = None,
        coop_params: dict = None
    ) -> str:
        """
        Generate text menggunakan endpoint OpenAI chat/completions lokal.
        """
        if not self._is_local_llm_enabled():
            return "⚠️ [Sistem] AI Lokal sedang dinonaktifkan di pengaturan untuk menghemat resource. Silakan gunakan tombol shutdown di setting untuk menyalakannya kembali jika ingin menggunakan AI lokal."

        self._wait_until_ready()

        # Ambil parameter kooperatif
        if coop_params is None:
            coop_params = CPUGovernor.get_cooperative_params()

        num_predict = coop_params.get("num_predict", coop_params.get("num_predict", 1024))
        # enable_thinking=False menonaktifkan chain-of-thought <think>...</think> Qwen3
        # sehingga respons jauh lebih cepat untuk query yang tidak butuh penalaran panjang
        enable_thinking = coop_params.get("enable_thinking", True)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Atur temperatur dinamis untuk presisi/akurasi
        temperature = coop_params.get("temperature", None)
        if temperature is None:
            # Deteksi tugas yang butuh presisi tinggi (kode, matematika, format angka, JSON)
            prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
            precise_keywords = ["code", "fungsi", "def ", "class ", "hitung", "math", "matematika", "json", "angka", "berapa", "speed", "kecepatan", "assert", "tulis fungsi"]
            if any(kw in prompt_lower for kw in precise_keywords):
                temperature = 0.0  # Deterministik penuh untuk akurasi maksimal
            else:
                temperature = 0.1 if not enable_thinking else 0.5

        payload = {
            "messages": messages,
            "max_tokens": num_predict,
            "temperature": temperature,
        }
        
        # Qwen3 thinking mode control: inject chat_template_kwargs jika thinking dimatikan
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            for attempt in range(12):  # Coba sampai 12 kali (total ~60 detik)
                response = requests.post(
                    f"{self.sovereign_url}/chat/completions",
                    json=payload,
                    timeout=300
                )
                if response.status_code == 200:
                    choices = response.json().get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "").strip()
                        if not content:
                            reasoning = msg.get("reasoning_content", "").strip()
                            if reasoning:
                                clean_reason = reasoning.replace("Thinking Process:\n\n", "")
                                content = f"*[Thinking]*\n{clean_reason}"
                        return content
                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model sedang memuat (503)... Mencoba lagi dalam 5 detik (Attempt {attempt+1}/12)")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass
                
                print(f"[ENGINE ERROR] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[ENGINE ERROR] Gagal menghubungi Sovereign Engine: {e}")

        return ""

    def generate_text_raw(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: str = None,
        coop_params: dict = None
    ) -> tuple:
        """
        Versi extended dari generate_text() yang juga mengembalikan finish_reason.

        Return: (content: str, finish_reason: str)
          - finish_reason = "stop"   → LLM selesai secara natural
          - finish_reason = "length" → Token habis, respons terpotong
          - finish_reason = "error"  → Terjadi kesalahan
        """
        if not self._is_local_llm_enabled():
            return "⚠️ [Sistem] AI Lokal dinonaktifkan.", "stop"

        self._wait_until_ready()

        if coop_params is None:
            coop_params = CPUGovernor.get_cooperative_params()

        num_predict     = coop_params.get("num_predict", 1024)
        enable_thinking = coop_params.get("enable_thinking", True)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Atur temperatur dinamis untuk presisi/akurasi di raw generation
        temperature = coop_params.get("temperature", None)
        if temperature is None:
            prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
            precise_keywords = ["code", "fungsi", "def ", "class ", "hitung", "math", "matematika", "json", "angka", "berapa", "speed", "kecepatan", "assert", "tulis fungsi"]
            if any(kw in prompt_lower for kw in precise_keywords):
                temperature = 0.0
            else:
                temperature = 0.1 if not enable_thinking else 0.5

        payload = {
            "messages":   messages,
            "max_tokens": num_predict,
            "temperature": temperature,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            for attempt in range(12):
                response = requests.post(
                    f"{self.sovereign_url}/chat/completions",
                    json=payload,
                    timeout=300
                )
                if response.status_code == 200:
                    resp_json     = response.json()
                    choices       = resp_json.get("choices", [])
                    if choices:
                        choice        = choices[0]
                        finish_reason = choice.get("finish_reason", "stop")
                        msg           = choice.get("message", {})
                        content       = msg.get("content", "").strip()
                        if not content:
                            reasoning = msg.get("reasoning_content", "").strip()
                            if reasoning:
                                clean_reason = reasoning.replace("Thinking Process:\n\n", "")
                                content = f"*[Thinking]*\n{clean_reason}"
                        return content, finish_reason
                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model sedang memuat (503)... Mencoba lagi (Attempt {attempt+1}/12)")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass

                print(f"[ENGINE ERROR] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[ENGINE ERROR] Gagal menghubungi Sovereign Engine: {e}")

        return "", "error"

    def generate_crypto_verified(
        self,
        prompt: str,
        system_prompt: str = "",
        model_override: str = None,
        coop_params: dict = None
    ) -> tuple:
        """
        Melakukan text generation sekaligus menangkap proof kriptografis
        yang disematkan oleh server di response headers.

        Return: (content: str, finish_reason: str, crypto_proof: dict)
        """
        if not self._is_local_llm_enabled():
            return "⚠️ [Sistem] AI Lokal sedang dinonaktifkan."

        self._wait_until_ready()

        if coop_params is None:
            coop_params = CPUGovernor.get_cooperative_params()

        num_predict     = coop_params.get("num_predict", 1024)
        enable_thinking = coop_params.get("enable_thinking", True)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = coop_params.get("temperature", None)
        if temperature is None:
            prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
            precise_keywords = ["code", "fungsi", "def ", "class ", "hitung", "math", "matematika", "json", "angka", "berapa", "speed", "kecepatan", "assert", "tulis fungsi"]
            if any(kw in prompt_lower for kw in precise_keywords):
                temperature = 0.0
            else:
                temperature = 0.1 if not enable_thinking else 0.5

        payload = {
            "messages":   messages,
            "max_tokens": num_predict,
            "temperature": temperature,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            for attempt in range(12):
                response = requests.post(
                    f"{self.sovereign_url}/chat/completions",
                    json=payload,
                    timeout=300
                )
                if response.status_code == 200:
                    resp_json     = response.json()
                    choices       = resp_json.get("choices", [])
                    content       = ""
                    finish_reason = "stop"
                    if choices:
                        choice        = choices[0]
                        finish_reason = choice.get("finish_reason", "stop")
                        msg           = choice.get("message", {})
                        content       = msg.get("content", "").strip()
                        if not content:
                            reasoning = msg.get("reasoning_content", "").strip()
                            if reasoning:
                                clean_reason = reasoning.replace("Thinking Process:\n\n", "")
                                content = f"*[Thinking]*\n{clean_reason}"

                    # Ekstrak cryptographic headers
                    crypto_proof = {
                        "chain_hash": response.headers.get("X-Moko-Chain", ""),
                        "signature": response.headers.get("X-Moko-Sig", ""),
                        "model_fingerprint": response.headers.get("X-Moko-Model-Fingerprint", ""),
                    }
                    return content, finish_reason, crypto_proof

                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model sedang memuat (503)... Mencoba lagi (Attempt {attempt+1}/12)")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass

                print(f"[ENGINE ERROR] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[ENGINE ERROR] Gagal menghubungi Sovereign Engine: {e}")

        return "", "error", {}


    def continue_generation(
        self,
        messages: list,
        partial_response: str,
        coop_params: dict = None
    ) -> tuple:
        """
        Lanjutkan generasi yang terpotong menggunakan riwayat pesan penuh.

        Mengirimkan messages array lengkap (riwayat + partial response)
        ke LLM dengan instruksi melanjutkan dari titik berhenti.

        Args:
            messages:         List pesan OpenAI format — sudah termasuk system prompt
                              dan riwayat percakapan terkini
            partial_response: Teks yang sudah dihasilkan LLM sebelum terpotong
            coop_params:      Parameter koperatif CPU governor

        Return: (continuation: str, finish_reason: str)
        """
        self._wait_until_ready()

        if coop_params is None:
            coop_params = CPUGovernor.get_cooperative_params()

        num_predict     = coop_params.get("num_predict", 1024)
        enable_thinking = coop_params.get("enable_thinking", False)

        # Susun messages: riwayat + partial response sebagai assistant turn
        continue_messages = list(messages) + [
            {"role": "assistant", "content": partial_response}
        ]

        payload = {
            "messages":   continue_messages,
            "max_tokens": num_predict,
            "temperature": 0.6,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            for attempt in range(12):
                response = requests.post(
                    f"{self.sovereign_url}/chat/completions",
                    json=payload,
                    timeout=300
                )
                if response.status_code == 200:
                    resp_json     = response.json()
                    choices       = resp_json.get("choices", [])
                    if choices:
                        choice        = choices[0]
                        finish_reason = choice.get("finish_reason", "stop")
                        msg           = choice.get("message", {})
                        content       = msg.get("content", "").strip()
                        return content, finish_reason
                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model loading (503)... retry {attempt+1}/12")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass
                print(f"[ENGINE ERROR continue] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[ENGINE ERROR continue] {e}")

        return "", "error"

    def get_embedding(self, text: str) -> list:
        self._wait_until_ready()

        payload = {
            "input": text,
            "model": settings.MODEL_EMBEDDER
        }
        try:
            for attempt in range(12):  # Coba sampai 12 kali (total ~60 detik)
                response = requests.post(
                    f"{self.embed_url}/embeddings",
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if data:
                        return data[0].get("embedding", [])
                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model embedder sedang memuat (503)... Mencoba lagi dalam 5 detik (Attempt {attempt+1}/12)")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass
                
                print(f"[EMBED ERROR] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[EMBED ERROR] {e}")
        return self._fallback_embedding(text)

    def get_embeddings_batch(self, texts: list) -> list:
        if not texts:
            return []
        self._wait_until_ready()

        payload = {
            "input": texts,
            "model": settings.MODEL_EMBEDDER
        }
        try:
            for attempt in range(12):  # Coba sampai 12 kali (total ~60 detik)
                response = requests.post(
                    f"{self.embed_url}/embeddings",
                    json=payload,
                    timeout=180
                )
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if data:
                        # Pastikan urutan data sesuai dengan input
                        if "index" in data[0]:
                            data = sorted(data, key=lambda x: x["index"])
                        return [item.get("embedding", []) for item in data]
                elif response.status_code == 503:
                    try:
                        err_msg = response.json().get("error", {}).get("message", "")
                        if "Loading model" in err_msg:
                            print(f"[MOKO ENGINE] Model embedder sedang memuat (503)... Mencoba lagi dalam 5 detik (Attempt {attempt+1}/12)")
                            time.sleep(5)
                            continue
                    except Exception:
                        pass
                
                print(f"[BATCH EMBED ERROR] Status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[BATCH EMBED ERROR] {e}")
        return [self._fallback_embedding(text) for text in texts]

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        coop_params: dict = None,
        session_messages: list = None,
        on_token: callable = None,
        stop_check: callable = None,
    ) -> str:
        """
        Streaming generation — token dikirim ke on_token() satu per satu.
        
        Menggunakan SSE (Server-Sent Events) dari llama-server:
          stream: true → setiap delta token dikirim segera tanpa menunggu selesai
        
        Args:
            on_token:   callback(token: str) — dipanggil setiap ada token baru
            stop_check: callback() -> bool — jika True, hentikan streaming
        
        Return: str — full response yang sudah dikumpulkan
        """
        import json as _json

        if not self._is_local_llm_enabled():
            msg = "⚠️ [Sistem] AI Lokal dinonaktifkan."
            if on_token:
                on_token(msg)
            return msg

        self._wait_until_ready()

        if coop_params is None:
            coop_params = CPUGovernor.get_cooperative_params()

        num_predict     = coop_params.get("num_predict", 512)
        enable_thinking = coop_params.get("enable_thinking", False)
        temperature     = coop_params.get("temperature", 0.3)

        # Susun messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if session_messages:
            # Saring pesan kosong agar tidak merusak Jinja template formatting di server
            clean_session = [m for m in session_messages if m.get("content") and str(m.get("content")).strip()]
            messages.extend(clean_session)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages":   messages,
            "max_tokens": num_predict,
            "temperature": temperature,
            "stream":     True,      # ← Kunci: SSE streaming
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        full_text = ""
        for attempt in range(12):  # Retry hingga 12x (max ~60 detik) jika model masih loading
            try:
                with requests.post(
                    f"{self.sovereign_url}/chat/completions",
                    json=payload,
                    stream=True,
                    timeout=300,
                ) as resp:
                    if resp.status_code == 503:
                        # Model masih loading — tunggu dan coba lagi
                        try:
                            err_data = resp.json()
                            err_msg  = err_data.get("error", {})
                            if isinstance(err_msg, dict):
                                err_msg = err_msg.get("message", "")
                            if "loading" in str(err_msg).lower() or "still loading" in str(err_msg).lower():
                                print(f"[MOKO ENGINE] Model streaming sedang memuat (503)... retry {attempt+1}/12")
                                time.sleep(5)
                                continue
                        except Exception:
                            pass
                        print(f"[STREAM ERROR] Status {resp.status_code}")
                        break

                    if resp.status_code != 200:
                        print(f"[STREAM ERROR] Status {resp.status_code}")
                        break

                    # Capture the headers
                    self.latest_crypto_proof = {
                        "chain_hash": resp.headers.get("X-Moko-Chain", ""),
                        "signature": resp.headers.get("X-Moko-Sig", ""),
                        "model_fingerprint": resp.headers.get("X-Moko-Model-Fingerprint", ""),
                    }

                    think_filter = ThinkFilter()
                    for raw_line in resp.iter_lines():
                        # Stop check — bisa dibatalkan dari luar
                        if stop_check and stop_check():
                            break

                        if not raw_line:
                            continue

                        # Format SSE: "data: {...}" atau "data: [DONE]"
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = _json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            
                            # Hanya ambil konten jawaban final, BUKAN reasoning_content
                            # reasoning_content adalah isi <think>...</think> internal model — JANGAN tampilkan ke user
                            token = delta.get("content", "")
                            
                            if token:
                                full_text += token
                                filtered_token = think_filter.feed(token)
                                if filtered_token and on_token:
                                    on_token(filtered_token)
                        except (_json.JSONDecodeError, IndexError, KeyError):
                            continue

                    # Flush any remaining text in the filter if not inside thinking
                    if think_filter.buffer and not think_filter.in_thinking:
                        if on_token:
                            on_token(think_filter.buffer)

                    break  # Streaming selesai normal — keluar dari retry loop

            except Exception as e:
                print(f"[STREAM ERROR] {e}")
                break

        return full_text


    def release_model(self, model_override: str = None):
        """
        Di LLaMA.cpp server, model tetap di memori.
        Fungsi ini dipertahankan demi kompatibilitas API.
        """
        return True

    def shutdown(self):
        from moko_inference.server_manager import MokoLocalInferenceServer
        MokoLocalInferenceServer.stop_servers()
        print("[MOKO ENGINE] Sovereign Engine terputus.")


# Singleton Engine
engine = MokoEngine()
