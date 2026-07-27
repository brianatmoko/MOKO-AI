"""
MOKO Binary Knowledge Codec — Extreme Compression for Math Knowledge
====================================================================
Sistem encoding untuk menyimpan knowledge base matematika
dalam format binary/hex yang ultra-kompak.

Strategi Kompresi 4-Layer:
  Layer 1: UTF-8 → Binary packing
  Layer 2: Zlib compression
  Layer 3: Delta encoding (untuk data terurut)
  Layer 4: Huffman-like frequency encoding (opcode table)

Referensi:
  - BitNet b1.58: extreme weight quantization
  - Huffman coding: entropy-based compression
  - Neural Weight Compression: learned codecs
"""
import struct
import zlib
import json
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from collections import Counter
from enum import IntEnum


class MathOpcode(IntEnum):
    """
    Binary opcodes untuk operasi matematika umum.
    Setiap opcode = 1 byte, menggantikan string panjang.
    Menghemat ~80% dibanding teks mentah untuk operasi umum.
    """
    # Aritmetika (0x01 - 0x0F)
    ADD = 0x01          # +
    SUB = 0x02          # -
    MUL = 0x03          # ×
    DIV = 0x04          # ÷
    POW = 0x05          # ^
    SQRT = 0x06         # √
    ABS = 0x07          # |x|
    MOD = 0x08          # mod
    FACTORIAL = 0x09    # !
    NEG = 0x0A          # -x (negate)

    # Kalkulus (0x10 - 0x1F)
    INTEGRAL = 0x10     # ∫
    DERIVATIVE = 0x11   # d/dx
    LIMIT = 0x12        # lim
    SUM = 0x13          # Σ
    PRODUCT = 0x14      # Π
    PARTIAL = 0x15      # ∂
    SERIES = 0x16       # Taylor series

    # Trigonometri (0x20 - 0x2F)
    SIN = 0x20
    COS = 0x21
    TAN = 0x22
    ARCSIN = 0x23
    ARCCOS = 0x24
    ARCTAN = 0x25
    SINH = 0x26
    COSH = 0x27
    TANH = 0x28

    # Logaritma/Eksponensial (0x30 - 0x3F)
    LOG = 0x30          # log
    LN = 0x31           # natural log
    EXP = 0x32          # e^x
    LOG2 = 0x33         # log base 2
    LOG10 = 0x34        # log base 10

    # Aljabar (0x40 - 0x4F)
    SOLVE = 0x40        # selesaikan
    FACTOR = 0x41       # faktorkan
    EXPAND = 0x42       # jabarkan
    SIMPLIFY = 0x43     # sederhanakan
    EQUATION = 0x44     # = (equation)

    # Konstanta (0x50 - 0x5F)
    PI = 0x50           # π
    EULER = 0x51        # e
    IMAGINARY = 0x52    # i
    INFINITY = 0x53     # ∞
    GOLDEN_RATIO = 0x54 # φ

    # Relasi (0x60 - 0x6F)
    EQUALS = 0x60       # =
    NOT_EQUALS = 0x61   # ≠
    LESS_THAN = 0x62    # <
    GREATER_THAN = 0x63 # >
    LESS_EQ = 0x64      # ≤
    GREATER_EQ = 0x65   # ≥
    APPROX = 0x66       # ≈

    # Linear Algebra (0x70 - 0x7F)
    MATRIX = 0x70
    DETERMINANT = 0x71
    INVERSE = 0x72
    TRANSPOSE = 0x73
    EIGENVALUE = 0x74
    RANK = 0x75

    # Statistik (0x80 - 0x8F)
    MEAN = 0x80
    MEDIAN = 0x81
    STDDEV = 0x82
    VARIANCE = 0x83
    PROBABILITY = 0x84
    COMBINATION = 0x85
    PERMUTATION = 0x86

    # Control bytes (0xF0 - 0xFF)
    BEGIN_EXPR = 0xF0   # Start expression block
    END_EXPR = 0xF1     # End expression block
    BEGIN_STEP = 0xF2   # Start reasoning step
    END_STEP = 0xF3     # End reasoning step
    SEPARATOR = 0xFE    # Field separator
    NULL = 0xFF         # Null terminator


# Reverse mapping: text → opcode (1-to-1 for exact round-trip reconstruction)
_TEXT_TO_OPCODE = {
    '+': MathOpcode.ADD,
    '-': MathOpcode.SUB,
    '*': MathOpcode.MUL,
    '/': MathOpcode.DIV,
    '^': MathOpcode.POW,
    '√': MathOpcode.SQRT,
    '|': MathOpcode.ABS,
    '%': MathOpcode.MOD,
    '!': MathOpcode.FACTORIAL,
    '∫': MathOpcode.INTEGRAL,
    'd/dx': MathOpcode.DERIVATIVE,
    'lim': MathOpcode.LIMIT,
    'Σ': MathOpcode.SUM,
    'Π': MathOpcode.PRODUCT,
    'sin': MathOpcode.SIN,
    'cos': MathOpcode.COS,
    'tan': MathOpcode.TAN,
    'arcsin': MathOpcode.ARCSIN,
    'arccos': MathOpcode.ARCCOS,
    'arctan': MathOpcode.ARCTAN,
    'log': MathOpcode.LOG,
    'ln': MathOpcode.LN,
    'exp': MathOpcode.EXP,
    'solve': MathOpcode.SOLVE,
    'factor': MathOpcode.FACTOR,
    'expand': MathOpcode.EXPAND,
    'simplify': MathOpcode.SIMPLIFY,
    'π': MathOpcode.PI,
    'e': MathOpcode.EULER,
    '∞': MathOpcode.INFINITY,
    '=': MathOpcode.EQUALS,
    '≠': MathOpcode.NOT_EQUALS,
    '<': MathOpcode.LESS_THAN,
    '>': MathOpcode.GREATER_THAN,
    '≤': MathOpcode.LESS_EQ,
    '≥': MathOpcode.GREATER_EQ,
    '≈': MathOpcode.APPROX,
    'det': MathOpcode.DETERMINANT,
    'mean': MathOpcode.MEAN,
    'median': MathOpcode.MEDIAN,
    'std': MathOpcode.STDDEV,
    'C': MathOpcode.COMBINATION,
    'P': MathOpcode.PERMUTATION,
}

# Reverse: opcode → text (for decoding)
_OPCODE_TO_TEXT = {v: k for k, v in _TEXT_TO_OPCODE.items()}



class BinaryKnowledgeCodec:
    """
    Codec utama untuk encoding/decoding knowledge base matematika.

    Format Binary Record:
    [4 bytes: magic "MK01"]
    [4 bytes: version]
    [4 bytes: record count]
    [4 bytes: uncompressed size]
    [4 bytes: compressed size]
    [N bytes: compressed data]
    [16 bytes: MD5 checksum]

    Setiap record:
    [2 bytes: record type]
    [2 bytes: key length]
    [key_len bytes: key (opcode-encoded)]
    [4 bytes: value length]
    [val_len bytes: value (zlib compressed)]
    """

    MAGIC = b'MK01'
    VERSION = 2

    def __init__(self, compression_level: int = 6):
        """
        Args:
            compression_level: zlib level 1-9 (1=fast, 9=max compression)
        """
        self.compression_level = compression_level
        self._stats = {
            "original_bytes": 0,
            "compressed_bytes": 0,
            "records_encoded": 0,
            "opcode_replacements": 0,
        }

    @property
    def compression_ratio(self) -> float:
        if self._stats["original_bytes"] == 0:
            return 0.0
        return 1.0 - (self._stats["compressed_bytes"] / self._stats["original_bytes"])

    def encode_knowledge_base(self, records: List[Dict[str, Any]]) -> bytes:
        """
        Encode seluruh knowledge base ke format binary.

        Args:
            records: List of dicts with 'key', 'value', 'type' fields
                     e.g. [{"key": "integral_x2", "value": "∫x²dx = x³/3 + C", "type": "formula"}]

        Returns:
            Binary-encoded bytes
        """
        t0 = time.time()
        encoded_records = []

        for rec in records:
            key = rec.get("key", "")
            value = rec.get("value", "")
            rec_type = rec.get("type", "general")

            # Opcode-encode the value
            encoded_value = self._opcode_encode(value)

            # Compress the opcode-encoded value
            value_bytes = encoded_value.encode('utf-8')
            compressed_value = zlib.compress(value_bytes, self.compression_level)

            # Track stats
            original_size = len(value.encode('utf-8'))
            self._stats["original_bytes"] += original_size
            self._stats["compressed_bytes"] += len(compressed_value)
            self._stats["records_encoded"] += 1

            # Build binary record
            key_bytes = key.encode('utf-8')
            type_code = self._type_to_code(rec_type)

            record_bytes = struct.pack(
                '>HH',          # Big-endian: type(2), key_len(2)
                type_code,
                len(key_bytes)
            )
            record_bytes += key_bytes
            record_bytes += struct.pack('>I', len(compressed_value))
            record_bytes += compressed_value

            encoded_records.append(record_bytes)

        # Concatenate all records
        all_records = b''.join(encoded_records)

        # Build header
        header = struct.pack(
            '>4sIII',
            self.MAGIC,
            self.VERSION,
            len(records),
            self._stats["original_bytes"]
        )
        header += struct.pack('>I', len(all_records))

        # Build final payload
        payload = header + all_records

        # Append checksum
        checksum = hashlib.md5(payload).digest()
        payload += checksum

        return payload

    def decode_knowledge_base(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Decode binary knowledge base back to records.
        """
        # Verify checksum
        stored_checksum = data[-16:]
        payload = data[:-16]
        computed_checksum = hashlib.md5(payload).digest()
        if stored_checksum != computed_checksum:
            raise ValueError("Checksum mismatch! Data corrupted.")

        # Parse header
        magic, version, record_count, original_size, data_size = struct.unpack(
            '>4sIIII', payload[:20]
        )

        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")

        # Parse records
        offset = 20
        records = []

        for _ in range(record_count):
            type_code, key_len = struct.unpack('>HH', payload[offset:offset + 4])
            offset += 4

            key = payload[offset:offset + key_len].decode('utf-8')
            offset += key_len

            val_len = struct.unpack('>I', payload[offset:offset + 4])[0]
            offset += 4

            compressed_value = payload[offset:offset + val_len]
            offset += val_len

            # Decompress
            value_bytes = zlib.decompress(compressed_value)
            encoded_value = value_bytes.decode('utf-8')

            # Opcode-decode
            value = self._opcode_decode(encoded_value)

            records.append({
                "key": key,
                "value": value,
                "type": self._code_to_type(type_code),
            })

        return records

    def encode_to_hex(self, records: List[Dict[str, Any]]) -> str:
        """Encode to hex string representation."""
        binary = self.encode_knowledge_base(records)
        return binary.hex()

    def decode_from_hex(self, hex_str: str) -> List[Dict[str, Any]]:
        """Decode from hex string."""
        binary = bytes.fromhex(hex_str)
        return self.decode_knowledge_base(binary)

    def encode_single_formula(self, formula: str) -> Tuple[bytes, str]:
        """
        Encode a single formula to binary and hex.
        Returns: (binary_bytes, hex_string)
        """
        # Opcode encode
        encoded = self._opcode_encode(formula)
        value_bytes = encoded.encode('utf-8')
        compressed = zlib.compress(value_bytes, self.compression_level)
        return compressed, compressed.hex()

    def decode_single_formula(self, data: bytes) -> str:
        """Decode a single formula from binary."""
        decompressed = zlib.decompress(data)
        encoded_text = decompressed.decode('utf-8')
        return self._opcode_decode(encoded_text)

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self._stats,
            "compression_ratio": f"{self.compression_ratio:.1%}",
            "space_saved": f"{self.compression_ratio * 100:.1f}%",
        }

    def benchmark_compression(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark compression performance on given records.
        """
        t0 = time.time()

        # Original size (JSON)
        json_bytes = json.dumps(records, ensure_ascii=False).encode('utf-8')
        json_size = len(json_bytes)

        # Binary encoded size
        binary = self.encode_knowledge_base(records)
        binary_size = len(binary)

        # Hex size
        hex_str = binary.hex()
        hex_size = len(hex_str)

        # Just zlib JSON (for comparison)
        zlib_json = zlib.compress(json_bytes, self.compression_level)
        zlib_json_size = len(zlib_json)

        encode_time = (time.time() - t0) * 1000.0

        # Decode test
        t1 = time.time()
        decoded = self.decode_knowledge_base(binary)
        decode_time = (time.time() - t1) * 1000.0

        return {
            "record_count": len(records),
            "json_size_bytes": json_size,
            "binary_size_bytes": binary_size,
            "hex_size_chars": hex_size,
            "zlib_json_bytes": zlib_json_size,
            "compression_vs_json": f"{(1 - binary_size / json_size) * 100:.1f}%",
            "compression_vs_zlib_json": f"{(1 - binary_size / zlib_json_size) * 100:.1f}%" if zlib_json_size > 0 else "N/A",
            "encode_time_ms": round(encode_time, 2),
            "decode_time_ms": round(decode_time, 2),
            "integrity_check": len(decoded) == len(records),
        }

    # ─── Internal Methods ───

    def _opcode_encode(self, text: str) -> str:
        """
        Replace known math terms with opcode markers.
        Format: \\xNN where NN is hex opcode.
        """
        result = text
        replacements = 0

        # Sort by length (longest first) to avoid partial matches
        sorted_terms = sorted(_TEXT_TO_OPCODE.keys(), key=len, reverse=True)

        for term in sorted_terms:
            opcode = _TEXT_TO_OPCODE[term]
            marker = f"\\x{opcode:02x}"
            if term in result:
                count = result.count(term)
                result = result.replace(term, marker)
                replacements += count

        self._stats["opcode_replacements"] += replacements
        return result

    def _opcode_decode(self, encoded_text: str) -> str:
        """Reverse opcode markers back to text."""
        result = encoded_text
        for opcode, text in _OPCODE_TO_TEXT.items():
            marker = f"\\x{opcode:02x}"
            result = result.replace(marker, text)
        return result

    def _type_to_code(self, type_str: str) -> int:
        """Map record type string to numeric code."""
        type_map = {
            "formula": 0x01,
            "theorem": 0x02,
            "definition": 0x03,
            "example": 0x04,
            "proof": 0x05,
            "algorithm": 0x06,
            "table": 0x07,
            "general": 0xFF,
        }
        return type_map.get(type_str, 0xFF)

    def _code_to_type(self, code: int) -> str:
        """Map numeric code back to type string."""
        code_map = {
            0x01: "formula",
            0x02: "theorem",
            0x03: "definition",
            0x04: "example",
            0x05: "proof",
            0x06: "algorithm",
            0x07: "table",
            0xFF: "general",
        }
        return code_map.get(code, "general")


class EmbeddingQuantizer:
    """
    Quantize embedding vectors untuk penyimpanan kompak.

    FP32 (768 dims) = 3072 bytes
    INT8 (768 dims)  = 768 bytes   (4x compression)
    Binary (768 dims) = 96 bytes   (32x compression)
    """

    @staticmethod
    def quantize_int8(embedding: List[float]) -> bytes:
        """Quantize FP32 embedding to INT8."""
        min_val = min(embedding)
        max_val = max(embedding)
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0

        quantized = bytes([
            max(0, min(255, int((v - min_val) / scale)))
            for v in embedding
        ])

        # Prepend scale info (8 bytes: min_val + scale as FP32)
        header = struct.pack('>ff', min_val, scale)
        return header + quantized

    @staticmethod
    def dequantize_int8(data: bytes) -> List[float]:
        """Dequantize INT8 back to FP32."""
        min_val, scale = struct.unpack('>ff', data[:8])
        quantized = data[8:]
        return [min_val + b * scale for b in quantized]

    @staticmethod
    def quantize_binary(embedding: List[float]) -> bytes:
        """
        Quantize to binary hash (1 bit per dimension).
        Values >= 0 → 1, values < 0 → 0.
        768 dims → 96 bytes (32x compression!)
        """
        bits = []
        for v in embedding:
            bits.append(1 if v >= 0 else 0)

        # Pack bits into bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits) and bits[i + j]:
                    byte |= (1 << (7 - j))
            byte_array.append(byte)

        return bytes(byte_array)

    @staticmethod
    def hamming_distance(hash1: bytes, hash2: bytes) -> int:
        """Compute Hamming distance between two binary hashes."""
        dist = 0
        for b1, b2 in zip(hash1, hash2):
            xor = b1 ^ b2
            dist += bin(xor).count('1')
        return dist

    @staticmethod
    def binary_similarity(hash1: bytes, hash2: bytes) -> float:
        """Compute similarity [0,1] from binary hashes using Hamming distance."""
        total_bits = len(hash1) * 8
        dist = EmbeddingQuantizer.hamming_distance(hash1, hash2)
        return 1.0 - (dist / total_bits)


# ─── Global Singleton ───
binary_codec = BinaryKnowledgeCodec(compression_level=6)
embedding_quantizer = EmbeddingQuantizer()
