"""
MOKO Security Package — Adversarial Cryptographic Defense System (ACDS)
  + Cryptographic Reasoning Gateway (MCRG)
=========================================================================
Red Team vs Blue Team adaptive security dengan Proof-Carrying Rule Chain.
Plus: Homomorphic-encrypted reasoning pipeline (LWE FHE + Masked SMT).

Komponen:
  - red_team_fuzzer:  Generator exploit adversarial (8 kategori CWE/OWASP)
  - crypto_chain:     Hash chain + Z3 proof ledger untuk integritas rule
  - blue_team_defender: Adaptive rule synthesis + live hotpatching
  - acds_engine:      Orchestrator Red↔Blue self-play campaign
  - crypto_gateway:   LWE FHE cipher + Masked SMT + Fiat-Shamir NIZK proof

Filosofi: Seperti sistem imun adaptif — setiap serangan menghasilkan
antibodi baru (rule) yang disimpan dalam memori imunologis permanen
(crypto chain) yang diverifikasi secara matematis (Z3 SMT proof).

MCRG Filosofi: Zombie Kripto — semua penalaran AI dikriptografikan dari
input hingga output. Model bisa menalar dalam domain terenkripsi tanpa
membuka plaintext ke memori global. Output didekripsi hanya setelah
proof integritas (NIZK) terbukti valid.
"""

from .red_team_fuzzer import RedTeamFuzzer, ExploitCategory, ExploitPayload
from .crypto_chain import CryptoChain, ChainBlock
from .blue_team_defender import BlueTeamDefender
from .acds_engine import ACDSEngine
from .crypto_gateway import (
    CryptoGateway,
    SecureEncoder,
    MaskedSMTEngine,
    zkReasoningCertificate,
    GatewayCertificate,
    GatewayCertificateV2,
    Ciphertext,
    NIZKProof,
)
from .crypto_red_team import (
    MilitaryCryptoRedTeam,
    AttackVector,
    Vulnerability,
    Severity,
    AttackReport,
)
from .ring_lwe_engine import (
    RingLWEEncoder,
    RingCiphertext,
    RingLWEParams,
    MOKO_BASIC,
    MOKO_SECURE,
    MOKO_ULTRA,
)
from .gaussian_noise_engine import (
    DiscreteGaussianSampler,
    CenteredBinomialSampler,
    NoiseSecurityValidator,
)
from .pqc_signatures import (
    MokoLatticeSign,
    QuantumNIZKCertificate,
    PQCSignature,
    PQCKeyPair,
)
from .mcrg_math_foundations import (
    QuantumHardnessProver,
)

__all__ = [
    # ACDS
    "RedTeamFuzzer",
    "ExploitCategory",
    "ExploitPayload",
    "CryptoChain",
    "ChainBlock",
    "BlueTeamDefender",
    "ACDSEngine",
    # MCRG
    "CryptoGateway",
    "SecureEncoder",
    "MaskedSMTEngine",
    "zkReasoningCertificate",
    "GatewayCertificate",
    "GatewayCertificateV2",
    "Ciphertext",
    "NIZKProof",
    # MCRG v2
    "RingLWEEncoder",
    "RingCiphertext",
    "RingLWEParams",
    "MOKO_BASIC",
    "MOKO_SECURE",
    "MOKO_ULTRA",
    "DiscreteGaussianSampler",
    "CenteredBinomialSampler",
    "NoiseSecurityValidator",
    "MokoLatticeSign",
    "QuantumNIZKCertificate",
    "PQCSignature",
    "PQCKeyPair",
    "QuantumHardnessProver",
    # MCRT — Military Red Team
    "MilitaryCryptoRedTeam",
    "AttackVector",
    "Vulnerability",
    "Severity",
    "AttackReport",
]
