"""Configuration, model capabilities, and constants for genome_entropy."""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Literal

DEFAULT_GENETIC_CODE_TABLE = 11
DEFAULT_MIN_NT_LENGTH = 90
DEFAULT_MIN_AA_LENGTH = 30

MODERNPROST_50M_MODEL = "gbouras13/modernprost-50M"
MODERNPROST_1B_MODEL = "gbouras13/modernprost-base"
MODERNPROST_BASE_DEPRECATED_MODEL = "gbouras13/modernprost-base-deprecated"
MODERNPROST_PROFILES_DEPRECATED_MODEL = "gbouras13/modernprost-profiles-deprecated"
# Backward-compatible constant imports use the current base model and the
# renamed legacy profiles repository respectively.
MODERNPROST_BASE_MODEL = MODERNPROST_1B_MODEL
MODERNPROST_PROFILES_MODEL = MODERNPROST_PROFILES_DEPRECATED_MODEL
PROSTT5_MODEL = "Rostlab/ProstT5"
PROSTT5_FP16_MODEL = "Rostlab/ProstT5_fp16"
DEFAULT_PROSTT5_MODEL = MODERNPROST_50M_MODEL
DEFAULT_ENCODING_SIZE = 10000


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities and provenance for one supported Hugging Face model."""

    model_name: str
    family: Literal["modernprost_multitask", "modernprost_legacy", "prostt5"]
    supports_3di: bool
    supports_12st: bool
    supports_profiles: bool = False
    deprecated: bool = False
    description: str = ""


MODEL_REGISTRY: Dict[str, ModelCapabilities] = {
    MODERNPROST_50M_MODEL: ModelCapabilities(
        MODERNPROST_50M_MODEL,
        "modernprost_multitask",
        True,
        True,
        description="Default approximately 50M-parameter dual-head ModernProst model",
    ),
    MODERNPROST_1B_MODEL: ModelCapabilities(
        MODERNPROST_1B_MODEL,
        "modernprost_multitask",
        True,
        True,
        description="Larger approximately 1B-parameter dual-head ModernProst model",
    ),
    MODERNPROST_BASE_DEPRECATED_MODEL: ModelCapabilities(
        MODERNPROST_BASE_DEPRECATED_MODEL,
        "modernprost_legacy",
        True,
        False,
        deprecated=True,
        description="Deprecated legacy ModernProst base model (3Di only)",
    ),
    MODERNPROST_PROFILES_DEPRECATED_MODEL: ModelCapabilities(
        MODERNPROST_PROFILES_DEPRECATED_MODEL,
        "modernprost_legacy",
        True,
        False,
        supports_profiles=True,
        deprecated=True,
        description="Deprecated legacy ModernProst profiles model (3Di only)",
    ),
    PROSTT5_MODEL: ModelCapabilities(
        PROSTT5_MODEL,
        "prostt5",
        True,
        False,
        description="Original ProstT5 model (3Di only)",
    ),
    PROSTT5_FP16_MODEL: ModelCapabilities(
        PROSTT5_FP16_MODEL,
        "prostt5",
        True,
        False,
        description="Original half-precision ProstT5 model (3Di only)",
    ),
}

MODEL_ALIASES = {
    "gbouras13/modernprost-profiles": MODERNPROST_PROFILES_DEPRECATED_MODEL,
}


def resolve_model_name(model_name: str, *, warn: bool = True) -> str:
    """Resolve old repository aliases and validate a supported model identifier."""
    resolved = MODEL_ALIASES.get(model_name, model_name)
    if resolved != model_name and warn:
        warnings.warn(
            f"Model {model_name!r} was renamed; using {resolved!r}.",
            FutureWarning,
            stacklevel=2,
        )
    if resolved not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unsupported model {model_name!r}. Supported models: {supported}"
        )
    return resolved


def get_model_capabilities(model_name: str, *, warn: bool = True) -> ModelCapabilities:
    """Return central capability metadata for a model or legacy alias."""
    return MODEL_REGISTRY[resolve_model_name(model_name, warn=warn)]


def supported_models_help() -> str:
    """Return CLI help text generated from the central model registry."""
    return "Supported models: " + ", ".join(
        f"{name} ({caps.description})" for name, caps in MODEL_REGISTRY.items()
    )


PROSTT5_MODELS = {
    name for name, caps in MODEL_REGISTRY.items() if caps.family == "prostt5"
}
MODERNPROST_MODELS = {
    name
    for name, caps in MODEL_REGISTRY.items()
    if caps.family.startswith("modernprost")
} | set(MODEL_ALIASES)

THREEDDI_ALPHABET_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
# No symbolic nomenclature is exposed by the model; these characters serialize IDs 0-11.
TWELVE_STATE_ALPHABET_ORDERED = "ABCDEFGHIJKL"
THREEDDI_ALPHABET = set(THREEDDI_ALPHABET_ORDERED)
TWELVE_STATE_ALPHABET = set(TWELVE_STATE_ALPHABET_ORDERED)
THREEDDI_ALPHABET_SIZE = 20
TWELVE_STATE_ALPHABET_SIZE = 12

DNA_ALPHABET = set("ACGT")
DNA_ALPHABET_WITH_N = set("ACGTN")
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
AA_ALPHABET_EXTENDED = set("ACDEFGHIKLMNPQRSTVWYX*")

GET_ORFS_BINARY = os.environ.get("GET_ORFS_PATH", "get_orfs")
AUTO_DEVICE = "auto"
CUDA_DEVICE = "cuda"
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"
FASTA_EXTENSIONS = {".fasta", ".fa", ".fna", ".faa"}
JSON_EXTENSIONS = {".json"}
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USER_ERROR = 2
EXIT_RUNTIME_ERROR = 3
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
