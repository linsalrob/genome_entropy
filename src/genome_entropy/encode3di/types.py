"""Data types for structural-state encoding."""

from dataclasses import dataclass
from ..translate.translator import ProteinRecord


@dataclass
class ThreeDiRecord:
    """Structural-state encodings predicted for a protein.

    Attributes:
        protein: The ProteinRecord that was encoded
        three_di: The 3Di token sequence
        method: Encoder method identifier
        model_name: Canonical model identifier used for inference
        inference_device: Device string, such as ``cuda``, ``mps``, or ``cpu``
        twelve_state: Optional 12-state sequence; ``None`` for 3Di-only models
    """

    protein: ProteinRecord
    three_di: str
    method: str
    model_name: str
    inference_device: str
    twelve_state: str | None = None


@dataclass(frozen=True)
class StructuralEncoding:
    """Associated structural encodings produced by one model forward pass."""

    three_di: str
    twelve_state: str | None


@dataclass(frozen=True)
class IndexedSeq:
    """A sequence paired with its original position in the input list."""

    idx: int
    seq: str
