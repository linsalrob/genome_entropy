"""Data types for structural-state encoding."""

from dataclasses import dataclass
from ..translate.translator import ProteinRecord


@dataclass
class ThreeDiRecord:
    """Represents a 3Di structural encoding of a protein.

    Attributes:
        protein: The ProteinRecord that was encoded
        three_di: The 3Di token sequence
        method: Method used for encoding (always "prostt5_aa2fold")
        model_name: Name of the ProstT5 model used
        inference_device: Device used for inference ("cuda", "mps", or "cpu")
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
