"""Data types for 3Di encoding."""

from dataclasses import dataclass
from typing import Literal

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
    method: Literal["prostt5_aa2fold"]
    model_name: str
    inference_device: str


@dataclass(frozen=True)
class IndexedSeq:
    """A sequence paired with its original position in the input list."""

    idx: int
    seq: str
