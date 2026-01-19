"""Data types for ORF representation."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class OrfRecord:
    """Represents a single Open Reading Frame (ORF).
    
    Attributes:
        parent_id: ID of the parent DNA sequence
        orf_id: Unique identifier for this ORF
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        strand: Strand orientation ('+' or '-')
        frame: Reading frame (0, 1, or 2)
        nt_sequence: Nucleotide sequence of the ORF
        aa_sequence: Amino acid sequence of the ORF
        table_id: NCBI genetic code table ID used
        has_start_codon: Whether the ORF has a start codon
        has_stop_codon: Whether the ORF has a stop codon
    """

    parent_id: str
    orf_id: str
    start: int
    end: int
    strand: Literal["+", "-"]
    frame: int
    nt_sequence: str
    aa_sequence: str
    table_id: int
    has_start_codon: bool
    has_stop_codon: bool
    
    def __post_init__(self) -> None:
        """Validate ORF attributes."""
        if self.strand not in ("+", "-"):
            raise ValueError(f"Invalid strand: {self.strand}")
        if self.frame not in (0, 1, 2):
            raise ValueError(f"Invalid frame: {self.frame}")
        if self.start < 0:
            raise ValueError(f"Invalid start position: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Invalid end position: {self.end} (must be > start {self.start})")
        # Note: We don't validate sequence length here because sequences may be empty
        # when initially created and filled in later (e.g., by ORF finder)
