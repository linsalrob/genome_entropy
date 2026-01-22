"""Unified data types for pipeline output format.

This module defines the unified feature structure that eliminates redundancy
by consolidating ORF, protein, and 3Di data into a single hierarchical format.

The unified structure addresses the problem where:
- The old `proteins` list duplicated entire ORF objects
- The old `three_dis` list duplicated entire protein objects (which contained ORFs)
- Each level repeated sequences, coordinates, and metadata

The new structure stores each piece of biological information exactly once,
organized hierarchically by biological concept.
"""

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class FeatureLocation:
    """Genomic location of a feature (ORF).
    
    Attributes:
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        strand: Strand orientation ('+' or '-')
        frame: Reading frame (0, 1, 2, or 3)
    """
    start: int
    end: int
    strand: Literal["+", "-"]
    frame: int


@dataclass
class FeatureDNA:
    """DNA-level information for a feature.
    
    Attributes:
        nt_sequence: Nucleotide sequence
        length: Length of nucleotide sequence
    """
    nt_sequence: str
    length: int


@dataclass
class FeatureProtein:
    """Protein-level information for a feature.
    
    Attributes:
        aa_sequence: Amino acid sequence
        length: Length of amino acid sequence
    """
    aa_sequence: str
    length: int


@dataclass
class FeatureThreeDi:
    """3Di structural encoding for a feature.
    
    Attributes:
        encoding: 3Di token sequence
        length: Length of 3Di sequence
        method: Method used for encoding (e.g., "prostt5_aa2fold")
        model_name: Name of the model used
        inference_device: Device used for inference ("cuda", "mps", or "cpu")
    """
    encoding: str
    length: int
    method: str
    model_name: str
    inference_device: str


@dataclass
class FeatureMetadata:
    """Metadata about a feature.
    
    Attributes:
        parent_id: ID of the parent DNA sequence
        table_id: NCBI genetic code table ID used
        has_start_codon: Whether the ORF has a start codon
        has_stop_codon: Whether the ORF has a stop codon
        in_genbank: Whether this ORF matches a CDS annotated in GenBank
    """
    parent_id: str
    table_id: int
    has_start_codon: bool
    has_stop_codon: bool
    in_genbank: bool


@dataclass
class FeatureEntropy:
    """Entropy values at different representation levels for a feature.
    
    Attributes:
        dna_entropy: Shannon entropy of nucleotide sequence
        protein_entropy: Shannon entropy of amino acid sequence
        three_di_entropy: Shannon entropy of 3Di encoding
    """
    dna_entropy: float
    protein_entropy: float
    three_di_entropy: float


@dataclass
class UnifiedFeature:
    """Unified representation of a biological feature (ORF and derived data).
    
    This structure consolidates all information about a single ORF into one
    hierarchical object, eliminating the redundancy present in the old format
    where ORF data was duplicated in proteins list and protein data was
    duplicated in three_dis list.
    
    Attributes:
        orf_id: Unique identifier for this feature
        location: Genomic coordinates
        dna: DNA sequence information
        protein: Protein sequence information
        three_di: 3Di structural encoding
        metadata: Additional metadata
        entropy: Entropy values at all representation levels
    """
    orf_id: str
    location: FeatureLocation
    dna: FeatureDNA
    protein: FeatureProtein
    three_di: FeatureThreeDi
    metadata: FeatureMetadata
    entropy: FeatureEntropy


@dataclass
class UnifiedPipelineResult:
    """Result of running the complete DNA to 3Di pipeline (unified format).
    
    This is the new format that eliminates redundancy by using a single
    dictionary of features keyed by orf_id, instead of separate parallel
    lists for orfs, proteins, and three_dis.
    
    Attributes:
        schema_version: Version of the output schema (for compatibility tracking)
        input_id: ID of the input DNA sequence
        input_dna_length: Length of the input DNA sequence
        dna_entropy_global: Entropy of the entire input DNA sequence
        alphabet_sizes: Dictionary with alphabet sizes for each representation
        features: Dictionary mapping orf_id to UnifiedFeature objects
    """
    schema_version: str
    input_id: str
    input_dna_length: int
    dna_entropy_global: float
    alphabet_sizes: Dict[str, int]
    features: Dict[str, UnifiedFeature]
