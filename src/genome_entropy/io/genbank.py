"""GenBank file reading and parsing utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GenBankCDS:
    """Represents a CDS (Coding Sequence) feature from GenBank.
    
    Attributes:
        parent_id: ID of the parent sequence
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        strand: Strand orientation ('+' or '-')
        protein_sequence: Translated protein sequence
    """
    
    parent_id: str
    start: int
    end: int
    strand: str
    protein_sequence: str


def read_genbank(genbank_path: Union[str, Path]) -> Dict[str, str]:
    """Read a GenBank file and return a dictionary of sequence_id -> DNA sequence.
    
    Args:
        genbank_path: Path to GenBank file
        
    Returns:
        Dictionary mapping sequence IDs to DNA sequences
        
    Raises:
        FileNotFoundError: If the GenBank file doesn't exist
        ValueError: If the GenBank file is malformed
    """
    genbank_path = Path(genbank_path)
    logger.info("Reading GenBank file: %s", genbank_path)
    
    if not genbank_path.exists():
        logger.error("GenBank file not found: %s", genbank_path)
        raise FileNotFoundError(f"GenBank file not found: {genbank_path}")
    
    sequences = {}
    
    try:
        for record in SeqIO.parse(genbank_path, "genbank"):
            seq_id = record.id
            dna_sequence = str(record.seq).upper()
            sequences[seq_id] = dna_sequence
            logger.debug("Read sequence '%s' (length=%d)", seq_id, len(dna_sequence))
    except Exception as e:
        logger.error("Failed to parse GenBank file: %s", e)
        raise ValueError(f"Failed to parse GenBank file: {e}")
    
    if not sequences:
        logger.error("No sequences found in GenBank file: %s", genbank_path)
        raise ValueError(f"No sequences found in GenBank file: {genbank_path}")
    
    logger.info("Successfully read %d sequence(s) from %s", len(sequences), genbank_path)
    return sequences


def extract_cds_features(genbank_path: Union[str, Path]) -> List[GenBankCDS]:
    """Extract CDS features from a GenBank file.
    
    Args:
        genbank_path: Path to GenBank file
        
    Returns:
        List of GenBankCDS objects
        
    Raises:
        FileNotFoundError: If the GenBank file doesn't exist
        ValueError: If the GenBank file is malformed
    """
    genbank_path = Path(genbank_path)
    logger.info("Extracting CDS features from GenBank file: %s", genbank_path)
    
    if not genbank_path.exists():
        logger.error("GenBank file not found: %s", genbank_path)
        raise FileNotFoundError(f"GenBank file not found: {genbank_path}")
    
    cds_features = []
    
    try:
        for record in SeqIO.parse(genbank_path, "genbank"):
            seq_id = record.id
            
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                
                # Get protein translation if available
                protein_seq = ""
                if "translation" in feature.qualifiers:
                    protein_seq = feature.qualifiers["translation"][0]
                
                # Convert strand
                strand = "+" if feature.location.strand == 1 else "-"
                
                # BioPython uses 0-based coordinates (inclusive start, exclusive end)
                start = int(feature.location.start)
                end = int(feature.location.end)
                
                cds = GenBankCDS(
                    parent_id=seq_id,
                    start=start,
                    end=end,
                    strand=strand,
                    protein_sequence=protein_seq,
                )
                cds_features.append(cds)
                logger.debug(
                    "Extracted CDS: %s %s:%d-%d (protein_len=%d)",
                    seq_id,
                    strand,
                    start,
                    end,
                    len(protein_seq),
                )
    except Exception as e:
        logger.error("Failed to extract CDS features: %s", e)
        raise ValueError(f"Failed to extract CDS features: {e}")
    
    logger.info("Extracted %d CDS feature(s)", len(cds_features))
    return cds_features


def match_orf_to_genbank_cds(
    orf_aa_sequence: str,
    genbank_cds_list: List[GenBankCDS],
    min_c_terminal_match: int = 10,
) -> bool:
    """Check if an ORF matches any GenBank CDS by C-terminal sequence.
    
    Matches are determined by comparing the C-terminal (end) sequences of the
    protein sequences. This accounts for cases where the predicted ORF may
    not exactly match the annotated CDS start position.
    
    Args:
        orf_aa_sequence: Amino acid sequence of the ORF
        genbank_cds_list: List of CDS features from GenBank
        min_c_terminal_match: Minimum length of C-terminal sequence to match (default: 10)
        
    Returns:
        True if the ORF C-terminal matches any GenBank CDS, False otherwise
    """
    if not orf_aa_sequence or not genbank_cds_list:
        return False
    
    # Remove stop codon (*) from ORF sequence for comparison
    orf_seq_clean = orf_aa_sequence.rstrip("*")
    
    # Get C-terminal sequence (last N amino acids)
    orf_c_terminal = orf_seq_clean[-min_c_terminal_match:]
    
    # If ORF is shorter than min match length, use the whole sequence
    if len(orf_seq_clean) < min_c_terminal_match:
        orf_c_terminal = orf_seq_clean
    
    # Check against all GenBank CDS features
    for cds in genbank_cds_list:
        if not cds.protein_sequence:
            continue
        
        cds_seq_clean = cds.protein_sequence.rstrip("*")
        
        # Get C-terminal of CDS
        cds_c_terminal = cds_seq_clean[-min_c_terminal_match:]
        
        # If CDS is shorter than min match length, use the whole sequence
        if len(cds_seq_clean) < min_c_terminal_match:
            cds_c_terminal = cds_seq_clean
        
        # Check if C-terminals match
        if orf_c_terminal == cds_c_terminal:
            logger.debug(
                "ORF C-terminal matches GenBank CDS (match_length=%d)",
                len(orf_c_terminal),
            )
            return True
    
    return False
