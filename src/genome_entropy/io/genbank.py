"""GenBank file reading and parsing utilities."""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from Bio import SeqIO

from ..logging_config import get_logger
from ..orf.types import OrfRecord

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

    Automatically detects and handles gzipped files (ending in .gz).

    Args:
        genbank_path: Path to GenBank file (plain text or gzipped)

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
        # Auto-detect gzipped files by extension
        is_gzipped = str(genbank_path).endswith(".gz")
        open_func = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"

        with open_func(genbank_path, mode) as handle:
            for record in SeqIO.parse(handle, "genbank"):
                seq_id = record.id
                dna_sequence = str(record.seq).upper()
                sequences[seq_id] = dna_sequence
                logger.debug(
                    "Read sequence '%s' (length=%d)", seq_id, len(dna_sequence)
                )
    except Exception as e:
        logger.error("Failed to parse GenBank file: %s", e)
        raise ValueError(f"Failed to parse GenBank file: {e}")

    if not sequences:
        logger.error("No sequences found in GenBank file: %s", genbank_path)
        raise ValueError(f"No sequences found in GenBank file: {genbank_path}")

    logger.info(
        "Successfully read %d sequence(s) from %s", len(sequences), genbank_path
    )
    return sequences


def extract_cds_features(genbank_path: Union[str, Path]) -> List[GenBankCDS]:
    """Extract CDS features from a GenBank file.

    Automatically detects and handles gzipped files (ending in .gz).

    Args:
        genbank_path: Path to GenBank file (plain text or gzipped)

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
        # Auto-detect gzipped files by extension
        is_gzipped = str(genbank_path).endswith(".gz")
        open_func = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"

        with open_func(genbank_path, mode) as handle:
            for record in SeqIO.parse(handle, "genbank"):
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


def orf_matches_genbank_cds(orf: OrfRecord, cds: GenBankCDS) -> bool:
    """Match proteins by their strand-aware stop and exact C-terminal sequence.

    C-terminal matching recognises partial CDS translations and alternative
    initiation residues without treating internal or N-terminal similarity as
    evidence that two proteins are the same.
    """
    if orf.parent_id != cds.parent_id or orf.strand != cds.strand:
        return False

    # get_orfs reports one-based inclusive coordinates. Biopython reports
    # zero-based, half-open locations, so only the reverse-strand low coordinate
    # needs an offset when comparing biological translation termination sites.
    orf_stop = orf.end if orf.strand == "+" else orf.start
    cds_stop = cds.end if cds.strand == "+" else cds.start + 1
    if orf_stop != cds_stop:
        return False

    def normalise(sequence: str) -> str:
        normalised = "".join(sequence.split()).upper()
        return normalised.removesuffix("*")

    orf_c_terminal = normalise(orf.aa_sequence)[1:]
    cds_c_terminal = normalise(cds.protein_sequence)[1:]
    if not orf_c_terminal or not cds_c_terminal:
        return False

    shorter, longer = sorted(
        (orf_c_terminal, cds_c_terminal),
        key=len,
    )
    return longer.endswith(shorter)


def match_orf_to_genbank_cds(
    orf: OrfRecord,
    genbank_cds_list: List[GenBankCDS],
) -> bool:
    """Return whether an ORF represents any annotated GenBank CDS."""
    for cds in genbank_cds_list:
        if orf_matches_genbank_cds(orf, cds):
            logger.debug(
                "ORF %s matches GenBank CDS at %s:%d-%d (%s)",
                orf.orf_id,
                cds.parent_id,
                cds.start,
                cds.end,
                cds.strand,
            )
            return True

    return False
