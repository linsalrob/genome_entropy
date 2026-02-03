"""Translation of nucleotide sequences to amino acids."""

from dataclasses import dataclass
from typing import List

import PyGeneticCode

from ..config import DEFAULT_GENETIC_CODE_TABLE
from ..errors import TranslationError
from ..logging_config import get_logger
from ..orf.types import OrfRecord

logger = get_logger(__name__)


@dataclass
class ProteinRecord:
    """Represents a translated protein from an ORF.

    Attributes:
        orf: The OrfRecord that was translated
        aa_sequence: The amino acid sequence
        aa_length: Length of the amino acid sequence
    """

    orf: OrfRecord
    aa_sequence: str
    aa_length: int

    def __post_init__(self) -> None:
        """Validate protein attributes."""
        if self.aa_length != len(self.aa_sequence):
            raise ValueError(
                f"aa_length {self.aa_length} doesn't match sequence length "
                f"{len(self.aa_sequence)}"
            )


def translate_orf(
    orf: OrfRecord, table_id: int = DEFAULT_GENETIC_CODE_TABLE
) -> ProteinRecord:
    """Translate an ORF to a protein sequence.

    Uses the pygenetic-code library for translation with NCBI genetic codes.
    Ambiguous codons (containing N or other IUPAC codes) are translated to 'X'.

    Args:
        orf: OrfRecord to translate
        table_id: NCBI genetic code table ID (default: from config)

    Returns:
        ProteinRecord with translated sequence

    Raises:
        TranslationError: If translation fails
    """

    logger.debug(
        "Translating ORF %s (length=%d nt) with table %d",
        orf.orf_id,
        len(orf.nt_sequence),
        table_id,
    )

    try:

        # Translate sequence
        aa_sequence = PyGeneticCode.translate(orf.nt_sequence, table_id)

        if aa_sequence != orf.aa_sequence:
            errmsg = f"""
            Translated amino acid sequence is not the same as read from the file:
            Provided:
            {orf.aa_sequence}
            Translated:
            {aa_sequence}
            DNA sequence:
            {orf.nt_sequence}
            Location:
            Start: {orf.start} Stop: {orf.end} Frame: {orf.frame} Strand: {orf.strand}
            """
            logger.error("Translation mismatch for ORF %s", orf.orf_id)
            raise ValueError(errmsg)

        # Remove stop codon (*) if present at the end
        if aa_sequence.endswith("*"):
            aa_sequence = aa_sequence[:-1]

        logger.debug(
            "Successfully translated ORF %s to %d amino acids",
            orf.orf_id,
            len(aa_sequence),
        )

        return ProteinRecord(
            orf=orf,
            aa_sequence=aa_sequence,
            aa_length=len(aa_sequence),
        )

    except Exception as e:
        logger.error("Failed to translate ORF %s: %s", orf.orf_id, e)
        raise TranslationError(f"Failed to translate ORF {orf.orf_id}: {e}")


def translate_orfs(
    orfs: List[OrfRecord], table_id: int = DEFAULT_GENETIC_CODE_TABLE
) -> List[ProteinRecord]:
    """Translate multiple ORFs to protein sequences.

    Args:
        orfs: List of OrfRecord objects to translate
        table_id: NCBI genetic code table ID

    Returns:
        List of ProteinRecord objects
    """
    logger.info("Translating %d ORF(s) with table %d", len(orfs), table_id)
    proteins = [translate_orf(orf, table_id=table_id) for orf in orfs]
    logger.info("Successfully translated %d ORF(s) to proteins", len(proteins))
    return proteins
