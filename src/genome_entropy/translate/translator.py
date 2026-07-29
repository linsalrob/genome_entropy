"""Translation of nucleotide sequences to amino acids."""

from dataclasses import dataclass
from typing import List

import PyGeneticCode
from Bio.Seq import Seq

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

    Uses pygenetic-code for unambiguous DNA and Biopython for sequences that
    contain IUPAC ambiguity codes. This prevents a multiply-resolvable codon
    such as ``AAN`` or ``NNN`` from being assigned an arbitrary amino acid while
    preserving specific translations for resolvable codons such as ``GCN``.

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

        # pygenetic-code 0.20 can resolve some ambiguous codons arbitrarily
        # (for example AAN and NNN as lysine). Biopython implements the IUPAC
        # semantics needed when ambiguity codes are present.
        nucleotide_sequence = orf.nt_sequence.upper()
        if set(nucleotide_sequence) <= set("ACGT"):
            aa_sequence = PyGeneticCode.translate(nucleotide_sequence, table_id)
        else:
            aa_sequence = str(Seq(nucleotide_sequence).translate(table=table_id))

        if aa_sequence != orf.aa_sequence:
            first_difference = next(
                (
                    index
                    for index, (provided, translated) in enumerate(
                        zip(orf.aa_sequence, aa_sequence),
                        start=1,
                    )
                    if provided != translated
                ),
                min(len(orf.aa_sequence), len(aa_sequence)) + 1,
            )
            logger.warning(
                "Translation mismatch for ORF %s; using translated sequence "
                "(provided length=%d, translated length=%d, first difference=%d)",
                orf.orf_id,
                len(orf.aa_sequence),
                len(aa_sequence),
                first_difference,
            )

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
