"""GenBank file reading and parsing utilities."""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from ..config import DEFAULT_GENETIC_CODE_TABLE
from ..logging_config import get_logger
from ..orf.types import OrfRecord

logger = get_logger(__name__)

VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBJZUOX")
MIN_CDS_OVERLAP_FRACTION = 0.90
MIN_SHARED_AA_IDENTITY = 0.98


@dataclass(frozen=True)
class CodingInterval:
    """A coding interval in zero-based, half-open genomic coordinates."""

    start: int
    end: int
    strand: Literal["+", "-"]

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid coding interval: [{self.start}, {self.end})")
        if self.strand not in ("+", "-"):
            raise ValueError(f"Invalid coding strand: {self.strand}")

    @property
    def length(self) -> int:
        """Return the genomic interval length in nucleotides."""
        return self.end - self.start


@dataclass(frozen=True)
class CdsMatchResult:
    """Diagnostics from one coordinate-anchored ORF/CDS comparison."""

    matched: bool
    overlap_nt: int = 0
    overlap_fraction: float = 0.0
    compared_aa: int = 0
    compatible_aa: int = 0
    wildcard_aa: int = 0
    identity: float = 0.0
    phase_compatible: bool = False
    reason: str = ""


@dataclass
class GenBankCDS:
    """Represents a CDS (Coding Sequence) feature from GenBank.

    Attributes:
        parent_id: ID of the parent sequence
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        strand: Strand orientation ('+' or '-')
        protein_sequence: Translated protein sequence
        record_length: Length of the parent sequence, needed to convert
            reverse-complement ORF coordinates to genomic coordinates
        feature_id: Stable CDS identifier used in diagnostics
        translation_table: NCBI genetic code used by this CDS
        codon_start: One-based offset of the first complete CDS codon
        partial: Whether either Biopython location boundary is partial
        skip_reason: Why this feature cannot safely be matched, if applicable
    """

    parent_id: str
    start: int
    end: int
    strand: Literal["+", "-"]
    protein_sequence: str
    record_length: Optional[int] = None
    feature_id: str = ""
    translation_table: int = DEFAULT_GENETIC_CODE_TABLE
    codon_start: int = 1
    partial: bool = False
    skip_reason: str = ""


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


def _first_qualifier(qualifiers: Dict[str, List[str]], name: str) -> Optional[str]:
    """Return the first non-empty value for a GenBank qualifier."""
    values = qualifiers.get(name, [])
    return values[0] if values else None


def _translation_table(
    qualifiers: Dict[str, List[str]],
    record_default: Optional[int],
    pipeline_default: int,
) -> int:
    """Resolve the CDS genetic code with increasingly broad fallbacks."""
    value = _first_qualifier(qualifiers, "transl_table")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            logger.debug("Ignoring invalid CDS transl_table=%r", value)
    return record_default or pipeline_default or DEFAULT_GENETIC_CODE_TABLE


def _record_translation_table(record: SeqRecord) -> Optional[int]:
    """Read a record-level translation table when one is explicitly present."""
    annotations = getattr(record, "annotations", {})
    value = annotations.get("transl_table")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid record transl_table=%r", value)

    for feature in getattr(record, "features", []):
        if feature.type != "source":
            continue
        source_value = _first_qualifier(feature.qualifiers, "transl_table")
        if source_value is not None:
            try:
                return int(source_value)
            except ValueError:
                logger.debug("Ignoring invalid source transl_table=%r", source_value)
    return None


def _fallback_feature_translation(
    feature: SeqFeature,
    record_sequence: Seq,
    codon_start: int,
    translation_table: int,
) -> str:
    """Translate one CDS feature when its annotation omits ``/translation``."""
    coding_sequence = feature.extract(record_sequence)
    coding_sequence = coding_sequence[codon_start - 1 :]
    complete_length = len(coding_sequence) - (len(coding_sequence) % 3)
    if complete_length == 0:
        return ""
    return str(
        coding_sequence[:complete_length].translate(
            table=translation_table,
            to_stop=False,
        )
    )


def extract_cds_features(
    genbank_path: Union[str, Path],
    pipeline_table_id: int = DEFAULT_GENETIC_CODE_TABLE,
) -> List[GenBankCDS]:
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
                record_length = len(record)
                record_table = _record_translation_table(record)

                for feature in record.features:
                    if feature.type != "CDS":
                        continue

                    qualifiers = feature.qualifiers
                    feature_id = (
                        _first_qualifier(qualifiers, "protein_id")
                        or _first_qualifier(qualifiers, "locus_tag")
                        or _first_qualifier(qualifiers, "gene")
                        or f"{seq_id}:{feature.location}"
                    )
                    try:
                        codon_start = int(
                            _first_qualifier(qualifiers, "codon_start") or "1"
                        )
                    except ValueError:
                        codon_start = 1
                    if codon_start not in (1, 2, 3):
                        logger.debug(
                            "Skipping CDS %s: invalid codon_start=%r",
                            feature_id,
                            _first_qualifier(qualifiers, "codon_start"),
                        )
                        continue
                    translation_table = _translation_table(
                        qualifiers,
                        record_table,
                        pipeline_table_id,
                    )

                    if isinstance(feature.location, CompoundLocation):
                        logger.debug(
                            "Skipping CDS %s: compound or origin-crossing location %s "
                            "cannot be mapped to one contiguous ORF",
                            feature_id,
                            feature.location,
                        )
                        continue
                    if feature.location.strand not in (1, -1):
                        logger.debug(
                            "Skipping CDS %s: location has no definite strand",
                            feature_id,
                        )
                        continue

                    strand: Literal["+", "-"] = (
                        "+" if feature.location.strand == 1 else "-"
                    )
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    offset = codon_start - 1
                    if strand == "+":
                        start += offset
                    else:
                        end -= offset
                    if end <= start:
                        logger.debug(
                            "Skipping CDS %s: codon_start leaves an empty interval",
                            feature_id,
                        )
                        continue

                    protein_seq = _first_qualifier(qualifiers, "translation")
                    if protein_seq is None:
                        protein_seq = _fallback_feature_translation(
                            feature,
                            record.seq,
                            codon_start,
                            translation_table,
                        )

                    cds = GenBankCDS(
                        parent_id=seq_id,
                        start=start,
                        end=end,
                        strand=strand,
                        protein_sequence=protein_seq,
                        record_length=record_length,
                        feature_id=feature_id,
                        translation_table=translation_table,
                        codon_start=codon_start,
                        partial=(
                            feature.location.start.__class__.__name__ != "ExactPosition"
                            or feature.location.end.__class__.__name__
                            != "ExactPosition"
                        ),
                    )
                    cds_features.append(cds)
                    logger.debug(
                        "Extracted CDS %s: %s %s:%d-%d "
                        "(protein_len=%d, table=%d, codon_start=%d, partial=%s)",
                        feature_id,
                        seq_id,
                        strand,
                        start,
                        end,
                        len(protein_seq),
                        translation_table,
                        codon_start,
                        cds.partial,
                    )
    except Exception as e:
        logger.error("Failed to extract CDS features: %s", e)
        raise ValueError(f"Failed to extract CDS features: {e}")

    logger.info("Extracted %d CDS feature(s)", len(cds_features))
    return cds_features


def normalise_protein_sequence(sequence: str) -> str:
    """Normalise a protein for GenBank matching.

    Whitespace is removed, residues are upper-cased, and one terminal stop
    marker is stripped. An internal stop marker makes the sequence invalid for
    matching and is represented by an empty result.
    """
    normalised = "".join(sequence.split()).upper().removesuffix("*")
    return "" if "*" in normalised else normalised


def amino_acids_are_compatible(residue_a: str, residue_b: str) -> bool:
    """Return whether two aligned protein residues are compatible.

    Equal valid residues match. ``X`` is an unknown-residue wildcard, but the
    more specific ambiguity symbols ``B``, ``Z``, and ``J`` are not themselves
    wildcards. ``U`` and ``O`` are also treated as specific residues.
    """
    if residue_a not in VALID_AMINO_ACIDS or residue_b not in VALID_AMINO_ACIDS:
        return False
    return residue_a == residue_b or residue_a == "X" or residue_b == "X"


def normalise_orf_interval(
    start: int,
    end: int,
    strand: str,
    record_length: Optional[int],
) -> CodingInterval:
    """Convert get_orfs one-based inclusive coordinates to genomic coordinates.

    Positive-strand coordinates index the source sequence. Negative-strand
    coordinates index its reverse complement and therefore require the parent
    record length to map them back to the genomic axis.

    This takes plain coordinates so that callers holding serialised ORF
    locations, rather than an :class:`~genome_entropy.orf.types.OrfRecord`,
    share one normalisation implementation.
    """
    if strand not in ("+", "-"):
        raise ValueError(f"Invalid strand: {strand}")
    if start < 1 or end < start:
        raise ValueError(f"Invalid ORF coordinates: start={start}, end={end}")
    if strand == "+":
        return CodingInterval(start - 1, end, "+")
    if record_length is None:
        raise ValueError("record length is required for a reverse-strand ORF")
    if end > record_length:
        raise ValueError(f"ORF end {end} exceeds record length {record_length}")
    return CodingInterval(record_length - end, record_length - start + 1, "-")


def normalise_orf_coordinates(
    orf: OrfRecord,
    record_length: Optional[int],
) -> CodingInterval:
    """Return an ORF record's genomic interval.

    This is a thin wrapper around :func:`normalise_orf_interval`.
    """
    return normalise_orf_interval(orf.start, orf.end, orf.strand, record_length)


def normalise_genbank_coordinates(cds: GenBankCDS) -> CodingInterval:
    """Return a CDS's already-normalised Biopython genomic interval."""
    return CodingInterval(cds.start, cds.end, cds.strand)


def coding_phase_is_compatible(
    orf_interval: CodingInterval,
    cds_interval: CodingInterval,
) -> bool:
    """Return whether biological translation starts share a codon phase."""
    if orf_interval.strand != cds_interval.strand:
        return False
    if orf_interval.strand == "+":
        difference = orf_interval.start - cds_interval.start
    else:
        difference = orf_interval.end - cds_interval.end
    return difference % 3 == 0


def calculate_interval_overlap(
    first: CodingInterval,
    second: CodingInterval,
) -> tuple[int, float]:
    """Return overlap length and its fraction of the shorter interval."""
    overlap = max(0, min(first.end, second.end) - max(first.start, second.start))
    fraction = overlap / min(first.length, second.length)
    return overlap, fraction


def _shared_protein_slices(
    orf_interval: CodingInterval,
    cds_interval: CodingInterval,
) -> tuple[int, int, int]:
    """Map the shared genomic codons to ORF/CDS protein offsets and a length."""
    overlap_start = max(orf_interval.start, cds_interval.start)
    overlap_end = min(orf_interval.end, cds_interval.end)
    if overlap_end <= overlap_start:
        return 0, 0, 0

    if orf_interval.strand == "+":
        shared_start = overlap_start
        shared_codons = (overlap_end - shared_start) // 3
        return (
            (shared_start - orf_interval.start) // 3,
            (shared_start - cds_interval.start) // 3,
            shared_codons,
        )

    shared_end = overlap_end
    shared_codons = (shared_end - overlap_start) // 3
    return (
        (orf_interval.end - shared_end) // 3,
        (cds_interval.end - shared_end) // 3,
        shared_codons,
    )


def compare_shared_translation(
    orf_sequence: str,
    cds_sequence: str,
    orf_offset: int,
    cds_offset: int,
    shared_codons: int,
) -> CdsMatchResult:
    """Compare coordinate-aligned amino acids without gaps or local alignment."""
    orf_protein = normalise_protein_sequence(orf_sequence)
    cds_protein = normalise_protein_sequence(cds_sequence)
    available = min(
        shared_codons,
        max(0, len(orf_protein) - orf_offset),
        max(0, len(cds_protein) - cds_offset),
    )
    minimum_coverage = max(1, shared_codons - 1)
    if available < minimum_coverage:
        return CdsMatchResult(
            False,
            compared_aa=available,
            reason=(
                "translations do not cover all shared codons except a possible "
                "terminal stop"
            ),
        )

    compatible = 0
    wildcards = 0
    for orf_residue, cds_residue in zip(
        orf_protein[orf_offset : orf_offset + available],
        cds_protein[cds_offset : cds_offset + available],
    ):
        if amino_acids_are_compatible(orf_residue, cds_residue):
            compatible += 1
            if orf_residue == "X" or cds_residue == "X":
                wildcards += 1
    identity = compatible / available
    return CdsMatchResult(
        identity >= MIN_SHARED_AA_IDENTITY,
        compared_aa=available,
        compatible_aa=compatible,
        wildcard_aa=wildcards,
        identity=identity,
        reason=(
            "shared translation identity passed"
            if identity >= MIN_SHARED_AA_IDENTITY
            else "shared translation identity below threshold"
        ),
    )


def evaluate_orf_genbank_cds_match(
    orf: OrfRecord,
    cds: GenBankCDS,
) -> CdsMatchResult:
    """Evaluate one genomic, strand, phase, overlap, and translation match."""
    if orf.parent_id != cds.parent_id:
        return CdsMatchResult(False, reason="different parent sequence")
    if orf.strand != cds.strand:
        return CdsMatchResult(False, reason="different strand")
    if cds.skip_reason:
        return CdsMatchResult(False, reason=cds.skip_reason)

    try:
        orf_interval = normalise_orf_coordinates(orf, cds.record_length)
        cds_interval = normalise_genbank_coordinates(cds)
    except ValueError as error:
        return CdsMatchResult(False, reason=str(error))

    phase_compatible = coding_phase_is_compatible(orf_interval, cds_interval)
    overlap_nt, overlap_fraction = calculate_interval_overlap(
        orf_interval,
        cds_interval,
    )
    if not phase_compatible:
        return CdsMatchResult(
            False,
            overlap_nt=overlap_nt,
            overlap_fraction=overlap_fraction,
            reason="different codon phase",
        )
    if overlap_fraction < MIN_CDS_OVERLAP_FRACTION:
        return CdsMatchResult(
            False,
            overlap_nt=overlap_nt,
            overlap_fraction=overlap_fraction,
            phase_compatible=True,
            reason="genomic overlap below threshold",
        )

    orf_offset, cds_offset, shared_codons = _shared_protein_slices(
        orf_interval,
        cds_interval,
    )
    translation_result = compare_shared_translation(
        orf.aa_sequence,
        cds.protein_sequence,
        orf_offset,
        cds_offset,
        shared_codons,
    )
    return CdsMatchResult(
        translation_result.matched,
        overlap_nt=overlap_nt,
        overlap_fraction=overlap_fraction,
        compared_aa=translation_result.compared_aa,
        compatible_aa=translation_result.compatible_aa,
        wildcard_aa=translation_result.wildcard_aa,
        identity=translation_result.identity,
        phase_compatible=True,
        reason=translation_result.reason,
    )


def _log_match_result(
    orf: OrfRecord,
    cds: GenBankCDS,
    result: CdsMatchResult,
) -> None:
    """Emit structured debug diagnostics without logging sequence content."""
    try:
        orf_interval = normalise_orf_coordinates(orf, cds.record_length)
        cds_interval = normalise_genbank_coordinates(cds)
        coordinates = (
            f"orf=[{orf_interval.start},{orf_interval.end}) "
            f"cds=[{cds_interval.start},{cds_interval.end})"
        )
    except ValueError:
        coordinates = "coordinates=invalid"
    logger.debug(
        "GenBank CDS match orf_id=%s cds_id=%s strand=%s %s phase=%s "
        "overlap_nt=%d overlap_fraction=%.4f compared_aa=%d wildcard_aa=%d "
        "identity=%.4f matched=%s reason=%s",
        orf.orf_id,
        cds.feature_id or "<unknown>",
        orf.strand,
        coordinates,
        result.phase_compatible,
        result.overlap_nt,
        result.overlap_fraction,
        result.compared_aa,
        result.wildcard_aa,
        result.identity,
        result.matched,
        result.reason,
    )


def orf_matches_genbank_cds(orf: OrfRecord, cds: GenBankCDS) -> bool:
    """Return whether an ORF and CDS represent the same coordinate-anchored gene."""
    result = evaluate_orf_genbank_cds_match(orf, cds)
    if orf.parent_id == cds.parent_id and orf.strand == cds.strand:
        _log_match_result(orf, cds, result)
    return result.matched


def match_orf_to_genbank_cds(
    orf: OrfRecord,
    genbank_cds_list: List[GenBankCDS],
) -> bool:
    """Return whether an ORF represents any annotated GenBank CDS."""
    for cds in genbank_cds_list:
        if orf_matches_genbank_cds(orf, cds):
            logger.debug(
                "ORF %s matches GenBank CDS %s at %s:%d-%d (%s)",
                orf.orf_id,
                cds.feature_id or "<unknown>",
                cds.parent_id,
                cds.start,
                cds.end,
                cds.strand,
            )
            return True

    return False
