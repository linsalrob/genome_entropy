"""ORF finder wrapper using get_orfs binary."""

import subprocess
import tempfile
import re
import sys
from pathlib import Path
from typing import Dict, List, Union

from ..config import DEFAULT_GENETIC_CODE_TABLE, GET_ORFS_BINARY
from ..errors import OrfFinderError
from ..logging_config import get_logger
from .types import OrfRecord

logger = get_logger(__name__)


def find_orfs(
    sequences: Dict[str, str],
    table_id: int = DEFAULT_GENETIC_CODE_TABLE,
    min_nt_length: int = 90,
    binary_path: str = GET_ORFS_BINARY,
) -> List[OrfRecord]:
    """Find ORFs in DNA sequences using get_orfs binary.
    
    This function wraps the external get_orfs binary (https://github.com/linsalrob/get_orfs).
    The binary must be installed and available in PATH or specified via binary_path.
    
    Args:
        sequences: Dictionary mapping sequence IDs to DNA sequences
        table_id: NCBI genetic code table ID (default: 11, bacterial)
        min_nt_length: Minimum ORF length in nucleotides (default: 90)
        binary_path: Path to get_orfs binary (default: from config/environment)
        
    Returns:
        List of OrfRecord objects
        
    Raises:
        OrfFinderError: If get_orfs binary is not found or fails
    """
    logger.info(
        "Starting ORF finding for %d sequence(s) (table=%d, min_length=%d)",
        len(sequences),
        table_id,
        min_nt_length,
    )
    
    # Check if binary exists
    try:
        logger.debug("Checking get_orfs binary at: %s", binary_path)
        result = subprocess.run(
            [binary_path, "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise OrfFinderError(f"get_orfs binary check failed: {result.stderr}")
        logger.debug("get_orfs binary check successful")
    except FileNotFoundError:
        logger.error("get_orfs binary not found at: %s", binary_path)
        raise OrfFinderError(
            f"get_orfs binary not found at: {binary_path}\n"
            "Please install from https://github.com/linsalrob/get_orfs\n"
            "or set GET_ORFS_PATH environment variable."
        )
    except subprocess.TimeoutExpired:
        logger.error("get_orfs binary check timed out")
        raise OrfFinderError("get_orfs binary check timed out")
    
    # Write sequences to temporary FASTA file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_fasta:
        tmp_fasta_path = tmp_fasta.name
        for seq_id, sequence in sequences.items():
            tmp_fasta.write(f">{seq_id}\n{sequence}\n")
    
    logger.debug("Wrote sequences to temporary file: %s", tmp_fasta_path)
    
    try:
        # Run get_orfs
        # Expected command: get_orfs -f fasta_file -t table_id -l min_length
        cmd = [
            binary_path,
            "-f", tmp_fasta_path,
            "-t", str(table_id),
            "-l", str(min_nt_length),
        ]
        
        logger.debug("Running command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error("get_orfs failed with return code %d: %s", result.returncode, result.stderr)
            raise OrfFinderError(f"get_orfs failed: {result.stderr}")
        
        logger.debug("get_orfs completed successfully")
        
        # Parse output
        orfs = _parse_get_orfs_output(result.stdout, sequences, table_id)
        logger.info("Found %d ORF(s) in %d sequence(s)", len(orfs), len(sequences))
        return orfs
        
    finally:
        # Clean up temporary file
        Path(tmp_fasta_path).unlink(missing_ok=True)
        logger.debug("Cleaned up temporary file: %s", tmp_fasta_path)

def _parse_orf_header_line(header: str, table_id: int) -> OrfRecord:

    FASTA_RE = re.compile(
    r"""
        ^>
        (?P<parent_id>[^-\s]+)      # JQ995537
        -
        (?P<orf_id>orf\d+)          # orf14635
        \s+
        \[
            (?P=parent_id)          # repeat accession
            \s+frame\s+
            (?P<frame>[+-]?\d+)
            \s+
            (?P<start>\d+)
            \s+
            (?P<end>\d+)
        \]
        $
    """,
    re.VERBOSE
    )

    m = FASTA_RE.match(header)
    if not m:
        raise ValueError("Header did not match expected format")

    data = m.groupdict()
    strand = '+'
    if int(data['frame']) < 0:
        strand = '-'

    orf = OrfRecord(
        parent_id=data['parent_id'],
        orf_id=data['orf_id'],
        start=int(data["start"]),
        end=int(data["end"]),
        frame=abs(int(data["frame"])),
        strand=strand,
        nt_sequence="",
        aa_sequence="",
        table_id=table_id,
        has_start_codon=False,
        has_stop_codon=False
    )

    return orf

def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def _extract_orf_dna_sequence(sequences: Dict[str, str], current_orf: OrfRecord) -> str:
    """
    Extract the sequence from sequences
    """

    if current_orf.parent_id not in sequences:
        raise ValueError(f"We were asked to extract an orf for {current_orf.parent_id} but it is not in our sequences: {sequences.keys()}")

    if current_orf.start < 1 or current_orf.end < 1:
        raise ValueError("Coordinates must be >= 1")

    if current_orf.end < current_orf.start:
        raise ValueError("End coordinate precedes start coordinate")

    # Convert to Python slicing (0-based, end-exclusive)
    dna = sequences[current_orf.parent_id]
    if current_orf.strand == '-':
        dna = reverse_complement(dna)
    
    orf_seq = dna[current_orf.start - 1 : current_orf.end]

    return orf_seq

def _parse_get_orfs_output(
    output: str, sequences: Dict[str, str], table_id: int
) -> List[OrfRecord]:
    """Parse get_orfs output into OrfRecord objects.
    
    Format: >JQ995537-orf14635 [JQ995537 frame -3 96951 97093]
    
    Args:
        output: stdout from get_orfs
        sequences: Original DNA sequences (for validation)
        table_id: Genetic code table used
        
    Returns:
        List of OrfRecord objects
    """
    logger.debug("Parsing get_orfs output")
    orfs = []
    current_orf = None
    
    for line in output.strip().split("\n"):
        try:
            line = line.rstrip('\r\n')
            if line.startswith(">"):
                if current_orf is not None:
                    current_orf.nt_sequence = _extract_orf_dna_sequence(sequences, current_orf)
                    current_orf.has_start_codon = True if 'M' in current_orf.aa_sequence else False
                    current_orf.has_stop_codon  = True if '*' in current_orf.aa_sequence else False
                    orfs.append(current_orf)
                    logger.debug("Parsed ORF: %s (length=%d nt)", current_orf.orf_id, len(current_orf.nt_sequence))
                current_orf = _parse_orf_header_line(line, table_id)
            else:
                current_orf.aa_sequence += line.strip()
        except (ValueError, IndexError) as e:
            logger.error("Error handling sequence line '%s': %s", line, e)
            print(f"Error handling sequence {line}", file=sys.stderr)
            raise Exception(f"Error as {e}")
            continue
    
    logger.debug("Parsed %d ORF(s) from get_orfs output", len(orfs))
    return orfs
