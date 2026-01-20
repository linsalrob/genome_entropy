"""FASTA file reading and writing utilities."""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

from ..logging_config import get_logger

logger = get_logger(__name__)


def read_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """Read a FASTA file and return a dictionary of sequence_id -> sequence.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence IDs to sequences
        
    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
        ValueError: If the FASTA file is malformed
    """
    fasta_path = Path(fasta_path)
    logger.info("Reading FASTA file: %s", fasta_path)
    
    if not fasta_path.exists():
        logger.error("FASTA file not found: %s", fasta_path)
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    sequences = {}
    current_id = None
    current_seq_parts = []
    
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq_parts)
                    logger.debug("Read sequence '%s' (length=%d)", current_id, len(sequences[current_id]))
                
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq_parts = []
            else:
                if current_id is None:
                    logger.error("FASTA sequence found before header in %s", fasta_path)
                    raise ValueError("FASTA sequence found before header")
                current_seq_parts.append(line.upper())
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq_parts)
            logger.debug("Read sequence '%s' (length=%d)", current_id, len(sequences[current_id]))
    
    if not sequences:
        logger.error("No sequences found in FASTA file: %s", fasta_path)
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
    
    logger.info("Successfully read %d sequence(s) from %s", len(sequences), fasta_path)
    return sequences


def read_fasta_iter(fasta_path: Union[str, Path]) -> Iterator[Tuple[str, str]]:
    """Read a FASTA file and yield (sequence_id, sequence) tuples.
    
    Memory-efficient iterator for large FASTA files.
    
    Args:
        fasta_path: Path to FASTA file
        
    Yields:
        Tuples of (sequence_id, sequence)
        
    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
        ValueError: If the FASTA file is malformed
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    current_id = None
    current_seq_parts = []
    
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(">"):
                # Yield previous sequence if exists
                if current_id is not None:
                    yield (current_id, "".join(current_seq_parts))
                
                # Start new sequence
                current_id = line[1:].split()[0]
                current_seq_parts = []
            else:
                if current_id is None:
                    raise ValueError("FASTA sequence found before header")
                current_seq_parts.append(line.upper())
        
        # Yield last sequence
        if current_id is not None:
            yield (current_id, "".join(current_seq_parts))


def write_fasta(sequences: Dict[str, str], output_path: Union[str, Path], line_width: int = 80) -> None:
    """Write sequences to a FASTA file.
    
    Args:
        sequences: Dictionary mapping sequence IDs to sequences
        output_path: Path to output FASTA file
        line_width: Maximum line width for sequence lines (default: 80)
    """
    output_path = Path(output_path)
    logger.info("Writing %d sequence(s) to FASTA file: %s", len(sequences), output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for seq_id, sequence in sequences.items():
            f.write(f">{seq_id}\n")
            # Write sequence in chunks of line_width
            for i in range(0, len(sequence), line_width):
                f.write(sequence[i:i+line_width] + "\n")
            logger.debug("Wrote sequence '%s' (length=%d)", seq_id, len(sequence))
    
    logger.info("Successfully wrote FASTA file: %s", output_path)
