"""End-to-end pipeline orchestration for DNA to 3Di with entropy calculation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import DEFAULT_GENETIC_CODE_TABLE, DEFAULT_MIN_AA_LENGTH, DEFAULT_PROSTT5_MODEL
from ..encode3di.prostt5 import ProstT5ThreeDiEncoder, ThreeDiRecord
from ..entropy.shannon import EntropyReport, calculate_sequence_entropy, calculate_entropies_for_sequences
from ..errors import PipelineError
from ..io.fasta import read_fasta
from ..io.jsonio import write_json
from ..logging_config import get_logger
from ..orf.finder import find_orfs
from ..orf.types import OrfRecord
from ..translate.translator import ProteinRecord, translate_orfs

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of running the complete DNA to 3Di pipeline.
    
    Attributes:
        input_id: ID of the input DNA sequence
        input_dna_length: Length of the input DNA sequence
        orfs: List of ORFs found in the sequence
        proteins: List of translated proteins
        three_dis: List of 3Di encoded structures
        entropy: Entropy report for all representations
    """

    input_id: str
    input_dna_length: int
    orfs: List[OrfRecord]
    proteins: List[ProteinRecord]
    three_dis: List[ThreeDiRecord]
    entropy: EntropyReport


def run_pipeline(
    input_fasta: Union[str, Path],
    table_id: int = DEFAULT_GENETIC_CODE_TABLE,
    min_aa_len: int = DEFAULT_MIN_AA_LENGTH,
    model_name: str = DEFAULT_PROSTT5_MODEL,
    compute_entropy: bool = True,
    output_json: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    use_multi_gpu: bool = False,
    gpu_ids: Optional[List[int]] = None,
) -> List[PipelineResult]:
    """Run the complete DNA to 3Di pipeline with entropy calculation.
    
    Pipeline steps:
    1. Read FASTA file
    2. Find ORFs in all 6 reading frames
    3. Translate ORFs to proteins
    4. Encode proteins to 3Di structural tokens
    5. Calculate entropy at all levels
    6. Optionally write results to JSON
    
    Args:
        input_fasta: Path to input FASTA file
        table_id: NCBI genetic code table ID
        min_aa_len: Minimum protein length in amino acids
        model_name: ProstT5 model name
        compute_entropy: Whether to compute entropy values
        output_json: Optional path to save results as JSON
        device: Device for 3Di encoding ("cuda", "mps", "cpu", or None for auto)
                Ignored if use_multi_gpu is True.
        use_multi_gpu: If True, use multi-GPU parallel encoding when available
        gpu_ids: Optional list of GPU IDs for multi-GPU encoding.
                If None and use_multi_gpu=True, auto-discover available GPUs.
        
    Returns:
        List of PipelineResult objects (one per input sequence)
        
    Raises:
        PipelineError: If any pipeline step fails
    """
    logger.info("=" * 60)
    logger.info("Starting DNA to 3Di pipeline")
    logger.info("=" * 60)
    logger.info("Input FASTA: %s", input_fasta)
    logger.info("Genetic code table: %d", table_id)
    logger.info("Minimum AA length: %d", min_aa_len)
    logger.info("Model: %s", model_name)
    logger.info("Compute entropy: %s", compute_entropy)
    if use_multi_gpu:
        logger.info("Multi-GPU encoding: enabled")
        if gpu_ids:
            logger.info("GPU IDs: %s", gpu_ids)
        else:
            logger.info("GPU IDs: auto-discover")
    else:
        logger.info("Device: %s", device if device else "auto")
    
    try:
        # Step 1: Read FASTA
        logger.info("Step 1: Reading FASTA file...")
        sequences = read_fasta(input_fasta)
        logger.info("Read %d sequence(s)", len(sequences))
        
        results = []
        
        for seq_idx, (seq_id, dna_sequence) in enumerate(sequences.items(), 1):
            logger.info("-" * 60)
            logger.info("Processing sequence %d/%d: %s (length=%d)", seq_idx, len(sequences), seq_id, len(dna_sequence))
            
            # Step 2: Find ORFs
            logger.info("Step 2: Finding ORFs...")
            min_nt_len = min_aa_len * 3
            orfs = find_orfs(
                {seq_id: dna_sequence},
                table_id=table_id,
                min_nt_length=min_nt_len,
            )
            
            if not orfs:
                logger.warning("No ORFs found for sequence '%s'", seq_id)
                # No ORFs found, create empty result
                empty_entropy = EntropyReport(
                    dna_entropy_global=calculate_sequence_entropy(dna_sequence) if compute_entropy else 0.0,
                    orf_nt_entropy={},
                    protein_aa_entropy={},
                    three_di_entropy={},
                    alphabet_sizes={},
                )
                results.append(PipelineResult(
                    input_id=seq_id,
                    input_dna_length=len(dna_sequence),
                    orfs=[],
                    proteins=[],
                    three_dis=[],
                    entropy=empty_entropy,
                ))
                continue
            
            logger.info("Found %d ORF(s)", len(orfs))
            
            # Step 3: Translate ORFs
            logger.info("Step 3: Translating ORFs to proteins...")
            proteins = translate_orfs(orfs, table_id=table_id)
            logger.info("Translated %d protein(s)", len(proteins))
            
            # Step 4: Encode to 3Di
            logger.info("Step 4: Encoding proteins to 3Di tokens...")
            encoder = ProstT5ThreeDiEncoder(model_name=model_name, device=device)
            three_dis = encoder.encode_proteins(
                proteins,
                use_multi_gpu=use_multi_gpu,
                gpu_ids=gpu_ids,
            )
            logger.info("Encoded %d 3Di sequence(s)", len(three_dis))
            
            # Step 5: Calculate entropy
            if compute_entropy:
                logger.info("Step 5: Calculating entropy at all levels...")
                entropy_report = calculate_pipeline_entropy(
                    dna_sequence, orfs, proteins, three_dis
                )
                logger.info("Calculated entropy values")
            else:
                logger.info("Step 5: Skipping entropy calculation")
                entropy_report = EntropyReport(
                    dna_entropy_global=0.0,
                    orf_nt_entropy={},
                    protein_aa_entropy={},
                    three_di_entropy={},
                    alphabet_sizes={},
                )
            
            # Create result
            result = PipelineResult(
                input_id=seq_id,
                input_dna_length=len(dna_sequence),
                orfs=orfs,
                proteins=proteins,
                three_dis=three_dis,
                entropy=entropy_report,
            )
            results.append(result)
            logger.info("Completed processing for sequence '%s'", seq_id)
        
        # Step 6: Write output if requested
        if output_json:
            logger.info("Step 6: Writing results to JSON...")
            write_json(results, output_json)
        
        logger.info("=" * 60)
        logger.info("Pipeline complete! Processed %d sequence(s)", len(results))
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=True)
        raise PipelineError(f"Pipeline execution failed: {e}")


def calculate_pipeline_entropy(
    dna_sequence: str,
    orfs: List[OrfRecord],
    proteins: List[ProteinRecord],
    three_dis: List[ThreeDiRecord],
) -> EntropyReport:
    """Calculate entropy at all representation levels.
    
    Args:
        dna_sequence: Original DNA sequence
        orfs: List of ORF records
        proteins: List of protein records
        three_dis: List of 3Di records
        
    Returns:
        EntropyReport with entropy values
    """
    logger.debug("Calculating entropy at all representation levels")
    
    # DNA entropy (global)
    dna_entropy = calculate_sequence_entropy(dna_sequence)
    logger.debug("DNA entropy: %.4f bits", dna_entropy)
    
    # ORF nucleotide entropy
    orf_nt_sequences = {orf.orf_id: orf.nt_sequence for orf in orfs}
    orf_nt_entropy = calculate_entropies_for_sequences(orf_nt_sequences)
    
    # Protein amino acid entropy
    protein_aa_sequences = {p.orf.orf_id: p.aa_sequence for p in proteins}
    protein_aa_entropy = calculate_entropies_for_sequences(protein_aa_sequences)
    
    # 3Di entropy
    three_di_sequences = {td.protein.orf.orf_id: td.three_di for td in three_dis}
    three_di_entropy = calculate_entropies_for_sequences(three_di_sequences)
    
    # Alphabet sizes
    alphabet_sizes = {
        "dna": 4,
        "protein": 20,
        "three_di": 20,
    }
    
    logger.debug("Entropy calculation complete")
    
    return EntropyReport(
        dna_entropy_global=dna_entropy,
        orf_nt_entropy=orf_nt_entropy,
        protein_aa_entropy=protein_aa_entropy,
        three_di_entropy=three_di_entropy,
        alphabet_sizes=alphabet_sizes,
    )
