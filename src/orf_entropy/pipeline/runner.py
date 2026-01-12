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
from ..orf.finder import find_orfs
from ..orf.types import OrfRecord
from ..translate.translator import ProteinRecord, translate_orfs


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
        
    Returns:
        List of PipelineResult objects (one per input sequence)
        
    Raises:
        PipelineError: If any pipeline step fails
    """
    try:
        # Step 1: Read FASTA
        sequences = read_fasta(input_fasta)
        
        results = []
        
        for seq_id, dna_sequence in sequences.items():
            # Step 2: Find ORFs
            min_nt_len = min_aa_len * 3
            orfs = find_orfs(
                {seq_id: dna_sequence},
                table_id=table_id,
                min_nt_length=min_nt_len,
            )
            
            if not orfs:
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
            
            # Step 3: Translate ORFs
            proteins = translate_orfs(orfs, table_id=table_id)
            
            # Step 4: Encode to 3Di
            encoder = ProstT5ThreeDiEncoder(model_name=model_name, device=device)
            three_dis = encoder.encode_proteins(proteins)
            
            # Step 5: Calculate entropy
            if compute_entropy:
                entropy_report = calculate_pipeline_entropy(
                    dna_sequence, orfs, proteins, three_dis
                )
            else:
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
        
        # Step 6: Write output if requested
        if output_json:
            write_json(results, output_json)
        
        return results
        
    except Exception as e:
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
    # DNA entropy (global)
    dna_entropy = calculate_sequence_entropy(dna_sequence)
    
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
    
    return EntropyReport(
        dna_entropy_global=dna_entropy,
        orf_nt_entropy=orf_nt_entropy,
        protein_aa_entropy=protein_aa_entropy,
        three_di_entropy=three_di_entropy,
        alphabet_sizes=alphabet_sizes,
    )
