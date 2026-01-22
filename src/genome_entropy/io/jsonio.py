"""JSON serialization for data models."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from ..logging_config import get_logger

logger = get_logger(__name__)

# Schema version for tracking output format changes
SCHEMA_VERSION = "2.0.0"


def to_json_dict(obj: Any) -> Any:
    """Convert a dataclass object to a JSON-serializable dictionary.
    
    Recursively handles nested dataclasses, lists, and dictionaries.
    
    Args:
        obj: Object to convert (typically a dataclass instance)
        
    Returns:
        JSON-serializable dictionary
    """
    if is_dataclass(obj):
        return {k: to_json_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [to_json_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    else:
        return obj


def convert_pipeline_result_to_unified(pipeline_result):
    """Convert PipelineResult to UnifiedPipelineResult format.
    
    This function transforms the old redundant format (separate orfs, proteins, 
    three_dis lists) into the new unified format where each feature appears 
    exactly once with all its related data organized hierarchically.
    
    Args:
        pipeline_result: PipelineResult object or list of PipelineResult objects
        
    Returns:
        UnifiedPipelineResult object or list of UnifiedPipelineResult objects
    """
    # Import here to avoid circular imports
    from ..pipeline.types import (
        UnifiedPipelineResult,
        UnifiedFeature,
        FeatureLocation,
        FeatureDNA,
        FeatureProtein,
        FeatureThreeDi,
        FeatureMetadata,
        FeatureEntropy,
    )
    
    # Handle both single result and list of results
    if isinstance(pipeline_result, list):
        return [convert_pipeline_result_to_unified(r) for r in pipeline_result]
    
    # Extract the PipelineResult object
    result = pipeline_result
    
    # Build a dictionary of features by orf_id
    features = {}
    
    # Create lookup dictionaries for efficient access
    proteins_by_orf_id = {p.orf.orf_id: p for p in result.proteins}
    three_dis_by_orf_id = {td.protein.orf.orf_id: td for td in result.three_dis}
    
    # Process each ORF
    for orf in result.orfs:
        orf_id = orf.orf_id
        
        # Get corresponding protein and 3Di records
        protein = proteins_by_orf_id.get(orf_id)
        three_di_record = three_dis_by_orf_id.get(orf_id)
        
        # Validate that we have all the data
        if protein is None:
            logger.warning(f"No protein found for ORF {orf_id}, skipping")
            continue
        if three_di_record is None:
            logger.warning(f"No 3Di encoding found for ORF {orf_id}, skipping")
            continue
        
        # Extract entropy values for this feature
        dna_entropy = result.entropy.orf_nt_entropy.get(orf_id, 0.0)
        protein_entropy = result.entropy.protein_aa_entropy.get(orf_id, 0.0)
        three_di_entropy = result.entropy.three_di_entropy.get(orf_id, 0.0)
        
        # Build the unified feature
        unified_feature = UnifiedFeature(
            orf_id=orf_id,
            location=FeatureLocation(
                start=orf.start,
                end=orf.end,
                strand=orf.strand,
                frame=orf.frame,
            ),
            dna=FeatureDNA(
                nt_sequence=orf.nt_sequence,
                length=len(orf.nt_sequence),
            ),
            protein=FeatureProtein(
                aa_sequence=protein.aa_sequence,
                length=protein.aa_length,
            ),
            three_di=FeatureThreeDi(
                encoding=three_di_record.three_di,
                length=len(three_di_record.three_di),
                method=three_di_record.method,
                model_name=three_di_record.model_name,
                inference_device=three_di_record.inference_device,
            ),
            metadata=FeatureMetadata(
                parent_id=orf.parent_id,
                table_id=orf.table_id,
                has_start_codon=orf.has_start_codon,
                has_stop_codon=orf.has_stop_codon,
                in_genbank=orf.in_genbank,
            ),
            entropy=FeatureEntropy(
                dna_entropy=dna_entropy,
                protein_entropy=protein_entropy,
                three_di_entropy=three_di_entropy,
            ),
        )
        
        features[orf_id] = unified_feature
    
    # Validate that no features were lost
    expected_count = len(result.orfs)
    actual_count = len(features)
    if actual_count != expected_count:
        logger.warning(
            f"Feature count mismatch: expected {expected_count}, got {actual_count}"
        )
    
    # Create the unified result
    unified_result = UnifiedPipelineResult(
        schema_version=SCHEMA_VERSION,
        input_id=result.input_id,
        input_dna_length=result.input_dna_length,
        dna_entropy_global=result.entropy.dna_entropy_global,
        alphabet_sizes=result.entropy.alphabet_sizes,
        features=features,
    )
    
    return unified_result


def write_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Write data to a JSON file.
    
    Automatically handles dataclass objects by converting them to dictionaries.
    If data contains PipelineResult objects, they are automatically converted
    to the new unified format to eliminate redundancy.
    
    Args:
        data: Data to write (dataclass, dict, list, etc.)
        output_path: Path to output JSON file
        indent: Indentation level for pretty printing (default: 2)
    """
    output_path = Path(output_path)
    logger.info("Writing JSON data to: %s", output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert PipelineResult to unified format if needed
    # Import here to avoid circular imports
    try:
        from ..pipeline.runner import PipelineResult
        if isinstance(data, (PipelineResult, list)):
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], PipelineResult):
                logger.info("Converting PipelineResult to unified format")
                data = convert_pipeline_result_to_unified(data)
            elif isinstance(data, PipelineResult):
                logger.info("Converting PipelineResult to unified format")
                data = convert_pipeline_result_to_unified(data)
    except ImportError:
        # PipelineResult not available, skip conversion
        pass
    
    json_data = to_json_dict(data)
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=indent)
    
    logger.info("Successfully wrote JSON file: %s", output_path)


def read_json(input_path: Union[str, Path]) -> Any:
    """Read JSON data from a file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Parsed JSON data (dict, list, etc.)
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    input_path = Path(input_path)
    logger.info("Reading JSON file: %s", input_path)
    
    if not input_path.exists():
        logger.error("JSON file not found: %s", input_path)
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    logger.info("Successfully read JSON file: %s", input_path)
    return data
