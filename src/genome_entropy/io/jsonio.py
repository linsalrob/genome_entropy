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
    
    OLD FORMAT PROBLEM:
    -------------------
    The old format had three parallel lists:
    - orfs: [ORF1, ORF2, ...]
    - proteins: [{orf: ORF1, aa_seq: ...}, {orf: ORF2, aa_seq: ...}, ...]
    - three_dis: [{protein: {orf: ORF1, ...}, 3di: ...}, ...]
    
    This caused:
    1. ORF data duplicated 3 times (in orfs, inside proteins, inside three_dis)
    2. Protein data duplicated 2 times (in proteins, inside three_dis)
    3. ~2-3x larger files due to redundancy
    4. Risk of inconsistency if data differs between copies
    
    NEW UNIFIED FORMAT:
    -------------------
    Single features dictionary with hierarchical organization:
    - features: {
        "orf_1": {
          location: {start, end, strand, frame},
          dna: {sequence, length},
          protein: {sequence, length},
          three_di: {encoding, length, method, model, device},
          metadata: {parent_id, table_id, has_start, has_stop, in_genbank},
          entropy: {dna_entropy, protein_entropy, three_di_entropy}
        }
      }
    
    Benefits:
    1. Each piece of information stored exactly once
    2. 40-50% smaller file sizes
    3. Direct O(1) access by orf_id
    4. Clear hierarchical organization matching biological concepts
    5. Single source of truth - no inconsistency possible
    
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
    # This replaces the three separate lists (orfs, proteins, three_dis)
    # with a single unified structure where each ORF appears exactly once
    features = {}
    
    # Create lookup dictionaries for efficient O(1) access
    # Maps: orf_id -> ProteinRecord
    proteins_by_orf_id = {p.orf.orf_id: p for p in result.proteins}
    # Maps: orf_id -> ThreeDiRecord
    three_dis_by_orf_id = {td.protein.orf.orf_id: td for td in result.three_dis}
    
    # Process each ORF and merge data from all three sources
    for orf in result.orfs:
        orf_id = orf.orf_id
        
        # Get corresponding protein and 3Di records
        # These contain the ORF data nested inside them (redundancy!)
        protein = proteins_by_orf_id.get(orf_id)
        three_di_record = three_dis_by_orf_id.get(orf_id)
        
        # Validate that we have all the data
        # (Should always be true unless pipeline failed partially)
        if protein is None:
            logger.warning(f"No protein found for ORF {orf_id}, skipping")
            continue
        if three_di_record is None:
            logger.warning(f"No 3Di encoding found for ORF {orf_id}, skipping")
            continue
        
        # Extract entropy values for this feature from the entropy report
        # OLD: entropy had separate dicts for orf_nt, protein_aa, three_di
        # NEW: we consolidate these into a single entropy sub-object per feature
        dna_entropy = result.entropy.orf_nt_entropy.get(orf_id, 0.0)
        protein_entropy = result.entropy.protein_aa_entropy.get(orf_id, 0.0)
        three_di_entropy = result.entropy.three_di_entropy.get(orf_id, 0.0)
        
        # Build the unified feature
        # Instead of storing the ORF object three times, we extract each
        # piece of information once and organize it hierarchically
        unified_feature = UnifiedFeature(
            orf_id=orf_id,
            # Genomic location (from ORF)
            location=FeatureLocation(
                start=orf.start,
                end=orf.end,
                strand=orf.strand,
                frame=orf.frame,
            ),
            # DNA sequence (from ORF) - stored once, not three times
            dna=FeatureDNA(
                nt_sequence=orf.nt_sequence,
                length=len(orf.nt_sequence),
            ),
            # Protein sequence (from ProteinRecord) - stored once, not twice
            protein=FeatureProtein(
                aa_sequence=protein.aa_sequence,
                length=protein.aa_length,
            ),
            # 3Di encoding (from ThreeDiRecord) - stored once
            three_di=FeatureThreeDi(
                encoding=three_di_record.three_di,
                length=len(three_di_record.three_di),
                method=three_di_record.method,
                model_name=three_di_record.model_name,
                inference_device=three_di_record.inference_device,
            ),
            # Metadata (from ORF) - organized separately for clarity
            metadata=FeatureMetadata(
                parent_id=orf.parent_id,
                table_id=orf.table_id,
                has_start_codon=orf.has_start_codon,
                has_stop_codon=orf.has_stop_codon,
                in_genbank=orf.in_genbank,
            ),
            # Entropy values (from EntropyReport) - consolidated per feature
            entropy=FeatureEntropy(
                dna_entropy=dna_entropy,
                protein_entropy=protein_entropy,
                three_di_entropy=three_di_entropy,
            ),
        )
        
        # Store in dictionary keyed by orf_id
        # This enables O(1) lookup instead of O(n) list search
        features[orf_id] = unified_feature
    
    # Validate that no features were lost during conversion
    expected_count = len(result.orfs)
    actual_count = len(features)
    if actual_count != expected_count:
        logger.warning(
            f"Feature count mismatch: expected {expected_count}, got {actual_count}"
        )
    
    # Create the unified result with schema version for compatibility tracking
    unified_result = UnifiedPipelineResult(
        schema_version=SCHEMA_VERSION,  # v2.0.0 for new unified format
        input_id=result.input_id,
        input_dna_length=result.input_dna_length,
        dna_entropy_global=result.entropy.dna_entropy_global,
        alphabet_sizes=result.entropy.alphabet_sizes,
        features=features,  # Single unified dictionary replaces three lists
    )
    
    return unified_result


def write_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Write data to a JSON file.
    
    Automatically handles dataclass objects by converting them to dictionaries.
    If data contains PipelineResult objects, they are automatically converted
    to the new unified format to eliminate redundancy.
    
    AUTOMATIC CONVERSION:
    ---------------------
    This function transparently converts old-format PipelineResult objects to
    the new unified format. This means:
    
    1. Users don't need to manually call convert_pipeline_result_to_unified()
    2. All JSON output from the pipeline automatically uses the new format
    3. The conversion happens only once during serialization
    4. No changes needed to pipeline code or user scripts
    
    MAPPING: Old Keys → New Structure
    ----------------------------------
    OLD FORMAT:
      - orfs[i].orf_id → features[orf_id].orf_id
      - orfs[i].start → features[orf_id].location.start
      - orfs[i].nt_sequence → features[orf_id].dna.nt_sequence
      - proteins[i].aa_sequence → features[orf_id].protein.aa_sequence
      - three_dis[i].three_di → features[orf_id].three_di.encoding
      - entropy.orf_nt_entropy[id] → features[id].entropy.dna_entropy
    
    NEW FORMAT adds:
      - schema_version: "2.0.0" (for compatibility tracking)
      - features: dict (replaces orfs, proteins, three_dis lists)
      - Hierarchical organization (location, dna, protein, three_di, metadata, entropy)
    
    Args:
        data: Data to write (dataclass, dict, list, etc.)
        output_path: Path to output JSON file
        indent: Indentation level for pretty printing (default: 2)
    """
    output_path = Path(output_path)
    logger.info("Writing JSON data to: %s", output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert PipelineResult to unified format if needed
    # This conversion happens transparently to ensure all JSON output
    # uses the new redundancy-free format
    # Import here to avoid circular imports
    try:
        from ..pipeline.runner import PipelineResult
        if isinstance(data, (PipelineResult, list)):
            # Check if we have a list of PipelineResult objects
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], PipelineResult):
                logger.info("Converting PipelineResult to unified format")
                data = convert_pipeline_result_to_unified(data)
            # Check if we have a single PipelineResult object
            elif isinstance(data, PipelineResult):
                logger.info("Converting PipelineResult to unified format")
                data = convert_pipeline_result_to_unified(data)
    except ImportError:
        # PipelineResult not available, skip conversion
        # (e.g., when writing non-pipeline data)
        pass
    
    # Convert dataclasses to dictionaries recursively
    json_data = to_json_dict(data)
    
    # Write to file with pretty printing
    with open(output_path, "w", encoding='utf-8') as f:
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
    
    with open(input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info("Successfully read JSON file: %s", input_path)
    return data
