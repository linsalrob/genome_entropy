# JSON Output Format Refactoring Summary

## Problem Statement

The genome_entropy pipeline previously generated JSON output with significant redundancy, where ORF data was duplicated in three separate parallel structures (`orfs`, `proteins`, `three_dis`), leading to:
- 2-3x larger file sizes
- Risk of data inconsistency
- Complex data access patterns

## Solution: Unified Format (v2.0.0)

We implemented a single unified `features` dictionary that stores each piece of biological information exactly once, organized hierarchically.

## Visual Comparison

### OLD FORMAT (Redundant)
```
Size: 2304 bytes (example with 2 ORFs)

{
  "orfs": [
    {
      "orf_id": "orf_1",
      "nt_sequence": "ATGGCTAGC...",  ← DNA sequence #1
      "aa_sequence": "MASSSSSS",      ← Protein sequence #1
      "start": 0, "end": 30,          ← Location #1
      "table_id": 11,                 ← Metadata #1
      ...
    }
  ],
  "proteins": [
    {
      "orf": {                         ← ENTIRE ORF DUPLICATED!
        "orf_id": "orf_1",
        "nt_sequence": "ATGGCTAGC...", ← DNA sequence #2
        "aa_sequence": "MASSSSSS",     ← Protein sequence #2
        "start": 0, "end": 30,         ← Location #2
        "table_id": 11,                ← Metadata #2
        ...
      },
      "aa_sequence": "MASSSSSS",       ← Protein sequence #3
      "aa_length": 8
    }
  ],
  "three_dis": [
    {
      "protein": {                     ← ENTIRE PROTEIN DUPLICATED!
        "orf": {                       ← ENTIRE ORF DUPLICATED AGAIN!
          "orf_id": "orf_1",
          "nt_sequence": "ATGGCTAGC...", ← DNA sequence #3
          "aa_sequence": "MASSSSSS",     ← Protein sequence #4
          "start": 0, "end": 30,         ← Location #3
          "table_id": 11,                ← Metadata #3
          ...
        },
        "aa_sequence": "MASSSSSS",     ← Protein sequence #5
        ...
      },
      "three_di": "AAAAAAAA"
    }
  ]
}

Problem: DNA appears 3x, protein appears 5x, location 3x, metadata 3x!
```

### NEW FORMAT (Unified)
```
Size: 1281 bytes (44.4% reduction!)

{
  "schema_version": "2.0.0",
  "input_id": "test_seq",
  "input_dna_length": 100,
  "dna_entropy_global": 1.85,
  "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20},
  
  "features": {
    "orf_1": {
      "orf_id": "orf_1",
      
      "location": {              ← Location: 1 time only
        "start": 0,
        "end": 30,
        "strand": "+",
        "frame": 0
      },
      
      "dna": {                   ← DNA: 1 time only
        "nt_sequence": "ATGGCTAGC...",
        "length": 28
      },
      
      "protein": {               ← Protein: 1 time only
        "aa_sequence": "MASSSSSS",
        "length": 8
      },
      
      "three_di": {              ← 3Di: 1 time only
        "encoding": "AAAAAAAA",
        "length": 8,
        "method": "prostt5_aa2fold",
        "model_name": "test",
        "inference_device": "cpu"
      },
      
      "metadata": {              ← Metadata: 1 time only
        "parent_id": "test_seq",
        "table_id": 11,
        "has_start_codon": true,
        "has_stop_codon": false,
        "in_genbank": false
      },
      
      "entropy": {               ← Entropy: consolidated per feature
        "dna_entropy": 1.2,
        "protein_entropy": 0.8,
        "three_di_entropy": 0.0
      }
    }
  }
}

Solution: Each piece of data appears exactly once!
```

## Implementation Details

### Data Structures

**New Types in `pipeline/types.py`:**
- `UnifiedPipelineResult` - Main container with schema_version
- `UnifiedFeature` - Single feature with hierarchical organization
- `FeatureLocation`, `FeatureDNA`, `FeatureProtein`, `FeatureThreeDi` - Logical sub-structures
- `FeatureMetadata`, `FeatureEntropy` - Additional context

### Conversion Logic

**In `io/jsonio.py`:**
```python
def convert_pipeline_result_to_unified(pipeline_result):
    """Converts old format to new unified format.
    
    Process:
    1. Create lookup dictionaries for proteins and three_dis by orf_id
    2. For each ORF:
       - Extract data from ORF, protein, and three_di records
       - Build single UnifiedFeature with hierarchical organization
       - Store in features dict keyed by orf_id
    3. Validate no features lost
    4. Return UnifiedPipelineResult with schema_version
    """
```

**Automatic Conversion in `write_json()`:**
- Detects PipelineResult objects
- Automatically converts to unified format
- Transparent to users - no code changes needed

### Testing

**Test Suite (`test_unified_output.py`):**
- ✓ Structure validation
- ✓ Conversion correctness
- ✓ Multiple features handling
- ✓ No data loss verification
- ✓ JSON serialization correctness
- ✓ List of results handling

**Results:**
- 6 new tests, all passing
- 114 total tests passing
- No regressions in existing functionality

## Benefits

| Aspect | Old Format | New Format | Improvement |
|--------|-----------|-----------|-------------|
| **File Size** | 2304 bytes | 1281 bytes | **44.4% reduction** |
| **Redundancy** | 3x ORF, 2x protein | 1x each | **No redundancy** |
| **Access Pattern** | O(n) list search | O(1) dict lookup | **Faster access** |
| **Data Integrity** | 3 copies (risk) | 1 copy (safe) | **Single source of truth** |
| **Structure** | Flat parallel lists | Hierarchical | **Clearer organization** |
| **Compatibility** | No version | schema_version | **Version tracking** |

## Usage

### Writing (Automatic)
```python
from genome_entropy.pipeline.runner import run_pipeline

# Run pipeline - output automatically uses new format
results = run_pipeline(
    input_fasta="sequences.fasta",
    output_json="results.json"  # Will be in unified format v2.0.0
)
```

### Reading
```python
import json

with open("results.json") as f:
    data = json.load(f)

# Check version
if data[0].get("schema_version", "1.0").startswith("2."):
    # New unified format
    for result in data:
        for orf_id, feature in result["features"].items():
            print(f"ORF {orf_id}:")
            print(f"  DNA: {feature['dna']['nt_sequence'][:20]}...")
            print(f"  Protein: {feature['protein']['aa_sequence'][:20]}...")
            print(f"  3Di: {feature['three_di']['encoding'][:20]}...")
```

## Migration

**For Existing Code:**
No changes needed! The conversion happens automatically when writing JSON.

**For Parsing Code:**
Update to access the new `features` dictionary structure. See `docs/unified_output_format.md` for examples.

**Backward Compatibility:**
The `schema_version` field allows tools to detect format version and handle appropriately.

## Files Changed

1. **src/genome_entropy/pipeline/types.py** (NEW)
   - Unified data structure definitions
   - 9 new dataclasses with clear hierarchical organization

2. **src/genome_entropy/io/jsonio.py** (MODIFIED)
   - Added SCHEMA_VERSION constant
   - Implemented convert_pipeline_result_to_unified()
   - Modified write_json() for automatic conversion
   - Extensive inline documentation

3. **src/genome_entropy/pipeline/__init__.py** (MODIFIED)
   - Export new unified types

4. **tests/test_unified_output.py** (NEW)
   - 6 comprehensive tests
   - Validates structure, conversion, and data preservation

5. **docs/unified_output_format.md** (NEW)
   - Complete format reference
   - Migration guide
   - Usage examples

## Validation

✅ All requirements from problem statement met:
1. ✅ Single canonical dictionary (`features`)
2. ✅ Stable unique identifier (`orf_id` as key)
3. ✅ All three sources merged
4. ✅ Hierarchical structure
5. ✅ Top-level context preserved
6. ✅ Referential integrity maintained
7. ✅ Backward compatibility with schema_version
8. ✅ Refactored serialization code
9. ✅ Validation and assertions

✅ Testing results:
- 114 tests passing (no regressions)
- 44.4% size reduction verified
- Zero data loss confirmed
- All fields preserved

## Summary

The refactoring successfully eliminates redundancy while preserving all information, reducing file sizes by ~44%, and providing clearer hierarchical organization. The conversion happens automatically, ensuring a smooth transition with full backward compatibility tracking via schema versioning.
