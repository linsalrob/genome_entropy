# Unified JSON Output Format (v2.0.0)

## Overview

Starting with schema version 2.0.0, the genome_entropy pipeline uses a **unified output format** that eliminates redundancy by storing each piece of biological information exactly once.

## Problem with Old Format

The previous output format (v1.x) contained significant redundancy:

```json
{
  "orfs": [
    {"orf_id": "orf_1", "nt_sequence": "ATG...", "aa_sequence": "M...", ...}
  ],
  "proteins": [
    {
      "orf": {"orf_id": "orf_1", "nt_sequence": "ATG...", "aa_sequence": "M...", ...},
      "aa_sequence": "M...",
      "aa_length": 100
    }
  ],
  "three_dis": [
    {
      "protein": {
        "orf": {"orf_id": "orf_1", "nt_sequence": "ATG...", ...},
        "aa_sequence": "M...",
        ...
      },
      "three_di": "AAA...",
      ...
    }
  ]
}
```

**Issues:**
- ORF data appears 3 times (in `orfs`, `proteins`, and `three_dis`)
- Protein data appears 2 times (in `proteins` and `three_dis`)
- Sequences, coordinates, and metadata are duplicated
- Result: ~2-3x larger file sizes, risk of inconsistencies

## New Unified Format (v2.0.0)

The new format uses a single `features` dictionary where each ORF appears exactly once with all its related data organized hierarchically:

```json
{
  "schema_version": "2.0.0",
  "input_id": "sequence_name",
  "input_dna_length": 1000,
  "dna_entropy_global": 1.85,
  "alphabet_sizes": {
    "dna": 4,
    "protein": 20,
    "three_di": 20
  },
  "features": {
    "orf_1": {
      "orf_id": "orf_1",
      "location": {
        "start": 0,
        "end": 300,
        "strand": "+",
        "frame": 0
      },
      "dna": {
        "nt_sequence": "ATGGCTAGC...",
        "length": 300
      },
      "protein": {
        "aa_sequence": "MASSSSSS...",
        "length": 100
      },
      "three_di": {
        "encoding": "AAAAAAAA...",
        "length": 100,
        "method": "prostt5_aa2fold",
        "model_name": "Rostlab/ProstT5",
        "inference_device": "cpu"
      },
      "metadata": {
        "parent_id": "sequence_name",
        "table_id": 11,
        "has_start_codon": true,
        "has_stop_codon": true,
        "in_genbank": false
      },
      "entropy": {
        "dna_entropy": 1.2,
        "protein_entropy": 0.8,
        "three_di_entropy": 0.0
      }
    }
  }
}
```

## Benefits

1. **No Redundancy**: Each sequence, coordinate, and metadata value appears exactly once
2. **Smaller Files**: Typically 40-50% reduction in file size
3. **Clear Organization**: Hierarchical structure groups related data logically
4. **Easier Access**: Direct lookup by `orf_id` instead of searching through lists
5. **Consistency**: Single source of truth for each piece of information

## Structure Reference

### Top Level

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Format version (e.g., "2.0.0") |
| `input_id` | string | ID of the input DNA sequence |
| `input_dna_length` | integer | Length of input DNA in nucleotides |
| `dna_entropy_global` | float | Shannon entropy of entire DNA sequence |
| `alphabet_sizes` | object | Size of each alphabet (dna: 4, protein: 20, three_di: 20) |
| `features` | object | Dictionary of features keyed by `orf_id` |

### Feature Structure

Each feature (ORF) contains:

#### `location`
Genomic coordinates:
- `start`: 0-based start position (inclusive)
- `end`: 0-based end position (exclusive)
- `strand`: "+" or "-"
- `frame`: 0, 1, 2, or 3

#### `dna`
DNA sequence information:
- `nt_sequence`: Nucleotide sequence
- `length`: Length in nucleotides

#### `protein`
Protein sequence information:
- `aa_sequence`: Amino acid sequence
- `length`: Length in amino acids

#### `three_di`
3Di structural encoding:
- `encoding`: 3Di token sequence
- `length`: Length of 3Di sequence
- `method`: Encoding method (e.g., "prostt5_aa2fold")
- `model_name`: Model used for encoding
- `inference_device`: Device used (cpu/cuda/mps)

#### `metadata`
Additional information:
- `parent_id`: ID of parent DNA sequence
- `table_id`: NCBI genetic code table ID
- `has_start_codon`: Boolean
- `has_stop_codon`: Boolean
- `in_genbank`: Boolean (matches GenBank CDS annotation)

#### `entropy`
Entropy values:
- `dna_entropy`: Shannon entropy of DNA sequence
- `protein_entropy`: Shannon entropy of protein sequence
- `three_di_entropy`: Shannon entropy of 3Di encoding

## Migration from Old Format

The conversion happens automatically when using `genome_entropy` v0.1.3+. 

If you have old JSON files and need to convert them:

```python
from genome_entropy.io.jsonio import read_json, write_json, convert_pipeline_result_to_unified

# Read old format
old_data = read_json("old_format.json")

# Convert (if PipelineResult objects are present, conversion happens automatically)
write_json(old_data, "new_format.json")
```

## Backward Compatibility

The `schema_version` field allows tools to detect and handle different format versions:

```python
import json

with open("results.json") as f:
    data = json.load(f)

for result in data:
    if result.get("schema_version", "1.0.0").startswith("2."):
        # New unified format
        features = result["features"]
    else:
        # Old format (convert if needed)
        features = convert_old_format(result)
```

## Design Rationale

### Why a Dictionary Instead of a List?

The `features` dictionary uses `orf_id` as keys because:

1. **Fast lookup**: O(1) access to any feature
2. **Natural grouping**: All data for one ORF is together
3. **Prevents duplication**: Each ORF appears exactly once
4. **Clear relationships**: Entropy values use same keys

### Why Hierarchical Sections?

The feature is organized into logical sections (`location`, `dna`, `protein`, etc.) to:

1. **Group related fields**: All genomic coordinates are together
2. **Match biological concepts**: DNA → protein → 3Di transformation is explicit
3. **Improve readability**: Structure mirrors the analysis workflow
4. **Enable selective access**: Can extract just protein data without parsing other fields

### Why Include Length Fields?

Explicit `length` fields are included alongside sequences because:

1. **Performance**: No need to compute `len()` when just checking sizes
2. **Validation**: Easy to verify consistency
3. **Clarity**: Makes the data self-documenting

## Example Usage

### Accessing Features

```python
import json

with open("results.json") as f:
    data = json.load(f)

result = data[0]  # First sequence result

# Get all ORF IDs
orf_ids = list(result["features"].keys())

# Access a specific feature
feature = result["features"]["orf_1"]

# Get sequences
dna_seq = feature["dna"]["nt_sequence"]
protein_seq = feature["protein"]["aa_sequence"]
three_di_seq = feature["three_di"]["encoding"]

# Get location
start = feature["location"]["start"]
end = feature["location"]["end"]
strand = feature["location"]["strand"]

# Get entropy values
dna_entropy = feature["entropy"]["dna_entropy"]
protein_entropy = feature["entropy"]["protein_entropy"]
```

### Filtering Features

```python
# Find ORFs on positive strand
positive_strand_features = {
    orf_id: feature
    for orf_id, feature in result["features"].items()
    if feature["location"]["strand"] == "+"
}

# Find ORFs with stop codons
with_stop = {
    orf_id: feature
    for orf_id, feature in result["features"].items()
    if feature["metadata"]["has_stop_codon"]
}

# Find ORFs matching GenBank annotations
genbank_matches = {
    orf_id: feature
    for orf_id, feature in result["features"].items()
    if feature["metadata"]["in_genbank"]
}
```

### Computing Statistics

```python
# Average protein length
avg_protein_length = sum(
    f["protein"]["length"] for f in result["features"].values()
) / len(result["features"])

# Average entropy by representation level
avg_dna_entropy = sum(
    f["entropy"]["dna_entropy"] for f in result["features"].values()
) / len(result["features"])

avg_protein_entropy = sum(
    f["entropy"]["protein_entropy"] for f in result["features"].values()
) / len(result["features"])
```

## Summary

The unified format (v2.0.0) eliminates redundancy while preserving all information and providing a clearer, more efficient representation of pipeline results. All conversions happen automatically, ensuring a smooth transition from the old format.
