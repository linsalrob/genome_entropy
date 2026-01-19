#!/bin/bash
# Example end-to-end pipeline execution

set -e  # Exit on error

echo "=== DNA to 3Di Pipeline Example ==="
echo ""

# Check if dna23di is installed
if ! command -v dna23di &> /dev/null; then
    echo "Error: dna23di not found. Install with: pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p output

echo "Step 1: Running complete pipeline..."
dna23di run \
    --input examples/example_small.fasta \
    --output output/results.json \
    --table 11 \
    --min-aa 10

echo ""
echo "✓ Pipeline complete!"
echo "  Results saved to: output/results.json"
echo ""
echo "You can also run individual steps:"
echo ""
echo "# Extract ORFs"
echo "dna23di orf --input examples/example_small.fasta --output output/orfs.json"
echo ""
echo "# Translate to proteins"
echo "dna23di translate --input output/orfs.json --output output/proteins.json"
echo ""
echo "# Encode to 3Di"
echo "dna23di encode3di --input output/proteins.json --output output/3di.json"
echo ""
echo "# Calculate entropy"
echo "dna23di entropy --input output/3di.json --output output/entropy.json"
