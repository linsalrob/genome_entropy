#!/bin/bash
# Example end-to-end pipeline execution

set -e  # Exit on error

echo "=== DNA to structural-state entropy pipeline example ==="
echo ""

# Check if genome_entropy is installed
if ! command -v genome_entropy &> /dev/null; then
    echo "Error: genome_entropy not found. Install with: pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p output

echo "Step 1: Running complete pipeline..."
genome_entropy run \
    --input example_data/JQ995537.fna \
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
echo "genome_entropy orf --input example_data/JQ995537.fna --output output/orfs.json"
echo ""
echo "# Translate to proteins"
echo "genome_entropy translate --input output/orfs.json --output output/proteins.json"
echo ""
echo "# Encode to 3Di and, with multitask ModernProst, 12-state"
echo "genome_entropy encode3di --input output/proteins.json --output output/3di.json"
echo ""
echo "# Calculate entropy"
echo "genome_entropy entropy --input output/3di.json --output output/entropy.json"
