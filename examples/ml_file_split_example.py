#!/usr/bin/env python3
"""
Example: Using file-based train/test splitting for ML classification

This example demonstrates how to use the --split-dir option to:
1. Randomly split JSON files 80/20 into training and test sets
2. Train a classifier on the training files
3. Evaluate on the test files
4. Generate a detailed JSON report with per-prediction results

This is useful when you want to evaluate how well the model generalizes
to completely new sequences (files), rather than just new ORFs from
sequences it has already seen.
"""

import json
import tempfile
from pathlib import Path
import subprocess
import sys


def create_sample_data(output_dir: Path, n_files: int = 10) -> None:
    """Create sample JSON files for demonstration."""
    print(f"Creating {n_files} sample JSON files in {output_dir}...")

    # Sample data in unified format
    sample_data = {
        "schema_version": "2.0.0",
        "input_id": "sample_seq",
        "input_dna_length": 1000,
        "dna_entropy_global": 1.8,
        "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20},
        "features": {
            "orf_1": {
                "orf_id": "orf_1",
                "location": {"start": 0, "end": 300, "strand": "+", "frame": 0},
                "dna": {"nt_sequence": "ATG" * 100, "length": 300},
                "protein": {"aa_sequence": "M" * 100, "length": 100},
                "three_di": {
                    "encoding": "A" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu",
                },
                "metadata": {
                    "parent_id": "sample_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": True,
                    "in_genbank": True,
                },
                "entropy": {
                    "dna_entropy": 1.2,
                    "protein_entropy": 0.5,
                    "three_di_entropy": 0.3,
                },
            },
            "orf_2": {
                "orf_id": "orf_2",
                "location": {"start": 400, "end": 700, "strand": "-", "frame": 1},
                "dna": {"nt_sequence": "ATG" * 100, "length": 300},
                "protein": {"aa_sequence": "K" * 100, "length": 100},
                "three_di": {
                    "encoding": "B" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu",
                },
                "metadata": {
                    "parent_id": "sample_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": False,
                    "in_genbank": False,
                },
                "entropy": {
                    "dna_entropy": 1.5,
                    "protein_entropy": 0.8,
                    "three_di_entropy": 0.6,
                },
            },
        },
    }

    # Create multiple files with varying data
    for i in range(n_files):
        data = sample_data.copy()
        data["input_id"] = f"sample_seq_{i}"

        json_file = output_dir / f"sample_{i:02d}.json"
        with open(json_file, "w") as f:
            json.dump([data], f, indent=2)

    print(f"✓ Created {n_files} JSON files")


def run_training(json_dir: Path, output_model: Path, json_output: Path) -> None:
    """Run the file-based training."""
    print("\n" + "=" * 60)
    print("Running file-based train/test split...")
    print("=" * 60)

    cmd = [
        "genome_entropy",
        "ml",
        "train",
        "--split-dir",
        str(json_dir),
        "--output",
        str(output_model),
        "--json-output",
        str(json_output),
        "--random-seed",
        "42",
        "--model-type",
        "xgboost",
        "--device",
        "cpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: Training failed!")
        print(result.stderr)
        sys.exit(1)

    # Print last 20 lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(line)


def analyze_results(json_output: Path) -> None:
    """Analyze the detailed JSON results."""
    print("\n" + "=" * 60)
    print("Analyzing Results")
    print("=" * 60)

    with open(json_output) as f:
        results = json.load(f)

    print(f"\n📁 Training Files ({len(results['training_files'])}):")
    for f in results["training_files"]:
        print(f"  • {f}")

    print(f"\n📁 Test Files ({len(results['test_files'])}):")
    for f in results["test_files"]:
        print(f"  • {f}")

    print("\n📊 Training Metrics:")
    for key, value in results["training_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n📊 Test Metrics:")
    for key, value in results["test_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n🎯 Test Predictions: {len(results['test_predictions'])} predictions")

    # Show some example predictions
    print("\nSample Predictions:")
    for pred in results["test_predictions"][:3]:
        correct_icon = "✓" if pred["correct"] else "✗"
        print(
            f"  {correct_icon} {pred['orf_id']}: "
            f"predicted={pred['predicted_label']}, "
            f"actual={pred['actual_label']}, "
            f"prob={pred['probability_in_genbank']:.3f}"
        )

    if len(results["test_predictions"]) > 3:
        print(f"  ... and {len(results['test_predictions']) - 3} more")

    # Calculate accuracy from predictions
    correct = sum(1 for p in results["test_predictions"] if p["correct"])
    total = len(results["test_predictions"])
    accuracy = correct / total if total > 0 else 0

    print(f"\n✓ Verified accuracy: {correct}/{total} = {accuracy:.4f}")

    print("\n🔍 Feature Importance (Top 5):")
    if "feature_importance" in results:
        sorted_features = sorted(
            results["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )
        for feat, importance in sorted_features[:5]:
            print(f"  {feat:20s}: {importance:.4f}")


def main():
    """Run the complete example."""
    print("=" * 60)
    print("File-Based Train/Test Split Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample data
        json_dir = tmpdir / "json_files"
        json_dir.mkdir()
        create_sample_data(json_dir, n_files=10)

        # Set output paths
        model_file = tmpdir / "model.ubj"
        results_file = tmpdir / "results.json"

        # Run training
        run_training(json_dir, model_file, results_file)

        # Analyze results
        analyze_results(results_file)

        print("\n" + "=" * 60)
        print("Example Complete!")
        print("=" * 60)
        print(f"\nFiles created in: {tmpdir}")
        print(f"  • Model: {model_file.name}")
        print(f"  • Results: {results_file.name}")
        print(
            "\nNote: Files are in a temporary directory and will be deleted "
            "when this script exits."
        )


if __name__ == "__main__":
    main()
