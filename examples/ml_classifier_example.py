#!/usr/bin/env python3
"""Example: Train and use ML classifier for GenBank annotation prediction.

This example demonstrates the complete workflow:
1. Generate synthetic training data (in real use, run genome_entropy pipeline on GenBank files)
2. Train an XGBoost classifier
3. Evaluate the model
4. Make predictions
5. Analyze feature importance
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from genome_entropy.ml import GenbankClassifier, load_json_data, extract_features


def create_synthetic_json_data(output_dir: Path, n_files: int = 5):
    """Create synthetic JSON data for demonstration.
    
    In real usage, you would generate these files by running:
        genome_entropy run --input genome.gbk --output results/genome.json
    """
    print(f"\n{'='*60}")
    print("Step 1: Creating Synthetic Training Data")
    print(f"{'='*60}")
    print(f"Creating {n_files} synthetic JSON files in {output_dir}")
    print("(In real usage, these would come from genome_entropy pipeline)\n")
    
    for file_idx in range(n_files):
        # Create data with realistic properties
        n_orfs = np.random.randint(10, 30)
        features = {}
        
        for orf_idx in range(n_orfs):
            orf_id = f"orf_{file_idx}_{orf_idx}"
            
            # Simulate realistic feature distributions
            # Real genes tend to have:
            # - Higher entropy (more complex)
            # - Longer length
            # - Start and stop codons
            is_real_gene = np.random.random() < 0.3  # 30% are real genes
            
            if is_real_gene:
                dna_entropy = np.random.uniform(1.5, 2.0)
                protein_entropy = np.random.uniform(2.5, 3.5)
                three_di_entropy = np.random.uniform(2.0, 3.0)
                length = np.random.randint(300, 2000)
                has_start = True
                has_stop = True
            else:
                dna_entropy = np.random.uniform(0.5, 1.8)
                protein_entropy = np.random.uniform(1.0, 3.0)
                three_di_entropy = np.random.uniform(1.0, 2.5)
                length = np.random.randint(90, 600)
                has_start = np.random.random() < 0.5
                has_stop = np.random.random() < 0.5
            
            features[orf_id] = {
                "orf_id": orf_id,
                "location": {
                    "start": orf_idx * 100,
                    "end": orf_idx * 100 + length,
                    "strand": "+" if np.random.random() < 0.5 else "-",
                    "frame": int(np.random.randint(0, 3))
                },
                "dna": {
                    "nt_sequence": "ATG" * (length // 3),
                    "length": length
                },
                "protein": {
                    "aa_sequence": "M" * (length // 3),
                    "length": length // 3
                },
                "three_di": {
                    "encoding": "A" * (length // 3),
                    "length": length // 3,
                    "method": "prostt5_aa2fold",
                    "model_name": "Rostlab/ProstT5",
                    "inference_device": "cpu"
                },
                "metadata": {
                    "parent_id": f"seq_{file_idx}",
                    "table_id": 11,
                    "has_start_codon": has_start,
                    "has_stop_codon": has_stop,
                    "in_genbank": is_real_gene
                },
                "entropy": {
                    "dna_entropy": float(dna_entropy),
                    "protein_entropy": float(protein_entropy),
                    "three_di_entropy": float(three_di_entropy)
                }
            }
        
        # Save as JSON
        data = {
            "schema_version": "2.0.0",
            "input_id": f"seq_{file_idx}",
            "input_dna_length": 10000,
            "dna_entropy_global": 1.8,
            "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20},
            "features": features
        }
        
        json_file = output_dir / f"synthetic_{file_idx}.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"✓ Created {n_files} JSON files")


def main():
    """Run the complete ML classifier example."""
    print("\n" + "="*60)
    print("ML Classifier for GenBank Annotation Prediction - Example")
    print("="*60)
    
    # Create temporary directory for demo data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        json_dir = tmpdir / "json_data"
        json_dir.mkdir()
        
        # Step 1: Create synthetic data
        create_synthetic_json_data(json_dir, n_files=10)
        
        # Step 2: Load and extract features
        print(f"\n{'='*60}")
        print("Step 2: Loading Data and Extracting Features")
        print(f"{'='*60}")
        
        json_data = load_json_data(json_dir)
        X, y, feature_names, _ = extract_features(json_data)
        
        print(f"✓ Loaded {len(json_data)} JSON files")
        print(f"✓ Extracted {len(X)} ORF samples")
        print(f"✓ Features: {len(feature_names)} total")
        print(f"  - {', '.join(feature_names[:6])}")
        print(f"  - {', '.join(feature_names[6:])}")
        print(f"\nLabel distribution:")
        print(f"  - In GenBank (positive): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
        print(f"  - Not in GenBank (negative): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        
        # Step 3: Train classifier
        print(f"\n{'='*60}")
        print("Step 3: Training XGBoost Classifier")
        print(f"{'='*60}")
        
        classifier = GenbankClassifier(model_type="xgboost", device="cpu")
        
        # Split data
        n_test = int(len(X) * 0.15)
        indices = np.random.permutation(len(X))
        X_train = X[indices[n_test:]]
        y_train = y[indices[n_test:]]
        X_test = X[indices[:n_test]]
        y_test = y[indices[:n_test]]
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        train_metrics = classifier.fit(
            X_train,
            y_train,
            feature_names=feature_names,
            validation_split=0.2
        )
        
        print(f"\nTraining Results:")
        print(f"  - Validation Accuracy: {train_metrics['val_accuracy']:.4f}")
        print(f"  - Validation AUC: {train_metrics.get('val_auc', 0.0):.4f}")
        
        # Step 4: Evaluate on test set
        print(f"\n{'='*60}")
        print("Step 4: Evaluating on Test Set")
        print(f"{'='*60}")
        
        test_metrics = classifier.evaluate(X_test, y_test)
        
        print(f"Test Set Performance:")
        print(f"  - Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  - Precision: {test_metrics['precision']:.4f}")
        print(f"  - Recall:    {test_metrics['recall']:.4f}")
        print(f"  - F1 Score:  {test_metrics['f1']:.4f}")
        if 'auc' in test_metrics:
            print(f"  - AUC:       {test_metrics['auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  - True Positives:  {test_metrics['true_positives']}")
        print(f"  - True Negatives:  {test_metrics['true_negatives']}")
        print(f"  - False Positives: {test_metrics['false_positives']}")
        print(f"  - False Negatives: {test_metrics['false_negatives']}")
        
        # Step 5: Feature importance
        print(f"\n{'='*60}")
        print("Step 5: Analyzing Feature Importance")
        print(f"{'='*60}")
        
        importance = classifier.get_feature_importance()
        if importance:
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("Top 10 Most Important Features:")
            for i, (feat_name, score) in enumerate(sorted_features[:10], 1):
                bar_length = int(score * 50)
                bar = "█" * bar_length
                print(f"{i:2d}. {feat_name:20s}: {score:6.4f} {bar}")
            
            print("\nInterpretation:")
            top_feature = sorted_features[0][0]
            if "entropy" in top_feature:
                print("→ Entropy features are most predictive, suggesting sequence")
                print("  complexity is key to distinguishing real genes from noise.")
            elif "length" in top_feature:
                print("→ Length features dominate, suggesting real genes have")
                print("  characteristic length distributions.")
            elif "codon" in top_feature:
                print("→ Start/stop codon presence is highly predictive, suggesting")
                print("  proper gene structure is the main signal.")
        
        # Step 6: Make predictions
        print(f"\n{'='*60}")
        print("Step 6: Making Predictions")
        print(f"{'='*60}")
        
        # Take a few test samples
        sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]
        
        predictions = classifier.predict(X_sample)
        probabilities = classifier.predict_proba(X_sample)
        
        print("Sample Predictions:")
        print(f"{'Sample':<10} {'True':^10} {'Predicted':^12} {'Confidence':^12} {'Correct?':^10}")
        print("-" * 60)
        for i, (pred, prob, true) in enumerate(zip(predictions, probabilities, y_sample)):
            true_label = "In GenBank" if true == 1 else "Not in GB"
            pred_label = "In GenBank" if pred == 1 else "Not in GB"
            confidence = prob[pred]
            correct = "✓" if pred == true else "✗"
            print(f"{i+1:<10} {true_label:^10} {pred_label:^12} {confidence:^12.4f} {correct:^10}")
        
        # Step 7: Save model
        print(f"\n{'='*60}")
        print("Step 7: Saving Model")
        print(f"{'='*60}")
        
        model_path = tmpdir / "genbank_classifier.xgb"
        classifier.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Verify loading works
        classifier_loaded = GenbankClassifier(model_type="xgboost")
        classifier_loaded.load(model_path)
        print(f"✓ Model loaded successfully")
        
        # Final summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print("\n✓ Successfully trained XGBoost classifier")
        print(f"✓ Achieved {test_metrics['accuracy']:.1%} accuracy on test set")
        print(f"✓ Model can distinguish GenBank annotations from spurious ORFs")
        print("\nNext steps:")
        print("1. Run genome_entropy pipeline on real GenBank files")
        print("2. Train classifier on real data:")
        print("   genome_entropy ml train --json-dir results/ --output model.xgb")
        print("3. Use trained model to predict annotations in new genomes")
        print("4. Analyze feature importance to understand biological patterns")
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
