# Machine Learning Classifier for GenBank Annotation Prediction

## Overview

The ML classifier module provides tools to train machine learning models that predict whether an Open Reading Frame (ORF) was annotated in the original GenBank file based on various sequence features.

## Why This Matters

Not all ORFs detected in a genome are annotated as genes in GenBank. The ability to predict which ORFs are likely to be real genes can help:
- Identify potentially missed gene annotations
- Filter out spurious ORFs from large-scale analyses
- Understand what features distinguish annotated genes from background ORFs
- Prioritize ORFs for experimental validation

## Model Selection: XGBoost (Recommended)

We recommend **XGBoost (Gradient Boosted Trees)** as the default model for the following reasons:

### 1. **Excellent Performance on Structured Data**
XGBoost consistently achieves state-of-the-art results on tabular/structured datasets like ORF features. It's specifically designed for this type of data, unlike neural networks which excel at unstructured data (images, text, etc.).

### 2. **Handles Mixed Feature Types Naturally**
Our features include:
- Continuous values (entropy scores, lengths)
- Categorical values (strand: +/-, frame: 0-3)
- Boolean values (has_start_codon, has_stop_codon)

XGBoost handles these mixed types without requiring extensive preprocessing or one-hot encoding.

### 3. **Built-in GPU Support**
XGBoost provides native GPU acceleration through CUDA, which aligns with the requirement "we are already working on GPUs." Training can be 10-100x faster on GPUs for large datasets.

### 4. **Feature Importance for Interpretability**
XGBoost provides feature importance scores that tell you which features are most predictive:
- Are entropy values more important than structural features?
- Is protein entropy more predictive than DNA entropy?
- Do positional features (start/end) matter?

This interpretability is crucial for understanding biology, not just making predictions.

### 5. **Robust to Overfitting**
With proper parameters, XGBoost is less prone to overfitting than neural networks, especially on small-to-medium datasets (hundreds to thousands of samples). It doesn't require extensive hyperparameter tuning to get good results.

### 6. **Handles Class Imbalance**
GenBank annotations are often imbalanced (typically more non-annotated ORFs than annotated ones). XGBoost handles this well through:
- Scale_pos_weight parameter
- Stratified sampling
- Built-in class weighting

### 7. **Fast Training and Inference**
XGBoost trains in seconds to minutes (vs. hours for neural networks) and makes predictions in milliseconds, making it practical for large-scale genomic analyses.

## Alternative: Neural Network

A PyTorch-based neural network is also available as an alternative. It may perform better when:
- You have a very large dataset (10k+ samples)
- You need to model highly complex non-linear relationships
- You want to extend the model with sequence embeddings or other deep learning features

However, for typical use cases with ORF features, XGBoost will likely outperform.

## Features Used for Prediction

The classifier uses the following 12 features extracted from JSON output:

### Entropy Features (3)
- `dna_entropy`: Shannon entropy of nucleotide sequence
- `protein_entropy`: Shannon entropy of amino acid sequence  
- `three_di_entropy`: Shannon entropy of 3Di structural encoding

### Length Features (3)
- `dna_length`: Length of nucleotide sequence
- `protein_length`: Length of amino acid sequence
- `three_di_length`: Length of 3Di encoding

### Position Features (2)
- `start`: Start position in genome
- `end`: End position in genome

### Structural Features (2)
- `strand_plus`: 1 if forward strand (+), 0 if reverse (-)
- `frame`: Reading frame (0, 1, 2, or 3)

### Boolean Features (2)
- `has_start_codon`: 1 if ORF has start codon, 0 otherwise
- `has_stop_codon`: 1 if ORF has stop codon, 0 otherwise

## Installation

Install the ML dependencies:

```bash
pip install "genome_entropy[ml]"
```

This installs:
- xgboost >= 2.0.0 (with GPU support)
- scikit-learn >= 1.3.0 (for metrics)
- numpy >= 1.24.0

## Usage

### 1. Generate Training Data

First, run the genome_entropy pipeline on GenBank files to generate JSON output:

```bash
genome_entropy run --input genome.gbk --output results/genome.json
```

This will automatically set `in_genbank: true` for ORFs that match annotated CDSs in the GenBank file.

### 2. Train the Classifier

Train on a directory of JSON files:

```bash
genome_entropy ml train \
    --json-dir results/ \
    --output model.xgb \
    --model-type xgboost \
    --validation-split 0.2 \
    --test-split 0.1
```

**Arguments:**
- `--json-dir`: Directory containing JSON output files
- `--output`: Path to save the trained model
- `--model-type`: "xgboost" (default) or "neural_net"
- `--validation-split`: Fraction for validation during training (default: 0.2)
- `--test-split`: Fraction to hold out for final testing (default: 0.1)
- `--device`: "cuda", "cpu", or None for auto-detect

**Output:**
```
Training XGBoost model on 800 samples
Validation accuracy: 0.8750
Validation AUC: 0.9234
Test accuracy: 0.8600
Test F1 score: 0.8421

Feature Importance (Top 5):
protein_entropy      : 0.2341
dna_length          : 0.1823
has_start_codon     : 0.1567
three_di_entropy    : 0.1234
has_stop_codon      : 0.0987
```

### 3. Make Predictions

Use the trained model on new data:

```bash
genome_entropy ml predict \
    --json-dir new_results/ \
    --model model.xgb \
    --output predictions.csv
```

### 4. Programmatic Usage

```python
from pathlib import Path
from genome_entropy.ml import GenbankClassifier, load_json_data, extract_features

# Load training data
json_data = load_json_data(Path("results/"))
X, y, feature_names, _ = extract_features(json_data)

# Train classifier
classifier = GenbankClassifier(model_type="xgboost")
classifier.fit(X, y, feature_names=feature_names)

# Evaluate
metrics = classifier.evaluate(X, y)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")

# Get feature importance
importance = classifier.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature:20s}: {score:.4f}")

# Save model
classifier.save(Path("model.xgb"))

# Make predictions on new data
new_data = load_json_data(Path("new_results/"))
X_new, _, _, _ = extract_features(new_data)
predictions = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)
```

## Understanding Results

### Metrics

- **Accuracy**: Fraction of correct predictions (TP + TN) / Total
- **Precision**: Of predicted positives, how many are correct? TP / (TP + FP)
- **Recall**: Of actual positives, how many did we find? TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (discrimination ability)

### Feature Importance

Feature importance shows which features are most predictive. Common patterns:

1. **Entropy dominates**: If entropy features rank highest, sequence complexity is the main signal for distinguishing real genes from spurious ORFs.

2. **Length matters**: If length features rank high, real genes tend to be longer/shorter than background ORFs.

3. **Structural features**: If has_start_codon/has_stop_codon rank high, proper gene structure is important.

4. **Position effects**: If start/end rank high, there may be positional biases (e.g., genes clustered in certain regions).

### Interpreting Predictions

- **High confidence (prob > 0.9)**: Very likely to be a real gene or non-gene
- **Medium confidence (0.6-0.9)**: Model is fairly certain
- **Low confidence (0.4-0.6)**: Uncertain, may need manual review
- **Threshold**: Default is 0.5, but you can adjust based on your use case:
  - Higher threshold (0.7): More stringent, fewer false positives
  - Lower threshold (0.3): More permissive, fewer false negatives

## Performance Considerations

### Dataset Size

- **Small (< 100 samples)**: May not have enough data, results unreliable
- **Medium (100-1000 samples)**: XGBoost should work well
- **Large (> 1000 samples)**: Both XGBoost and neural networks viable
- **Very large (> 10k samples)**: Neural networks may start to shine

### GPU Acceleration

XGBoost automatically uses GPU if available:
```python
classifier = GenbankClassifier(model_type="xgboost", device="cuda")
```

Neural networks also support GPU:
```python
classifier = GenbankClassifier(model_type="neural_net", device="cuda")
```

For large datasets (> 10k samples), GPU acceleration can provide 10-100x speedup.

### Training Time

Typical training times (CPU, 1000 samples):
- XGBoost: 5-30 seconds
- Neural Network: 1-5 minutes

With GPU (1000 samples):
- XGBoost: 1-5 seconds
- Neural Network: 10-30 seconds

## Troubleshooting

### "No JSON files found"
Check that your directory contains `.json` files from the genome_entropy pipeline.

### "No features could be extracted"
Verify that your JSON files contain the required fields. Check the schema version.

### "XGBoost not installed"
Install with: `pip install "genome_entropy[ml]"` or `pip install xgboost`

### Low accuracy (< 0.6)
Possible causes:
- Not enough training data
- Class imbalance too severe
- Features not informative for your dataset
- Data quality issues

### Class imbalance
If you have far more non-genes than genes (or vice versa), consider:
- Collecting more balanced data
- Using stratified sampling
- Adjusting class weights
- Using F1 score instead of accuracy as the metric

## Citation

If you use this ML classifier in your research, please cite:

```bibtex
@software{genome_entropy_ml,
  author = {Edwards, Rob},
  title = {genome_entropy: Machine Learning for GenBank Annotation Prediction},
  year = {2024},
  url = {https://github.com/linsalrob/genome_entropy}
}
```

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
3. Heinzinger, M., et al. (2023). ProstT5: Bilingual language model for protein sequence and structure.
