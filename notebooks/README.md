# Jupyter Notebooks for Analysing Data

These notebooks show how to access, analyse, visualise, and model output from
_genome_entropy_. Start Jupyter from the repository environment so the local
`genome_entropy` package and optional analysis dependencies are importable:

```bash
source venv/bin/activate
pip install -e ".[ml]" jupyterlab pandas matplotlib
jupyter lab notebooks/
```

## Available notebooks

### [Plot JSON](plot_json.ipynb)

Reads selected fields from pipeline JSON output into a pandas DataFrame and
creates an entropy scatter plot. This is a compact introduction to exploring
the unified output schema.

### [Machine learning](machine_learning.ipynb)

Provides an interactive counterpart to the `genome_entropy ml` commands. It:

1. Reads a single `.json` or `.json.gz` pipeline output containing multiple
   top-level sequence records.
2. Uses the package's production JSON loader and feature extractor, so the
   notebook sees the same 14 ORF features as the command-line workflow,
   including nullable 12-state entropy and length.
3. Removes records with no usable ORFs before splitting. Empty records are a
   normal pipeline outcome when an input sequence has no ORFs passing the
   configured thresholds.
4. Splits at the top-level sequence-record boundary. Every ORF from a sequence
   stays entirely in either training or testing, preventing sequence-level
   leakage between partitions.
5. Builds an XGBoost random forest with `XGBRFClassifier`. This is distinct
   from the gradient-boosted XGBoost classifier used by the default CLI model:
   the forest trains randomized trees in parallel using row and feature
   subsampling.
6. Evaluates held-out ORFs with a classification report, ROC AUC when both
   classes are present, and a confusion-matrix plot.
7. Creates a pandas table containing every held-out prediction, including both
   `input_id` and `orf_id`, the actual and predicted labels, probability, and
   correctness.
8. Optionally displays a variable-importance table and horizontal bar plot.
9. Optionally saves the trained forest in XGBoost's JSON model format.

Edit the configuration cell before running the notebook. At minimum, set
`JSON_PATH` to an existing consolidated pipeline output; no such generated file
is committed with the repository. The file must contain at least
two top-level records with usable ORFs so one or more records can be held out
for testing.

The main tuning controls are:

- `TEST_SPLIT`: fraction of usable sequence records reserved for testing.
- `RANDOM_SEED`: makes the record split and forest reproducible.
- `N_ESTIMATORS`: number of trees in the forest.
- `MAX_DEPTH`: maximum complexity of each tree.
- `SUBSAMPLE`: fraction of training ORFs sampled for each tree.
- `COLSAMPLE_BYNODE`: fraction of variables considered at each tree node.
- `DEVICE`: `cpu`, or `cuda` when using a GPU-enabled XGBoost installation.
- `PLOT_FEATURE_IMPORTANCE`: enables or suppresses the importance plot.
- `MODEL_OUTPUT`: optional destination for the trained model.

Feature importance describes how strongly the fitted forest used each
variable; it does not establish biological causation. Correlated entropy,
length, and location features may divide importance between themselves, so the
plot should be interpreted together with held-out performance and domain
knowledge.
