# Machine-learning implementation history

This file records the original machine-learning implementation work and is not
the current user guide. The implementation has since gained 12-state features,
additional split modes, and revised persistence and device behaviour.

Use the maintained [machine-learning guide](docs/source/ml.rst) for the current
feature order, estimator types, missing-value handling, split semantics, output
columns, reproducibility limits, and CPU/GPU requirements. The CLI's default
`xgboost` model uses gradient-boosted trees through `xgboost.train`; it is not
an `XGBRFClassifier` random forest. The example notebook intentionally uses
`XGBRFClassifier` as a distinct exploratory model.

The original implementation introduced:

- `genome_entropy.ml` data loading, feature extraction, model wrappers, and
  `GenbankClassifier`;
- `genome_entropy ml train` and `genome_entropy ml predict`;
- XGBoost and PyTorch neural-network backends;
- model persistence, evaluation, prediction probabilities, and feature
  importance reporting.

Historical test counts and performance claims have been removed because they
are not stable documentation. Run the repository's current test suite and use
the validation commands in the [development guide](docs/source/development.rst).
