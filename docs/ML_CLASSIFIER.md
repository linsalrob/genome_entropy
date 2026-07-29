# Machine-learning classifier

The maintained machine-learning guide is [`docs/source/ml.rst`](source/ml.rst) and is published as the [ML workflow documentation](https://genome-entropy.readthedocs.io/en/latest/ml.html).

Key facts:

- the default is gradient-boosted `xgboost.train`, not `XGBRFClassifier`;
- `neural_net` is the supported PyTorch alternative;
- features include nullable `twelve_state_entropy` and `twelve_state_length`;
- single-file mode splits top-level records, directory mode splits ORF samples, and split-directory mode uses an 80/20 file split;
- prediction output is a six-column TSV;
- classifier probabilities estimate the `in_genbank` label and are not biological proof;
- PyTorch ROCm support does not imply AMD GPU support in the installed XGBoost build.

Install with `pip install "genome_entropy[ml]"` and run `genome_entropy ml --help` for the installed CLI.
