# Copilot instructions for `genome_entropy`

Use the current implementation and tests as the source of truth. The maintained
developer, API, CLI, schema, model, ML, and HPC references are linked from
`docs/source/index.rst`; do not duplicate a second manual here.

## Project constraints

- Python 3.10+ is required and CI tests 3.10, 3.11, and 3.12.
- Preserve the unified JSON schema (`2.1.0`) and its nullable 12-state fields.
- Standard JSON contains raw Shannon entropy only. Downstream code may call the
  generic or representation-specific normalisation helpers.
- The default model is `gbouras13/modernprost-50M`; it and
  `gbouras13/modernprost-base` emit both 3Di and 12-state encodings.
- Deprecated ModernProst and ProstT5 models emit 3Di only and serialise
  12-state values as `null`.
- ModernProst uses `trust_remote_code=True`; do not obscure that security and
  reproducibility consideration.
- ORF coordinates are 1-based and inclusive. GenBank matching is strand-aware
  and uses the tested C-terminal subset heuristic, not exact full-length
  equality.
- `get_orfs` is an external executable and is not installed by this package.
- The CLI's `xgboost` backend uses gradient-boosted trees via `xgboost.train`,
  not `XGBRFClassifier`. XGBoost GPU mode requires a CUDA-capable build; PyTorch
  ROCm support does not provide XGBoost AMD GPU support.

## Working practices

- Keep changes focused and preserve public behaviour unless the task explicitly
  requires a runtime change.
- Add or update tests for behavioural changes.
- Run `pytest`, Black, Ruff, mypy, CLI help smoke tests, and the Sphinx
  warnings-as-errors command documented in `docs/source/development.rst`.
- Do not make tests download large model repositories by default. Integration
  tests must remain explicitly enabled.
- Do not commit generated documentation, model caches, virtual environments,
  test caches, cluster logs, or user data.
- Use Australian/British spelling in narrative documentation while preserving
  literal API and dependency names.
