# Version Management

## Single Source of Truth

The version for the `genome_entropy` package is now managed in a single location: **`pyproject.toml`**

```toml
[project]
version = "0.1.2"
```

## How It Works

All other parts of the codebase dynamically read the version from the installed package metadata using Python's built-in `importlib.metadata` module (available in Python 3.8+):

- **`src/genome_entropy/__init__.py`**: Uses `importlib.metadata.version("genome_entropy")`
- **`docs/source/conf.py`**: Uses `importlib.metadata.version("genome_entropy")`
- **CLI and other code**: Import `__version__` from the main package

## Making a New Release

To update the version for a new release:

1. Update the version string in `pyproject.toml` **only**
2. Commit and tag the release
3. All other locations will automatically use the new version

## Special Files

### citation.cff
The `citation.cff` file contains its own version field (`version: v0.1.1`) which should be updated manually when creating releases with DOIs. This is intentionally separate as it tracks released versions with citation metadata.

## Benefits

✅ Single place to update the version  
✅ No risk of version mismatches  
✅ Standard Python packaging approach (PEP 621)  
✅ Works with all Python 3.8+ installations  

## Testing

The version is tested in:
- `tests/test_basic.py::test_version` - Verifies `__version__` is accessible
- `tests/test_cli_smoke.py::test_cli_version` - Verifies CLI `--version` flag works

Run tests with:
```bash
pytest tests/test_basic.py::test_version tests/test_cli_smoke.py::test_cli_version -v
```
