# Version management

`pyproject.toml` is the single authoritative package-version source. The current version is `0.2.0`.

`src/genome_entropy/__init__.py` and `docs/source/conf.py` read installed metadata through `importlib.metadata.version("genome_entropy")`; the CLI imports that value for `--version`. The package requires Python 3.10 or newer and CI tests 3.10, 3.11, and 3.12.

For a release, update only the `[project].version` value required by the established release process, update the changelog, build and test distributions, and create the reviewed release. Do not edit generated `src/genome_entropy.egg-info` metadata manually.

Validate version reporting with:

```bash
pytest tests/test_basic.py::test_version tests/test_cli_smoke.py::test_cli_version -v
genome_entropy --version
```
