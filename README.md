# genome_entropy

[![Python CI](https://github.com/linsalrob/genome_entropy/actions/workflows/python-ci.yml/badge.svg)](https://github.com/linsalrob/genome_entropy/actions/workflows/python-ci.yml)
[![Documentation](https://github.com/linsalrob/genome_entropy/actions/workflows/docs.yml/badge.svg)](https://linsalrob.github.io/genome_entropy/)
[![Read the Docs](https://readthedocs.org/projects/genome-entropy/badge/?version=latest)](https://genome-entropy.readthedocs.io/en/latest/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`genome_entropy` quantifies Shannon entropy across biological representations derived from genomic DNA. It finds open reading frames (ORFs), translates proteins, predicts structural-state encodings, and writes a non-redundant JSON record for downstream analysis.

Current multitask ModernProst models produce both Foldseek 3Di and 12-state (`12st`) encodings. The command remains named `encode3di` for compatibility. Legacy ModernProst and ProstT5 models produce 3Di only, with 12-state fields written as `null`.

## Capabilities

- DNA FASTA and GenBank input, including gzip-compressed GenBank files
- six-frame ORF discovery through the external [`get_orfs`](https://github.com/linsalrob/get_orfs) program
- translation with [`pygenetic-code`](https://github.com/linsalrob/genetic_codes)
- 3Di and optional 12-state prediction with ModernProst or ProstT5
- raw Shannon entropy for DNA, protein, 3Di, and 12-state representations
- CUDA, ROCm-through-PyTorch's CUDA API, Apple MPS, CPU, and multi-GPU encoding
- optional XGBoost or PyTorch classification of whether an ORF matches a GenBank CDS

ORF discovery is gene calling, not functional annotation. Likewise, a classifier score is a model estimate of agreement with the supplied GenBank annotations, not biological proof of a gene or function.

## Installation

Python 3.10 or newer is required.

```bash
pip install genome_entropy
pip install "genome_entropy[ml]"  # optional ML dependencies
```

`get_orfs` is not installed by `pip`; install it separately and ensure the executable is on `PATH`, or set `GET_ORFS_PATH`. GPU users should install a PyTorch build appropriate for their CUDA or ROCm platform before installing this package. See the [installation guide](https://genome-entropy.readthedocs.io/en/latest/installation.html) for development, CPU, GPU, HPC, caching, and offline-job instructions.

## Quick start

```bash
# Pre-cache the default model when login nodes have internet access
genome_entropy download --model gbouras13/modernprost-50M

# DNA FASTA to unified JSON
genome_entropy run --input genome.fasta --output results.json

# GenBank matching accepts aligned ambiguous X residues in C-terminal suffixes
genome_entropy run --genbank genome.gbk.gz --output results.json

# Protein FASTA directly to structural-state records
genome_entropy encode3di --input proteins.faa --output structures.json
```

Run `genome_entropy --help` and `genome_entropy COMMAND --help` for the installed version's authoritative option list. Detailed examples are in the [quick-start guide](https://genome-entropy.readthedocs.io/en/latest/quickstart.html) and [CLI reference](https://genome-entropy.readthedocs.io/en/latest/cli.html).

## Supported encoders

| Canonical model | Approximate size | 3Di | 12-state | Status |
|---|---:|:---:|:---:|---|
| `gbouras13/modernprost-50M` | 52.6M | yes | yes | default |
| `gbouras13/modernprost-base` | approximately 1B | yes | yes | supported |
| `gbouras13/modernprost-base-deprecated` | legacy | yes | no | deprecated |
| `gbouras13/modernprost-profiles-deprecated` | legacy | yes | no | deprecated |
| `Rostlab/ProstT5` | approximately 3B | yes | no | supported |
| `Rostlab/ProstT5_fp16` | approximately 3B | yes | no | supported, half precision |

ModernProst loads model-provided Python code with `trust_remote_code=True`. Review and pin model revisions when your threat model requires reproducible, audited remote code. The legacy alias `gbouras13/modernprost-profiles` resolves to `gbouras13/modernprost-profiles-deprecated` with a warning.

See the [model guide](https://genome-entropy.readthedocs.io/en/latest/models.html) for provenance, precision, devices, multi-GPU behaviour, output semantics, and verified repository links.

## Output and entropy

The pipeline writes schema `2.1.0`, with features keyed by ORF identifier. Each feature contains location, DNA, protein, 3Di, optional 12-state, metadata, and raw entropy. JSON and JSON-gzip input are supported where documented.

Normalised entropy is deliberately not serialised. Derive it downstream:

```python
from genome_entropy.entropy import normalise_protein_entropy

value = normalise_protein_entropy(feature["entropy"]["protein_entropy"])
```

The generic formula is `raw_entropy / math.log2(alphabet_size)`, using theoretical alphabet sizes 4 (DNA), 20 (protein), 20 (3Di), and 12 (12-state). See [data formats and entropy](https://genome-entropy.readthedocs.io/en/latest/data_formats.html) for the complete schema, coordinate conventions, null semantics, and normalisation helpers.

## Documentation and support

- [GitHub Pages documentation](https://linsalrob.github.io/genome_entropy/)
- [Read the Docs](https://genome-entropy.readthedocs.io/en/latest/)
- [Issue tracker](https://github.com/linsalrob/genome_entropy/issues)
- [Machine-learning guide](https://genome-entropy.readthedocs.io/en/latest/ml.html)
- [NVIDIA and ROCm SLURM notes](slurm/README.md)

This project is alpha software. Report reproducible bugs through the issue tracker and include the package version, command, platform, accelerator, and relevant log output without credentials or sensitive sequence data.

## Citation and attribution

Please cite the software release used in your analysis and the methods relevant to your workflow:

- Heinzinger et al., *Bilingual language model for protein sequence and structure* (ProstT5), **NAR Genomics and Bioinformatics** (2024), [doi:10.1093/nargab/lqae150](https://doi.org/10.1093/nargab/lqae150).
- van Kempen et al., *Fast and accurate protein structure search with Foldseek*, **Nature Biotechnology** (2024), [doi:10.1038/s41587-023-01773-0](https://doi.org/10.1038/s41587-023-01773-0).
- Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD (2016), [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785), when using the ML workflow.

ModernProst model repositories and integration were provided by George Bouras. See the [full attribution page](https://genome-entropy.readthedocs.io/en/latest/models.html#citations-and-provenance) for model and dependency links.

## Licence

`genome_entropy` is distributed under the [MIT License](LICENSE).
