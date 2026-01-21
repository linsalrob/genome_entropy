"""Configuration and constants for genome_entropy."""

import os
from pathlib import Path

# Default NCBI translation table (Bacterial/Archaeal)
DEFAULT_GENETIC_CODE_TABLE = 11

# Minimum ORF lengths
DEFAULT_MIN_NT_LENGTH = 90  # nucleotides
DEFAULT_MIN_AA_LENGTH = 30  # amino acids

# ProstT5 model configuration
DEFAULT_PROSTT5_MODEL = "Rostlab/ProstT5_fp16"
DEFAULT_ENCODING_SIZE = 10000

# 3Di alphabet (20 structural states)
THREEDDI_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# DNA alphabet
DNA_ALPHABET = set("ACGT")
DNA_ALPHABET_WITH_N = set("ACGTN")

# Amino acid alphabet (standard 20 + ambiguous X + stop *)
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
AA_ALPHABET_EXTENDED = set("ACDEFGHIKLMNPQRSTVWYX*")

# External binary paths
GET_ORFS_BINARY = os.environ.get("GET_ORFS_PATH", "get_orfs")

# Device selection for PyTorch
AUTO_DEVICE = "auto"
CUDA_DEVICE = "cuda"
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"

# File formats
FASTA_EXTENSIONS = {".fasta", ".fa", ".fna", ".faa"}
JSON_EXTENSIONS = {".json"}

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USER_ERROR = 2
EXIT_RUNTIME_ERROR = 3

# Logging configuration
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
