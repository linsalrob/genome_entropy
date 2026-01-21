"""genome_entropy: Compare and contrast the entropy of sequences, ORFs, proteins, and 3Di encodings."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("genome_entropy")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
