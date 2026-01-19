"""ProstT5 encoder for amino acid to 3Di structural token conversion.

This module maintains backward compatibility by re-exporting classes and functions
that have been moved to separate modules for better organization.
"""

# For backward compatibility, re-export from new modules
from .encoder import ProstT5ThreeDiEncoder
from .types import IndexedSeq, ThreeDiRecord

__all__ = [
    "ProstT5ThreeDiEncoder",
    "ThreeDiRecord",
    "IndexedSeq",
]
