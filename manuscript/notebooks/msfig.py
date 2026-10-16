"""Shared helpers for the manuscript figure notebooks.

Every notebook in this directory imports this module, reads its plotting data
with ``data()`` and writes its outputs with ``save()``. Keeping style and
output handling here means all figures share typography and palette, and
every output file is byte-reproducible from the committed data, which is what
``../tools/run_notebooks.py`` checks.

Paths are resolved relative to this file, never to a machine path, so the
notebooks run from any checkout.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
# run_notebooks.py points this at a scratch directory so a verification run
# never overwrites the committed images.
IMAGES = Path(os.environ.get("MS_IMAGES_DIR", HERE.parent / "images"))

# Okabe-Ito: distinguishable under the common colour-vision deficiencies.
PALETTE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "grey": "#999999", "black": "#000000",
}


def style():
    """Apply the manuscript style. Call once at the top of a notebook."""
    sns.set_theme(context="paper", style="ticks", palette=list(PALETTE.values()))
    plt.rcParams.update({
        "font.family": "DejaVu Sans",  # ships with matplotlib: no font drift
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 300, "figure.dpi": 100,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "genome_entropy",  # deterministic SVG ids
    })


def data(name, **kwargs):
    """Read a plotting-ready table from ../data (TSV, optionally .gz)."""
    return pd.read_csv(DATA / name, sep="\t", **kwargs)


def save(fig, slug, formats=("pdf", "png")):
    """Write ``<slug>.<fmt>`` to the images directory, without timestamps.

    Matplotlib embeds a creation date and its own version in PDF and PNG
    metadata; both are blanked so that re-running a notebook reproduces the
    committed file byte for byte.
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    meta = {
        "pdf": {"CreationDate": None, "ModDate": None, "Producer": None,
                "Creator": None},
        "png": {"Software": None},
        "svg": {"Date": None, "Creator": None},
    }
    paths = []
    for fmt in formats:
        p = IMAGES / f"{slug}.{fmt}"
        fig.savefig(p, bbox_inches="tight", metadata=meta.get(fmt))
        paths.append(p)
    return paths
