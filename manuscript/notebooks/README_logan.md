# LOGAN figure packages

These are the figures for the LOGAN metagenome validation (genome_entropy #102, closed by PR #115): 50,009 LOGAN v1.2 metagenomes, 89.99 B ORFs. The integrator assigns final figure numbers, so the `figL*` / `tabL*` IDs are provisional.

| Path | What it is |
|---|---|
| `notebooks/figL*.ipynb` | One notebook per figure. The first cell holds the draft caption and the data provenance. The notebook reads `../data/` only and writes `../images/<name>.{pdf,png}`. Stored outputs are stripped. |
| `data/figL*.tsv.gz`, `data/tabL*.tsv.gz` | Plotting-ready data and table data, about 570 kB in total. |
| `data/logan_provenance.json` | For every data file: its source path on Setonix and the sha256 of both source and output, plus the commit of the LOGAN scripts. |
| `prep/logan_prepare_data.py` | Reduces the final LOGAN outputs on Setonix to the files in `data/`. Not needed to redraw the figures. |
| `prep/build_logan_notebooks.py` | Writes the notebooks. They share one style. |
| `figures_manifest.tsv` | LOGAN rows (package `L…`, `LS1`, `tabL…`) are merged into the single manuscript manifest, with final figure and table numbers. |
| `notebooks/requirements_logan.txt` | The tested versions. |

**Redraw everything:**

```bash
cd manuscript/notebooks
for nb in figL*.ipynb; do jupyter nbconvert --to notebook --execute --output "/tmp/$nb" "$nb"; done
```

**Verification:**
- All 11 notebooks were executed twice in a clean conda environment with the versions in `requirements_logan.txt`.
- Both runs produced byte-identical PDF and PNG files (22 files). The PDFs have fixed metadata; the PNGs are 300 dpi.

**Upstream data.** The data are archived on Teams at `genome_entropy:General/LOGAN/`, checked with `rclone check`:
- `analysis/v1/`: the analysis tables;
- `qc/`: all summary outputs, including Foldseek results and contigs;
- `modernprost-50M/v1/`: the AA, 3Di and 12-state cache, 5.95 TB.
