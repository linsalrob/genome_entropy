# Figure notebooks

Every figure and table in the manuscript, including supplementary material, is
drawn by one notebook here from plotting-ready data committed in
[`../data/`](../data/). Nothing needs HPC filesystems, Teams or network access
to redraw a published figure. This is the standing requirement from
[agent-handoffs#24](https://github.com/linsalrob/agent-handoffs/issues/24).

## Layout

| path | contents |
|---|---|
| `notebooks/figNN_<slug>.ipynb` | main figure NN (all panels) |
| `notebooks/figSNN_<slug>.ipynb`, `tabNN_<slug>.ipynb` | supplementary figures, tables |
| `notebooks/_template.ipynb` | copy this to start a new figure |
| `notebooks/msfig.py` | shared style, data loading and deterministic saving |
| `data/figNN_<slug>.tsv.gz` | the exact rows the notebook plots, and nothing larger |
| `images/figNN_<slug>.{pdf,png}` | rendered outputs (vector + 300 dpi raster) |
| `figures_manifest.tsv` | one row per panel: notebook, data, upstream source, code, outputs, caption, Results section, owner, status |
| `tools/run_notebooks.py` | executes notebooks and checks they reproduce `images/` byte for byte |
| `environment.yml` | the only environment the notebooks may assume |

## Rules

1. **Data in, figure out.** A notebook reads only `../data/` via
   `msfig.data()` and writes only via `msfig.save()`. No absolute paths, no
   `/g/data`, `/scratch` or `genome_entropy:` paths.
2. **Plotting-ready means small.** Aggregate or sample upstream before
   committing, and record how in the provenance cell. Keep each data file well
   under 50 MB (GitHub's warning size); split if needed. Anything that only
   exists at genome scale stays on Teams or HPC and is listed in
   `../roads_deposition_candidates.md` if it should be published.
3. **Preparation code lives in git too.** If plotting data needs substantial
   preparation, commit that script (under `Validation/<analysis>/scripts/` or
   `manuscript/prep/`) and cite it and its commit in the provenance cell.
4. **Final numbers only.** Use the matched-comparator (v7) GTDB results and
   the v2 corrected phage strand labels. If an alternative reading matters,
   record it in `../decisions.md`; don't choose silently.
5. **Assert what the caption says.** Put `assert` statements on the n values
   and key numbers the caption quotes, so a changed input fails at execution.
6. **Commit without outputs.** Run `nbstripout <notebook>` before committing.
   The images are the outputs.
7. **Check before opening a PR:**

   ```bash
   mamba env create -f manuscript/environment.yml -p <prefix>
   <prefix>/bin/python manuscript/tools/run_notebooks.py --update notebooks/figNN_<slug>.ipynb
   <prefix>/bin/python manuscript/tools/run_notebooks.py notebooks/figNN_<slug>.ipynb   # must print ok
   ```

   The check runs each notebook in a temporary copy of `manuscript/`, so it
   never writes into the checkout, and works whether a notebook saves through
   `msfig.save()` or directly to `../images/`. Then add or update the panel's
   rows in `figures_manifest.tsv`. (The first LOGAN and GTDB packages keep
   their rows in `figures_manifest_{logan,gtdb}.tsv` until the integrator
   merges them.)
