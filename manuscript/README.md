# genome_entropy manuscript

This directory contains the **draft manuscript** describing [`genome_entropy`](../README.md):

> *Joint sequence and structural entropy distinguish protein-coding genes from spurious open reading frames in bacterial genomes and metagenomes.*

The manuscript is a work in progress. Expect the text, figures, and author list to change.

**To edit the paper,** follow [`EDITING.md`](EDITING.md). It explains, step by step, how to fork this repository, make a token, edit the manuscript in the browser with [ManuMD](https://linsalrob.github.io/manumd/), and send your changes back as a pull request. No previous GitHub experience is needed.

## Contents

| File / directory | Description |
|---|---|
| [`genome_entropy.md`](genome_entropy.md) | The manuscript text (Markdown) |
| [`EDITING.md`](EDITING.md) | How co-authors edit the manuscript with ManuMD and send changes back |
| [`images/`](images/) | Figures referenced by the manuscript |
| [`Author_Information.xlsx`](Author_Information.xlsx) | Author details used to build the author list, affiliations, contributions, and funding statements |
| [`results_inventory.md`](results_inventory.md) | Evidence matrix: every result the paper may state, its final value and source |
| [`decisions.md`](decisions.md) | Open scientific and editorial decisions for the authors |
| [`figure_plan.md`](figure_plan.md) | Proposed main and supplementary figures and tables |
| [`methods_provenance.md`](methods_provenance.md) | Data locations, code and configuration for each dataset |
| [`manuscript_audit.md`](manuscript_audit.md) | Line-level check of the current text against the final results |
| [`references.bib`](references.bib) | Bibliography for the `[@key]` citations in the manuscript |

## Being included as an author

If you have contributed to `genome_entropy` or to this manuscript and want to be included on the paper, **add your details to [`Author_Information.xlsx`](Author_Information.xlsx)**. Add one row per person and fill in:

- ORCID
- Initials (the `acronym` column, e.g. `RAE`), used in the author contributions statement
- Last name, first name, and middle initial
- Email address
- Department, institution, and full postal address (street, city, state/region, ZIP/postcode, country)
- Your contribution to the work
- Any funding that supported your contribution (agency and grant number)

Please leave the **"Added"** column and the **"Order in the author list"** column blank. These are filled in when your details are added to the manuscript.

To add your details, open a pull request against the `manuscript` branch with the updated spreadsheet. If you can't do that, send your details to [Rob Edwards](mailto:robert.edwards@flinders.edu.au).

## Rendering

The manuscript uses pandoc citations (`[@key]`) and TeX equations. To render it:

```bash
pandoc genome_entropy.md --citeproc --bibliography references.bib -s -o genome_entropy.html   # or .docx
```

HTML comments of the form `<!-- src: … -->` record the source of every number, for the final audit. They are not rendered, and are stripped before submission.

