# Drafting guide for Methods and Results sections (checkpoint D, agent-handoffs#24)

Temporary file for section drafters; the integrator removes it before the PR merges.

## Figure and table numbering (final; D4 as implemented)

Main figures:
- **Fig. 1** = package G01 (`images/figG01_entropy_landscape`), panels a–d as in its notebook.
- **Fig. 2** = L02 (`figL02_models_beyond_length`), panels a–h.
- **Fig. 3** = L05 (`figL05_six_frame`), panels a–g.
- **Fig. 4** = G04 (`figG04_missed_genes`), panels a–d.
- **Fig. 5** = L06 (`figL06_missed_gene_evidence`), panels a–b.
- **Fig. 6** = P01 (`figP01_phage`), panels a–d.

Supplementary figures:

| fig. | package | content |
|---|---|---|
| S1 | G02 | estimator bias vs biology |
| S2 | GS02 | reference calibration |
| S3 | L01 | LOGAN feature distributions |
| S4 | L03 | within-length AUC |
| S5 | L04 | fragments |
| S6 | LS1 | biome/batch |
| S7 | G05 | GTDB Prodigal agreement |
| S8 | GS03 | candidate burden |
| S9 | G06 | functions and examples |
| S10 | L07 | candidate evidence detail |
| S11 | L08 | contig taxonomy |
| S12 | L09 | Prodigal skip |
| S13 | L10 | low-scoring Prodigal genes |
| S14 | P04 | phage vs cellular 3Di |
| S15 | PS01 | phage label correction |
| S16 | S01 | smORF separation |
| S17 | SS01 | smORF density |
| S18 | SS02 | smORF stripes |

Tables:
- **Table 1:** datasets and encoding configurations (integrator).
- **Table 2:** LOGAN ORF classes (tabL01).
- **Table 3:** GTDB direct-evidence ladder C3–C7 (G04 data).

Supplementary tables:
- **Table S1:** encoding determinism (tabG01).
- **Tables S2–S9:** tabL02–tabL09, in that order.
- **Tables S10–S13:** tabS01–tabS04.

Read each package's notebook caption cell (`notebooks/<name>.ipynb`, first cell) to see what each panel shows before citing a panel letter.

## Rules
1. **Final numbers only.** The authority is `results_inventory.md`, plus the primary sources it cites. Check every number you use against the primary source, not only the inventory. Never use anything in the inventory's *Superseded* list.
2. **Decisions D1–D15 in `decisions.md` are accepted.** Follow them. In particular:
   - D1 (headline numbers);
   - D2 and D6 (state as limitations);
   - D7 (the OM869708 reframing);
   - D8 (3Di carries most of the signal);
   - D9 (configuration per dataset);
   - D11 (word mechanistic claims as observations);
   - D12 (no unsupported claims).
3. **Provenance comment.** After every sentence that states a number, add an HTML comment naming its source, e.g. `<!-- src: Validation/gtdb/scientific_report.md §6.1 table -->`. The integrator uses these for the audit (checkpoint F) and strips them later.
4. **Style.** Plain scientific prose, past tense for what was done and present tense for what figures show.
   - Methods should be reproducible: dataset, inclusion rules, counts, software and versions, parameters, statistics.
   - Results give n, effect sizes and uncertainty, and cite figures and tables.
   - Keep paragraphs short. Introduce each abbreviation once.
   - Say what a result does *not* show where that matters, e.g. structural homology is evidence of protein-likeness, not proof of expression or function.
5. **Length.** Methods about 500–900 words per dataset. Results about 600–1,100 words per dataset. Be lean; the supplementary figures carry the detail.
6. **Software names:**
   - genome_entropy (with version per dataset);
   - ModernProst (`gbouras13/modernprost-50M`, revision 2d68d506 where recorded);
   - Foldseek 10;
   - Prodigal 2.6.3;
   - Pharokka/phold (versions not recorded: say so);
   - Kraken2 2.17.2.

   Cite tools as `[@key]` with a plausible key (e.g. `[@vanKempen2024]`); the integrator builds the reference list.
7. **Output.** Write only the files you are told to write. Don't edit any other file and don't run git.
