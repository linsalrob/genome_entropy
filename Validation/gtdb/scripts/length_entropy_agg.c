/* Length-conditioned entropy aggregator for genome_entropy GTDB entropy_rows.
 *
 * Reads per-ORF TSV on stdin (with header) and accumulates a 2-D histogram of
 * (aa_length bin) x (entropy bin) for the three sequence alphabets, stratified
 * exactly as population_agg.c stratifies -- (genome has >=1 deposited CDS,
 * in_genbank) -- so the marginals of this table must reproduce stage 29's
 * numbers, and that is the cross-check.
 *
 * WHY BINARY OUTPUT.  The array is 6 x 3 x 1042 x 4400 cells.  Written sparsely
 * as text that is tens of millions of lines per worker and hundreds of millions
 * over the run -- more text to recombine than to produce, which is the same
 * trade stage 29 called out.  Each worker therefore dumps its raw uint32 array
 * (330 MB) plus a small text sidecar of exact counts and sums, and the merge is
 * a numpy add.
 *
 * uint32 per cell is safe: the whole bacterial domain is 2.57e9 rows, below
 * UINT32_MAX, so no single cell can overflow even if every row landed in one.
 *
 * usage: length_agg <annotated_genomes.txt> <out.bin> < rows.tsv > sidecar.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NENT   4400         /* entropy bins, 1e-3 wide over [0, 4.4) */
#define ENTW   1e-3
#define NLEN   1042
#define NMET   3            /* protein, three_di, twelve_state */
#define NSTRAT 6            /* (has_cds, in_genbank) 2x2, plus 2 unknown */

static uint32_t *hist;      /* [NSTRAT][NMET][NLEN][NENT] */
static uint64_t cell_n[NSTRAT][NMET][NLEN];
static double   cell_s1[NSTRAT][NMET][NLEN];
static double   cell_s2[NSTRAT][NMET][NLEN];
static uint64_t nan_ct[NSTRAT][NMET], oob_ct[NSTRAT][NMET];
static uint64_t n_rows[NSTRAT], n_short[NSTRAT];

/* Length binning: one bin per residue where the ORFs are, coarsening above.
 * Must stay in lockstep with lenbin_edges() in 33_length_null_summary.py. */
static int lenbin(long L) {
    if (L < 90)    return 1041;                       /* below the ORF floor */
    if (L < 1000)  return (int)(L - 90);              /* 0 .. 909, 1 aa wide */
    if (L < 2000)  return 910  + (int)((L - 1000) / 10);
    if (L < 5000)  return 1010 + (int)((L - 2000) / 100);
    return 1040;                                      /* >= 5000, overflow */
}

/* ---- open-addressing hash set of annotated genome accessions (as stage 29) ---- */
static char  **hkey;
static char   *hval;
static size_t  hcap;
static uint64_t fnv(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}
static void hset_init(size_t cap) {
    hcap = 1; while (hcap < cap * 4) hcap <<= 1;
    hkey = calloc(hcap, sizeof(char *));
    hval = calloc(hcap, 1);
    if (!hkey || !hval) { fprintf(stderr, "length_agg: calloc hash failed\n"); exit(2); }
}
static void hset_add(const char *k, char v) {
    size_t i = fnv(k) & (hcap - 1);
    while (hkey[i]) { if (!strcmp(hkey[i], k)) { hval[i] = v; return; } i = (i + 1) & (hcap - 1); }
    hkey[i] = strdup(k); hval[i] = v;
}
static int hset_get(const char *k) {
    size_t i = fnv(k) & (hcap - 1);
    while (hkey[i]) { if (!strcmp(hkey[i], k)) return hval[i]; i = (i + 1) & (hcap - 1); }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: length_agg <annotated_genomes.txt> <out.bin>\n"); return 2; }

    FILE *g = fopen(argv[1], "r");
    if (!g) { perror("length_agg: annotated genome list"); return 2; }
    hset_init(300000);
    char gl[512];
    while (fgets(gl, sizeof gl, g)) {
        char *e = gl + strcspn(gl, "\r\n"); *e = 0;
        char *tab = strchr(gl, '\t');
        if (!tab || tab == gl) continue;
        *tab = 0;
        hset_add(gl, tab[1] == '1' ? 1 : 0);
    }
    fclose(g);

    size_t ncell = (size_t)NSTRAT * NMET * NLEN * NENT;
    hist = calloc(ncell, sizeof(uint32_t));
    if (!hist) { fprintf(stderr, "length_agg: calloc hist failed (%zu cells)\n", ncell); return 2; }

    const int C_GENOME = 2, C_AALEN = 8, C_GB = 9;
    const int col[NMET] = { 11, 12, 13 };   /* protein, three_di, twelve_state */

    size_t cap = 1 << 16;
    char *line = malloc(cap);
    ssize_t len;
    uint64_t nline = 0, bad = 0;

    if ((len = getline(&line, &cap, stdin)) <= 0) { fprintf(stderr, "length_agg: empty input\n"); return 2; }
    if (strncmp(line, "domain\t", 7) != 0) { fprintf(stderr, "length_agg: unexpected header: %.40s\n", line); return 2; }

    char *f[24];
    while ((len = getline(&line, &cap, stdin)) > 0) {
        if (len && line[len - 1] == '\n') line[--len] = 0;
        if (!len) continue;
        if (line[0] == 'd' && !strncmp(line, "domain\t", 7)) continue;  /* concatenated stream */
        int nf = 0;
        char *p = line;
        f[nf++] = p;
        while (*p && nf < 24) { if (*p == '\t') { *p = 0; f[nf++] = p + 1; } p++; }
        if (nf < 16) { bad++; continue; }
        nline++;

        int ann = hset_get(f[C_GENOME]);
        int gb = (f[C_GB][0] == 'T');
        int st = (ann < 0) ? (4 + gb) : (ann * 2 + gb);
        n_rows[st]++;

        char *ep;
        long L = strtol(f[C_AALEN], &ep, 10);
        if (ep == f[C_AALEN]) { bad++; continue; }
        int lb = lenbin(L);
        if (lb == 1041) n_short[st]++;

        for (int m = 0; m < NMET; m++) {
            char *v = f[col[m]];
            if (!*v) { nan_ct[st][m]++; continue; }
            double x = strtod(v, &ep);
            if (ep == v || x != x) { nan_ct[st][m]++; continue; }
            cell_n[st][m][lb]++;
            cell_s1[st][m][lb] += x;
            cell_s2[st][m][lb] += x * x;
            long b = (long)(x / ENTW);
            if (b < 0 || b >= NENT) { oob_ct[st][m]++; b = (b < 0) ? 0 : NENT - 1; }
            hist[(((size_t)st * NMET + m) * NLEN + lb) * NENT + b]++;
        }
    }

    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("length_agg: output"); return 2; }
    if (fwrite(hist, sizeof(uint32_t), ncell, out) != ncell) {
        fprintf(stderr, "length_agg: short write\n"); fclose(out); return 2;
    }
    if (fclose(out)) { perror("length_agg: close"); return 2; }

    static const char *mname[NMET] = { "protein", "three_di", "twelve_state" };
    printf("#length_agg v1 nent=%d entw=%.10g nlen=%d nmet=%d nstrat=%d\n",
           NENT, ENTW, NLEN, NMET, NSTRAT);
    printf("#lines=%llu malformed=%llu\n", (unsigned long long)nline, (unsigned long long)bad);
    for (int s = 0; s < NSTRAT; s++) {
        printf("N\t%d\t%llu\t%llu\n", s,
               (unsigned long long)n_rows[s], (unsigned long long)n_short[s]);
        for (int m = 0; m < NMET; m++)
            printf("M\t%d\t%s\t%llu\t%llu\n", s, mname[m],
                   (unsigned long long)nan_ct[s][m], (unsigned long long)oob_ct[s][m]);
    }
    for (int s = 0; s < NSTRAT; s++)
        for (int m = 0; m < NMET; m++)
            for (int l = 0; l < NLEN; l++)
                if (cell_n[s][m][l])
                    printf("C\t%d\t%s\t%d\t%llu\t%.17g\t%.17g\n", s, mname[m], l,
                           (unsigned long long)cell_n[s][m][l],
                           cell_s1[s][m][l], cell_s2[s][m][l]);
    return 0;
}
