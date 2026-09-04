/* Full-population entropy aggregator for genome_entropy GTDB entropy_rows TSVs.
 *
 * Reads per-ORF TSV on stdin (with header), stratifies every row by
 * (genome has >=1 deposited CDS, in_genbank) and accumulates exact counts,
 * sums, sums of squares and fixed-width histograms for the four entropy
 * columns.  Histograms are 1e-4 wide over [0, 4.4), which bounds any
 * quantile read from them to +/-5e-5.
 *
 * usage: agg <annotated_genomes.txt> < rows.tsv > partial.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NBIN   44000
#define BINW   1e-4
#define NMET   5            /* 3di, protein, 12st, mi, dna */
#define NSTRAT 6            /* (ann,gb) 2x2, plus 2 unknown-annotation */

static uint64_t n[NSTRAT];
static double   s1[NSTRAT][NMET], s2[NSTRAT][NMET];
static double   mn[NSTRAT][NMET], mx[NSTRAT][NMET];
static uint64_t aa_n[NSTRAT];
static double   aa_s1[NSTRAT];
static uint64_t *hist;      /* [NSTRAT][NMET][NBIN] */
static uint64_t nan_ct[NSTRAT][NMET];
static uint64_t oob_ct[NSTRAT][NMET];

/* ---- open-addressing hash set of annotated genome accessions ---- */
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
    if (!hkey || !hval) { fprintf(stderr, "agg: calloc hash failed\n"); exit(2); }
}
static void hset_add(const char *k, char v) {
    size_t i = fnv(k) & (hcap - 1);
    while (hkey[i]) { if (!strcmp(hkey[i], k)) { hval[i] = v; return; } i = (i + 1) & (hcap - 1); }
    hkey[i] = strdup(k); hval[i] = v;
}
/* 1 = genome has >=1 deposited CDS, 0 = has none, -1 = genome not in table */
static int hset_get(const char *k) {
    size_t i = fnv(k) & (hcap - 1);
    while (hkey[i]) { if (!strcmp(hkey[i], k)) return hval[i]; i = (i + 1) & (hcap - 1); }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: agg <annotated_genomes.txt>\n"); return 2; }

    FILE *g = fopen(argv[1], "r");
    if (!g) { perror("agg: annotated genome list"); return 2; }
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

    hist = calloc((size_t)NSTRAT * NMET * NBIN, sizeof(uint64_t));
    if (!hist) { fprintf(stderr, "agg: calloc hist failed\n"); return 2; }
    for (int s = 0; s < NSTRAT; s++)
        for (int m = 0; m < NMET; m++) { mn[s][m] = 1e300; mx[s][m] = -1e300; }

    /* column indices (0-based) in the entropy_rows TSV */
    const int C_GENOME = 2, C_AALEN = 8, C_GB = 9;
    const int col[NMET] = { 12, 11, 13, 14, 10 };   /* 3di, protein, 12st, mi, dna */

    size_t cap = 1 << 16;
    char *line = malloc(cap);
    ssize_t len;
    uint64_t nline = 0, bad = 0;

    /* discard header */
    if ((len = getline(&line, &cap, stdin)) <= 0) { fprintf(stderr, "agg: empty input\n"); return 2; }
    if (strncmp(line, "domain\t", 7) != 0) { fprintf(stderr, "agg: unexpected header: %.40s\n", line); return 2; }

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

        /* strata 0..3 = (has_cds, in_genbank); 4,5 = genome absent from CDS table */
        int ann = hset_get(f[C_GENOME]);
        int gb = (f[C_GB][0] == 'T');
        int st = (ann < 0) ? (4 + gb) : (ann * 2 + gb);

        n[st]++;
        char *ep;
        double a = strtod(f[C_AALEN], &ep);
        if (ep != f[C_AALEN]) { aa_n[st]++; aa_s1[st] += a; }

        for (int m = 0; m < NMET; m++) {
            char *v = f[col[m]];
            if (!*v) { nan_ct[st][m]++; continue; }
            double x = strtod(v, &ep);
            if (ep == v || x != x) { nan_ct[st][m]++; continue; }
            s1[st][m] += x; s2[st][m] += x * x;
            if (x < mn[st][m]) mn[st][m] = x;
            if (x > mx[st][m]) mx[st][m] = x;
            long b = (long)(x / BINW);
            if (b < 0 || b >= NBIN) { oob_ct[st][m]++; if (b < 0) b = 0; else b = NBIN - 1; }
            hist[((size_t)st * NMET + m) * NBIN + b]++;
        }
    }

    static const char *mname[NMET] = { "three_di", "protein", "twelve_state", "mi", "dna" };
    printf("#agg v1 nbin=%d binw=%.10g\n", NBIN, BINW);
    printf("#lines=%llu malformed=%llu\n",
           (unsigned long long)nline, (unsigned long long)bad);
    for (int s = 0; s < NSTRAT; s++) {
        printf("N\t%d\t%llu\t%llu\t%.17g\n", s,
               (unsigned long long)n[s], (unsigned long long)aa_n[s], aa_s1[s]);
        for (int m = 0; m < NMET; m++)
            printf("S\t%d\t%s\t%.17g\t%.17g\t%llu\t%llu\t%.17g\t%.17g\n", s, mname[m],
                   s1[s][m], s2[s][m],
                   (unsigned long long)nan_ct[s][m], (unsigned long long)oob_ct[s][m],
                   mn[s][m] > 1e299 ? 0.0 / 0.0 : mn[s][m],
                   mx[s][m] < -1e299 ? 0.0 / 0.0 : mx[s][m]);
    }
    for (int s = 0; s < NSTRAT; s++)
        for (int m = 0; m < NMET; m++) {
            uint64_t *h = hist + ((size_t)s * NMET + m) * NBIN;
            for (int b = 0; b < NBIN; b++)
                if (h[b]) printf("H\t%d\t%s\t%d\t%llu\n", s, mname[m], b,
                                 (unsigned long long)h[b]);
        }
    return 0;
}
