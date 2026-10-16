<!-- Draft Methods text, Gadi agent, for agent-handoffs#24. The integrator merges it into genome_entropy.md. -->
<!-- Source: agent-handoffs projects/genome_entropy_logan/inphared_gadi/README.md §5 and tables/determinism_50genomes.txt (commit 4b81cfb); PBS job 181181761. -->

### Numerical reproducibility of the structural encoding

To separate run-to-run variation from the effects of configuration, we re-encoded the first 50 genomes of the Inphared phage set (6,284 ORFs). The encoder was `genome_entropy` 0.3.0 with `gbouras13/modernprost-50M`, run on one NVIDIA V100 at encoding sizes of 10,000 and 20,000 residues per batch. We compared the re-encodings with the production run (V100, 10,000) and with an independent run of the same ORFs on AMD GPUs on Pawsey Setonix (ROCm, fp16, `genome_entropy` 0.2.0, 20,000) (Table G01).

- **Repeat run:** repeating the production configuration reproduced every 3Di entropy, 12-state entropy and 3Di–12-state mutual information value bit for bit.
- **Encoding size:** changing the encoding size altered 0.3–0.7% of ORFs, by at most 0.076 bits.
- **Hardware, precision and package version:** across the two systems only 37.8% of 3Di entropies were bit-identical. The median absolute difference was 0.007 bits, the 99th percentile 0.31 bits and the maximum 0.92 bits (Pearson r = 0.997). Matching the encoding size left this spread unchanged.

The design cannot apportion that residual between hardware, numerical precision and package version, because the second system could not be re-run in the same configuration. The 50 genomes reproduce the whole-population cross-system agreement (p99 0.310 against 0.317 bits; r 0.99719 against 0.99698).

Population-level quantities are unaffected: class means move by about 0.01 bits against a 1.46-bit difference between classes. Per-ORF entropies are reproducible to within a few tenths of a bit across platforms but not bit for bit. Per-ORF thresholds and candidate rankings should therefore be reported together with the hardware, numerical precision, encoding size and software version that produced them.
