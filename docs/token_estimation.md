# Encoding-budget estimation

The maintained guide is [`docs/source/token_estimation.md`](source/token_estimation.md) and is published as [Encoding-budget estimation](https://genome-entropy.readthedocs.io/en/latest/token_estimation.html).

`encoding_size` is an approximate padded amino-acid budget, not an exact tokenizer count. The estimator runs real model inference and recommends 90% of the largest successful synthetic workload. Measure on the same model and accelerator intended for production.
