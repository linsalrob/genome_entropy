# Encoding-budget estimation

`--encoding-size` is an approximate padded amino-acid budget for one structural-state inference batch. A value that is too large can exhaust accelerator memory; a very small value increases overhead. It is not an exact model-token count.

## CLI

```bash
genome_entropy estimate-tokens
genome_entropy estimate-tokens \
  --model gbouras13/modernprost-50M \
  --device cuda --start 3000 --end 10000 --step 1000 --trials 3
```

The command generates random standard-amino-acid sequences, tests increasing combined lengths with the selected encoder, and recommends 90% of the largest length for which every trial succeeds. Defaults are start 3000, end 10000, step 1000, three trials, and approximate individual protein length 100.

This is real inference: it loads or downloads the model and consumes compute. Run it on the same model, PyTorch build, device type, and memory allocation intended for production. Synthetic sequences do not cover every real workload, so retain a safety margin and monitor representative runs.

## Python API

```python
from genome_entropy.encode3di import (
    ModernProstThreeDiEncoder,
    estimate_token_size,
)

encoder = ModernProstThreeDiEncoder(
    model_name="gbouras13/modernprost-50M",
    device="cuda",
)
result = estimate_token_size(
    encoder,
    start_length=3000,
    end_length=10000,
    step=1000,
    num_trials=3,
    base_protein_length=100,
)
print(result["recommended_token_size"])
```

The result contains `max_length`, `recommended_token_size`, `trials_per_length`, and `device`. See `examples/token_estimation_example.py` for a deliberately non-executing demonstration; uncommenting inference may download a model.
