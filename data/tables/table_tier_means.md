**Per-tier sealed-panel means across all repetitions (extension included where present), with naive and method references. Valid-answer F1 scores only parsed answers; coverage is the fraction of profiles with a parsed answer.**

| Executor | Arm | Trajectories | Prompt changed | Test F1 (mean ± sample SD) | Test recall | Valid-answer F1 | Coverage | Temporal F1 | Held-out F1 |
|---|---|---|---|---|---|---|---|---|---|
| Granite 4.2 30B | Tier 1 | 8 | 8 | 0.395 ± 0.030 | 0.581 | 0.419 | 0.977 | 0.392 | 0.315 |
| Granite 4.2 30B | Tier 2 | 8 | 7 | 0.362 ± 0.034 | 0.481 | 0.378 | 0.983 | 0.361 | 0.285 |
| Granite 4.2 30B | Tier 3 | 8 | 7 | 0.387 ± 0.032 | 0.477 | 0.421 | 0.971 | 0.361 | 0.302 |
| Granite 4.2 30B | Naive prompt |  |  | 0.335 | 0.558 | 0.386 | 0.938 | 0.320 | 0.311 |
| Qwen 3.8 27B | Tier 1 | 5 | 1 | 0.753 ± 0.022 | 0.802 | 0.753 | 1.000 | 0.757 | 0.783 |
| Qwen 3.8 27B | Tier 2 | 5 | 2 | 0.752 ± 0.016 | 0.802 | 0.752 | 1.000 | 0.764 | 0.777 |
| Qwen 3.8 27B | Tier 3 | 5 | 2 | 0.765 ± 0.004 | 0.808 | 0.765 | 1.000 | 0.776 | 0.788 |
| Qwen 3.8 27B | Naive prompt |  |  | 0.763 | 0.790 | 0.763 | 1.000 | 0.780 | 0.798 |
| Qwen 3.8 27B | GEPA, Tier 2 | 3 | 1 | 0.768 ± 0.010 | 0.793 | 0.768 | 1.000 | 0.770 | 0.789 |
| Qwen 3.8 27B | DeepSeek V4 Pro, Tier 2 | 3 | 0 | 0.763 ± 0.000 | 0.790 | 0.763 | 1.000 | 0.780 | 0.798 |
