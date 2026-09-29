**Reference rows on the sealed panels: hosted and local models with the unchanged naive prompt (one pass), and the input-matched deterministic rule baseline that parses the same rendered text.**

| Model | Run | Test F1 / recall | Temporal F1 / recall | Held-out F1 / recall | Coverage | Temperature 0 |
|---|---|---|---|---|---|---|
| GLM-5.3 (hosted) | ceiling-glm-akashml | 0.880 / 0.935 | 0.830 / 0.974 | 0.930 / 1.000 | 1.000, 0.990, 1.000 | yes |
| Claude Opus 5 (hosted) | ceiling-opus-and-glm-reka | 0.970 / 0.997 | 0.956 / 1.000 | 0.971 / 1.000 | 1.000, 1.000, 1.000 | no |
| GLM-5.3 (hosted) | ceiling-opus-and-glm-reka | — | — | 0.899 / 0.987 | 1.000 | yes |
| Granite 4.2 30B, naive prompt | naive-local | 0.335 / 0.558 | 0.320 / 0.458 | 0.311 / 0.494 | 0.938, 0.969, 0.958 | yes |
| Qwen 3.8 27B, naive prompt | naive-local | 0.763 / 0.790 | 0.780 / 0.791 | 0.798 / 0.874 | 1.000, 1.000, 1.000 | yes |
| Rule baseline (deterministic) | rule-baseline | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000, 1.000, 1.000 | n/a (deterministic) |
