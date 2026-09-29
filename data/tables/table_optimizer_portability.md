**Tier 2 optimizer portability on the sealed test panel, repetitions 1 to 3, matched budgets.**

| Optimizer | Executor | Trajectories | Prompt changed | Test F1 per repetition | Mean F1 | Mean recall |
|---|---|---|---|---|---|---|
| GLM-5.3 (main loop) | Granite 4.2 30B | 3 | 3 | 0.335, 0.309, 0.358 | 0.334 | 0.375 |
| GLM-5.3 (main loop) | Qwen 3.8 27B | 3 | 1 | 0.763, 0.748, 0.763 | 0.758 | 0.810 |
| GLM-5.3 via GEPA | Qwen 3.8 27B | 3 | 1 | 0.763, 0.780, 0.763 | 0.768 | 0.793 |
| DeepSeek V4 Pro | Qwen 3.8 27B | 3 | 0 | 0.763, 0.763, 0.763 | 0.763 | 0.790 |
| Claude Opus 5 | Granite 4.2 30B | 3 | 2 | 0.419, 0.335, 0.344 | 0.366 | 0.570 |
| Claude Opus 5 | Qwen 3.8 27B | 3 | 1 | 0.749, 0.763, 0.763 | 0.758 | 0.788 |
