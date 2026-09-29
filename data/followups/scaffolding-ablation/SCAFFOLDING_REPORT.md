# Task 3 fixed scaffolding controls

Completed 1,920 paired local evaluations on all 192 sealed-test profiles of benchmark 07: two gate-promoted Executors, five frozen variants, no search. All 1,920 raw outcomes and ten variant reports reconstruct identically; no retries or unknown attempts. Completed replay adds zero inference.

| Executor | Variant | F1 (failure-aware) | Recall | Valid / 192 | Mean output tokens | Extra output tokens / profile | Mean latency (s) | Extra latency / profile (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3.8:27b/off | naive | 0.7702 | 0.7947 | 192 | 57.08 | +0.00 | 0.919 | +0.000 |
| qwen3.8:27b/off | output_scaffold | 0.8029 | 0.8523 | 192 | 624.03 | +566.94 | 4.058 | +3.140 |
| qwen3.8:27b/off | structured_inventory | 0.7689 | 0.7916 | 192 | 56.30 | -0.78 | 1.150 | +0.232 |
| qwen3.8:27b/off | schema_enum | 0.7680 | 0.7900 | 192 | 56.85 | -0.23 | 0.881 | -0.037 |
| qwen3.8:27b/off | neutral | 0.7394 | 0.7963 | 192 | 59.90 | +2.82 | 1.340 | +0.422 |
| granite4.2:30b/off | naive | 0.3494 | 0.5863 | 182 | 77.78 | +0.00 | 1.447 | +0.000 |
| granite4.2:30b/off | output_scaffold | 0.4546 | 0.7481 | 174 | 568.85 | +491.07 | 8.180 | +6.733 |
| granite4.2:30b/off | structured_inventory | 0.2968 | 0.6936 | 170 | 156.36 | +78.58 | 2.562 | +1.115 |
| granite4.2:30b/off | schema_enum | 0.3306 | 0.5614 | 183 | 80.09 | +2.32 | 1.280 | -0.167 |
| granite4.2:30b/off | neutral | 0.3087 | 0.4619 | 179 | 104.72 | +26.95 | 2.143 | +0.696 |

Differences use the same profiles and the unchanged naive control within each Executor. All outcomes, including invalid answers, remain in failure-aware metrics; valid-answer metrics, precision, per-stratum F1/recall, p90 and every raw paired difference are in `exports/summary.json` and `reports/complete.json`. No control result changes the benchmark, prompts, calibration gate or family promotions.

The output scaffold increases failure-aware F1 in this fixed comparison, with substantial additional output and latency. This is fixed-control evidence, not a feedback-tier optimization result or a non-inferiority finding. Qwen has valid output on every control. Granite retains 72 invalid outputs across the five controls: naive 10, neutral 13, scaffold 18, enum 9 and structured inventory 22. Its naive sealed-control coverage is 94.79%; calibration passed on its separate 48-profile development sample. Flag this generalization/format limitation without tuning on these test outcomes.

All five controls use thinking off, temperature 0, seed 11, context 16,384 and output cap 4,096. Calibration-off used cap 512 and different profiles, so its scores are not directly interchangeable with these results. The minimal scaffold emits only ID/boolean decisions, not v1's full reasoning fields. The neutral addition matches 271 ASCII characters/bytes, not tokenizer counts. Provider-reported mean extra input tokens are:

| Executor | Scaffold addition | Neutral addition |
|---|---:|---:|
| qwen3.8:27b/off | +64.00 | +44.00 |
| granite4.2:30b/off | +66.00 | +45.00 |

Latency includes the committed HTTP attempt, including model loading when applicable; raw load durations and all physical-attempt durations remain available. Variant order rotates by profile index within fixed model blocks. Character matching and deterministic order do not eliminate every tokenizer/cache/order effect.

Dedicated-key spend was USD 0.507254571 before and after, including the original pilot; this block adds zero hosted inference and zero hosted cost. Final charged control time is 5200.348797 seconds (1.4445 hours); cumulative charged preparation time is 27068.734708 seconds (7.5191 hours). These clocks conservatively include controller/metadata overhead. Sampled peak VRAM is 24,035 MiB; observed whole-device energy is 2,535,721.407 J over 5,199.571 seconds, without idle subtraction or exclusive attribution. No GPU sensor samples are missing.

Evidence: `manifest.json`, `pre-execution-audit.json`, `reports/complete.json`, `exports/summary.json`, `exports/summary.csv`, `scientific-integrity-audit.json`, `completed-replay-audit.json`, `resource-closeout.json`, `operator.log`, and `telemetry/`. Raw-response reconstruction: `../reconstruction-20260913T162800831657Z/report.json` with `attempts.jsonl`, `evaluations.jsonl` and the auditor source snapshot in `lineage.json`. The live source remains frozen; later offline audit support adds only the already-declared scaffold parser flag. See the final Task 3 closeout for completed engineering validation.

Stop after Task 3. Task 4 remains unassigned, Task 5 has not run, and Phase 3 is not admitted.
