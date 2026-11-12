# Supplementary material: figures and tables removed from the manuscript for length

These figures and tables were part of the revised manuscript of "Disclosure-Constrained Prompt Optimization for Local LLMs: A Controlled Study on Synthetic CVE Applicability Rules" (CMC submission 94508, round 1 revision). They were removed to shorten the paper, because their content is also given in the paper's text or tables. They are kept here unchanged for interested readers. The manuscript refers to them as "Supplementary Fig. S1–S3" and "Supplementary Table S1–S2".

Each figure is given as a vector PDF and as a PNG.

## Supplementary Fig. S1: paired tier differences (`S1_paired_differences.pdf`)

This was Figure 2 of the revised manuscript. Table 3 of the paper gives the means and intervals; this figure also shows the individual paired differences.

Paired differences in failure-aware micro-F1 on the sealed test panel. Dots show the repetition pairs. For each contrast, the solid bar is the mean with the prespecified 95% repetition-bootstrap interval. The dotted bar is the same mean with the 95% interval over repetitions, profiles and CVEs (Section 3.6). The contrasts are:
- Tier 2 − Tier 3 (T2 − T3, prespecified)
- Tier 1 − Tier 2 (secondary)
- Tier 1 − Tier 3 (exploratory)

Panels A and B show the five prespecified repetitions for Qwen and Granite. Panel C shows Granite's eight-repetition extension. The extension is an exploratory analysis, decided after the primary result was known. Negative values favour the second tier. The dashed line is the −0.02 margin, which applies to the Tier 2 − Tier 3 retained-utility comparison.

## Supplementary Fig. S2: operating points in precision–recall space (`S2_precision_recall.pdf`)

This was Figure 3 of the revised manuscript. Its tier means are in Table 4 of the paper.

Operating points on the sealed test panel in precision–recall space, Qwen (A) and Granite (B). The plot shows the following:
- every selected prompt of the campaign and of the Granite extension, coloured by tier
- the naive prompt (hollow)
- the fixed output contract (star)
- the reference rows that perform the task directly (triangles): the hosted models and the deterministic rule baseline

The output limits differ by condition:

| Condition | Output limit (tokens) |
|---|---|
| Fixed output contract (with its own schema) | 4,096 |
| Local naive and selected prompts | 2,048 |
| Hosted models | 4,096 |
| Rule baseline | none |

## Supplementary Fig. S3: micro-F1 by difficulty stratum (`S3_strata_heatmap.pdf`)

This was Figure A2 of the revised manuscript. It is summarized in Section 4.3 of the paper.

Sealed-test micro-F1 by difficulty stratum, Qwen (A) and Granite (B). The grid covers the following conditions:
- the naive prompt
- the tier means over all repetitions (eight for Granite, five for Qwen)
- the fixed output contract (with its own schema)
- the hosted models
- the deterministic rule baseline

The output limits differ by condition:

| Condition | Output limit (tokens) |
|---|---|
| Fixed output contract | 4,096 |
| Local naive and tier rows | 2,048 |
| Hosted models | 4,096 |
| Rule baseline | none |

## Supplementary Table S1: benchmark calibration history

This was Table A1 of the revised manuscript. Section 3.4 of the paper describes the calibration gate.

Naive-prompt failure-aware micro-F1 of the Executors that were eventually promoted, on pilot-development profiles. Qwen and Granite ran with reasoning off.

| Version | Change | Qwen 27B | Granite 30B | Hosted reference | Verdict |
|---|---|---|---|---|---|
| 03 | canonical names, exact versions, structured clauses, 2–8 advisories, schema enum | 1.000 | 0.915 | 1.000 (pilot) | no family in the 0.30–0.70 band |
| 06 | 6–10 aliased components, 15–25 advisories, ≥30% distractors, near-miss versions, 30% prose bounds, no structured inventory | 0.822 | 0.375 | 0.992 | one family in the band |
| 07 | near-miss probability 0.40/0.80/1.00 by stratum, weighted appended-segment and lexicographic traps, package-manager-heavy aliases, 21–25 advisories | 0.664 | 0.373 | 1.000 | **adopted** |

## Supplementary Table S2: all fixed-prompt controls

This was Table 7 of the revised manuscript. The paper keeps the naive and fixed-output-contract rows and gives the F1 of the other variants in the text.

Fixed-prompt controls on the sealed test panel, without optimization:
- All conditions use a 4,096-token output limit. The campaign conditions use 2,048.
- Each variant is paired against the naive variant of the same Executor.
- The fixed output contract changes the instruction, the response schema and the validator together.

| Executor | Fixed prompt | F1 | Recall | Precision | Coverage | Valid-answer F1 | Mean output tokens | Δ latency s |
|---|---|---|---|---|---|---|---|---|
| Qwen 3.8 27B | naive | 0.770 | 0.795 | 0.747 | 1.000 | 0.770 | 57 | |
| Qwen 3.8 27B | naive + fixed output contract | **0.803** | 0.852 | 0.759 | 1.000 | 0.803 | 624 | +3.14 |
| Qwen 3.8 27B | naive + length-matched neutral text | 0.739 | 0.796 | 0.690 | 1.000 | 0.739 | 60 | +0.42 |
| Qwen 3.8 27B | naive + structured inventory | 0.769 | 0.792 | 0.747 | 1.000 | 0.769 | 56 | +0.23 |
| Qwen 3.8 27B | naive + schema enum | 0.768 | 0.790 | 0.747 | 1.000 | 0.768 | 57 | −0.04 |
| Granite 4.2 30B | naive | 0.349 | 0.586 | 0.249 | 0.948 | 0.393 | 78 | |
| Granite 4.2 30B | naive + fixed output contract | **0.455** | 0.748 | 0.327 | 0.906 | 0.569 | 569 | +6.73 |
| Granite 4.2 30B | naive + length-matched neutral text | 0.309 | 0.462 | 0.232 | 0.932 | 0.366 | 105 | +0.70 |
| Granite 4.2 30B | naive + structured inventory | 0.297 | 0.694 | 0.189 | 0.885 | 0.358 | 156 | +1.12 |
| Granite 4.2 30B | naive + schema enum | 0.331 | 0.561 | 0.234 | 0.953 | 0.366 | 80 | −0.17 |

## Provenance

The figures are the ones printed in the revised manuscript. They were generated from `data/` by `harness/scripts/make_tables.py` (S2, S3; written to `data/tables/` as `fig_precision_recall.*` and `fig_strata_heatmap.*`) and by `harness/scripts/fig_paired_deltas.py` (S1, with the intervals of `harness/scripts/revision_analyses.py`). A regeneration gives visually equivalent figures; byte identity depends on the plotting library and font versions. The data rows of the tables are unchanged from the manuscript; the headers of Table S1 were simplified. Table S1's values are in `data/tables/table_calibration.*`, Table S2's in `data/tables/table_scaffolding_ablation.*`.
