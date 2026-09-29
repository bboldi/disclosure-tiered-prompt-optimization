# Data release

Evidence for *Automatic Iterative Prompt Optimization for Local LLMs: Disclosure Tiers in CVE Applicability Matching*. Every table and figure in the paper is regenerated from this directory by `../harness/scripts/make_tables.py`. Raw provider requests and responses are not included; each section's `PROVENANCE.json` names the source run, its manifest hash, and the journal totals (calls, running time, hosted cost) derived from the unpublished raw records.

## Layout

| Path | Content |
|---|---|
| `benchmark/` | Benchmark 07: `profiles.jsonl` (840 synthetic inventories with hidden labels), `advisories.jsonl`, full selected NVD and CNA source records, `partitions.json`, `dataset-statistics.json`, `DATA_CARD.md`, and the oracle, split and CNA-decision audits. |
| `sources/` | Source recipe: NVD feed metadata and receipts (2023 to 2025), the cvelistV5 repository revision used for CNA records, the candidate-population procedure, and the complete machine-corroboration outcomes with every exclusion and the textual source review. Raw feed and CNA snapshots are not included; the harness re-downloads them at the recorded revision. |
| `calibration/` | Calibration gate on benchmark 07: per-condition evaluations for six local conditions with naive and expert prompts, the hosted decidability reference, and the gate report. `history/` keeps the summary reports of the two earlier benchmark versions that failed the gate. |
| `campaign/` | The campaign: 36 trajectories (30 main across two Executors, three tiers and five repetitions; 3 GEPA; 3 DeepSeek reference), every round decision, every Optimizer request (`disclosures/`), every evaluation, selection and sealed-panel summary, the GEPA engine records, hosted generation metadata, the analysis export, and `lineage/` with the plans, pauses and operator notes of the three earlier continuation directories. |
| `campaign-extension/` | The prospectively planned extension: Granite repetitions 6 to 8 at all three tiers with the campaign's schedule generator and settings, in the same record layout as `campaign/`; `lineage/` holds the six earlier continuation directories (cost gate, two tariff changes, two replay fixes) with their plans, pauses and journal totals. |
| `followups/ceiling-opus-and-glm-reka/` | Claude Opus 5 doing the task with the naive prompt on all three sealed panels; GLM-5.3 on the Reka endpoint on the product held-out panel (its remaining panels were blocked by the provider's firewall; see `pauses/`). |
| `followups/ceiling-glm-akashml/` | GLM-5.3 doing the task on all three panels via the AkashML endpoint. |
| `followups/opus-optimizer/` | Claude Opus 5 as Tier 2 Optimizer, both Executors, three repetitions on the campaign's first three batch schedules. |
| `followups/scaffolding-ablation/` | Five fixed prompt variants per Executor on the 192 sealed test profiles, 1,920 evaluations, with paired token and latency differences. |
| `followups/naive-local/` | Both local Executors with the naive prompt on all three sealed panels at the campaign output cap. |
| `followups/rule-baseline/` | The input-matched deterministic rule baseline on all three sealed panels: per-profile evaluations, panel summaries, the parse rules (`manifest.json`) and parse coverage on the optimization partition (`development.json`). |
| `exports/` | Flat tables over all sections: `evaluations.csv` (one row per scored profile), `trajectories.csv` (51 trajectories), `decisions.csv`, `panels.csv`, `disclosures.jsonl` (the exact Optimizer view per proposal), `selected_prompts.jsonl`. |
| `tables/` | Output of `make_tables.py`: the paper's tables as Markdown and CSV, figures as PNG (including the Figure 1 design diagram), and `SUMMARY.json`. |
| `INDEX.json` | Section-to-source-run map. |

## Record format

Files under record folders are JSON envelopes `{"payload": ..., "sha256": ...}` where the hash is over the canonical JSON of the payload (UTF-8, sorted keys, compact separators). Some payloads keep absolute paths from the machine the study ran on (for example in `benchmark/lineage.json`); they are provenance only and are left unchanged because they are covered by the hashes. Flat exports in `exports/` are plain CSV or JSONL for direct loading.

## Metrics

Micro-F1, precision and recall pool every CVE decision across profiles. The **failure-aware** variant counts an invalid or missing Executor answer as every true candidate missed and every false candidate flagged; the **valid-answer** variant scores only parsed answers. Coverage is the share of parsed answers. Both variants are present in every panel summary and in `exports/trajectories.csv` (`test_valid_f1`, `test_coverage`).

## Partitions

`pilot_development` (72) and `pilot_validation` (48) for screening and calibration; `optimization` (240) for feedback batches; `validation` (96) for selection between the start prompt and a trajectory's final prompt; `test` (192), `temporal` (96, publication year 2025) and `product_heldout` (96, disjoint product groups) as sealed panels evaluated once per selected prompt. CVEs never cross partitions.

## Analyses

`tables/table_primary_contrast.*` reports the paired tier contrasts on the original five campaign repetitions (the prespecified primary analysis) and on all eight Granite repetitions including the extension, each with the percentile bootstrap over repetitions, a paired-t sensitivity interval and the sample SD. `campaign/reports/analysis.json` holds the campaign-time analysis with the profile-level bootstrap.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). NVD content is provided by the U.S. National Institute of Standards and Technology; CVE Program records are provided by the CVE Program and its CNAs under their terms. Model outputs are the outputs of the named models under their providers' terms.

## Citation

Bednárik, B., Gogolák, L., Sárosi, J. Automatic Iterative Prompt Optimization for Local LLMs: Disclosure Tiers in CVE Applicability Matching. 2026.

This release: Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-tiered prompt optimization: code and data. Version v1.0.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.23044381
