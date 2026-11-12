# Data release

Evidence for *Disclosure-Constrained Prompt Optimization for Local LLMs: A Controlled Study on Synthetic CVE Applicability Rules*. Every table and figure in the paper is regenerated from this directory by `../harness/scripts/make_tables.py`, and the analyses added in the round 1 revision by the scripts listed in `../harness/README.md` (section 5c). Raw provider requests and responses are not included; each section's `PROVENANCE.json` names the source run, its manifest hash, and the journal totals (calls, running time, hosted cost) derived from the unpublished raw records.

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
| `followups/naive-recheck-granite/`, `followups/naive-recheck-qwen/` | Round 1 revision: the naive prompt evaluated again on all three sealed panels before the revision runs; every test-panel answer equals the campaign's (`scripts/verify_pinned_stack.py`). |
| `followups/local-self-optimization/` | Round 1 revision: each Executor's own model as Optimizer at Tier 3, five repetitions per Executor, paired with the campaign by repetition. |
| `followups/shuffled-abstraction/` | Round 1 revision: Granite, Tier 2 feedback with each relational field permuted across the records of the request, five repetitions, on a new GLM-5.3 endpoint (the campaign's was no longer offered). |
| `followups/shuffled-abstraction-plain-tier2/` | Round 1 revision: the plain Tier 2 arm on the same new endpoint, the shuffled arm's primary comparator. It paused once on a DNS failure and was resumed without repeating committed work. |
| `followups/real-record-panel/`, `followups/real-record-panel-hosted/` | Round 1 revision: fixed prompts (naive, output scaffold, every changed selected prompt of the main arm) on the Real-Record Panel. The first run paused on its cost cap before the hosted pass (see `pauses/`); the hosted pass ran in the second directory under a higher cap, on the same panel. |
| `real-record-panel/` | The Real-Record Panel: 48 sub-profiles of real System Profiles, each with 23 candidate CVE records as published (`profiles.jsonl`, `advisories.jsonl`), the cells and their matcher labels (`cells.csv`), the build manifest, and the blank labelling sheet. `labels/` holds the hand labels of the undecided cell of each sub-profile, frozen before any model run, with the labelling guide. `label-audit/` holds the blind label audit: sheet, answer key, guide, manifest, adjudication, and the post-hoc label corrections used only for the sensitivity rescoring. |
| `exports/` | Flat tables over all sections: `evaluations.csv` (one row per scored profile), `trajectories.csv` (71 trajectories), `decisions.csv`, `panels.csv`, `disclosures.jsonl` (the exact Optimizer view per proposal), `selected_prompts.jsonl`. |
| `tables/` | Output of `make_tables.py`: the paper's tables as Markdown and CSV, figures as PNG (including the Figure 1 design diagram), and `SUMMARY.json`. |
| `INDEX.json` | Section-to-source-run map. |

Each run section under `followups/` (from release 1.1 on) also has `work-summary.jsonl`: per committed work item, the attempt count, worker seconds, model-load seconds and token counts, derived from the unpublished journal.

## Record format

Files under record folders are JSON envelopes `{"payload": ..., "sha256": ...}` where the hash is over the canonical JSON of the payload (UTF-8, sorted keys, compact separators). Some payloads keep absolute paths from the machine the study ran on (for example in `benchmark/lineage.json`); they are provenance only and are left unchanged because they are covered by the hashes. For the same reason, the `adr` fields in the Real-Record Panel and label-audit manifests point to the authors' unpublished design record and are left as written. Flat exports in `exports/` are plain CSV or JSONL for direct loading.

## Metrics

Micro-F1, precision and recall pool every CVE decision across profiles. The **failure-aware** variant counts an invalid or missing Executor answer as every true candidate missed and every false candidate flagged; the **valid-answer** variant scores only parsed answers. Coverage is the share of parsed answers. Both variants are present in every panel summary and in `exports/trajectories.csv` (`test_valid_f1`, `test_coverage`).

## Partitions

`pilot_development` (72) and `pilot_validation` (48) for screening and calibration; `optimization` (240) for feedback batches; `validation` (96) for selection between the start prompt and a trajectory's final prompt; `test` (192), `temporal` (96, publication year 2025) and `product_heldout` (96, disjoint product groups) as sealed panels evaluated once per selected prompt. CVEs never cross partitions.

## Analyses

`tables/table_primary_contrast.*` reports the paired tier contrasts on the original five campaign repetitions (the prespecified primary analysis) and on all eight Granite repetitions including the extension, each with the percentile bootstrap over repetitions, a paired-t sensitivity interval and the sample SD. `campaign/reports/analysis.json` holds the campaign-time analysis with the profile-level bootstrap.

## Errata

- The `reviewer` field of `sources/admission-source-review.json` (and its copy `benchmark/inputs/source-review.json`) names the wrong AI tool for the 26-record source review; the paper names the tool that was used. The files are unchanged because their hashes are part of the admission and benchmark manifests.

## Changes in release 1.1

Release 1.1 accompanies the round 1 revision of the paper. It adds the revision's follow-up runs, the Real-Record Panel with its labels and audit, `work-summary.jsonl` per new run, and the matching harness code and analysis scripts. The records of release 1.0 are unchanged; `exports/` and `INDEX.json` were rebuilt to include the new sections (the rows of release 1.0 are unchanged). In `tables/`, the 2025-records and disjoint-product panels are labelled as in the revised paper (the data keys stay `temporal` and `product_heldout`), the revision runs are added to the chronology, cost and trajectory tables, and the figures were re-rendered. The supplementary figures and tables of the revised paper are in `../supplementary/`.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). NVD content is provided by the U.S. National Institute of Standards and Technology; CVE Program records are provided by the CVE Program and its CNAs under their terms. Model outputs are the outputs of the named models under their providers' terms.

## Citation

Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-Constrained Prompt Optimization for Local LLMs: A Controlled Study on Synthetic CVE Applicability Rules. 2026.

This release: Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-tiered prompt optimization: code and data. Version v1.1.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.24167238
