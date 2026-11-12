# Disclosure-tiered prompt optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24167238.svg)](https://doi.org/10.5281/zenodo.24167238)

Code and data for the paper *Disclosure-Constrained Prompt Optimization for Local LLMs: A Controlled Study on Synthetic CVE Applicability Rules* by Boldizsár Bednárik, László Gogolák and József Sárosi.

In the study, a hosted model improves the instructions (the prompt) of a smaller model that runs on local hardware. The local model decides which supplied CVE advisories apply to a software inventory. A **disclosure tier** controls what the hosted model may learn about the local model's mistakes:

1. **Tier 1:** aggregate metrics and error-category counts only.
2. **Tier 2:** locally computed relational abstractions of each error, with no names, versions or text.
3. **Tier 3:** verbatim failing examples.

Every request and response was written to an append-only journal, and every number in the paper can be regenerated from the records in this repository.

## Contents

| Folder | What it holds | License |
|---|---|---|
| [`harness/`](harness/) | The experiment code, tests, model configuration and run instructions. | MIT |
| [`data/`](data/) | Benchmark 07, the source-corroboration recipe, the calibration gate, the campaign and its extension, the follow-up runs (including the round 1 revision runs), the Real-Record Panel with its hand labels and label audit, flat CSV/JSONL exports, and the paper's tables and figures. | CC BY 4.0 |
| [`supplementary/`](supplementary/) | Supplementary Figs. S1–S3 and Tables S1–S2 of the revised paper. | CC BY 4.0 |

Start with [`harness/README.md`](harness/README.md) for the code and [`data/README.md`](data/README.md) for the layout and record format of the data.

## Regenerate the paper's tables and figures

This needs no GPU, no model and no API key. It reads only the files in `data/`.

```sh
cd harness
python3.13 -m venv .venv && . .venv/bin/activate   # or: uv venv -p 3.13 && . .venv/bin/activate
pip install -e ".[analysis]"
python scripts/make_tables.py --data ../data --out ../tables-check
```

The Markdown and CSV tables in `tables-check/` should match `data/tables/` exactly. Figures are visually equivalent; byte identity depends on the plotting library and font versions (the versions used are pinned in `harness/uv.lock`).

The analyses added in the round 1 revision (dependence-aware intervals, supplementary metrics, the local self-optimization, Shuffled Abstraction and Real-Record Panel follow-ups, the label audit) also run from `data/` alone; the commands are in section 5c of [`harness/README.md`](harness/README.md).

## Rerun the study

The full pipeline runs from source download to analysis; for exact reproduction start from the released frozen benchmark in `data/benchmark/` (see the note in `harness/README.md`). It needs Linux, Python 3.13, [Ollama](https://ollama.com) serving the local models, a GPU with about 32 GB of memory, and an [OpenRouter](https://openrouter.ai) API key for the hosted models. The steps, commands and configuration are in [`harness/README.md`](harness/README.md).

Hosted endpoints change prices and availability over time, so a rerun gives new hosted samples, not the published ones. Locally served models are not bit-reproducible across serving stacks either. The published records in `data/` are the reference.

## What is not included

- **The raw request and response journal** of about 7 GB. Each section's `PROVENANCE.json` names its source run, manifest hash and journal totals.
- **The raw NVD feed and CNA record snapshots.** The harness downloads them again at the recorded feed dates and repository revision.

## Citation

> Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-Constrained Prompt Optimization for Local LLMs: A Controlled Study on Synthetic CVE Applicability Rules. 2026.

The code and data themselves, as cited in the revised paper (release v1.1.0):

> Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-tiered prompt optimization: code and data. Version v1.1.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.24167238

Release v1.0.0 (https://doi.org/10.5281/zenodo.23044381) is the version cited in the originally submitted paper.

## License

The code in `harness/` is under the MIT License, and the data in `data/` is under CC BY 4.0. See [`LICENSE`](LICENSE) for details and third-party terms.
