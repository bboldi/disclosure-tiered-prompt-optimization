# Disclosure-tiered prompt optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23044381.svg)](https://doi.org/10.5281/zenodo.23044381)

Code and data for the paper *Automatic Iterative Prompt Optimization for Local LLMs: Disclosure Tiers in CVE Applicability Matching* by Boldizsár Bednárik, László Gogolák and József Sárosi.

In the study, a hosted model improves the instructions (the prompt) of a smaller model that runs on local hardware. The local model decides which supplied CVE advisories apply to a software inventory. A **disclosure tier** controls what the hosted model may learn about the local model's mistakes:

1. **Tier 1:** aggregate metrics and error-category counts only.
2. **Tier 2:** locally computed relational abstractions of each error, with no names, versions or text.
3. **Tier 3:** verbatim failing examples.

Every request and response was written to an append-only journal, and every number in the paper can be regenerated from the records in this repository.

## Contents

| Folder | What it holds | License |
|---|---|---|
| [`harness/`](harness/) | The experiment code, tests, model configuration and run instructions. | MIT |
| [`data/`](data/) | Benchmark 07, the source-corroboration recipe, the calibration gate, the campaign and its extension, the follow-up runs, flat CSV/JSONL exports, and the paper's tables and figures. | CC BY 4.0 |

Start with [`harness/README.md`](harness/README.md) for the code and [`data/README.md`](data/README.md) for the layout and record format of the data.

## Regenerate the paper's tables and figures

This needs no GPU, no model and no API key. It reads only the files in `data/`.

```sh
cd harness
python3.13 -m venv .venv && . .venv/bin/activate   # or: uv venv -p 3.13 && . .venv/bin/activate
pip install -e ".[analysis]"
python scripts/make_tables.py --data ../data --out ../tables-check
```

The Markdown and CSV tables in `tables-check/` should match `data/tables/` exactly. Figures are visually equivalent; they are byte-identical only with the pinned plotting versions in `harness/uv.lock`.

## Rerun the study

The full pipeline runs from source download to analysis; for exact reproduction start from the released frozen benchmark in `data/benchmark/` (see the note in `harness/README.md`). It needs Linux, Python 3.13, [Ollama](https://ollama.com) serving the local models, a GPU with about 32 GB of memory, and an [OpenRouter](https://openrouter.ai) API key for the hosted models. The steps, commands and configuration are in [`harness/README.md`](harness/README.md).

Hosted endpoints change prices and availability over time, so a rerun gives new hosted samples, not the published ones. Locally served models are not bit-reproducible across serving stacks either. The published records in `data/` are the reference.

## What is not included

- **The raw request and response journal** of about 7 GB. Each section's `PROVENANCE.json` names its source run, manifest hash and journal totals.
- **The raw NVD feed and CNA record snapshots.** The harness downloads them again at the recorded feed dates and repository revision.

## Citation

> Bednárik, B., Gogolák, L., Sárosi, J. Automatic Iterative Prompt Optimization for Local LLMs: Disclosure Tiers in CVE Applicability Matching. 2026.

The code and data themselves, as cited in the paper (release v1.0.0):

> Bednárik, B., Gogolák, L., Sárosi, J. Disclosure-tiered prompt optimization: code and data. Version v1.0.0. Zenodo, 2026. https://doi.org/10.5281/zenodo.23044381

## License

The code in `harness/` is under the MIT License, and the data in `data/` is under CC BY 4.0. See [`LICENSE`](LICENSE) for details and third-party terms.
