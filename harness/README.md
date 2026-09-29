# Privacy-bounded prompt optimization harness

Code for the experiments in *Automatic Iterative Prompt Optimization for Local LLMs: Disclosure Tiers in CVE Applicability Matching*. A hosted **Optimizer** model rewrites the instruction of a local **Executor** model that matches a software inventory against supplied CVE advisories; a **disclosure tier** controls what the Optimizer may learn about the Executor's mistakes. Everything the harness does is written to an immutable, resumable file journal so that every reported number traces back to a stored request and response.

The companion data release (`../data/`) contains the benchmark, the source-corroboration recipe, and the scientific records of every run in the paper. `scripts/make_tables.py` regenerates all tables and figures from it without any model calls.

## Requirements

- Linux, Python 3.13.
- [Ollama](https://ollama.com) serving the local Executor models (the study used Ollama 0.32.13 and Q4_K_M tags; see `models.toml`). An NVIDIA GPU with 32 GB of memory ran 27B to 31B models at 1.5 to 2 seconds per call.
- An [OpenRouter](https://openrouter.ai) API key for hosted Optimizer and reference models. Put it in `.env` as `OPENROUTER_API_KEY=...` (copy `.env.example`). The harness reads only that assignment and never writes the key to disk.
- Optional: the pinned GEPA engine for the established-optimizer baseline, `pip install -e ".[gepa]"`; matplotlib for figures, `pip install -e ".[analysis]"`.

```sh
python3.13 -m venv .venv && . .venv/bin/activate   # or: uv venv -p 3.13 && . .venv/bin/activate
pip install -e ".[analysis]"          # add ,gepa for the GEPA arm
pip install -r requirements-dev.lock  # linters, type checker, security scanner (optional)
cp .env.example .env                  # then paste your key
python scripts/validate.py            # runs the full offline test suite, ~5 minutes
```

## Configuration

`models.toml` declares the Ollama and OpenRouter hosts, the local tags to screen, the hosted models with one pinned endpoint each, the calibration conditions, and the ablation executors. Every run copies this file into its `inputs/` for provenance, and a run refuses to resume if the code, the Python version, or a frozen input changed. Set `PROMPTBENCH_CONFIG=/path/to/models.toml` to use a different file.

## Reproducing the paper

All commands run from this directory with the virtual environment active. Every stage writes one self-contained run directory; pass a new directory name each time. Interrupt any stage with Ctrl-C or `kill`; rerun the same `resume.py` inside the run directory to continue without repeating committed work.

### 1. Model registry

```sh
python -m promptbench.live preflight --run-dir runs/preflight-01 --env-file .env
```

Read-only: records Ollama version, model digests, templates and parameters; OpenRouter catalog entries, endpoints and prices; and current key usage. No inference.

### 2. Sources and benchmark

```sh
# NVD annual feeds (2023, 2024, 2025), verified and retained
python -m promptbench.benchmark fetch --run-dir runs/sources-01

# Candidate population: supported OR-only numeric-version advisories
# (procedure documented in ../data/sources/candidates-REPRODUCE.md)

# Official CNA records for every candidate at one recorded cvelistV5 revision
python -m promptbench.benchmark.cna_sources --run-dir runs/cna-01 --candidate-dir runs/candidates-01

# Machine corroboration of NVD against CNA, with every exclusion retained
python -m promptbench.benchmark.admission --run-dir runs/admission-01 \
    --candidate-dir runs/candidates-01 --cna-dir runs/cna-01 --parent-dir runs/sources-01

# Difficulty-controlled profiles with sealed partitions
python -m promptbench.benchmark build --run-dir runs/benchmark-01 --source-dir runs/sources-01 \
    --admission-dir runs/admission-01 --review-file runs/admission-01/source-review.json
python -m promptbench.benchmark audit --run-dir runs/benchmark-01
```

The build is deterministic for a given source snapshot, seed and settings. The audit re-derives every label with an independent matcher and checks partition disjointness.

**Supported starting point.** The released benchmark in `../data/benchmark/` is the exact artifact used in the paper, and it is the supported starting point for reproduction: point later stages at that directory instead of building. The commands above document how it was made. They are not a turnkey rebuild: the candidate population is a documented procedure, not a command; the textual source review and the original partition map are inputs from the original run; and current NVD feeds differ from the recorded snapshots.

### 3. Calibration gate

```sh
python -m promptbench.live.calibrate prepare --run-dir runs/calibration-01 \
    --benchmark-dir ../data/benchmark --registry-dir runs/preflight-01 \
    --expert-prompt fixtures/expert_prompt.txt --env-file .env
python runs/calibration-01/resume.py
```

Screens every calibration condition with the naive and the historical expert prompt on the pilot-development partition, plus one hosted Executor as a decidability reference, and applies the frozen gate: naive F1 between 0.30 and 0.70 for at least two Executor families, hosted reference at or above 0.90, nonzero Tier 2 feedback in at least 80% of batches, output validity at or above 95%. A failed gate means the benchmark difficulty must change; the criteria do not.

### 4. Campaign

```sh
python -m promptbench.live.study prepare --run-dir runs/campaign-01 --phase phase3 \
    --benchmark-dir ../data/benchmark --registry-dir runs/preflight-01 \
    --calibration-dir runs/calibration-01 --env-file .env \
    --optimizer z-ai/glm-5.3 --reference-optimizer deepseek/deepseek-v4-pro \
    --tiers 1 2 3 --repetitions 5 --depth 8 --candidates 2 --batch-size 24 \
    --gepa-repetitions 3 --reference-repetitions 3 --max-hours 72 --max-cost-usd 15
python runs/campaign-01/resume.py
```

Trajectories are `arm/executor/T<tier>/R<repetition>`. Repetition *r* changes only the seeded feedback-batch schedule (shared across tiers and executors) and the hosted sampling seed where the endpoint supports one; Executors run at temperature 0 with a fixed seed. Selection uses the validation partition; each selected prompt is evaluated once on the three sealed panels. The paper's campaign took 34 running hours on one workstation and about USD 5 of hosted calls.

If a run pauses (time or cost cap, provider tariff change, identity change), `pauses/` states why. Continue under a recorded change with:

```sh
python -m promptbench.live.study continue --run-dir runs/campaign-02 --parent-run runs/campaign-01 \
    --max-hours 40 [--max-cost-usd 30] [--accept-tariff MODEL] [--refresh-runtime] --reason "..."
python runs/campaign-02/resume.py
```

After completion, fetch hosted generation metadata offline: `python -m promptbench.live.study reconcile --run-dir runs/campaign-02`.

### 5. Reference rows and ablation

```sh
# Models doing the task themselves with the naive prompt (hosted ceiling, local naive baselines)
python -m promptbench.live.ceiling prepare --run-dir runs/ceiling-01 --benchmark-dir ../data/benchmark \
    --registry-dir runs/preflight-01 --env-file .env --models anthropic/claude-opus-5 z-ai/glm-5.3 \
    [--endpoint z-ai/glm-5.3=akashml/fp8]
python -m promptbench.live.ceiling prepare --run-dir runs/naive-01 --benchmark-dir ../data/benchmark \
    --registry-dir runs/preflight-01 --env-file .env --models qwen3.8:27b/off granite4.2:30b/off --output-tokens 2048
python runs/ceiling-01/resume.py && python runs/naive-01/resume.py

# Fixed-prompt scaffolding ablation on the sealed test panel
python -m promptbench.live.scaffolding prepare --run-dir runs/ablation-01 --benchmark-dir ../data/benchmark \
    --registry-dir runs/preflight-01 --calibration-dir runs/calibration-01 --env-file .env
python runs/ablation-01/resume.py

# Input-matched deterministic rule baseline (no model calls; seconds)
python -m promptbench.baseline --benchmark-dir ../data/benchmark --run-dir runs/rule-baseline-01
```

The rule baseline parses exactly what the Executor sees (the rendered inventory text and each advisory's affected clause), normalizes names, compares dotted numeric versions, and writes panel summaries in the same format as the reference rows. Its rules were fixed on the optimization partition; `development.json` in the run directory records parse coverage there.

### 5b. Prospectively planned repetition extension

```sh
python -m promptbench.live.study prepare --run-dir runs/extension-01 --phase phase3 \
    --benchmark-dir ../data/benchmark --registry-dir runs/preflight-01 \
    --calibration-dir runs/calibration-01 --env-file .env \
    --only-executors granite4.2:30b/off --tiers 1 2 3 --repetitions 8 --first-repetition 6 \
    --gepa-repetitions 0 --reference-repetitions 0 --max-hours 14 --max-cost-usd 10
python runs/extension-01/resume.py
```

`--first-repetition N` runs repetitions N..`--repetitions` only. The seeded schedule generator produces the same batch schedules for lower indices, so the new repetitions are exactly the ones a longer original campaign would have run, and they pair with the original repetitions by index.

### 6. Tables and figures

```sh
python scripts/export_release.py --out release/data --benchmark runs/benchmark-01 ... --campaign runs/campaign-02 \
    [--lineage runs/campaign-01] [--extension runs/extension-01] --followup rule-baseline=runs/rule-baseline-01 ...
python scripts/make_tables.py --data ../data --out ../data/tables
```

`export_release.py --help` lists every argument. `--extension` releases a repetition-extension run as `campaign-extension`; `make_tables.py` then reports the paired contrasts on the original repetitions and on all repetitions, both labelled.

`make_tables.py` reads only the flat exports and record folders in the data release and writes every table (Markdown and CSV) and figure (PNG, plus vector PDF for the figures in the paper) of the paper. Run it on `../data/` to reproduce the published numbers exactly. The Markdown and CSV tables match byte for byte; the published figures were rendered with matplotlib 3.11.2 (pinned in `uv.lock`) and its bundled DejaVu fonts, and other versions give visually equivalent but not byte-identical images.

## Operations, one line each

- `python -m promptbench.live smoke ...`: three-profile connectivity check with one hosted proposal.
- `python -m promptbench.live.soak prepare|run ...`: long local recovery soak with injected interruptions.
- `python -m promptbench.live.pilot prepare|run ...`: short exploratory model-selection pilot (Tier 2 only).
- `python -m promptbench.live.reconstruct --run-dir R --output-dir NEW`: rebuild attempt and evaluation exports offline from raw responses.
- `python -m promptbench.benchmark audit --run-dir B`: recheck labels and partitions of a built benchmark.
- `python -m promptbench` : the fake-provider fixture protocol used by the test suite.

## Journal layout

Each run directory contains `plan.json` and `manifest.json` (frozen design and input hashes), `inputs/` (profiles, advisories, prompts, conditions, model registry, configuration, source snapshot), `runtime/` (the exact code that executes on resume), `work/<spec>/` (specification, every attempt's request and raw response, one committed result), `evaluations/`, `disclosures/` (the full Optimizer request per proposal), `decisions/`, `trajectories/`, `selections/`, `sealed-panels/`, `sessions/` (cumulative clock), `telemetry/`, `pauses/`, and `reports/`. JSON records carry a SHA-256 of their canonical payload. Nothing in a run directory is edited after it is written.

## License

MIT. The data release is licensed separately (CC BY 4.0); NVD and CVE Program content within it remains subject to its own terms.
