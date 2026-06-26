<div align="center">

# Foundations without Fundamentals
### Zero-Shot Blind Spots in Time Series Foundation Models

[![Workshop](https://img.shields.io/badge/ICML%202026-FMSD%20Workshop-7B1FA2)](https://icml-structured-fm-workshop.github.io/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

**Nafiseh Ghoroghchian, Haipeng Zhang, Shuyi Han, Alex Labach, George Stein**

*Layer 6 AI*

[**Paper**](#citation) · [**Poster**](#citation) · [**SimpleTimeBench**](#simpletimebench) · [**Reproducing results**](paper_result_replication.md)

</div>

---

## Overview

Time-series foundation models (TSFMs) post impressive **aggregate** scores on public leaderboards, yet quietly fail on tasks a junior analyst would solve with a ruler. This repository is the official implementation accompanying our paper **“Foundations without Fundamentals: Zero-Shot Blind Spots in Time Series FMs”** (ICML 2026 FMSD Workshop).

We diagnose these failures with **SimpleTimeBench**, a controlled, GIFT-Eval-compatible benchmark of **28 generative processes × 3 evaluation modes** (univariate, multivariate, leading-covariate). Alongside the benchmark we ship:

- A **clean, config-driven pipeline** (`ts_toolkit`) for running zero-shot and fine-tuning experiments across **Chronos-2, Moirai, Toto, Sundial**, classic baselines, and custom models.
- Four **diagnostic metrics** beyond aggregate MASE/CRPS — `relMAE`, `MBR`, `MAR`, `PVR` — designed to expose mean-hedging, amplitude collapse, and path-shape distortions.
- **Real-world covariate stress tests** (Delaware River streamflow, NVIDIA leak test) that confirm synthetic findings transfer to the wild.
- Fully reproducible **GIFT-Eval multi-vs-univariate runs** showing that a 840-parameter DLinear baseline can beat billion-parameter TSFMs when covariates matter.

> **TL;DR.** Like LLMs’ jagged frontier, today’s TSFMs miss primitives where near-perfect forecasts should be trivial — and they barely benefit from perfect leading indicators. Targeted fine-tuning recovers individual primitives but **struggles to retain them jointly**, pointing to pre-training data composition (not architecture alone) as a central bottleneck.

## Table of Contents

1. [Key Findings](#key-findings)
2. [SimpleTimeBench](#simpletimebench)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Quickstart](#quickstart)
7. [Configuration Reference](#configuration-reference)
8. [Reproducing the Paper](#reproducing-the-paper)
9. [Supported Models](#supported-models)
10. [Fine-Tuning Chronos-2](#fine-tuning-chronos-2)
11. [Real-World Case Studies](#real-world-case-studies)
12. [Notebooks Index](#notebooks-index)
13. [Development](#development)
14. [Citation](#citation)
15. [Acknowledgements](#acknowledgements)
16. [License](#license)

## Key Findings

| # | Failure mode | Symptom | Evidence |
| - | --- | --- | --- |
| 1 | **Mean-reversion bias** | Forecasts hedge toward the historical mean on trend & periodic tasks | MBR up to 0.80 (ideal 0.5) for Moirai |
| 2 | **Anti-exponential bias** | Accelerating growth is flattened into an exponential decay-like curve | `MAR > 1` and `PVR > 1` across all three models on `exponential` |
| 3 | **Covariate underutilization** | Perfect future-leaking covariates barely change predictions | Chronos-2 random-walk `relMAE_last`: 1.02 (uni) → 0.98 (lead) |
| 4 | **Capacity bottleneck on joint primitives** | Fine-tuning on a single primitive helps, all-in-one fine-tuning regresses | All-in-one yields only **+4.6% MAPE / +1.4% relMAE** vs. zero-shot Chronos-2 |

These are not pathologies of a single architecture — they appear in **Chronos-2, Moirai, and Toto**, the three TSFMs with native past-covariate support.

## SimpleTimeBench

`SimpleTimeBench` is a diagnostic “unit test” suite for time-series models, released in GIFT-Eval format so it slots directly into existing evaluation harnesses.

### Evaluation Modes

| Mode | What it tests |
| --- | --- |
| **Univariate** | Can the model extrapolate trends, periodicity, and noise from a single channel? |
| **Multivariate** | Does adding an *independent* extra channel from the same generative process degrade performance? (It shouldn’t.) |
| **Leading covariate** | When a past-only covariate exactly reveals the future target, can the model copy-paste? |

### Regime Taxonomy

| Regime | Difficulty | # | Distributions |
| --- | --- | --- | --- |
| **Deterministic** | Trivial | 4 | `constant`, `linear`, `staircase`, `exponential` |
| **Periodic / Harmonic** | Low | 6 | `sine`, `sawtooth`, `square`, `triangle`, `noise_patch`, `fourier` |
| **Evolving periodic** | Moderate | 3 | `damped_sine`, `growing_sine`, `chirp` |
| **I.I.D. white noise** | Impossible | 9 | `normal`, `uniform`, `poisson`, `binary`, `student_t`, `lognormal`, `laplace`, `cauchy`, `skew_normal` |
| **Stochastic correlated** | Impossible | 6 | `random_walk`, `piecewise_constant`, `gbm`, `intermittent`, `impulse`, `logistic` |

Trivial/Low regimes should be **solved**; Impossible regimes serve as a control to ensure the model does not hallucinate structure in pure randomness.

### Diagnostic Metrics

| Metric | Ideal | Interpretation |
| --- | --- | --- |
| **relMAE** | < 1 | MAE relative to a naive baseline (historical mean or last-value persistence). |
| **MBR** (Mean-Bias Ratio) | 0.5 | How much the forecast hedges toward the global mean. |
| **MAR** (Mean-Amplitude Ratio) | 1.0 | Global amplitude vs. historical mean. `>1` under-sizes moves; `<1` overshoots peaks. |
| **PVR** (Path-Variation Ratio) | 1.0 | High-frequency path shape. `>1` too smooth; `<1` adds spurious jitter. |

Implementations live in [`src/ts_toolkit/pipeline/eval.py`](src/ts_toolkit/pipeline/eval.py).

## Repository Layout

```
timeseries-research/
├── configs/
│   ├── dataset_properties.json
│   └── experiments/                 # YAML experiment configs (entry points)
│       ├── simpletime*.yaml         # SimpleTimeBench configs
│       ├── gifteval*.yaml           # GIFT-Eval configs
│       ├── multi_vs_uni/            # Multivariate vs. univariate sweep (paper Table)
│       └── real.yaml                # Delaware River / NVIDIA leak case studies
├── src/ts_toolkit/                  # Pipeline + model wrappers
│   ├── dataloaders/                 # GIFT-Eval & generic loaders
│   ├── models/
│   │   ├── foundational/            # Chronos2, Moirai, Toto, Sundial, TimesFM, ...
│   │   ├── benchmarks/              # ARIMA, ETS, Theta, SeasonalNaive, ...
│   │   └── ensembles/median.py
│   └── pipeline/                    # Runner, evaluator, GluonTS adapter, visualization
├── scripts/
│   ├── run_gifteval_config.py       # CLI batch runner (entry point)
│   └── aggregate_validated_experiment_results.py
├── notebooks/                       # Analysis & paper-figure notebooks
├── paper_result_replication.md      # Step-by-step paper reproduction guide
└── pyproject.toml
```

## Installation

We recommend [`uv`](https://github.com/astral-sh/uv); [`pdm`](https://pdm-project.org/) is also supported.

```bash
git clone https://github.com/layer6ai-labs/timeseries-research.git
cd timeseries-research

# Option A — uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv sync

# Option B — pdm
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
pdm use 3.11
pdm sync --clean
```

> If you hit `ImportError`s when launching the pipeline, double-check that your virtual environment uses **Python 3.11** (`uv venv --python 3.11`). A CUDA-capable GPU is strongly recommended for foundation-model inference.

### Environment variables

The pipeline expects two paths at the repository root:

```bash
echo "REPO_ROOT=$(pwd)"            >> .env
echo "GIFT_EVAL=$(pwd)/data/gift_eval" >> .env
# Optional: where SimpleTimeBench data should be generated/loaded from
echo "SYNTHETIC=$(pwd)/data/simpletime/datasets" >> .env
```

## Data Preparation

### 1. GIFT-Eval

Download via the official Hugging Face dataset:

```bash
hf download Salesforce/GiftEval --repo-type=dataset --local-dir data/gift_eval
```

(If you use `huggingface-cli` instead: `huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir data/gift_eval`.)

### 2. SimpleTimeBench (synthetic)

Regenerate the benchmark with the bundled CLI. The default writes to the path stored in the `SYNTHETIC` env var.

```bash
python data/simpletime/simpletime.py \
  --length 256 \
  --num_samples 100 \
  --num_series 2 \
  --lag_amount 48 \
  --plot_examples
```

Each distribution is emitted in three variants: `univariate/<name>`, `multivariate/<name>`, and `multivariate_lagged/<name>` (the last is the perfect-leading-indicator setting). See [`data/simpletime/README.md`](data/simpletime/README.md) for the full CLI reference.

To explore the benchmark visually first, open [`data/simpletime/create_and_plot_simpletime.ipynb`](data/simpletime/create_and_plot_simpletime.ipynb).

### 3. Real-world case studies (optional)

The Delaware River streamflow and NVIDIA leak-test datasets used in the paper live under `data/Hahn_survey/`. Additional placeholders go under `data/real/` (gitignored) and are wired up via `configs/experiments/real.yaml`.

## Quickstart

Every experiment is described by a single YAML file in `configs/experiments/`. Pick one (or copy a template) and launch it:

```bash
# Smoke test: a 2-dataset SimpleTimeBench run with Chronos-2
python scripts/run_gifteval_config.py \
  --config configs/experiments/simpletime_sample.yaml \
  --gpu 0
```

The runner streams through every dataset in the config, saves per-dataset artefacts under a timestamped folder, writes a combined CSV of successful runs, logs failures, and emits a JSON summary next to the log file. You can invoke it from anywhere inside the repository — paths are resolved relative to the repo root.

| Flag | Description |
| --- | --- |
| `--config` | Path to an experiment YAML (relative to repo root or absolute). |
| `--gpu`    | GPU index exposed via `CUDA_VISIBLE_DEVICES` (e.g. `--gpu 0`). Omit for default device visibility. |

A notebook walkthrough of the same pipeline is available in [`notebooks/run_pipeline.ipynb`](notebooks/run_pipeline.ipynb).

### Generated artefacts

Outputs land in the directory referenced by `paths.output_path` (default: `results/`):

```
results/<experiment_name>_YYYYMMDD-HHMMSS/
  ├── <Alias>/
  │   ├── <dataset>/<term>/
  │   │   ├── metrics.csv
  │   │   ├── forecasts.parquet
  │   │   └── plots/
  │   └── combined_results.csv
  ├── dataset_run_summary.json
  └── run.log
```

## Configuration Reference

A minimal config looks like this:

```yaml
experiment_name: "simpletime_sample"

paths:
  storage_path: "data/simpletime/datasets"   # Where the dataset lives
  output_path: "results"                      # Where artefacts go

forecaster:
  name: "MedianEnsemble"                      # Ensemble strategy
  max_length: 512                             # Context window cap
  models:
    - name: "Chronos2"                        # Any key in MODEL_REGISTRY
      params:
        batch_size: 64
  alias: "Chronos2-SimpleTime-Sample"         # Identifier for this run

evaluation:
  to_univariate: false                        # Force univariate slicing
  show_plot: true                             # Save sample forecast plots
  eval_substr: "dim0"                         # Metric channel selector

datasets:
  - name: "univariate/linear"
    term: "short"
  - name: "multivariate_lagged/sine"
    term: "short"
```

**Key knobs**

- `forecaster.max_length` controls the context window handed to each model (the runner clips inputs accordingly).
- `evaluation.to_univariate=true` collapses multivariate datasets to univariate evaluation — useful for univariate-only baselines.
- `evaluation.eval_substr` selects which variate dimension to score (e.g. `"dim0"` evaluates only the target channel in leading-covariate setups).
- Omitting `datasets` makes the runner **auto-discover** every `{group}/{name}` directory under `paths.storage_path`.

## Reproducing the Paper

A standalone, step-by-step guide for reproducing the headline results lives in [`paper_result_replication.md`](paper_result_replication.md). At a glance:

```bash
# 1. Multivariate vs. univariate GIFT-Eval sweep (one config per model × mode)
python scripts/run_gifteval_config.py \
  --config configs/experiments/multi_vs_uni/gifteval_all_terms_chronos2_multi_4096.yaml \
  --gpu 0
# ... repeat for each YAML under configs/experiments/multi_vs_uni/

# 2. Aggregate and validate
#    Run, top-to-bottom, both notebooks under notebooks/04_merge-results/:
#      - 01_load_and_merge.ipynb
#      - 02_analysis.ipynb

# 3. Produce the final summary CSV used in the paper
python scripts/aggregate_validated_experiment_results.py \
  --experiment-results results_shared/pipeline_result/validated_experiment_results.csv \
  --baseline-results   results_shared/seasonal_naive/all_results.csv \
  --output             results_shared/pipeline_result/aggregated_detailed_results.csv \
  --summary-output     results_shared/pipeline_result/aggregated_summary_metrics.csv
```

Pre-computed pipeline outputs are checked in under `results_shared/pipeline_result/` so you can jump straight to the analysis notebooks without re-running the full sweep.

## Supported Models

The pipeline wraps each model behind a uniform [`Forecaster`](src/ts_toolkit/models/utils/forecaster.py) interface, then combines them through [`MedianEnsemble`](src/ts_toolkit/models/ensembles/median.py).

| Family | Models | Multivariate / Covariate Support | Notes |
| --- | --- | --- | --- |
| **Foundation** | `Chronos2`, `Moirai`, `Toto`, `Sundial`, `TimesFM`, `TiRex`, `TabPFN-TS`, `TimeGPT`, `Chronos` | Chronos-2, Moirai, Toto support past covariates natively | See `src/ts_toolkit/models/foundational/` |
| **Statistical** | `AutoARIMA`, `AutoETS`, `AutoCES`, `Theta`, `DynamicOptimizedTheta`, `SeasonalNaive`, `HistoricAverage`, `ADIDA`, `IMAPA`, `CrostonClassic`, `ZeroModel` | Univariate | Backed by StatsForecast |
| **Neural / ML** | DLinear and assorted neural baselines, gradient-boosted models | Univariate | See `src/ts_toolkit/models/benchmarks/` |

Adding a new model is a matter of writing a thin `Forecaster` subclass and registering it in `MODEL_REGISTRY` (`src/ts_toolkit/models/__init__.py`).

## Fine-Tuning Chronos-2

The paper’s “Nature vs. Nurture” section uses targeted fine-tuning of Chronos-2 on SimpleTimeBench subsets to ask whether blind spots are architectural or data-driven. The recipe is intentionally short:

```bash
python notebooks/chronos_finetune_all.py
```

This script:

1. Builds train/validation splits from `data/simpletime/datasets_*` via `ts_toolkit.utils.chronos_tune_input_generator`.
2. Loads `amazon/chronos-2` with `Chronos2Pipeline.from_pretrained`.
3. Fine-tunes (full or LoRA) and writes checkpoints to `results/finetuned_models/chronos2/<timestamp>/`.

To evaluate a checkpoint, point a SimpleTimeBench config at it:

```yaml
forecaster:
  models:
    - name: "Chronos2"
      params:
        repo_id: "results/finetuned_models/chronos2/<timestamp>/finetuned-ckpt"
        batch_size: 64
```

See [`configs/experiments/simpletime_finetuned.yaml`](configs/experiments/simpletime_finetuned.yaml) and the `chronos_finetune_*.ipynb` notebooks for trend-only, copy-paste-only, normalization-only, and joint-fine-tuning variants used in the paper.

## Real-World Case Studies

| Dataset | Setting | Horizon | What it tests |
| --- | --- | --- | --- |
| **Delaware River** | Montague → Trenton streamflow, 100 mi apart, **14 h** lag | `H=14` | Realistic past-only leading indicator. |
| **NVIDIA leak test** | Close price; covariate is the same series shifted by 16 days | `H=16d` | Stress test of copy-paste behaviour under non-stationary, noisy data. |

These are wired up by `configs/experiments/real.yaml`, with plotting code in `notebooks/real_data_plots.ipynb` and aggregated outputs under `results_shared/real_data/`.

Headline numbers (MASE, lower is better):

| Dataset | Model | Univariate | + Covariate | Gain |
| --- | --- | ---: | ---: | ---: |
| Delaware River | Chronos-2 | 7.16 | 6.74 | 5.9% |
| Delaware River | DLinear (840 params) | 8.82 | **6.69** | **24.1%** |
| NVIDIA leak | Chronos-2 | 10.56 | 9.90 | 6.3% |
| NVIDIA leak | DLinear (840 params) | 11.29 | **4.29** | **62.0%** |

When covariates are highly predictive, an 840-parameter DLinear trained on CPU can outperform billion-parameter TSFMs.

## Notebooks Index

| Notebook | Purpose |
| --- | --- |
| `notebooks/run_pipeline.ipynb` | Interactive walkthrough of the full pipeline. |
| `notebooks/01_sota-uni-vs-multi/01_roc_sota-uni-vs-multi.ipynb` | Multivariate-vs-univariate analysis on GIFT-Eval. |
| `notebooks/02_mean-bias-experiment/02_mean-bias-experiment.ipynb` | Mean-hedging diagnostic. |
| `notebooks/03_mar-mbr/run_pipeline_MAR_MBR_*.ipynb` | MAR / MBR investigations across horizons. |
| `notebooks/04_merge-results/{01,02}_*.ipynb` | Aggregation pipeline used for the paper tables. |
| `notebooks/polished_*.ipynb` | Paper-figure rendering (forecasts, training curves, post-FT comparisons). |
| `notebooks/real_data_{baseline,plots}.ipynb` | Delaware River & NVIDIA case studies. |
| `notebooks/chronos_finetune_*.ipynb` | Fine-tuning recipes per primitive (bias, copy-paste, norm, all-in-one). |
| `data/simpletime/create_and_plot_simpletime.ipynb` | Visualise and sanity-check SimpleTimeBench. |

## Development

This repo uses [`pre-commit`](https://pre-commit.com/) to keep environments synced and code consistent.

| Hook | Trigger | Effect |
| --- | --- | --- |
| `uv sync` | `git checkout`, `git merge` | Keep `.venv` aligned with `pyproject.toml`. |
| `black` | `git commit` | Auto-format Python. |
| `flake8` | `git commit` | Lint Python. |

**Setup:**

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type post-checkout --hook-type post-merge --hook-type post-commit
```

**Skip hooks for one commit (avoid unless necessary):**

```bash
git commit -m "<message>" --no-verify
```

### Dependency management

```bash
uv add <package>                   # Runtime dep
uv add --group dev <pkg1> <pkg2>   # Dev-only deps
uv sync                            # Refresh the venv
```

### Personal vs. shared experiments

| Purpose | Folder | Tracked? |
| --- | --- | --- |
| Private, local experiments | `results/`, `configs/experiments/personal/` | **Ignored by git** |
| Shared, reviewable experiments | `results_shared/`, `configs/experiments/` | **Tracked** |

Please scrub absolute paths and machine-specific aliases before committing anything under the shared folders.

## Citation

If you use SimpleTimeBench, the `ts_toolkit` pipeline, or any results from this repository, please cite:

```bibtex
@inproceedings{ghoroghchian2026foundations,
  title     = {Foundations without Fundamentals: Zero-Shot Blind Spots in Time Series Foundation Models},
  author    = {Ghoroghchian, Nafiseh and Zhang, Haipeng and Han, Shuyi and Labach, Alex and Stein, George},
  booktitle = {2nd ICML Workshop on Foundation Models for Structured Data (FMSD)},
  year      = {2026}
}
```

## Acknowledgements

This work builds on the open ecosystem around time-series foundation models. We gratefully acknowledge the authors and maintainers of:

- [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) — the benchmark format SimpleTimeBench is released in.
- [Chronos](https://github.com/autogluon/chronos), [Moirai / uni2ts](https://github.com/SalesforceAIResearch/uni2ts), [Toto](https://github.com/datadog/toto), [TimesFM](https://github.com/google-research/timesfm), [TabPFN-TS](https://github.com/PriorLabs/tabpfn), [TiRex](https://github.com/NX-AI/tirex), [Sundial](https://github.com/thuml/Sundial), [TimeCopilot](https://github.com/AzulGarza/timecopilot), [Nixtla](https://github.com/Nixtla) for the model implementations we wrap.
- [GluonTS](https://github.com/awslabs/gluonts) for the evaluation primitives underlying our pipeline.

## License

Distributed under the **Apache 2.0** license — see [`pyproject.toml`](pyproject.toml) for the project metadata and the third-party `model_zoo/chronos/LICENSE` for the upstream Chronos license.

---

<div align="center">

Made with care at <a href="https://layer6.ai/">Layer 6 AI</a>. Issues, PRs, and benchmark contributions are very welcome.

</div>
