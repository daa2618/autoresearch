# autoresearch — UK House Price Index Prediction

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted from LLM pretraining to **UK house price forecasting** using classical ML.

The original autoresearch idea: give an AI agent a small but real ML setup and let it experiment autonomously. It modifies the code, runs the model, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and a better model.

This fork replaces the GPU-based LLM training loop with a tabular regression task — predicting next-month UK average house prices from Land Registry data (1990-2026). No GPU required; experiments run in under a second on any machine.

## Results

Starting from a default Ridge baseline, the agent ran 12 experiments and reduced RMSE by **77%**:

| Stage | val_rmse | val_mape | Change |
|-------|----------|----------|--------|
| Baseline (Ridge alpha=1.0) | £10,256 | 3.56% | — |
| Alpha tuning (alpha=270) | £2,965 | 1.06% | -71% |
| + Feature engineering | £2,571 | 0.93% | -13% |
| + Feature selection | **£2,322** | **0.86%** | -10% |

Full experiment log and analysis in [`docs/analysis.md`](docs/analysis.md).

## How it works

Same philosophy as the original — three files that matter:

- **`prepare.py`** — data loading, feature engineering (lags, rolling stats, percentage changes), train/val split. Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the model, preprocessing, and any additional feature engineering. Everything is fair game: model type, hyperparameters, target transforms, feature selection, ensembling.
- **`program.md`** — instructions for the agent. Defines the experiment loop, rules, allowed changes, and the running experiment log.

### Dataset

- **Source**: UK House Price Index (Land Registry), 1990-2026
- **Granularity**: Monthly, national UK aggregate (~400 rows)
- **Target**: `average_price` — next month's average UK house price in GBP
- **Split**: Strictly chronological — train (~1990-2022), val (~2022-2024), test (held out)

### Best model

Ridge regression with tuned regularisation, log-target transform, engineered features, and feature selection:

```
Pipeline: StandardScaler -> Ridge(alpha=280)
Target:   log1p(y) / expm1(pred)
Features: 19 selected from 23 (4 dropped as redundant/noisy)
```

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/) (or pip/poetry).

```bash
# 1. Install dependencies
uv sync

# 2. Run a single experiment
uv run train.py

# 3. Check output
uv run train.py | grep val_rmse
```

If using pip or poetry instead of uv:

```bash
# pip
pip install -r pyproject.toml

# poetry
poetry install

# Then run with the virtualenv's python
.venv/bin/python train.py
```

## Running the agent

Point your AI coding agent (Claude Code, Codex, etc.) at this repo and prompt:

```
Read program.md and run the next experiment.
```

The agent will read the current state, propose a change to `train.py`, run it, and log the result in `program.md`. Disable permission prompts for a fully autonomous loop.

## Project structure

```
prepare.py                  — data loading, feature engineering, splits (do not modify)
train.py                    — model and training (agent modifies this)
program.md                  — agent instructions and experiment log
uk_hpi_1990_2026.csv        — raw dataset
docs/analysis.md            — detailed analysis of all experiments and usage guide
pyproject.toml              — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs are small and reviewable.
- **Sub-second experiments.** Classical ML on ~400 rows means each experiment completes in under a second, enabling rapid iteration. Grid searches over hyperparameters still finish in well under the 60-second budget.
- **No GPU required.** Scikit-learn, NumPy, and Pandas are the only runtime dependencies that matter. Runs on any laptop.
- **Chronological split.** Train/val/test are strictly time-ordered — no data leakage. The agent never sees the test set.

## Key findings

1. **Regularisation tuning dominates.** Changing Ridge alpha from 1.0 to 280 cut RMSE by 71%. Default hyperparameters are rarely optimal.
2. **Linear models win for extrapolation.** Tree-based methods (HistGradientBoostingRegressor) scored 3x worse because they cannot predict prices beyond the training range.
3. **Log-target transform is essential.** Prices span £50k-£290k over 35 years. Log-space fitting captures proportional growth.
4. **Simplicity wins on small data.** With ~300 training samples, every added parameter is a liability. No ensemble, polynomial, or stacking approach beat a single well-tuned Ridge.

See [`docs/analysis.md`](docs/analysis.md) for the full writeup including algorithms evaluated, feature engineering details, and a guide for using the model on new data.

## Attribution

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Original concept by Andrej Karpathy.

## License

MIT
