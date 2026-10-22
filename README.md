# Loan balance forecasting model

Selects a small set of macro drivers (at most 3 by default) for a loan balance, using two approaches side by side, then backtests, diagnoses and forecasts each one under every scenario.

- **Approach A, screen first:** test every candidate one at a time, cluster near-duplicates, keep one per cluster, then search.
- **Approach B, Elastic Net first:** run an Elastic Net on all candidates (with optional stability selection), group correlated picks, keep one per group, then search.

Both approaches finish with the same final step, so their results are directly comparable.

## Run

```
python run_pipeline.py                                    # example data and config
python run_pipeline.py --config config/my_portfolio.yaml  # your data
python run_pipeline.py --config config/my_portfolio.yaml --approach en_first   # one approach only
```

Missing Python packages are installed automatically the first time. A run takes about 30 seconds.

## Using your own files: only the config changes

Copy `config/example_config.yaml`, rename it, and edit it. Everything that depends on a data set lives there:

| What | Where in the config |
|---|---|
| Bank file, sheet, date column, balance column, monthly or quarterly | `data.bank` |
| Inorganic events file and its columns (optional) | `data.events` |
| One Moody's file per scenario, any names, plus the date column | `data.macro.scenarios`, `data.macro.date_column` |
| Metadata file and **your** column names for mnemonic, description, geography, theme, transform, expected sign | `data.metadata.columns` |
| Extra variables built from macro columns (spreads, real rates) | `data.constructed` |
| Dates: last actual quarter, estimation start, backtest start, dummy quarters | `sample` |
| Selection settings for each approach and the final model | `features`, `screen_first`, `en_first`, `final_model` |

A few details on the inputs:

- Files can be `.csv` or `.xlsx`; give `sheet:` for Excel.
- Dates can be quarter labels (`2020Q1`) or real dates (`2020-03-31`).
- Monthly balances are converted to quarter-end values.
- `run_name` sets the output folder, so runs on different portfolios never overwrite each other.

## Skipping unchanged steps (hash-based cache)

Each step gets a fingerprint, a SHA-256 hash, built from everything it depends on:

| Step | Its fingerprint includes |
|---|---|
| `data` | the loaded values of the target, balance, metadata, macro history and every scenario |
| `features` | macro history, metadata, the `features` and `sample` settings, the code in `features/` |
| `screen_first` / `en_first` | the `features` fingerprint, the target, that approach's settings, `final_model`, `sample`, its code |
| `evaluate:<model>` | the chosen drivers, the target, the scenarios, `sample`, the evaluation code |

If a step's fingerprint matches a result already in the cache, the result is loaded instead of recomputed. A full rerun with nothing changed takes about 4 seconds instead of 30. Some examples from testing:

- **New scenario path only:** only the `evaluate` steps re-run.
- **One revised history value:** everything downstream re-runs.
- **Settings change that leaves the drivers the same:** the evaluation is reused, because it depends on the drivers, not on how they were found.
- **Same values under new file or column names:** everything is loaded from the cache. Data are hashed by value, not by file name.
- **Edited code:** only the steps that use that code re-run.

Each run writes `outputs/<run_name>/run_manifest.json` and appends to `run_history.csv`. These record, for every step, whether it was cached or computed and which inputs changed since the previous run, for example `changed: config.en_first` or `changed: scenarios`.

Options:

```
python run_pipeline.py --force                      # recompute everything
python run_pipeline.py --force-step en_first        # recompute one step (also: features, screen_first, evaluate)
python run_pipeline.py --no-cache                   # neither read nor write the cache
```

The cache lives in `cache/` (set by `cache.dir` in the config) and is shared by all configs. Delete the folder to clear it.

## Folders

```
run_pipeline.py          entry point
config/                  one YAML per data set
data/                    example (simulated) inputs
tools/simulate_data.py   rebuilds the example data
src/utils/               config loading, file reading, package set-up, cache.py (hashing)
src/features/            transformations, stationarity tests, candidate features
src/selection/           screening.py, clustering.py, elastic_net.py (incl. stability selection)
src/modelling/           ardl.py (final model), search.py (shared final search), evaluate.py
src/pipelines/           screen_first.py (A), en_first.py (B)
src/reporting/           tables, charts, summary
outputs/<run_name>/      common/, screen_first/, en_first/, comparison/, summary.md
```

## What each approach does

**Shared start**
- Relevance filter, transformations (q/q, y/y, 2-quarter, acceleration, gap, real growth; changes and levels for rates).
- ADF stationarity test, lags 0–4, giving about 2,300 candidate features on the example data.

**A: screen first** (`src/pipelines/screen_first.py`)
1. Regress loan growth on lagged growth and the dummies plus each candidate on its own. A candidate passes if p < 0.05 (Newey-West) and has the expected sign.
2. Keep the best form and lag per variable.
3. Cluster at |correlation| > 0.75 and keep one per cluster (an Ontario series is preferred if nearly as strong).
4. The top 25 go to the final search.

**B: Elastic Net first** (`src/pipelines/en_first.py`, `src/selection/elastic_net.py`)
1. Lagged loan growth and the dummies are left unpenalized: they are partialled out first, which is the same as glmnet's `penalty.factor = 0`.
2. X and y are standardized as glmnet does.
3. Coefficients are constrained to their expected sign (optional).
4. Lambda is the first point on the path where `target_n` features are selected. There is no cross-validation, so the result is deterministic for given data.
5. **Stability selection** (optional, on by default): repeat on 100 subsamples built from blocks of consecutive quarters, and keep features selected in at least 60% of them.
6. Group the picks at |correlation| > 0.75 and keep one per group.

**Shared final step** (`src/modelling/search.py`)
1. Try every combination of 1–3 candidates. A combination is kept only if:
   - every driver is significant;
   - every sign is as expected;
   - the VIF is below 5;
   - there is at most one driver per theme.
2. Rank the survivors by out-of-sample balance error in the rolling backtest.
3. Choose the simplest model within 5% of the best.

**Evaluation** (`src/modelling/evaluate.py`)
- Rolling backtest 1–9 quarters ahead, with dummy quarters excluded from the error.
- Residual diagnostics.
- A plain-language driver table.
- Forecasts under every scenario.
- An AR(1) benchmark, run alongside for comparison.

## Example results (simulated data)

| Model | Average error, 1–9Q | Drivers |
|---|---|---|
| AR(1) benchmark | 1.50% | none |
| Screen first | 0.61% | Toronto single-detached prices, Ontario income |
| Elastic Net first | 0.72% | Ontario house prices, Ontario income |

At `target_n = 10`, the full-sample Elastic Net spent 9 of its 10 picks on city house-price series. Stability selection and grouping reduced these to one house-price driver plus income.
