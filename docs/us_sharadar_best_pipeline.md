# US Sharadar Best Weekly Pipeline (Qlib)

This document describes the **current best weekly pipeline** in this repo for US equities using Sharadar PIT fundamentals and daily prices.

## Canonical config

- Config file: `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best.yaml`
- Experiment name: `us_sharadar_weekly_pit_best`

## Data sources (Sharadar bundle)

The pipeline assumes you already ingested the Sharadar bundle into Qlib format:

- **Prices**: Sharadar Equity Prices (SEP) with dividend-adjusted close used in Qlib.
- **Fundamentals (PIT)**: Sharadar Core US Fundamentals (SF1), accessed via PIT operators in Qlib.
- **Security master**: TICKERS table to build the instrument universe.

Data location used by the pipeline:
- Qlib data root: `/root/.qlib/qlib_data/us_data`

## Universe

- `pit_mrq_large` (Qlib instrument list built from Sharadar PIT fundamentals, MRQ, larger cap focus).
- Instruments file: `/root/.qlib/qlib_data/us_data/instruments/pit_mrq_large.txt`

## Features

Base features:
- Alpha158 price/volume features (daily, from Qlib standard feature set).

PIT fundamentals (quarterly):
- `assets, liabilities, equity, revenue, netinc, ebit, ebitda, cashneq, debt, fcf, capex, grossmargin, netmargin, divyield, dps, eps, epsdil, bvps, shareswa`

Extra PIT ratio features:
- ROE, ROA, EBITDA margin, FCF margin, leverage, cash/assets, capex/assets, asset turnover
- Dividend yield (PIT/price), earnings yield (PIT/price), book/price, FCF yield

Implementation:
- Data handler: `Alpha158WithPIT` in `qlib.contrib.data.handler_sharadar`
- PIT interval: `q` (quarterly)

## Label

- Weekly label = next 10 trading day return:
  - `Ref($close, -11)/Ref($close, -1) - 1`

## Filters (quality)

Applied to the instrument universe during dataset preparation:
- Price filter: `$close > 5`
- Liquidity filter: `Mean($close * $volume, 20) > 10,000,000`

## Preprocessing

- Feature normalization: cross‑sectional robust z‑score
- Label normalization: cross‑sectional rank normalization

## Model

- Model class: `LGBModel` (LightGBM)
- Loss: `mse`
- Hyperparameters (from best run):
  - `learning_rate: 0.0421`
  - `num_leaves: 210`
  - `max_depth: 8`
  - `subsample: 0.8789`
  - `colsample_bytree: 0.8879`
  - `lambda_l1: 205.6999`
  - `lambda_l2: 580.9768`
  - `num_threads: 20`

## Train/valid/test splits

- Train: 2000‑01‑01 to 2018‑12‑31
- Valid: 2019‑01‑01 to 2021‑12‑31
- Test: 2022‑01‑01 to 2026‑01‑30

## Strategy (portfolio construction)

- Strategy: `WeeklyTopkDropoutStrategy`
  - `topk: 50`
  - `n_drop: 10`
  - `rebalance_weekday: 0` (Monday)
  - `hold_thresh: 5`
  - `only_tradable: true`

Trading costs (backtest):
- Open cost: `0.0005`
- Close cost: `0.0015`
- Min cost: `5`
- Deal price: `close`

## Evaluation

### Default backtest

The `qrun` pipeline produces a default backtest report vs the benchmark specified in the config (AAPL). This is mainly for sanity checks.

### ETF basket benchmark (canonical evaluation)

We evaluate against an equal‑weight ETF basket:
- `SPY, QQQ, IWM, DIA, IVV, VOO, VTI`

Benchmark file used:
- `/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl`

Evaluation command:
```bash
python scripts/eval_pred_backtest.py \
  --pred <pred.pkl path> \
  --start 2022-01-01 --end 2026-01-30 \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --skip_optimized
```

### Latest confirmed run (ETF benchmark)

- Run id: `21a9ae8a26d840ecbe716c1cc8eb00f7`
- Period: 2022-01-01 to 2026-01-30
- Strategy (with cost): annualized return **0.1458**, IR **0.4393**, max drawdown **-0.4042**
- ETF benchmark: annualized return **0.1104**
- Excess (strategy - benchmark - cost): annualized return **0.0354**

## How to run (end‑to‑end)

Train + backtest (AAPL benchmark):
```bash
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best.yaml
```

Find the prediction file (`pred.pkl`) in mlruns:
```bash
ls -la mlruns/907430387469519321/<RUN_ID>/artifacts/pred.pkl
```

Evaluate vs ETF basket:
```bash
python scripts/eval_pred_backtest.py \
  --pred mlruns/907430387469519321/<RUN_ID>/artifacts/pred.pkl \
  --start 2022-01-01 --end 2026-01-30 \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --skip_optimized
```

## Suggestions output (top/bottom picks)

```bash
python scripts/us_sharadar_suggestions.py \
  --experiment_name us_sharadar_weekly_pit_best \
  --weekly --weekday 0 --topk 20
```

## Validation & robustness checks

Run the validation script to sanity‑check config/data and stress‑test the backtest with sliced metrics:

```bash
python scripts/validate_us_sharadar_pipeline.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best.yaml \
  --pred mlruns/907430387469519321/<RUN_ID>/artifacts/pred.pkl \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --start 2022-01-01 --end 2026-01-30 \
  --check_data_quality \
  --fail_on_data_quality_fail \
  --check_gates \
  --fail_on_gate_fail \
  --check_rolling \
  --fail_on_rolling_fail \
  --rolling_window_days 504 \
  --rolling_step_days 252 \
  --rolling_min_days 252 \
  --by_year
```

Notes:
- Add `--skip_data_checks` if you only want performance slices.
- Adjust `--stress_cost_mult` and `--stress_deal_price` to test execution sensitivity.
- Gate thresholds are configurable via `--gate_*` flags.
- `--gate_min_year_days` (default `200`) excludes partial-year slices from yearly robustness gates.
- Add `--check_rolling` to run walk-forward robustness windows; tune with `--rolling_*` flags.
- Add `--check_data_quality` to fail fast on NaN/coverage issues in predictions, benchmark, and generated reports.

### Risk-managed strategy variant

To reduce drawdown and regime sensitivity without changing the model, use:

`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2.yaml`

Notes:
- `risk_v2` is the current low-drawdown baseline.
- `risk_v3` is more defensive (lower drawdown, but weaker excess return in stress).
- Optional stability knobs (`risk_smoothing_up/down`, `max_risk_step_up/down`) exist in strategy code but are not enabled in the active `risk_v2` config.
- Controlled proxy A/B showed benchmark-excess training can improve rolling stability and excess return, but drawdown tuning still needs work before promoting to active baseline.

### Benchmark-excess training variant (experimental)

Config:

`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess.yaml`

Notes:
- Uses `BenchmarkExcessLabel` to train on ETF-basket-relative labels.
- Processor alignment was corrected to match qlib label semantics (`Ref($close, -(s+N))/Ref($close, -s) - 1`, default `s=1`).
- Treat as experimental until full retrain + rolling validation are completed.

### Excess stability candidate (experimental)

Config:

`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability.yaml`

Notes:
- Adds optional regime-guard knobs in strategy:
  - include current holdings in risk basket (`include_current_positions`)
  - market drawdown penalty (`market_drawdown_*`)
- Proxy validation (2025-01-02 -> 2026-01-29) improved drawdown modestly and reduced turnover while keeping positive excess:
  - full excess ann: `0.0320`
  - full MDD: `-0.2750`
  - stress excess ann: `0.0211`
- Keep as candidate until full-universe rolling validation confirms improvement over `risk_v2_excess`.

### Candidate ranking (same benchmark/gates)

Use a shared YAML spec to compare multiple configs/preds with identical evaluation settings:

```bash
python scripts/rank_us_sharadar_candidates.py \
  --candidates_yaml examples/benchmarks/LightGBM/us_sharadar_candidate_ranking_etf.yaml \
  --out_csv /tmp/us_sharadar_candidate_ranking.csv
```

The ranking script now supports rolling walk-forward stability scoring.
- Enable from YAML via `rolling.enabled: true` (already set in the ETF ranking YAML).
- Or force via CLI: `--check_rolling --rolling_*`.
