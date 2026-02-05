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
- Strategy (with cost): annualized return **0.1942**, IR **0.5633**, max drawdown **-0.3803**
- ETF benchmark: annualized return **0.1104**
- Excess (strategy - benchmark - cost): annualized return **0.0838**

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
