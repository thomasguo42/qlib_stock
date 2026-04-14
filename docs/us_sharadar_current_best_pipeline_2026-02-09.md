# US Sharadar Current Best Pipeline Report (2026-02-09)

This document records the current best-performing US Sharadar pipeline variant in this repository, with reproducible metrics and a readiness assessment for live trading use.

## 1) What is considered "current best"

Definition used here:
- Best annualized return on the comparable full test window `2022-01-03` to `2026-01-29`.
- Evaluated against the ETF basket benchmark (not AAPL).
- Includes transaction costs.

Current best variant:
- Config: `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml`
- Prediction artifact used in validation: `mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl`
- Validator output (rechecked): `/tmp/validate_recheck_v2_hold10_etf.txt`

Important reproducibility note:
- There is currently no completed `qrun` entry using `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml` directly in `mlruns/907430387469519321`.
- The `0.1719` result is a validated strategy/config evaluation using the above `pred.pkl`.

## 2) Pipeline specification

Core configuration:
- Region/data root: US, `/root/.qlib/qlib_data/us_data`
- Universe: `pit_mrq_large`
- Train segment: `2016-01-01` to `2020-12-31`
- Valid segment: `2021-01-01` to `2021-12-31`
- Test segment: `2022-01-01` to `2026-01-30`
- Label: `Ref($close, -11)/Ref($close, -1) - 1` (10 trading days forward return)
- Model: `LGBModel` (LightGBM), `num_boost_round=2000`, `early_stopping_rounds=200`
- Strategy: `WeeklyTopkDropoutStrategy`
- Portfolio params: `topk=50`, `n_drop=10`, `hold_thresh=10`, `rebalance_weekday=0`, `only_tradable=true`
- Cost model: `open_cost=0.0005`, `close_cost=0.0015`, `min_cost=5`, `deal_price=close`

## 3) Data and feature details

Price/liquidity filters:
- `$close > 5`
- `Mean($close * $volume, 20) > 10,000,000`

PIT fundamentals:
- Includes fields such as `assets`, `liabilities`, `equity`, `revenue`, `netinc`, `ebit`, `ebitda`, `fcf`, `capex`, `workingcapital`, `currentratio`, `divyield`, and others in the config.
- Includes missingness-indicator features via `If(IsNull(...), 1, 0)` for key PIT items.

Observed data quality from prior validator run (`/tmp/validate_v2_hold10_etf_new.txt`):
- Sample months checked: `24`
- Universe count range in sample: min `705`, median `720`, max `744`
- PIT missing ratio top fields: around `34%` (example: assets/equity/netinc-based fields)
- PIT staleness: median `65` days, p95 `130` days
- Missing target rebalance weekdays: `24` weeks in sample (handled by next-trading-day rebalance logic)

## 4) Confirmed performance metrics (ETF benchmark)

Evaluation benchmark:
- `/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl`

Full window (`2022-01-03` to `2026-01-29`, `1022` trading days):
- Annualized return: `0.1719`
- Information ratio: `0.5471`
- Max drawdown: `-0.4258`
- Benchmark annualized return: `0.1104`
- Excess annualized return: `0.0615`
- Average turnover: `0.0618`

Year-by-year:
- `2022`: ann `-0.2848`, excess ann `-0.1106`
- `2023`: ann `0.3979`, excess ann `0.1598`
- `2024`: ann `0.0680`, excess ann `-0.1252`
- `2025`: ann `0.2541`, excess ann `0.0865`
- `2026` (partial year, 19 days): ann `-0.6807`, excess ann `-1.0196`

Stress test (2x transaction costs):
- Annualized return: `0.1572`
- Information ratio: `0.5004`
- Max drawdown: `-0.4301`
- Excess annualized return: `0.0469`
- Average turnover: `0.0617`

Gate results (from validator):
- `full_excess_ann`: PASS
- `full_ir`: PASS
- `full_mdd_abs`: PASS (threshold `0.45`)
- `stress_excess_ann`: PASS
- `full_avg_turnover`: PASS
- `positive_excess_years`: PASS (eligible full years)
- `worst_year_excess_ann`: PASS
- Overall: `PASS`

## 5) Why this is still not trading-ready

Even as current best by annual return, key trading-readiness gaps remain:

- Drawdown remains too large for most production mandates.
  - Current max drawdown is `-42.58%`.
  - This is near the validator gate ceiling and can be unacceptable for capital deployment.

- Regime instability is material.
  - Excess return flips sign by year (`2022` and `2024` negative excess; `2023` and `2025` positive).
  - Performance concentration in a subset of years indicates unstable edge.

- Partial-year fragility is visible.
  - In `2026` YTD slice (`19` days), strategy and excess both collapse sharply.
  - This slice is short and noisy, but it still indicates downside sensitivity in stress regimes.

- Reproducibility path is not yet clean end-to-end.
  - Best reported result is tied to a validated `pred.pkl` + hold10 strategy setup.
  - There is not yet a completed dedicated `qrun` training record for `best_v2_hold10` in mlruns.

- Data limitations remain structural.
  - PIT coverage missingness is substantial (~`34%` for key fields in sample).
  - PIT freshness is lagged (median `65` days, p95 `130` days), which can reduce timeliness.

- Execution realism is still simplified.
  - Backtest uses fixed proportional costs and `deal_price=close`.
  - No explicit market impact/slippage model beyond simple costs.
  - No broker/live fill uncertainty, borrow constraints, or venue liquidity modeling.

## 6) What blocks live deployment right now

Before live trading, at minimum:
- A completed, reproducible training run for `best_v2_hold10` with stored artifacts and rerun consistency checks.
- Stronger downside controls to push drawdown materially below current level.
- Better regime robustness (less year-to-year excess sign flip).
- Rolling/walk-forward validation with strict fail criteria promoted from "analysis" to "release gate".
- More realistic execution simulation and paper-trading evidence under production-like constraints.

## 7) Reproduction commands

Recheck (used for this report):

```bash
python scripts/validate_us_sharadar_pipeline.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml \
  --pred mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --start 2022-01-01 --end 2026-01-30 \
  --skip_data_checks \
  --by_year \
  --check_gates
```

Output file used:
- `/tmp/validate_recheck_v2_hold10_etf.txt`

## 8) Snapshot conclusion

As of `2026-02-09`, the best annual-return pipeline variant is `best_v2_hold10` at `0.1719` annualized return with `0.0615` ETF-benchmark excess annualized return. It is the top performer in current comparable evaluations, but it is not yet sufficiently robust for live trading due to high drawdown and regime instability.
