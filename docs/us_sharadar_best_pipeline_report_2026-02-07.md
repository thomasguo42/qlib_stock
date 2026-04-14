# US Sharadar PIT Weekly Pipeline - Work Log and Current State (2026-02-07)

This document captures everything I changed, how I validated it, and the current status of the US Sharadar PIT weekly pipeline. It is written so you can reproduce every step and see what is robust, what is not, and what remains open.

---

## 1) Scope and goal

Goal: verify the pipeline in `docs/us_sharadar_best_pipeline.md`, fix robustness issues (especially PIT alignment and missing PIT), retrain and validate, and report whether the pipeline is ready for trading.

I focused on:
- PIT correctness (no lookahead, proper dates)
- Missing PIT handling
- Strategy calendar robustness (weekly rebalance when target weekday is a holiday)
- Validation and stress testing
- Re-training and backtest evaluation

---

## 2) Files added or modified

### A) Pipeline config (documentation/clarity)
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best.yaml`
  - Updated the comment on the label to match the expression:
    - from “next 5 trading days” to “next 10 trading days”.

### B) Validator tooling
- Added: `scripts/validate_us_sharadar_pipeline.py`
  - Checks config validity, dataset coverage, PIT staleness, missing ratios, and backtest robustness.
  - Supports `--by_year`, `--skip_data_checks`, and stress testing (2x costs).
- Updated: `docs/us_sharadar_best_pipeline.md`
  - Added usage for the validator.

### C) PIT data build fixes (critical)
- `scripts/data_collector/sharadar/prepare_sf1_pit.py`
  - Default `--date_col` changed to `datekey` (instead of `lastupdated`).
  - Added `--date_offset_days` (default 45) to shift PIT availability forward.
  - `_as_int_date` applies offset correctly.
- `scripts/data_collector/sharadar/README.md`
  - Updated example commands to use `datekey` and `--date_offset_days 45`.

### D) Weekly rebalance robustness
- `qlib/contrib/strategy/weekly.py`
  - Added `_is_rebalance_day` to both `WeeklyTopkDropoutStrategy` and `WeeklyScoreWeightedStrategy`.
  - Logic: if the target weekday is a holiday, rebalance on the next trading day.

### E) New configs created for experiments
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2.yaml`
  - Updated PIT fields, added missingness indicators, adjusted training window to 2016+, model params.
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml`
  - Same model as v2; strategy `hold_thresh` set to 10 days to match 10‑day label.
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v3_label5.yaml`
  - 5‑day label variant (performed worse).

---

## 3) Root cause found: PIT data was effectively unusable

### Problem
The PIT dataset was built using `lastupdated`, which in SF1 often contains a single late date (e.g., 2026-01-30). That meant all PIT data became available at the same future date and was effectively missing for earlier years.

### Fix
Switched PIT date to `datekey` and applied a realistic availability offset.

### Rebuild commands
```
python scripts/data_collector/sharadar/prepare_sf1_pit.py \
  --sf1 /root/.qlib/sharadar/raw/sf1_MRQ.csv \
  --out_dir /root/.qlib/sharadar/pit_normalized/mrq \
  --dimension MRQ \
  --date_col datekey \
  --period_col calendardate \
  --date_offset_days 45

python scripts/dump_pit.py \
  --csv_path /root/.qlib/sharadar/pit_normalized/mrq \
  --qlib_dir /root/.qlib/qlib_data/us_data dump \
  --interval quarterly --overwrite
```

### Result
- PIT dates now span 2016‑05‑10 to 2026‑02‑10.
- PIT fields are no longer 100% missing.

---

## 4) Current PIT missingness and staleness

From validator (2016‑01‑04 → 2017‑12‑01 sample):
- Missing ratios for PIT fields ~32–36% (varies by field).
- PIT staleness median: 65 days; p95: 130 days.

This level of missingness is expected for fundamentals. It’s not imputed; LightGBM handles NaNs, and missingness indicators were added in v2.

---

## 5) Current pipeline config (best candidate)

### Candidate: `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2.yaml`

Key points:
- Training window: 2016‑01‑01 → 2020‑12‑31
- Validation: 2021‑01‑01 → 2021‑12‑31
- Test: 2022‑01‑01 → 2026‑01‑30
- Label: next 10 trading days return
- PIT fields expanded to include `workingcapital`, `currentratio`, `shareswadil`, `divyield` and others.
- Missingness indicators added for key PIT fields.
- Custom operator registered to support `IsNull` in expressions:
  - `qlib_init.custom_ops` includes `IsNull` from `qlib.contrib.ops.high_freq`.
- Model: LightGBM with larger `num_boost_round` and `early_stopping_rounds`.

Note: I removed PIT fields `roe`, `roa`, `roic` from v2 because they are not present in the Qlib financial directory and caused “period data not found” warnings.

---

## 6) Training runs and outputs

### A) Base retrain after PIT fix (older)
- Run id: `4720c01cb3674c568c4569ea92d5f9de`
- Path: `mlruns/907430387469519321/4720c01cb3674c568c4569ea92d5f9de/artifacts/pred.pkl`
- Performance (2022‑01‑01 → 2026‑01‑29):
  - ann_return 0.1055, IR 0.3656, MDD ‑0.4240, excess ‑0.0048

### B) v2 retrain (missingness indicators + updated PIT fields)
- Run id: `d00d2865e3994cc795586ed0b89fd6ff`
- Path: `mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl`
- Validation is still weak (early stopping at iteration 1).

### C) label 5 variant (worse)
- Run id: `b1d0c2e951de4c0a9e205e08a1bd874e`
- Path: `mlruns/907430387469519321/b1d0c2e951de4c0a9e205e08a1bd874e/artifacts/pred.pkl`
- Annualized excess return negative (not a candidate).

---

## 7) Backtest validation results (ETF basket benchmark, canonical)

All metrics in this section use:
- `/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl`

### v2 (default hold 5 days)
`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2.yaml`

Full period 2022‑01‑03 → 2026‑01‑29:
- ann_return 0.1408
- IR 0.4426
- MDD ‑0.4112
- bench ann_return 0.1104
- excess ann_return 0.0304
- avg turnover 0.0819

Year-by-year:
- 2022: ‑0.2492 (excess ‑0.0749)
- 2023: 0.3599 (excess 0.1218)
- 2024: 0.0809 (excess ‑0.1123)
- 2025: 0.2643 (excess 0.0967)
- 2026 YTD: ‑0.6287 (excess ‑0.9676)

Stress test (2x costs):
- ann_return 0.1214
- IR 0.3816
- excess ann_return 0.0110

Robustness gates (with `gate_min_year_days=200`):
- PASS overall
- Note: partial 2026 slice (19 trading days) is excluded from yearly gate scoring.

### v2 hold 10 days (aligns label horizon)
`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml`

Full period 2022‑01‑03 → 2026‑01‑29:
- ann_return 0.1719
- IR 0.5471
- MDD ‑0.4258
- bench ann_return 0.1104
- excess ann_return 0.0615
- avg turnover 0.0618

Year-by-year:
- 2022: ‑0.2848 (excess ‑0.1106)
- 2023: 0.3979 (excess 0.1598)
- 2024: 0.0680 (excess ‑0.1252)
- 2025: 0.2541 (excess 0.0865)
- 2026 YTD: ‑0.6807 (excess ‑1.0196)

Stress test (2x costs):
- ann_return 0.1572
- IR 0.5004
- excess ann_return 0.0469

Robustness gates (with `gate_min_year_days=200`):
- PASS overall
- Note: partial 2026 slice (19 trading days) is excluded from yearly gate scoring.

### Strategy variations tested (same predictions)
- `hold10_drop10`: ann 0.1719, excess 0.0615 (best among tested dropout variants)
- `hold10_drop5`: ann 0.0614, excess ‑0.0490
- `topk30_hold10`: ann 0.0978, excess ‑0.0126

### Score-weighted strategy (exploratory, same predictions)
- `WeeklyScoreWeightedStrategy` tested with:
  - `weighting=softmax, topk=50, temperature=1.0, score_clip=3.0, max_weight=0.05`
  - ann 0.1247, IR 0.4192, MDD ‑0.2935, excess 0.0143
  - turnover higher (~0.1725)

This remains exploratory and underperforms `v2_hold10` on annualized and excess return.

---

## 8) Differences between pred.pkl files

Predictions differ significantly between runs due to:
- Different label horizon (5‑day vs 10‑day)
- Different training start (2000 vs 2016)
- Different PIT fields / extra fields
- Different processors
- Different model hyperparameters

Example: correlation between two earlier preds (5‑day vs 10‑day) was ~0.142, even though shapes matched.

---

## 9) Open issues and risks

1) Validation instability
- Even in the best variant (hold 10), 2022, 2024, and 2026 YTD are negative.
- The pipeline is not stable enough to be called “robust” or “trading ready.”

2) LightGBM is not learning a strong signal
- Early stopping at iteration 1 suggests weak predictive power.

3) Missing PIT fields persist
- Missingness is real and should be handled carefully; indicators help, but signal quality still low.

4) Calendar holiday handling
- Rebalance weekday now correctly moves to next trading day if holiday.
- The validator still reports “weeks_missing_rebalance_day” but that is informational.

---

## 10) Current pipeline recommendation

The best candidate at this moment is:
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml`

But it is **not yet trading‑ready** due to negative years and weak robustness.

---

## 11) Suggested next steps (if you want higher accuracy + robustness)

1) Hyperparameter search with fixed PIT and missingness indicators
- Optimize for a target metric (e.g., IC/ICIR or excess ann_return) on validation.

2) Feature selection
- Drop high‑missing PIT fields or add more missingness indicators.

3) Strategy upgrade
- Consider score‑weighted strategies as an exploratory branch, but require explicit parameter tuning and validation.

4) Walk‑forward CV
- Ensure stability across multiple test windows (not just a single long test period).

---

## 12) Quick reproduction commands

Run pipeline (v2):
```
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2.yaml
```

Validate results:
```
python scripts/validate_us_sharadar_pipeline.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2.yaml \
  --pred mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --by_year \
  --check_gates \
  --check_rolling
```

Hold 10 variant (best so far):
```
python scripts/validate_us_sharadar_pipeline.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10.yaml \
  --pred mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --check_data_quality \
  --fail_on_data_quality_fail \
  --by_year \
  --check_gates \
  --check_rolling \
  --rolling_window_days 504 \
  --rolling_step_days 252 \
  --rolling_min_days 252
```

Risk-managed hold10 variant (lower drawdown focus):
```
python scripts/validate_us_sharadar_pipeline.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk.yaml \
  --pred mlruns/907430387469519321/d00d2865e3994cc795586ed0b89fd6ff/artifacts/pred.pkl \
  --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
  --check_data_quality \
  --fail_on_data_quality_fail \
  --check_gates \
  --check_rolling \
  --rolling_window_days 504 \
  --rolling_step_days 252 \
  --rolling_min_days 252
```

---

## 13) Summary

- PIT data leak was fixed (datekey + 45‑day offset).
- Missing PIT is now real and manageable, not 100% missing.
- Best configuration so far improves excess return but is not robust across years.
- The pipeline is **not ready for live trading** without further tuning and stability work.

---

## 14) 2026-02-08 Drawdown/Stability Addendum

Additional risk-managed tuning was completed and validated against the ETF basket benchmark using:
- strict data/report quality checks
- stress test (`cost x2`)
- yearly gates
- rolling walk-forward (`window=504`, `step=252`, `min_days=252`)

### A) Risk variant comparison (full period)

| Variant | Full Excess Ann | Full MDD | Full Turnover | Stress Excess Ann |
| --- | ---: | ---: | ---: | ---: |
| `v2_hold10_risk` | 0.0390 | -0.3393 | 0.0557 | 0.0258 |
| `v2_hold10_risk_v2` (pre-boost) | 0.0208 | -0.2995 | 0.0501 | 0.0090 |
| `v2_hold10_risk_v2` (boost v1) | 0.0229 | -0.3009 | 0.0504 | 0.0110 |
| `v2_hold10_risk_v2` (boost v2, current) | 0.0253 | -0.3029 | 0.0507 | 0.0134 |
| `v2_hold10_risk_v3` | 0.0049 | -0.2474 | 0.0445 | -0.0056 |

Interpretation:
- `risk_v3` gives the lowest drawdown but stress excess is negative (too defensive).
- `risk_v2` family is the best low-drawdown compromise.
- `risk_v2` boost v2 improved excess and stress excess vs pre-boost with only a small drawdown increase.

### B) Rolling stability result (key blocker)

For `v2_hold10_risk_v2` boost v2:
- `rolling_pass_rate = 0.50` (2/4 windows, threshold 0.60) -> **FAIL**
- worst rolling excess annualized: `-0.0271`

Window details:
- 2022-01-03 -> 2024-01-04: PASS (`excess 0.0147`)
- 2023-01-04 -> 2025-01-06: FAIL (`excess -0.0271`)
- 2024-01-05 -> 2026-01-08: FAIL (`excess -0.0001`)
- 2025-01-07 -> 2026-01-29: PASS (`excess 0.0903`)

Conclusion:
- Drawdown reduction work succeeded.
- Regime stability is improved but still insufficient for trading readiness.

### C) Stability-tooling update (implemented)

- `scripts/rank_us_sharadar_candidates.py` now supports rolling walk-forward checks and stability-aware scoring:
  - tracks `rolling_ok`, `rolling_pass_rate`, `rolling_worst_excess`, `rolling_worst_mdd_abs`
  - sorts candidates by gate pass, then rolling pass, then rolling pass rate
- `examples/benchmarks/LightGBM/us_sharadar_candidate_ranking_etf.yaml` now includes a `rolling:` section so ranking is stability-first by default.

### D) Rejected tuning (kept optional, disabled in config)

- Added optional strategy parameters for bullish regime floor:
  - `market_bull_floor_thresh`
  - `market_bull_risk_floor`
- Directly enabling this floor increased full/stress drawdown materially (around `-0.37/-0.38`) and was rolled back from `risk_v2`.

### E) Benchmark-excess label correction (2026-02-08)

The benchmark-relative training path was kept, but the label processor was corrected:
- `BenchmarkExcessLabel` now aligns benchmark forward return with qlib label semantics:
  - from `Ref($close, -(s+N))/Ref($close, -s) - 1`
  - with defaults `s=1`, `N=10` for this weekly pipeline.
- A missing-data branch bug was fixed:
  - removed invalid `bench_vals.loc[df.index]` (MultiIndex mismatch risk)
  - now uses position-based filtering aligned with dropped rows.

Quick synthetic sanity checks passed:
- expected benchmark subtraction alignment produced consistent transformed labels
- missing benchmark values correctly reduced rows without exceptions

### F) Asymmetric risk ramp experiment (rejected for active config)

Added optional strategy knobs:
- `risk_smoothing_up`, `risk_smoothing_down`
- `max_risk_step_up`, `max_risk_step_down`

A short A/B (2025-01-02 -> 2025-03-31, ETF basket benchmark, same `pred.pkl`) showed no improvement when enabled:

| Variant | Full Excess Ann | Full MDD | Full Turnover | Stress Excess Ann |
| --- | ---: | ---: | ---: | ---: |
| baseline (`risk_v2`) | -0.0550 | -0.1526 | 0.0526 | -0.0664 |
| asymmetric-on | -0.0605 | -0.1543 | 0.0513 | -0.0716 |

Decision:
- keep these knobs available in code for future tuning
- keep active `risk_v2` config unchanged (feature disabled by default)

### G) Controlled proxy A/B: baseline vs benchmark-excess training (2026-02-08)

To unblock decisions while full-universe retrain remained expensive, a matched proxy A/B was run:
- same date range (`2025-01-02 -> 2026-01-29`)
- same strategy family (`risk_v2`)
- same ETF basket benchmark and validator gates
- only difference: training label path
  - baseline: raw forward return label
  - excess: `BenchmarkExcessLabel` adjusted label

Results:

| Variant | Full Excess Ann | Full MDD | Stress Excess Ann | Turnover | Rolling Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy baseline | -0.0633 | -0.2251 | -0.0750 | 0.0524 | 0.0000 |
| Proxy excess-label | 0.0520 | -0.2763 | 0.0397 | 0.0528 | 1.0000 |

Interpretation:
- Excess-label training materially improved excess return and rolling stability in this controlled proxy.
- Drawdown worsened vs proxy baseline.
- Therefore, instability improved, but drawdown objective still needs additional work.

Data quality note from the same proxy validation:
- baseline: `data_quality_overall = FAIL` (missing top-k close/volume ratio too high)
- excess-label: `data_quality_overall = PASS`

### H) Defensive risk parameter checks on excess predictions (2026-02-08)

Two defensive strategy variants were evaluated on the same excess predictions:

| Variant | Full Excess Ann | Full MDD | Stress Excess Ann | Turnover | Rolling Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excess current proxy | 0.0520 | -0.2763 | 0.0397 | 0.0528 | 1.0000 |
| Excess `def1` | 0.0536 | -0.2767 | 0.0416 | 0.0522 | 1.0000 |
| Excess `ultradef` | 0.0460 | -0.2836 | 0.0371 | 0.0387 | 1.0000 |
| Excess `lowcap` | 0.0721 | -0.3009 | 0.0608 | 0.0488 | 1.0000 |

Interpretation:
- Defensive tuning in this tested direction did not reduce drawdown.
- More aggressive de-risking reduced turnover but made MDD worse.
- Current excess proxy setting remains the best drawdown among tested excess variants.

### I) Regime guard extension + proxy retune (2026-02-08)

`qlib/contrib/strategy/weekly.py` was extended with optional (default-off) controls:
- `include_current_positions`
- `market_drawdown_window`, `market_drawdown_limit`, `market_drawdown_penalty`
- `crash_guard`, `crash_drawdown_limit`, `crash_return_lookback`, `crash_return_limit`, `crash_penalty`, `crash_cooldown_steps`

Goal:
- reduce drawdown instability after sharp selloffs
- avoid changing behavior of existing configs unless explicitly enabled

Matched proxy retune (same excess `pred.pkl`, same ETF benchmark, same validator settings):

| Variant | Full Excess Ann | Full MDD | Stress Excess Ann | Turnover | Rolling Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excess current proxy (recheck) | 0.0531 | -0.2790 | 0.0409 | 0.0528 | 1.0000 |
| `stability_v4` (very defensive) | -0.0251 | -0.2239 | -0.0338 | 0.0382 | 0.0000 |
| `stability_v5_balanced` | 0.0273 | -0.2773 | 0.0166 | 0.0469 | 1.0000 |
| `stability_v6_marketdd` | 0.0320 | -0.2750 | 0.0211 | 0.0478 | 1.0000 |
| `stability_v7_crash` | 0.0307 | -0.2751 | 0.0198 | 0.0476 | 1.0000 |

Interpretation:
- Strong de-risking can cut drawdown a lot, but it can also destroy excess and fail robustness.
- The best balanced candidate in this proxy was `stability_v6_marketdd`:
  - modest drawdown improvement vs current proxy
  - lower turnover
  - still positive full/stress excess
- A new config was added for full-universe confirmation:
  - `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability.yaml`

Promotion decision:
- keep `risk_v2_excess` as the active excess baseline until full-universe run confirms `stability` is better across rolling windows.
