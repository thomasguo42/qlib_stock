# US Sharadar Live Readiness Checklist

This pipeline is not considered ready for live trading unless every item below passes on current data.

## Required Release Gates

- Train/valid/test segments must have an embargo of at least the label horizon. For the 10-day label (`Ref($close, -11) / Ref($close, -1) - 1`), require `10` trading days between train and valid, and between valid and test.
- Run `scripts/validate_us_sharadar_pipeline.py` with `--fail_on_input_check_fail`, `--check_data_quality`, `--fail_on_data_quality_fail`, `--check_gates`, `--fail_on_gate_fail`, `--check_rolling`, and `--fail_on_rolling_fail`.
- External benchmark gates must include `QQQ,SPY,IXIC`. `QQQ` and `SPY` are loaded from qlib ticker data; `IXIC` is the Nasdaq Composite baseline and should be supplied as `/root/.qlib/qlib_data/us_data/bench_ixic.pkl` when it is not present in qlib features. The validator accepts the user-facing alias `ICIC` and canonicalizes it to `IXIC`.
- Growth-track candidates must pass the latest rolling external-baseline gate. Current default retirement warning threshold is latest rolling annualized excess below `-10%` versus each required benchmark.
- Prediction artifacts must cover the backtest range and must not extend outside the configured test segment unless the run is explicitly a tail-extension validation using `--allow_pred_beyond_test_segment`.
- PIT fundamentals freshness must be checked with `--pit_staleness_max_p95_days`; the release runner uses a `189` calendar-day threshold over a `252` trading-day tail window.
- Data update runs should use `--fail_on_update_gaps` so missing SEP coverage, empty deltas, or unexpected instrument clamps fail instead of only printing warnings.
- Release validation must require SF3A and SF1 provenance markers. Missing provenance means the local qlib store may contain stale no-lag event bins or stale PIT fundamentals.
- Release validation must include the live-stress profile used by `scripts/run_us_sharadar_release.py`: `--stress_cost_mult 3.0 --stress_deal_price open`.
- Ensemble release validation should pass `--ensemble_manifest` and `--model_quality_use_ensemble_source_horizons` so fallback h60 dates are judged against the h60 forward horizon instead of the primary h20 horizon.
- Release candidates should append a trial registry record with `--trial_registry`. Use the registry count, or an explicit `--trial_count`, as the multiple-testing haircut denominator.

## Data Availability Rules

- SF2 insider features use `filingdate`.
- SF3A/13F snapshot features must not use `calendardate` as if it were known immediately. The update pipeline now applies `--sf3a_availability_lag_days` and defaults to `45` calendar days.
- Existing SF3A bins must be rebuilt once with `scripts/rebuild_us_sharadar_sf3a_features.py --dump_to_qlib`. Incremental updates alone do not repair historical no-lag bins.
- SF1 PIT fundamentals should be refreshed with `scripts/refresh_us_sharadar_pit.py --dump_to_qlib --overwrite` before a release run. Best practice is `--date_col datekey --date_offset_days 0` so the PIT date is the publication date.
- The SF1 refresh wrapper verifies post-download ticker coverage before writing provenance; keep `--max_missing_sf1_ticker_ratio` strict for release refreshes.
- Benchmark ETF tickers should come from `_sfp_benchmark_tickers.txt` in this repo or an explicit `--benchmark_tickers_file`.
- Build qlib ETF benchmark return pickles with `scripts/build_us_qlib_ticker_benchmark.py --tickers QQQ,SPY`. Build the Nasdaq Composite return pickle with `scripts/build_external_index_benchmark.py --ticker IXIC`.

## Ensemble & Fallback Policy

- The normal fallback state is the defensive h60 model, not cash. Severe live deterioration should retire the active overlay to QQQ-only operationally until a replacement candidate passes release gates.
- The preferred ensemble gate is no-lookahead: score-dispersion collapse can be combined with shifted QQQ/benchmark trend, drawdown, and volatility using `scripts/ensemble_us_sharadar_predictions.py --method regime_confidence_gate`.
- The release report must show source attribution by date so the reviewer can see how often the model used h20, h60, or regime fallback.

## Manual Trade Plan Inputs

- `scripts/generate_us_trade_plan_from_pred.py` validates the strategy score date, not the trade date. A Monday trade plan can validly consume Friday scores.
- When passing `--positions_csv`, include `count_day`, `holding_days`, or `days_held`. Strategies with `hold_thresh` need this state to decide which holdings are sellable.
- `--cash` is available cash for new buys and is required when `--positions_csv` is supplied. Use `--cash 0` for a fully invested rebalance.
- `--capital` remains only as a deprecated flat-start alias when no positions file is supplied; `--total_equity` is reporting-only.

## Stock Selector Report Categories

- `scripts/generate_us_stock_selector_report.py` is a decision-support report, not an order generator. It now has two layers: a strict absolute assessment and an actionable selector action.
- Expected edge means model-calibrated QQQ-excess opportunity over the primary horizon. It combines score percentile/z-score with candidate-specific and bucket-level historical calibration.
- Risk combines QQQ beta, realized volatility, drawdowns, liquidity, and historical downside-tail calibration. A stock can have a high model rank and still fail strict assessment if calibrated edge is weak or the downside tail is too wide.
- Confidence is separate from risk. It reflects calibration sample size, core field availability, FMP freshness/coverage when available, and sleeve/driver attribution.
- Strict assessment meanings: `Core Candidate` is high edge with non-high risk; `Aggressive Upside` is high edge with high risk; `Low Priority / Defensive` is medium or low edge with acceptable risk; `Avoid / Watch Only` is weak edge or poor risk/reward; `Needs Review` means confidence or data quality is too weak for a normal strict category.
- Selector action meanings: `Best Current Candidates` are the best risk-adjusted current review candidates, `Speculative / High Upside` are higher-risk names with stronger upside evidence or utility rank, `Watchlist Only` are secondary review names, `Avoid` are outside the diversified selector watchlist, and `Needs Review` means confidence is too weak.
- The report exports `candidate_diagnostics.csv` for the full candidate pool, including model rank, selector rank, utility components, strict assessment, selector action, and selected/rejected status.
- `scripts/score_us_stock_selector_reports.py` scores realized outcomes by selector action, strict assessment, utility decile, selector-rank bucket, and edge/risk/confidence tiers once 5/10/20-day exits are available.

## Current Main Config

The current topk40 DD config has been updated to use a 10-trading-day static embargo:

- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable_topk40_dd_v1.yaml`

Any older config with adjacent train/valid/test windows should be treated as research-only until its segments are rebuilt or generated with `scripts/walkforward_train_us_sharadar.py`.

## Rebuild Order

1. Refresh SF1 PIT with `scripts/refresh_us_sharadar_pit.py --dump_to_qlib --overwrite --date_col datekey --date_offset_days 0`.
2. Rebuild lag-corrected SF3A features with `scripts/rebuild_us_sharadar_sf3a_features.py --dump_to_qlib`.
3. Run `scripts/update_us_sharadar_data.py --fail_on_update_gaps` for current SEP/SFP/SF2/SF3A deltas.
4. Rebuild benchmark return pickles: `scripts/build_us_qlib_ticker_benchmark.py --tickers QQQ,SPY` and `scripts/build_external_index_benchmark.py --ticker IXIC`.
5. Retrain the controlled candidate set with embargoed walk-forward splits.
6. Build the ensemble prediction artifact and manifest.
7. Validate with `scripts/run_us_sharadar_release.py --baseline_tickers QQQ,SPY,IXIC --baseline_pkl_map IXIC=/root/.qlib/qlib_data/us_data/bench_ixic.pkl --ensemble_manifest <manifest> --trial_registry <registry.jsonl>`.
8. Generate a trade plan with explicit cash and holdings age only after every required gate passes.
