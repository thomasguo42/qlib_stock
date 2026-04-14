# US Sharadar COR Upgrade Progress (2026-02-10)

## What was implemented

- Verified live Sharadar datatable access with current `NDL_API_KEY`.
- Confirmed accessible core + COR tables:
  - `TICKERS, SEP, SFP, SF1, DAILY, ACTIONS, EVENTS, SF2, SF3, SF3A, SF3B, SP500, INDICATORS`.
- Upgraded bundle map to real COR table codes and correct date keys:
  - insiders: `SF2` (`filingdate`)
  - institutional snapshots: `SF3A` (`calendardate`)
  - institutional investor-level (very large): `SF3`
- Added safer map ingestion support:
  - `download_from_map --max_tickers` global cap
  - per-entry `max_tickers` override support
- Extended `prepare_event_features.py`:
  - supports input directory of per-ticker CSVs
  - supports `aggregation_mode=event|snapshot`
  - supports explicit output range via `--resample_start/--resample_end`
- Added tests:
  - `tests/test_prepare_event_features.py`

## Pilot data run completed

Universe used for pilot:
- `pit_mrq_large_cor100` (100 symbols with SF2/SF3A event features)

Pulled and prepared:
- `raw/daily` (100 files)
- `raw/actions` (100 files)
- `raw/events` (100 files)
- `raw/sf2` (100 files)
- `raw/sf3a` (100 files)

Feature outputs:
- `prepared/sf2_features` (100 files)
- `prepared/sf3a_features` (100 files)
- dumped into qlib bin feature store (`features/<symbol>/...`) for pilot symbols

## Full-universe data expansion completed

Expanded the same flow from pilot to full `pit_mrq_large` universe:
- `raw/sf2`: `1034` files
- `raw/sf3a`: `1034` files
- `prepared/sf2_features`: `1032` files (only symbols with usable SF2 rows)
- `prepared/sf3a_features`: `1033` files (only symbols with usable SF3A rows)

Then refreshed qlib bins with `dump_bin.py dump_fix` for both feature sets.

## Universe integrity hardening + backfill (new)

Identified a structural issue: `pit_mrq_large` excludes many primary large-cap names
(`AAPL/MSFT/NVDA/AMZN/GOOGL/...`) despite those symbols existing in `all.txt` and
having full PIT/price data. This is a release blocker for trading use.

Implemented:
- `scripts/audit_us_universe_integrity.py`
  - audits overlap vs reference universes (default `sp500,nasdaq100`)
  - checks anchor symbols (supports alias groups like `META|FB`)
  - supports fail-fast mode for CI/release gates
- `scripts/build_us_union_universe.py`
  - builds merged instrument files from existing sources
  - supports `all.txt` bounds clamping and forced anchor inclusion
- `scripts/run_us_sharadar_release.py`
  - now runs universe integrity audit before train/validate (unless skipped)

Built improved market:
- `/root/.qlib/qlib_data/us_data/instruments/pit_mrq_large_idx.txt`
  - sources: `pit_mrq_large + sp500 + nasdaq100` (+ forced anchors)
  - symbols: `1127`
  - integrity audit (`2016-01-01 -> 2026-01-30`):
    - S&P500 overlap: `547/615 = 88.94%`
    - Nasdaq100 overlap: `124/157 = 78.98%`
    - anchor coverage: `PASS` (0 missing)

Backfilled COR features for new symbols in `pit_mrq_large_idx`:
- computed missing list from new market vs existing COR raw: `93` symbols
- downloaded `SF2` + `SF3A` for those symbols
- prepared/dumped additional features:
  - prepared missing batch: `83` usable symbols for SF2 and `83` for SF3A
  - qlib feature-store coverage on `pit_mrq_large_idx`:
    - insider key field present for `1115/1127` symbols (`98.94%`)
    - inst13f key field present for `1116/1127` symbols (`99.02%`)
  - anchor symbols (`AAPL/MSFT/NVDA/AMZN/GOOGL/META/...`) now all have SF2+SF3A bins

## Pilot model integration

Created pilot config:
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_pilot.yaml`

Adds COR event features into `extra_fields`:
- insider rolling sums/means
- SF3A snapshot change/pct features
- missingness indicators for event fields

Also created full-market config:
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full.yaml`

Created improved full-market config on repaired universe:
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable.yaml`
- changes:
  - market: `pit_mrq_large_idx`
  - replaced unstable SF3A `%` features with more stable change/count variants:
    - `inst13f_totalvalue_63d_chg`, `inst13f_shrunits_63d_chg`, `inst13f_count_63d_sum`
  - shifted insider window features from `20d` to `63d` sums to reduce sparsity

## Pilot training + validation result

Run:
- `run_id: f686d8c5c19c42dbb472328c30476bb3`

Release-gate validation (ETF benchmark, 2022-01-03 to 2026-01-29):
- Full annualized return: `0.1319`
- Full excess annualized return: `0.0215`
- IR: `0.7068`
- Max drawdown: `-0.2071`
- Stress excess annualized return: `0.0127`

Gate outcome:
- `full_excess_ann`: FAIL (`0.0215 < 0.0300`)
- `positive_excess_years`: FAIL (`1 < 3`)
- Overall release gates: **FAIL**

## Conclusion

- COR ingestion and feature plumbing are now operational.
- The first integrated pilot with SF2/SF3A features is **not release-ready**.
- It improved drawdown profile but did not deliver sufficient benchmark excess robustness for deployment.
- Full-universe COR feature data is ready; full-market retrain/validation is the remaining compute step.

## Recommended next implementation steps

1. Scale SF2/SF3A ingestion from 100 symbols to full `pit_mrq_large` and retrain.
2. Keep `SF3` off by default (too heavy) unless explicitly needed for a targeted experiment.
3. Add event-feature-specific regularization and feature selection (drop unstable event fields).
4. Run candidate ranking with strict release gates and rolling walk-forward before promoting any config.

## Current run status (latest)

- Old full run on flawed universe was stopped:
  - run id: `36518f28c25c47a394542458be9285ca`
  - reason: `pit_mrq_large` integrity failure (missing primary large caps)
- Active full run on repaired universe + stabilized event set:
  - config:
    - `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable.yaml`
  - run id: `905a3dcfcfb74798b9ea9ca7a50f9dbf`
  - status at update time: training/data-processing in progress; `pred.pkl` not yet produced
  - strict validator is queued to auto-run immediately after `pred.pkl` appears:
    - output target: `/tmp/validate_cor_events_full_uplus_stable.txt`

## 2026-02-11 update: strategy-only tuning result + retrain ablation prep

Strategy-only tuning on the same trained predictions (`run_id: 248d0bb3bbf94625b13a257d55f78c44`)
did not improve release viability:

- `..._uplus_stable_ddguard_v1.yaml`
  - full ann: `0.0991`
  - full excess ann: `-0.0113`
  - full mdd: `-0.3573`
- `..._uplus_stable_topk40_dd_v1.yaml`
  - full ann: `0.0561`
  - full excess ann: `-0.0542`
  - full mdd: `-0.3951`

Conclusion:
- Portfolio/risk parameter changes alone degraded excess return.
- The next step must be model/data ablation retraining rather than additional strategy-only tuning.

Implemented for retrain ablation:

- new helper script:
  - `scripts/build_us_sharadar_ablation_configs.py`
  - generates COR ablation variants from a base config
- generated configs:
  - `..._uplus_stable_nocor.yaml`
  - `..._uplus_stable_sf2only.yaml`
  - `..._uplus_stable_sf3aonly.yaml`
  - `..._uplus_stable_cor_full.yaml`
- new batch runner:
  - `scripts/run_us_sharadar_ablation_batch.sh`
  - runs train + release validation sequentially for all four ablation configs
