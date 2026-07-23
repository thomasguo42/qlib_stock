# US Sharadar/FMP Qlib Repo Status and Handoff

Date: 2026-05-14

This document records the current state of the customized US equity research
pipeline in this repository. It summarizes what has been built, which model
families and experiments have been tried, what is currently failing, what FMP
adds, and how the important files fit together.

No API keys are written here. The pipeline expects secrets in environment
variables such as `NDL_API_KEY` and `FMP_API_KEY`.

## Executive Verdict

The repo now has a serious research and validation pipeline for US equities
using Sharadar plus FMP data. The mechanics are much stronger than the early
versions:

- Sharadar SEP/SFP/SF1/SF2/SF3A ingestion is implemented.
- Point-in-time SF1 fundamentals are refreshed and stamped with provenance.
- SF3A/13F snapshot features are rebuilt with an explicit 45 calendar-day lag.
- FMP event features are downloaded, lagged, transformed, dumped to Qlib, and
  stamped with provenance.
- QQQ, SPY, and IXIC benchmark files are wired into validation.
- Walk-forward training, release validation, failure maps, and candidate trial
  registries are implemented.
- The validator checks train/valid/test embargoes, prediction coverage,
  benchmark gates, rolling robustness, training diagnostics, model quality,
  strategy-weighted label quality, rebalance interval quality, active risk,
  data quality, and provenance.

The current model set is not release-grade. No current candidate passes the
strict `qqq_release` profile. FMP is helping as a signal source, but it has not
yet produced a robust QQQ-beating portfolio candidate.

The best strict FMP candidates are still short of the release bar:

| Candidate | Strict status | QQQ excess ann | QQQ stress excess ann | QQQ rolling pass | Worst rolling QQQ excess | QQQ down-day excess |
|---|---:|---:|---:|---:|---:|---:|
| `score_growth_v8_h20_qqqexcess_fmpv2_stack_core35_topk30` | FAIL | `0.0559` | `0.0308` | `0.6250` | `-0.0913` | `0.9532` |
| `score_growth_v9_h5_qqqexcess_fmpv2_ranker_core55_topk30` | FAIL | `0.0527` | `0.0348` | `0.6667` | `-0.1261` | `-0.3530` |
| `score_growth_v11_h5_qqqutility_fmpv2_ranker_mildrisk_core55_topk30` | FAIL | `-0.0795` | `-0.0964` | `0.0000` | `-0.1597` | `0.4167` |
| `score_growth_v12_h5_qqqsector_fmpv2_ranker_mildrisk_core55_topk30` | FAIL | `-0.0162` | `-0.0329` | `0.4444` | `-0.0509` | `-0.0133` |

The strict `qqq_release` external baseline gates require roughly:

- Full annualized excess versus QQQ/SPY/IXIC at least `0.08`.
- Stress annualized excess at least `0.04`.
- Rolling pass rate at least `0.85`.
- Worst rolling excess at least `0.00`.
- Latest rolling excess at least `0.02`.
- QQQ down-day regime excess at least `0.20`.
- Max drawdown and drawdown gap within strict limits.

The best current FMP candidates get to about `5.3%` to `5.6%` annualized QQQ
excess, but not the required `8%`, and their rolling/regime behavior is not
consistent enough.

## What The Repo Can Do

This repo is a customized pyqlib tree. In addition to standard Qlib examples,
docs, tests, and library code, it can:

1. Download and update Sharadar data from Nasdaq Data Link.
2. Normalize Sharadar daily prices into Qlib binary format.
3. Refresh Sharadar SF1 point-in-time quarterly fundamentals.
4. Build Sharadar SF1 ratio features.
5. Build Sharadar SF2 insider event features.
6. Build Sharadar SF3A institutional snapshot features with availability lag.
7. Build stock-level risk, market-regime, breadth, sector, and metadata
   features.
8. Download FMP event-like raw data while redacting API keys from logs and
   manifests.
9. Build PIT-conservative FMP event features.
10. Dump FMP features into the Qlib provider.
11. Build QQQ, SPY, IXIC, ETF-basket, and SFP benchmark return pickles.
12. Train Qlib LightGBM and deterministic score models.
13. Run embargoed walk-forward training and stitch predictions.
14. Run strict release validation against external benchmarks.
15. Produce failure maps from validation logs.
16. Rank candidates using common robustness gates.
17. Blend prediction files with ensemble/fallback logic.
18. Sweep strategy parameters for an existing prediction artifact.
19. Generate manual trade plans from `pred.pkl` plus current holdings/cash.

## Current Local Data State

Important local stores:

| Path | Role |
|---|---|
| `/root/.qlib/qlib_data/us_data` | Main Qlib provider used by configs and validators. |
| `/root/.qlib/sharadar` | Raw/prepared Sharadar working area. |
| `/root/.qlib/fmp` | Raw/prepared FMP working area. |
| `artifacts/research` | Experiment outputs, summaries, logs, manifests, and failure maps. |
| `artifacts/release` | Older release manifests and repair artifacts. |
| `mlruns` | MLflow local tracking store for model runs. |
| `workspace` | Older operational/workflow outputs, walk-forward runs, logs, and trade plans. |

Current provider metadata files:

| File | Meaning |
|---|---|
| `/root/.qlib/qlib_data/us_data/metadata/sharadar_sf1_pit.json` | SF1 PIT refresh provenance. Current record uses `datekey`, no artificial date offset, `MRQ`, `pit_mrq_large_idx`, and `1125/1137` ticker coverage. |
| `/root/.qlib/qlib_data/us_data/metadata/sharadar_sf3a_features.json` | SF3A feature rebuild provenance. Current record uses 45 calendar-day availability lag and wrote `49940` feature bins. |
| `/root/.qlib/qlib_data/us_data/metadata/sharadar_sf1_ratio_features.json` | Provenance for precomputed SF1 ratio features. |
| `/root/.qlib/qlib_data/us_data/metadata/sharadar_model_features.json` | Provenance for risk/regime/static model features. Current record includes `86108` bins and market features through `2026-05-05`. |
| `/root/.qlib/qlib_data/us_data/metadata/fmp_event_features.json` | FMP event feature provenance. Current record uses 1 calendar-day lag, 171 columns, directional feature version 2, and `194427` bins. |

Current benchmark pickles in the provider:

| File | Role |
|---|---|
| `/root/.qlib/qlib_data/us_data/bench_qqq.pkl` | QQQ daily return benchmark. |
| `/root/.qlib/qlib_data/us_data/bench_spy.pkl` | SPY daily return benchmark. |
| `/root/.qlib/qlib_data/us_data/bench_ixic.pkl` | Nasdaq Composite / IXIC daily return benchmark. The user previously referred to this as ICIC; the validator canonicalizes that alias. |
| `/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl` | Equal-weight SFP ETF basket benchmark used by older runs. |

## FMP Data Status

FMP has been integrated, but only PIT-safer event history is used for
historical backtests. The current FMP provenance classifies datasets this way:

| Classification | FMP datasets |
|---|---|
| PIT-safe event history | `earnings`, `grades`, `grades_historical`, `price_target_news` |
| Excluded for historical modeling | `analyst_estimates_annual`, `analyst_estimates_quarter`, `price_target_summary`, `price_target_consensus` |

Reason for exclusion: the excluded datasets are current-only or do not provide
sufficient as-of history to prove they were known at each historical signal
date. They can be useful for live/current dashboards, but not for historical
training unless we obtain reliable historical as-of snapshots.

FMP standalone feature IC is real but mixed. Stronger directional columns:

| FMP feature | Mean IC | ICIR | Positive IC rate | Notes |
|---|---:|---:|---:|---|
| `FMP_ALPHA_EARN_SURPRISE_LATEST` | `0.0183` | `3.6451` | `0.7407` | Strongest FMP event signal. |
| `FMP_ALPHA_RATING_BULLISH` | `0.0166` | `3.2649` | `0.6667` | Useful analyst-rating signal. |
| `FMP_ALPHA_RATING_BEARISH_PENALTY` | `0.0164` | `5.2562` | `0.7407` | Useful downside-avoidance signal. |
| `FMP_ALPHA_EVENT_COMPOSITE` | `0.0159` | `3.1400` | `0.6296` | Broad event composite. |
| `FMP_ALPHA_GRADE_SCORE_LATEST` | `0.0137` | `3.4054` | `0.6296` | Useful but less robust than earnings/rating signals. |

Weak or negative FMP components include some price-target upside windows,
grade-revision windows, and event-coverage features. FMP should be used
selectively. Broadly adding every FMP field is not enough and can hurt.

## Main Pipeline

The production-like pipeline is:

1. Data refresh and repair.
2. Feature rebuild.
3. Benchmark rebuild.
4. Config generation or selection.
5. Walk-forward training.
6. Strict validation.
7. Failure-map summary.
8. Candidate ranking.
9. Trade-plan generation only after all gates pass.

### Data Refresh

Sharadar update flow:

```bash
python scripts/update_us_sharadar_data.py \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --market pit_mrq_large_idx \
  --fail_on_update_gaps
```

Main effects:

- Downloads SEP/SFP/SF2/SF3A deltas.
- Rebuilds warmup SF2/SF3A feature deltas.
- Dumps daily price and event-feature deltas into the Qlib provider.
- Updates benchmark ETF SFP data.
- Extends/clamps instrument market dates if needed.
- Writes update reports under `/root/.qlib/sharadar/reports`.

SF1 PIT refresh:

```bash
python scripts/refresh_us_sharadar_pit.py \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --market pit_mrq_large_idx \
  --date_col datekey \
  --date_offset_days 0 \
  --dump_to_qlib \
  --overwrite
```

SF3A full lag-correct rebuild:

```bash
python scripts/rebuild_us_sharadar_sf3a_features.py \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --availability_lag_days 45 \
  --dump_to_qlib
```

FMP refresh flow:

```bash
FMP_API_KEY=<redacted> python scripts/download_fmp_data.py \
  --symbols_file <universe-or-symbol-file> \
  --datasets earnings,grades,grades_historical,price_target_news \
  --out_root /root/.qlib/fmp

python scripts/build_fmp_event_features.py \
  --raw_root /root/.qlib/fmp/raw \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --availability_lag_days 1 \
  --dump_to_qlib
```

### Training

Release training should use:

```bash
python scripts/run_us_sharadar_release.py \
  --config <config.yaml> \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --market pit_mrq_large_idx \
  --train_mode walkforward \
  --gate_profile qqq_release \
  --baseline_tickers QQQ,SPY,IXIC \
  --baseline_pkl_map IXIC=/root/.qlib/qlib_data/us_data/bench_ixic.pkl \
  --collect_all_diagnostics
```

The runner performs preflight audits, builds a runtime config, runs
walk-forward training through `scripts/walkforward_train_us_sharadar.py`, then
runs `scripts/validate_us_sharadar_pipeline.py`.

### Validation

Strict validation requires these major check families:

| Check family | Purpose |
|---|---|
| `training_diagnostics` | Reject weak/near-constant or broken walk-forward training folds. |
| `input_check` | Validate config, prediction window, embargo, benchmark, and segment consistency. |
| `data_quality` | Check score coverage, NaN rates, top-k close/volume availability, benchmark missingness, PIT freshness, and provenance. |
| `model_quality` | Check rank IC, recent IC, positive IC rate, top-k forward labels, and top/bottom quantile spread. |
| `strategy_weighted_quality` | Evaluate forward labels under approximate strategy weights. |
| `rebalance_interval_quality` | Check realized interval returns from one rebalance to the next. |
| `active_risk` | Check benchmark drift, active share, stock overlay, sector active weight, and core weight gaps. |
| `external_baseline_gates` | Compare full/stress/year/rolling/regime returns against QQQ/SPY/IXIC. |
| `robustness_gates` | Generic full/stress/year drawdown, IR, turnover, and excess gates. |
| `rolling` | Rolling-window pass-rate and worst-window checks. |

### Trade Plan

Trade plans are generated from validated predictions:

```bash
python scripts/generate_us_trade_plan_from_pred.py \
  --config <config.yaml> \
  --pred <pred.pkl> \
  --positions_csv <positions.csv> \
  --cash <cash-available>
```

This is not a release approval step. It should only be used after validation
passes.

## Models And Approaches Tried

The repository currently has 168 US Sharadar workflow configs under
`examples/benchmarks/LightGBM`. Their model/strategy composition from the repo
scan is:

| Category | Count |
|---|---:|
| `LGBModel` configs | 93 |
| `LGBRankerModel` configs | 44 |
| `ICSelectedScoreModel` configs | 13 |
| `StackedSignalScoreModel` configs | 6 |
| `FeatureWeightedScoreModel` configs | 6 |
| `RegimeSleeveScoreModel` configs | 4 |
| `WeeklyBenchmarkAwareScoreWeightedStrategy` configs | 105 |
| `WeeklyRiskManagedTopkDropoutStrategy` configs | 38 |
| `WeeklyTopkDropoutStrategy` configs | 18 |

Targets/horizons represented:

| Target or processor | Count |
|---|---:|
| `BenchmarkExcessLabel` | 65 |
| `GroupNeutralize` label target | 64 |
| Raw forward return | 27 |
| `PortfolioUtilityExcessLabel` | 7 |
| `ResidualForwardReturnLabel` | 5 |

| Horizon | Count |
|---|---:|
| 5 trading days | 39 |
| 10 trading days | 46 |
| 20 trading days | 30 |
| 40 trading days | 29 |
| 60 trading days | 21 |

### Experiment Family Ledger

| Family | Main config pattern | Purpose | Outcome |
|---|---|---|---|
| Early baseline | `workflow_config_lightgbm_Alpha158_us_sharadar*.yaml` | Establish Qlib/Sharadar weekly PIT training. | Useful bootstrap, not release-grade. |
| Best v2/v3 hold/risk | `best_v2_hold10*`, `risk_v2*`, `ddguard*`, `topk40_dd*` | Tune hold period, risk management, drawdown guard, and early COR features. | Strategy-only risk tuning reduced drawdown sometimes but often erased excess. |
| COR/SF2/SF3A | `cor_events_*`, `sf2only`, `sf3aonly`, `cor_full` | Test insider and 13F event features. | Mechanics work; alpha lift insufficient. |
| Value/stable | `value*`, `stable*` | Conservative value, factor, cap/momentum, benchmark-aware variants. | Some older candidates had attractive loose-gate excess, but failed strict/recent/rolling gates. |
| QV/risk/SF3A regime | `qv_risk_sf3a_regime*` | Quality/value/risk and SF3A features with regime/ETF-core logic. | Failed release grids, usually due robustness/model quality. |
| Growth v1/v2 | `stable*_growth_v1*`, `growth_v2*` | Benchmark-excess, absolute, beta-residual, and vol-scaled labels for higher return. | Some promising loose-gate QQQ excess, but rolling/worst-window failures remained. |
| Growth v3 / nextgen | `growth_v3*`, `nextgen_growth_v1*`, `nextgen_growth_v2*` | More benchmark-relative, residual, downside, utility, regime, and ranker approaches. | Failed strict checks; target changes did not fix top-bucket reliability. |
| Score growth v1-v3 | `score_growth_v1*` to `score_growth_v3*` | Deterministic factor composites and regime sleeve scoring. | Interpretable but not enough alpha or robustness. |
| Score growth v4 | `score_growth_v4_*_icselect*` | Train-only IC-selected deterministic feature scoring. | Mean IC could be positive, but top-bucket spread remained weak/negative. |
| Score growth v5 | `score_growth_v5_*_regimeic*` | Regime-conditioned IC feature selection. | Best pre-FMP deterministic direction, still failed strict top-bucket and worst-year checks. |
| Score growth v6 | `score_growth_v6_*_tailregimeic*` | Tail/top-quantile-aware feature selection. | Did not fix adverse-regime top-bucket problem. |
| FMP v7 | `score_growth_v7_*fmpcore*`, `*fmponly*` | First FMP feature integration. | Early failures and feature/provenance issues led to v8/v9 redesign. |
| FMP v8 | `score_growth_v8_*fmpv2_stack*` | FMP v2 directional features with stacked signal sleeves. | Best h20/core35 FMP candidate; strong down-day behavior but still fails strict release. |
| FMP v9 | `score_growth_v9_*fmpv2_ranker*` | FMP v2 features with LightGBM ranker. | Closest strict full/stress excess, but poor QQQ down-day regime and rolling failures. |
| FMP v10 | `score_growth_v10_*dynrisk*` | Dynamic risk and utility variants. | Too defensive; full QQQ excess became negative in tested run. |
| FMP v11 | `score_growth_v11_*mildrisk*` | Milder risk settings with utility target. | Fixed some downside behavior but destroyed full/stress return. |
| FMP v12 | `score_growth_v12_*qqqsector*` | Sector-aware target with mild risk. | Less bad than v11 but still negative QQQ excess and failed rolling/model quality. |

## Why Current Candidates Are Not Usable

The key failure is not basic pipeline mechanics anymore. The recurring problem
is alpha conversion.

Raw signals have some information, especially selected FMP event features and
some Sharadar/risk/regime features. However, the model plus portfolio
construction has not converted that information into stable, investable,
benchmark-beating returns after realistic costs and stress checks.

Main blockers:

1. The selected top basket is not reliable enough.
   A model can have positive mean IC while still selecting a weak top-k basket.
   Trading performance depends on top-bucket return, not average rank
   correlation.

2. Rolling robustness is poor.
   The best strict FMP rolling pass rates are `0.6250` to `0.6667`, below the
   `0.85` release requirement.

3. Regime behavior is asymmetric.
   FMP v8 does well on QQQ down days but misses full/stress/rolling thresholds.
   FMP v9 has better full/stress excess but loses heavily on QQQ down days.

4. Risk controls are too blunt.
   v10/v11/v12 show that global defensive overlays can improve one dimension
   while destroying total excess return.

5. FMP feature quality is mixed.
   Earnings surprise and ratings are useful. Price-target and event-coverage
   features are not reliably useful. A whitelist/ablation approach is safer
   than using all FMP features.

6. Multiple-testing risk is high.
   Many candidates have been tried. A candidate that only barely passes loose
   gates is not trustworthy without strict nested/walk-forward validation.

## Best Current Interpretation

FMP is helping. It provides real incremental information through event and
analyst-action features. But it is not currently enough to make this system
release-grade.

Sharadar plus FMP is still a plausible data foundation for a medium-frequency
US equity strategy. The current approach should not be abandoned, but it should
not be promoted as live-ready. The next serious research phase should be more
structural:

- Freeze current v8/v9 FMP candidates as baselines, not releases.
- Run clean Sharadar-only versus Sharadar-plus-FMP ablations under identical
  strict gates.
- Whitelist high-quality FMP features.
- Treat FMP as an explicit event/ratings sleeve, not a broad feature dump.
- Build regime-conditioned portfolio sleeves rather than one global risk
  throttle.
- Make the objective care directly about top-bucket and strategy-weighted
  forward returns.
- Only add more paid data after a clean ablation proves whether FMP's
  incremental lift is insufficient.

## Important Files And What They Do

This section focuses on the custom Sharadar/FMP/release surface. The repo also
contains the upstream Qlib library, examples, docs, and tests. Those upstream
files remain relevant because the custom pipeline runs inside Qlib, but they
are not all individually modified for this project.

### Root Files

| File | Role |
|---|---|
| `README.md`, `CHANGES.rst`, `CHANGELOG.md`, `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md` | Standard pyqlib project documentation and legal/project metadata. |
| `setup.py`, `pyproject.toml`, `MANIFEST.in`, `Makefile`, `Dockerfile`, `build_docker_image.sh` | Packaging, build, test, and Docker support for the Qlib repo. |
| `_sfp_benchmark_tickers.txt`, `_sfp_benchmark_tickers.csv` | Benchmark ETF ticker lists used when importing SFP ETF data and building ETF benchmarks. |
| `_sfp_hedge_tickers.txt` | Inverse/hedge ETF ticker list used by hedge feasibility and hedged benchmark-aware strategy experiments. |
| `_sharadar_check/` | Local schema/sample snapshots from Sharadar table discovery. Useful for debugging table access and columns. |
| `2026-01-30` | Local dated marker/file from earlier work. It is not part of the main pipeline. |

### Documentation Files

| File | Role |
|---|---|
| `docs/us_sharadar_best_pipeline.md` | Older baseline best-pipeline report from the early Sharadar PIT setup. |
| `docs/us_sharadar_best_pipeline_report_2026-02-07.md` | Earlier best-pipeline report. Historical context only. |
| `docs/us_sharadar_current_best_pipeline_2026-02-09.md` | Documents the old `best_v2_hold10` result and why it was not trading-ready. |
| `docs/us_sharadar_cor_upgrade_progress_2026-02-10.md` | Documents COR/SF2/SF3A ingestion, universe repair, and early COR results. |
| `docs/us_sharadar_experiment_detailed_status_2026-02-12.md` | Earlier detailed experiment ledger before the May 2026 FMP work. |
| `docs/us_sharadar_external_data_requirements_2026-05-11.md` | Data-needs document explaining why Sharadar alone was not enough and what external data to seek. |
| `docs/us_sharadar_live_readiness_checklist.md` | Current live-readiness checklist and required release gates. |
| `docs/us_sharadar_repo_current_status_2026-05-14.md` | This document. |
| `docs/*` standard Qlib docs | Upstream Qlib documentation for data, models, workflow, strategy, installation, and APIs. |

### Core Custom Qlib Library Files

| File | Role |
|---|---|
| `qlib/contrib/data/handler_sharadar.py` | Defines `Alpha158WithPIT` and `SharadarFeatureHandler`. This is the bridge between Qlib configs and Sharadar/FMP PIT/extra features. |
| `qlib/contrib/data/processor.py` | Adds custom processors: benchmark excess labels, beta residual labels, vol-scaled labels, downside-adjusted labels, portfolio-utility labels, group neutralization, and selective cross-sectional normalization. |
| `qlib/contrib/model/gbdt.py` | Extends LightGBM support with `LGBRankerModel`, a per-date cross-sectional LambdaRank model for top-k stock selection. |
| `qlib/contrib/model/score.py` | Adds deterministic score models: `FeatureWeightedScoreModel`, `RegimeSleeveScoreModel`, `ICSelectedScoreModel`, and `StackedSignalScoreModel`. These make interpretable factor/FMP sleeve experiments possible. |
| `qlib/data/dataset/weight.py` | Adds `RecencyReweighter` and `RegimeRecencyReweighter` for walk-forward LightGBM sample weighting. |
| `qlib/contrib/strategy/signal_strategy.py` | Adds feature score controls and sector-cap behavior to top-k strategies. |
| `qlib/contrib/strategy/weekly.py` | Adds weekly, risk-managed, score-weighted, benchmark-aware, dynamic-risk, and hedged strategies used by the pipeline. |
| `qlib/contrib/strategy/benchmark_aware.py` | Helper functions for long-only benchmark-core plus alpha-sleeve portfolio weights, turnover limiting, active weight metrics, and capping. |
| `qlib/contrib/model/__init__.py`, `qlib/contrib/strategy/__init__.py`, `qlib/contrib/strategy/optimizer/__init__.py` | Registration/import plumbing so configs can reference the custom classes. |
| `qlib/data/data.py` | Modified Qlib data access layer; relevant because custom PIT/extra fields rely on provider behavior. |
| `qlib/data/_libs/rolling.cpp`, `qlib/data/_libs/expanding.cpp` and compiled `.so` files | Qlib rolling/expanding C extensions. The compiled files are generated artifacts required for fast local execution. |
| `qlib/_version.py`, `pyqlib.egg-info/*` | Local package metadata touched by editable/build operations. Not strategy logic. |

### Main Scripts

| File | Role |
|---|---|
| `scripts/data_collector/sharadar/collector.py` | Nasdaq Data Link Sharadar collector. Downloads tables, per-ticker SEP/SFP/SF2/SF3A, discovers schemas, and supports bundle maps. |
| `scripts/data_collector/sharadar/prepare_sf1_pit.py` | Converts consolidated SF1 into per-ticker PIT-normalized files for `dump_pit.py`. |
| `scripts/data_collector/sharadar/prepare_event_features.py` | Converts event/snapshot tables such as SF2/SF3A into daily rolling features. Supports availability lag and output date ranges. |
| `scripts/data_collector/sharadar/table_map_us_core_bundle.yaml` | Bundle map describing Sharadar core/COR tables and download modes. |
| `scripts/update_us_sharadar_data.py` | Main incremental Sharadar update wrapper for SEP/SFP/SF2/SF3A, event-feature deltas, benchmark import, and instrument end-date repair. |
| `scripts/refresh_us_sharadar_pit.py` | Full SF1 PIT refresh wrapper with coverage checks and provenance stamping. |
| `scripts/rebuild_us_sharadar_sf3a_features.py` | Full SF3A/13F lag-correct feature rebuild and Qlib bin overwrite. |
| `scripts/rebuild_us_sharadar_pit_ratio_features.py` | Precomputes clean SF1 PIT ratio features into daily Qlib feature bins. |
| `scripts/rebuild_us_sharadar_model_features.py` | Builds stock risk, market regime, breadth, sector, and metadata features. |
| `scripts/download_fmp_data.py` | Downloads selected FMP endpoints, redacts API keys, writes raw JSON, manifest, and audit CSV. |
| `scripts/build_fmp_event_features.py` | Builds PIT-conservative FMP event/rating/earnings/price-target-news features and dumps them to Qlib. |
| `scripts/build_us_qlib_ticker_benchmark.py` | Builds benchmark return pickles from Qlib ticker close data, especially QQQ/SPY. |
| `scripts/build_external_index_benchmark.py` | Builds external benchmark pickle such as IXIC when not available as a Qlib instrument. |
| `scripts/import_sfp_benchmark_to_qlib.py` | Imports Sharadar SFP benchmark ETFs into the Qlib provider. |
| `scripts/build_us_union_universe.py` | Builds repaired union instrument universes from existing Qlib market files. |
| `scripts/audit_us_universe_integrity.py` | Audits universe overlap versus reference markets and required anchor symbols. |
| `scripts/validate_us_sharadar_price_adjustments.py` | Compares raw Sharadar adjusted prices against Qlib provider prices. |
| `scripts/audit_us_hedge_feasibility.py` | Checks whether hedge/inverse ETF tickers are present and usable in the provider. |
| `scripts/audit_us_sharadar_targets.py` | Audits target/horizon stability, feature IC, and rank buckets before model training. |
| `scripts/evaluate_us_sharadar_feature_ic.py` | Computes configured feature coverage and rank IC. Used for FMP and Sharadar feature screening. |
| `scripts/build_us_sharadar_ablation_configs.py` | Generates older COR ablation configs from a base config. |
| `scripts/build_us_sharadar_clean_candidate_configs.py` | Large config generator for conservative, growth, score, FMP, and release candidate families. |
| `scripts/walkforward_train_us_sharadar.py` | Embargoed walk-forward trainer. Produces stitched `pred.pkl` and manifest. |
| `scripts/run_us_sharadar_release.py` | Main release orchestrator: preflight audits, walk-forward train/qrun, strict validator. |
| `scripts/validate_us_sharadar_pipeline.py` | Main validator. Implements data, model-quality, benchmark, rolling, active-risk, strategy-weighted, and release-decision checks. |
| `scripts/us_sharadar_release_checks.py` | Shared helper functions for release readiness and strategy feasibility. |
| `scripts/run_us_sharadar_research_grid.py` | Runs comparable release grids over many configs and writes summaries/failure maps. Includes timeout support. |
| `scripts/summarize_us_sharadar_release_failures.py` | Parses validator logs into failure-map CSV/Markdown. |
| `scripts/rank_us_sharadar_candidates.py` | Ranks candidates with unified robustness gates and provenance. |
| `scripts/diagnose_us_sharadar_candidate.py` | Deep candidate diagnostics for known release-readiness failure modes. |
| `scripts/ensemble_us_sharadar_predictions.py` | Blends two prediction files with confidence/regime fallback logic. |
| `scripts/sweep_us_sharadar_strategy_params.py` | Sweeps strategy parameters for an existing `pred.pkl`. |
| `scripts/sweep_us_sharadar_strategy_fast.py` | Faster strategy-sweep variant. |
| `scripts/generate_us_trade_plan_from_pred.py` | Builds manual orders/trade plan from predictions, config, holdings, and cash. |
| `scripts/eval_pred_backtest.py` | Legacy helper for backtesting a saved `pred.pkl`. |
| `scripts/inspect_pred_coverage.py` | Reports prediction date coverage. |
| `scripts/merge_pred_pkls.py` | Merges prediction pickle files, with tail values overwriting base overlap. |
| `scripts/run_validate_logged.py` | Runs validator and tees output to a log file. |
| `scripts/post_catchup_sanity_check.py` | Simple post-update sanity checks. |
| `scripts/sharadar_price_utils.py` | Shared price-adjustment/SEP preparation helpers. |
| `scripts/run_us_sharadar_ablation_batch.sh` | Older shell batch runner for COR ablations. |
| `scripts/verify_sharadar_bundle.sh` | Shell helper for checking Sharadar bundle availability. |
| `scripts/us_sharadar_suggestions.py` | Legacy helper to print top/bottom picks from latest Qlib predictions. |
| `scripts/dump_bin.py`, `scripts/dump_pit.py`, `scripts/get_data.py`, `scripts/check_dump_bin.py`, `scripts/check_data_health.py`, `scripts/collect_info.py`, `scripts/rolling_train.py` | Upstream/general Qlib utility scripts still used by the custom pipeline. |

Other `scripts/data_collector/*` directories are upstream/general collectors
for Yahoo, US index, CN index, crypto, funds, PIT, and other Qlib examples.
The custom US Sharadar/FMP work primarily depends on `data_collector/sharadar`.

### Tests

The custom test surface is broad and important. Relevant tests include:

| File | What it validates |
|---|---|
| `tests/test_sharadar_collector_helpers.py` | Sharadar collector helper behavior. |
| `tests/test_sharadar_collector_price_prep.py` | SEP price preparation and adjusted-price handling. |
| `tests/test_prepare_event_features.py` | SF2/SF3A event/snapshot feature construction. |
| `tests/test_refresh_us_sharadar_pit.py` | SF1 PIT refresh helper behavior. |
| `tests/test_rebuild_us_sharadar_pit_ratio_features.py` | SF1 ratio feature rebuild logic. |
| `tests/test_rebuild_us_sharadar_sf3a_features.py` | SF3A lagged full rebuild logic. |
| `tests/test_rebuild_us_sharadar_model_features.py` | Risk/regime/model feature rebuild logic. |
| `tests/test_download_fmp_data.py` | FMP download, audit, redaction, endpoint behavior. |
| `tests/test_build_fmp_event_features.py` | FMP PIT event feature construction and directional alpha columns. |
| `tests/test_build_external_index_benchmark.py` | External benchmark pickle build logic. |
| `tests/test_build_us_qlib_ticker_benchmark.py` | Qlib ticker benchmark return pickle construction. |
| `tests/test_audit_us_universe_integrity.py` | Universe overlap and anchor audits. |
| `tests/test_audit_us_hedge_feasibility.py` | Hedge ETF feasibility audit. |
| `tests/test_audit_us_sharadar_targets.py` | Target audit and screen logic. |
| `tests/test_validate_us_sharadar_price_adjustments.py` | Price adjustment validator. |
| `tests/test_validate_us_sharadar_pipeline_helpers.py` | Main validator helper behavior. |
| `tests/test_run_us_sharadar_release_helpers.py` | Release runner helper behavior. |
| `tests/test_run_us_sharadar_research_grid.py` | Grid runner commands, summaries, timeout/failure map behavior. |
| `tests/test_summarize_us_sharadar_release_failures.py` | Failure-map parser. |
| `tests/test_rank_us_sharadar_candidates_provenance.py` | Candidate ranking and provenance handling. |
| `tests/test_diagnose_us_sharadar_candidate.py` | Candidate diagnostic logic. |
| `tests/test_evaluate_us_sharadar_feature_ic.py` | Feature IC evaluator. |
| `tests/test_ensemble_us_sharadar_predictions.py` | Prediction blending and fallback gates. |
| `tests/test_sweep_us_sharadar_strategy_params.py` | Strategy sweep logic. |
| `tests/test_generate_us_trade_plan_from_pred.py` | Trade-plan generation, positions/cash handling. |
| `tests/test_benchmark_aware_strategy.py` | Benchmark-aware strategy weight construction and risk behavior. |
| `tests/test_topk_dropout_sector_cap.py` | Sector cap behavior in top-k strategy. |
| `tests/test_feature_weighted_score_model.py` | Deterministic factor scoring. |
| `tests/test_lgb_ranker_model.py` | LightGBM ranker data preparation and prediction. |
| `tests/test_group_neutralize_processor.py` | Group/sector neutralization. |
| `tests/test_selective_cszscore_processor.py` | Selective cross-sectional normalization. |
| `tests/test_local_pit_provider.py`, `tests/test_sharadar_feature_handler.py` | PIT/extra field handler behavior. |
| Standard `tests/*` and subdirectory tests | Upstream Qlib test coverage for data, model, workflow, backtest, storage, rolling, and RL modules. |

The most recent focused custom test run from prior implementation work passed:
`116 passed, 4 warnings`.

### Workflow Configs

All candidate configs live under:

```text
examples/benchmarks/LightGBM/
```

Naming convention:

```text
workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_<family>_<horizon>_<target>_<model-or-strategy>_<core/topk>.yaml
```

Important config families:

| Pattern | Role |
|---|---|
| `workflow_config_lightgbm_Alpha158_us_sharadar*.yaml` | Early base Sharadar PIT examples. |
| `*_best*.yaml`, `*_best_v2*.yaml`, `*_best_v3*.yaml` | Historical best-pipeline/risk/hold/COR configs. |
| `*_value*.yaml` | Value-factor experiments. |
| `*_stable*.yaml` | Stable factor, cap/momentum, benchmark-aware, and growth experiments. |
| `*_qv_risk_sf3a_regime*.yaml` | Quality/value/risk/SF3A/regime experiments. |
| `*_nextgen_growth_v1*.yaml`, `*_nextgen_growth_v2*.yaml` | Higher-return target and regime experiments. |
| `*_score_growth_v1*.yaml` to `*_score_growth_v6*.yaml` | Deterministic score, regime sleeve, IC selection, and tail IC experiments. |
| `*_score_growth_v7*.yaml` to `*_score_growth_v12*.yaml` | FMP-integrated score/ranker/utility/sector experiments. |
| `us_sharadar_candidate_*.yaml` | Older candidate search/ranking configs. |

The current most relevant FMP configs are:

- `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_score_growth_v8_h20_qqqexcess_fmpv2_stack_core35_topk30.yaml`
- `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_score_growth_v9_h5_qqqexcess_fmpv2_ranker_core55_topk30.yaml`
- `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_score_growth_v11_h5_qqqutility_fmpv2_ranker_mildrisk_core55_topk30.yaml`
- `workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_score_growth_v12_h5_qqqsector_fmpv2_ranker_mildrisk_core55_topk30.yaml`

### Artifacts

Important artifact directories:

| Path | Role |
|---|---|
| `artifacts/research/target_audit` | Early target/feature IC/rank-bucket audits. |
| `artifacts/research/grid` | Early release grids. |
| `artifacts/research/release_v2_20260506` | Stable60 release-v2 attempts and strategy sweeps. |
| `artifacts/research/growth_v1_20260506` | First growth-track runs. |
| `artifacts/research/growth_v2_20260507` | Growth v2 summary, logs, manifests, and diagnostics. |
| `artifacts/research/growth_v3_20260507` | Residual/beta/absolute growth experiments and target audits. |
| `artifacts/research/nextgen_growth_v1_20260508` | Nextgen growth v1 target screen and grids. |
| `artifacts/research/nextgen_growth_v2_20260510` | Nextgen growth v2 screened runs. |
| `artifacts/research/nextgen_growth_v2_noreweight_20260510` | No-reweight comparison. |
| `artifacts/research/nextgen_growth_v2_ranker_20260510` | Ranker comparison. |
| `artifacts/research/score_growth_v1_20260508` | Score-growth v1 composites. |
| `artifacts/research/score_growth_v2_20260508` | Score-growth v2 regime QQQ composites. |
| `artifacts/research/score_growth_v3_20260510` | Regime sleeve scoring. |
| `artifacts/research/score_growth_v4_icselect_single_20260510` | IC-selected deterministic score model. |
| `artifacts/research/score_growth_v5_regimeic_single_20260511` | Regime-conditioned IC-selected score model. |
| `artifacts/research/score_growth_v6_tailregimeic_single_20260511` | Tail/top-quantile-aware IC selection. |
| `artifacts/research/fmp_smoke_20260512` | FMP smoke download. |
| `artifacts/research/fmp_universe_pilot_20260512` | FMP universe pilot download. |
| `artifacts/research/fmp_full_event_pull_20260512` | Full FMP raw pull artifacts. |
| `artifacts/research/fmp_v7_grid` | First FMP feature grid. |
| `artifacts/research/fmp_v8_grid_20260513` | FMP v2 stacked signal experiments and feature IC. |
| `artifacts/research/fmp_v9_grid_20260513` | FMP v2 ranker experiments and strict diagnostics. |
| `artifacts/research/fmp_v10_grid_20260513` | Dynamic-risk/utility FMP experiments. |
| `artifacts/research/fmp_v11_grid_20260513` | Mild-risk utility FMP experiments. |
| `artifacts/research/fmp_v12_grid_20260513` | Sector-aware mild-risk FMP experiments. |

Key files to inspect first:

- `artifacts/research/fmp_v8_feature_ic_20260513.csv`
- `artifacts/research/fmp_v8_grid_20260513/fmpv8_h20_core35_final/qqq_release_full_diagnostics_20260513.log`
- `artifacts/research/fmp_v9_grid_20260513/release_failure_map_20260513.md`
- `artifacts/research/fmp_v11_grid_20260513/h5_comparison_failure_map.md`
- `artifacts/research/fmp_v12_grid_20260513/h5_experiment_comparison_failure_map.md`
- `artifacts/research/growth_v2_20260507/diagnostic_summary.csv`
- `artifacts/research/growth_v3_20260507/diagnostic_summary.csv`

## Recommended Next Work

The next step should not be another broad parameter sweep over the same
architecture. The highest-value path is:

1. Freeze v8 h20 core35 and v9 h5 core55 as strict FMP baselines.
2. Run a controlled Sharadar-only versus Sharadar-plus-selected-FMP ablation
   using identical walk-forward splits and strict `qqq_release` gates.
3. Build an explicit FMP whitelist:
   - keep earnings surprise latest/windows,
   - keep bullish/bearish rating signals,
   - keep event composite,
   - keep grade score latest if stable,
   - drop or heavily penalize event coverage, weak grade revisions, and
     price-target windows unless ablations prove value.
4. Build regime-conditioned sleeves at the portfolio level:
   - one upside sleeve for normal/risk-on periods,
   - one defensive event/quality sleeve for weak/down/volatile periods,
   - avoid global de-risking that suppresses all alpha.
5. Add top-bucket-aware objective checks to candidate selection:
   - strategy-weighted forward label,
   - rebalance interval excess,
   - top-k realized label,
   - top/bottom quantile spread,
   - rolling QQQ excess.
6. Keep `qqq_release` gates as the release target. Do not loosen gates to
   declare success.
7. Consider more external data only after the clean FMP ablation proves that
   selected FMP signals are not enough.

Most useful future data categories, if needed:

- Point-in-time analyst estimate revisions with true historical as-of dates.
- Higher-quality earnings calendar and surprise history.
- News/key-development data with timestamps.
- Short interest, borrow, and crowding.
- Options/implied-volatility and event-risk data.
- Macro/liquidity regime data.

## Current Bottom Line

This repo is no longer blocked by basic data plumbing. It is blocked by
release-grade alpha robustness.

FMP helps, especially earnings surprise and ratings features, but the current
Sharadar plus FMP candidates are not robust enough for live trading. The path is
still promising if the next phase is a controlled, feature-selective,
regime-aware rebuild rather than more broad tuning.
