# US Sharadar Experiment Log and Status (2026-02-12)

This document is a detailed ledger of what has been implemented, what models/configs were tried, what metrics were observed, and where the project currently stands for trading robustness.

## 1) Scope and objective

Goal:
- Build a US Sharadar-based pipeline that is robust enough for trading under strict release gates.

Current strict release gate profile used in this repo:
- `full_excess_ann >= 0.03`
- `full_ir >= 0.40`
- `abs(full_mdd) <= 0.35`
- `stress_excess_ann >= 0.01` (2x costs, close execution)
- `avg_turnover <= 0.10`
- `positive_excess_years >= 3` (eligible years, min 200 trading days)
- `worst_year_excess_ann >= -0.20`

## 2) Infrastructure and implementation work completed

### 2.1 Data and universe integrity hardening

Implemented:
- `scripts/audit_us_universe_integrity.py`
  - overlap checks vs reference markets (`sp500,nasdaq100`)
  - anchor checks (`AAPL,MSFT,NVDA,AMZN,GOOGL,META|FB,TSLA,JPM,XOM,AVGO,LLY,V,MA,HD,COST`)
- `scripts/build_us_union_universe.py`
  - built repaired universe `pit_mrq_large_idx`
- release runner integration:
  - `scripts/run_us_sharadar_release.py` now runs universe integrity audit pre-train

Observed integrity check (from release logs):
- market: `pit_mrq_large_idx`
- base_count: `992`
- overlap:
  - S&P500: `459/505 = 90.89%`
  - Nasdaq100: `88/103 = 85.44%`
- anchors missing: `0`
- result: `PASS`

### 2.2 COR ingestion and feature plumbing

Completed:
- Sharadar COR table support + ingestion pipeline for:
  - SF2 (insiders)
  - SF3A (institutional snapshots)
- preparation + bin dump flow integrated
- repaired-universe backfill performed earlier for missing symbols

### 2.3 Validation and release tooling

Implemented:
- strict validator:
  - `scripts/validate_us_sharadar_pipeline.py`
- candidate ranking:
  - `scripts/rank_us_sharadar_candidates.py`
- release orchestration:
  - `scripts/run_us_sharadar_release.py`
  - robust run lookup fixes for MLflow experiment-name mismatch cases

### 2.4 New ablation tooling added

Implemented in this phase:
- `scripts/build_us_sharadar_ablation_configs.py`
  - auto-generates COR ablation configs from base config
- `scripts/run_us_sharadar_ablation_batch.sh`
  - sequential release runs for:
    - `..._nocor.yaml`
    - `..._sf2only.yaml`
    - `..._sf3aonly.yaml`
    - `..._cor_full.yaml`

Note:
- Batch script uses `set -e`; it stops at first failing variant.

## 3) Experiment/run ledger

## 3.1 Key training runs (MLflow run ids)

| Run ID | Config / Approach | Status | pred.pkl | Notes |
|---|---|---:|---:|---|
| `d00d2865e3994cc795586ed0b89fd6ff` | `..._v2.yaml` baseline train | done | yes | Used by multiple later strategy/eval comparisons |
| `f6f5144513184a7a96b6346fb1e5b102` | `/tmp/risk_v2_excess_full_sig.yaml` | done | yes | Experimental risk-v2-era full run |
| `f686d8c5c19c42dbb472328c30476bb3` | `..._cor_events_pilot.yaml` | done | yes | COR pilot (small/pilot scope) |
| `36518f28c25c47a394542458be9285ca` | `..._cor_events_full.yaml` | incomplete | no | Stopped after discovering flawed universe issue |
| `905a3dcfcfb74798b9ea9ca7a50f9dbf` | `..._full_uplus_stable.yaml` | incomplete | no | Orphaned/incomplete attempt |
| `248d0bb3bbf94625b13a257d55f78c44` | `..._full_uplus_stable.yaml` | done | yes | Main repaired-universe COR full run |
| `fb39e098335e4b118e99431a9ec8b272` | `..._full_uplus_stable_nocor.yaml` | done | yes | First ablation retrain (no COR extras) |

## 3.2 Major evaluated approaches and results

### A) Legacy best-return baseline (evaluation path)

Config/eval:
- `..._v2_hold10.yaml` behavior evaluated on `d00d.../pred.pkl`
- Source file: `/tmp/validate_recheck_v2_hold10_etf.txt`

Metrics (`2022-01-03 -> 2026-01-29`, ETF benchmark):
- full: ann `0.1719`, IR `0.5471`, MDD `-0.4258`, excess ann `0.0615`, turnover `0.0618`
- stress (2x cost): excess ann `0.0469`

Gate context:
- Passes older/looser gate profile (research-like thresholds: MDD cap `0.45`, stress floor `0.00`, positive years `>=2`)
- Fails strict release profile due:
  - drawdown too high vs `0.35`
  - insufficient positive excess years (`2 < 3`)
- Strict-gate check example: `/tmp/validate_v2_hold10_gates_strict_noroll_20260209.txt`

### B) Risk-managed tuning era

Representative outputs:
- `/tmp/validate_risk_v2_full.txt`
  - full: ann `0.1312`, IR `0.5446`, MDD `-0.2995`, excess `0.0208`
  - stress excess `0.0090`
  - Fails excess floor vs `0.03`
- `/tmp/validate_risk_full_again.txt`
  - full: ann `0.1494`, IR `0.5526`, MDD `-0.3393`, excess `0.0390`
  - stress excess `0.0258`

Observation:
- Risk controls improved MDD significantly vs legacy v2_hold10, but often reduced excess.
- Passing all strict gates remained inconsistent.

### C) COR pilot run

Run:
- `f686d8c5c19c42dbb472328c30476bb3`
- Config: `..._cor_events_pilot.yaml`

Documented metrics (from prior project note):
- full ann `0.1319`
- full excess ann `0.0215`
- IR `0.7068`
- MDD `-0.2071`
- stress excess ann `0.0127`

Release outcome:
- FAIL (`full_excess_ann` and yearly robustness issues)

### D) Repaired-universe COR full run (main recent)

Run:
- `248d0bb3bbf94625b13a257d55f78c44`
- Config: `..._full_uplus_stable.yaml`

User-reported release validation output (strict profile):
- full excess ann: `0.0195` (FAIL)
- full IR: `0.4705` (PASS)
- full MDD: `-0.4345` (FAIL)
- stress excess ann: `0.0085` (FAIL)
- turnover: `0.0463` (PASS)
- positive excess years: `2` (FAIL)
- worst year excess: `-0.1819` (PASS)
- overall: FAIL

### E) Strategy-only post-training tuning (no retrain)

Evaluated on `248.../pred.pkl`:

1) `..._ddguard_v1.yaml`
- full: ann `0.0991`, IR `0.4597`, MDD `-0.3573`, excess `-0.0113`, turnover `0.0352`
- Result: clearly non-viable (negative excess)

2) `..._topk40_dd_v1.yaml`
- full: ann `0.0561`, IR `0.2508`, MDD `-0.3951`, excess `-0.0542`, turnover `0.0367`
- Result: clearly worse (negative excess, weaker IR)

Conclusion:
- Strategy/risk parameter tweaks on fixed predictions did not rescue release robustness.

### F) COR ablation retrain batch (current stage)

Command run:
- `bash scripts/run_us_sharadar_ablation_batch.sh`

What happened:
- first variant (`nocor`) trained and validated; failed release gates
- batch stopped immediately (expected with `set -e`)
- remaining variants not executed yet:
  - `sf2only`
  - `sf3aonly`
  - `cor_full`

`nocor` user-reported gate summary:
- full excess ann: `-0.0199` (FAIL)
- full IR: `0.3390` (FAIL)
- full MDD: `-0.3574` (FAIL)
- stress excess ann: `-0.0288` (FAIL)
- turnover: pass
- positive years: fail (`2 < 3`)
- overall: FAIL

## 4) What we are observing (patterns)

### 4.1 Main failure mode is excess robustness, not turnover

Across recent strict checks:
- turnover is generally within limits (`~0.03-0.06`, gate max `0.10`)
- recurring failures are:
  - `full_excess_ann`
  - `stress_excess_ann`
  - `positive_excess_years`
  - sometimes `full_mdd_abs`

### 4.2 Risk controls help drawdown but can erase alpha

- Stronger defensive settings often push MDD down but reduce or invert excess.
- Strategy-only tuning on `248` confirmed this: both tested variants went negative excess.

### 4.3 Regime instability remains material

Typical pattern in yearly slices:
- 2023/2025 positive
- 2022/2024 often negative excess
- strict gate (`>=3` positive years) keeps failing

### 4.4 Pipeline/data integrity is much better than before

- Universe integrity now audited and passing.
- Anchor coverage and COR feature coverage materially improved.
- Strict data-quality failures from stale symbols (e.g., old ABMD issue) were addressed.

## 5) Best run so far (two definitions)

### 5.1 Best by raw ETF excess/return seen so far

- Legacy `v2_hold10` eval on `d00d...` prediction:
  - full ann `0.1719`
  - excess ann `0.0615`
  - IR `0.5471`
  - MDD `-0.4258`

But:
- Not acceptable under current strict release risk standards.

### 5.2 Best on current repaired-universe COR production track

- `248...` (`..._full_uplus_stable.yaml`) is currently the strongest recent COR full run,
  but still fails strict release gates:
  - excess too low (`0.0195`)
  - MDD too high (`-0.4345`)
  - stress excess too low (`0.0085`)
  - yearly consistency insufficient (`2/4` positive years)

## 6) Current status and gap-to-trading

Current status:
- Infra + data pipeline: substantially hardened.
- Modeling: still below strict trading robustness requirements.
- No currently validated strict-release-pass model in the latest repaired-universe/COR path.

Remaining concrete gap:
- Need a model configuration that simultaneously satisfies:
  - excess floors (full + stress)
  - drawdown cap
  - yearly consistency

## 7) Is a trading-robust model still feasible here?

Assessment:
- **Yes, feasible**, but not yet demonstrated by current model family/settings.
- Evidence suggests next gains must come from signal-level ablation/selection and retraining (not just portfolio parameter tuning).

Immediate next high-value step:
- Complete remaining ablation retrains and release checks:
  1. `sf2only`
  2. `sf3aonly`
  3. `cor_full`
- Then rank by strict release gates and rolling robustness.

---

## 8) File references used for this report

- Docs:
  - `docs/us_sharadar_cor_upgrade_progress_2026-02-10.md`
  - `docs/us_sharadar_current_best_pipeline_2026-02-09.md`
- Validation outputs:
  - `/tmp/validate_recheck_v2_hold10_etf.txt`
  - `/tmp/validate_v2_hold10_gates_strict_noroll_20260209.txt`
  - `/tmp/validate_risk_v2_full.txt`
  - `/tmp/validate_risk_full_again.txt`
  - `/tmp/validate_full_f6f51445.txt`
  - `/tmp/validate_ddguard_v1_20260211.txt`
  - `/tmp/validate_topk40_dd_v1_20260211.txt`
- MLflow runs:
  - `/workspace/qlib/mlruns/907430387469519321/*`
