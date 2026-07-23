#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.us_sharadar_release_checks import strategy_feasibility_check_rows


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config structure: {path}")
    return data


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _available_instrument_markets(provider_uri: str) -> Set[str]:
    inst_dir = Path(provider_uri).expanduser().resolve() / "instruments"
    if not inst_dir.exists():
        return set()
    return {path.stem for path in inst_dir.glob("*.txt") if path.is_file()}


def _filter_available_reference_markets(reference_markets: str, provider_uri: str) -> str:
    requested = _parse_csv(reference_markets)
    if not requested:
        return ""
    available = _available_instrument_markets(provider_uri)
    if not available:
        return ",".join(requested)
    return ",".join(market for market in requested if market in available)


def _override_benchmark_pkl(obj, benchmark_pkl: str) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "benchmark_pkl":
                obj[key] = benchmark_pkl
            else:
                _override_benchmark_pkl(val, benchmark_pkl)
    elif isinstance(obj, list):
        for item in obj:
            _override_benchmark_pkl(item, benchmark_pkl)


def _normalize_mlruns_uri(uri: str) -> str:
    text = str(uri or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    return str(Path(text).expanduser().resolve())


def _mlruns_local_path(uri: str) -> Optional[Path]:
    text = str(uri or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    if not parsed.scheme:
        return Path(text).expanduser().resolve()
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return None


def _latest_provider_calendar_date(provider_uri: str) -> Optional[str]:
    cal_path = Path(provider_uri).expanduser().resolve() / "calendars" / "day.txt"
    try:
        last = ""
        with cal_path.open("r", encoding="utf-8") as f:
            for line in f:
                val = line.strip()
                if val:
                    last = val
        if not last:
            return None
        # Validate the date format while preserving plain YYYY-MM-DD output.
        return datetime.strptime(last, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def _config_test_segment(cfg: dict) -> Tuple[Optional[str], Optional[str]]:
    seg = (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments", {})
    test = seg.get("test") if isinstance(seg, dict) else None
    if isinstance(test, (list, tuple)) and len(test) == 2:
        return str(test[0]), str(test[1])
    return None, None


def _set_runtime_date_ranges(cfg: dict, *, test_start: Optional[str], test_end: Optional[str]) -> None:
    def set_nested_key(obj, target_key: str, value: str) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key == target_key:
                    obj[key] = value
                else:
                    set_nested_key(val, target_key, value)
        elif isinstance(obj, list):
            for item in obj:
                set_nested_key(item, target_key, value)

    dh_cfg = cfg.get("data_handler_config")
    if isinstance(dh_cfg, dict) and test_end:
        dh_cfg["end_time"] = test_end
        set_nested_key(dh_cfg.get("filter_pipe", []), "filter_end_time", test_end)

    port_bt = ((cfg.get("port_analysis_config") or {}).get("backtest") or {})
    if isinstance(port_bt, dict):
        if test_start:
            port_bt["start_time"] = test_start
        if test_end:
            port_bt["end_time"] = test_end

    seg = (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments", {})
    test = seg.get("test") if isinstance(seg, dict) else None
    if isinstance(test, list) and len(test) == 2:
        if test_start:
            test[0] = test_start
        if test_end:
            test[1] = test_end


def _write_runtime_config(
    cfg: dict,
    *,
    source_config: Path,
    provider_uri: str,
    mlruns_uri: str,
    benchmark_pkl: str,
    tmp_dir: Path,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    override_label_benchmark_pkl: bool = True,
) -> Path:
    runtime_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    qlib_init = runtime_cfg.setdefault("qlib_init", {})
    qlib_init["provider_uri"] = str(provider_uri)
    expm = qlib_init.setdefault("exp_manager", {})
    kwargs = expm.setdefault("kwargs", {})
    kwargs["uri"] = _normalize_mlruns_uri(str(mlruns_uri))
    if override_label_benchmark_pkl:
        _override_benchmark_pkl(runtime_cfg, str(Path(benchmark_pkl).expanduser().resolve()))
    _set_runtime_date_ranges(runtime_cfg, test_start=test_start, test_end=test_end)

    out = tmp_dir / f"{source_config.stem}.runtime.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)
    return out


def _parse_experiment_info(cfg: dict) -> Tuple[str, str]:
    qlib_init = cfg.get("qlib_init", {}) or {}
    expm = qlib_init.get("exp_manager", {}) or {}
    kwargs = expm.get("kwargs", {}) or {}
    exp_name = str(kwargs.get("default_exp_name", "")).strip()
    uri = str(kwargs.get("uri", "")).strip()
    if not exp_name:
        raise ValueError("Missing qlib_init.exp_manager.kwargs.default_exp_name in config")
    if not uri:
        uri = str(Path(__file__).resolve().parents[1] / "mlruns")
    return exp_name, _normalize_mlruns_uri(uri)


def _read_meta_name(meta_path: Path) -> str:
    try:
        txt = meta_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    for line in txt.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _find_experiment_dir(mlruns_uri: Path, exp_name: str) -> Path:
    if not mlruns_uri.exists():
        raise FileNotFoundError(f"mlruns uri not found: {mlruns_uri}")
    for p in sorted(mlruns_uri.iterdir()):
        if not p.is_dir():
            continue
        meta = p / "meta.yaml"
        if not meta.exists():
            continue
        if _read_meta_name(meta) == exp_name:
            return p
    raise FileNotFoundError(f"Experiment '{exp_name}' not found in {mlruns_uri}")


def _list_experiment_dirs(mlruns_uri: Path) -> List[Path]:
    out: List[Path] = []
    if not mlruns_uri.exists():
        return out
    for p in sorted(mlruns_uri.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "meta.yaml").exists():
            continue
        out.append(p)
    return out


def _find_run_dir_by_id(mlruns_uri: Path, run_id: str) -> Path:
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id cannot be empty")
    for exp_dir in _list_experiment_dirs(mlruns_uri):
        cand = exp_dir / run_id
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(f"Run ID not found under any experiment in {mlruns_uri}: {run_id}")


def _load_cmd(run_dir: Path) -> str:
    fp = run_dir / "params" / "cmd-sys.argv"
    if not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _find_latest_matching_run(exp_dir: Path, config_path: Path) -> Path:
    cfg_name = config_path.name
    candidates = []
    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        cmd = _load_cmd(p)
        if not cmd:
            continue
        if cfg_name not in cmd and str(config_path) not in cmd:
            continue
        art_pred = p / "artifacts" / "pred.pkl"
        if not art_pred.exists():
            continue
        mtime = art_pred.stat().st_mtime
        candidates.append((mtime, p))
    if not candidates:
        raise FileNotFoundError(
            f"No run with pred.pkl matched config {config_path} under experiment dir {exp_dir}"
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_latest_matching_run_any_experiment(mlruns_uri: Path, config_path: Path) -> Path:
    candidates = []
    for exp_dir in _list_experiment_dirs(mlruns_uri):
        try:
            run_dir = _find_latest_matching_run(exp_dir, config_path)
            pred = run_dir / "artifacts" / "pred.pkl"
            mtime = pred.stat().st_mtime
            candidates.append((mtime, run_dir))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No run with pred.pkl matched config {config_path} under any experiment in {mlruns_uri}"
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _append_release_quality_gate_args(validate_cmd: List[str], args: argparse.Namespace) -> None:
    fail_fast = not bool(getattr(args, "collect_all_diagnostics", False))
    if not args.skip_strategy_weighted_quality:
        validate_cmd.append("--check_strategy_weighted_model_quality")
        if fail_fast:
            validate_cmd.append("--fail_on_strategy_weighted_model_quality_fail")
    if not args.skip_rebalance_interval_quality:
        validate_cmd.append("--check_rebalance_interval_quality")
        if fail_fast:
            validate_cmd.append("--fail_on_rebalance_interval_quality_fail")
    if not args.skip_active_risk:
        validate_cmd.append("--check_active_risk")
        if fail_fast:
            validate_cmd.append("--fail_on_active_risk_fail")


def _append_external_baseline_gate_args(validate_cmd: List[str], args: argparse.Namespace) -> None:
    if bool(args.skip_baseline_gates):
        return
    should_check = bool(args.check_baseline_gates) or str(args.gate_profile) in {"release", "growth", "qqq_release"}
    if not should_check:
        return
    validate_cmd += [
        "--check_baseline_gates",
        "--baseline_tickers",
        str(args.baseline_tickers),
    ]
    if str(args.baseline_pkl_map or "").strip():
        validate_cmd += ["--baseline_pkl_map", str(args.baseline_pkl_map)]
    fail_fast = not bool(getattr(args, "collect_all_diagnostics", False))
    if fail_fast and (
        bool(args.fail_on_baseline_gate_fail) or str(args.gate_profile) in {"release", "growth", "qqq_release"}
    ):
        validate_cmd.append("--fail_on_baseline_gate_fail")


def _print_check_table(rows: List[Tuple[str, bool, str]]) -> None:
    print("check | status | detail")
    print("--- | --- | ---")
    for name, ok, detail in rows:
        print(f"{name} | {'PASS' if ok else 'FAIL'} | {detail}")


def _config_requires_hedge_feasibility(cfg: dict) -> bool:
    strategy = ((cfg.get("port_analysis_config") or {}).get("strategy") or {})
    strategy_class = str(strategy.get("class") or "")
    kwargs = strategy.get("kwargs") or {}
    if "Hedged" in strategy_class:
        return True
    if not isinstance(kwargs, dict):
        return False
    return bool(kwargs.get("hedge_tickers") or kwargs.get("hedge_tickers_file"))


def _build_hedge_feasibility_cmd(
    args: argparse.Namespace,
    *,
    hedge_auditor_path: Path,
    train_config_path: Path,
    test_start: str,
    test_end: str,
) -> List[str]:
    cmd = [
        args.python_bin,
        str(hedge_auditor_path),
        "--config",
        str(train_config_path),
        "--provider_uri",
        str(args.provider_uri),
        "--raw_sfp_dir",
        str(Path(args.raw_sfp_dir).expanduser().resolve()),
        "--min_history_days",
        str(int(args.hedge_min_history_days)),
        "--max_missing_ratio",
        str(float(args.hedge_max_missing_ratio)),
        "--require_inverse",
        "--fail_on_no_hedge",
    ]
    start = str(args.start or test_start or "").strip()
    end = str(args.end or test_end or "").strip()
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    return cmd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run walk-forward training + strict release validation for US Sharadar pipeline."
    )
    p.add_argument("--config", required=True, help="Workflow config YAML for qrun")
    p.add_argument(
        "--benchmark_pkl",
        default=str(Path(os.getenv("QLIB_PROVIDER_URI", "/root/.qlib/qlib_data/us_data")) / "bench_qqq.pkl"),
        help="Benchmark return series pickle for validator",
    )
    label_benchmark_group = p.add_mutually_exclusive_group()
    label_benchmark_group.add_argument(
        "--preserve_label_benchmark_pkl",
        dest="preserve_label_benchmark_pkl",
        action="store_true",
        default=True,
        help="Keep benchmark-relative label processor pkls from the config. This is the default.",
    )
    label_benchmark_group.add_argument(
        "--override_label_benchmark_pkl",
        dest="preserve_label_benchmark_pkl",
        action="store_false",
        help="Rewrite benchmark-relative label processor pkls to --benchmark_pkl.",
    )
    p.add_argument(
        "--provider_uri",
        default=os.getenv("QLIB_PROVIDER_URI", "/root/.qlib/qlib_data/us_data"),
        help="Qlib provider URI",
    )
    p.add_argument("--market", default="pit_mrq_large_idx", help="Instrument universe name")
    p.add_argument("--start", default="", help="Validation start date YYYY-MM-DD")
    p.add_argument("--end", default="", help="Validation end date YYYY-MM-DD")
    p.add_argument("--qrun_bin", default="qrun", help="qrun executable")
    p.add_argument("--python_bin", default=sys.executable, help="Python executable for validator")
    p.add_argument(
        "--train_mode",
        choices=["walkforward", "ensemble", "qrun"],
        default="walkforward",
        help="Training mode. walkforward is the release-safe default; ensemble trains multiple walk-forward members; qrun is legacy/debug.",
    )
    p.add_argument(
        "--walkforward_script",
        default="scripts/walkforward_train_us_sharadar.py",
        help="Walk-forward trainer script path",
    )
    p.add_argument(
        "--walkforward_ensemble_script",
        default="scripts/walkforward_ensemble_us_sharadar.py",
        help="Walk-forward ensemble trainer script path",
    )
    p.add_argument("--ensemble_primary_config", default="", help="Primary member config for --train_mode ensemble")
    p.add_argument("--ensemble_defensive_config", default="", help="Defensive/member-2 config for --train_mode ensemble")
    p.add_argument("--ensemble_primary_name", default="primary", help="Display name for the primary ensemble member")
    p.add_argument("--ensemble_defensive_name", default="defensive", help="Display name for the defensive ensemble member")
    p.add_argument("--ensemble_primary_weight", type=float, default=0.70, help="Static blend weight for primary member")
    p.add_argument("--ensemble_defensive_weight", type=float, default=0.30, help="Static blend weight for defensive member")
    p.add_argument(
        "--ensemble_normalize",
        choices=["none", "zscore", "rank_zscore"],
        default="rank_zscore",
        help="Per-date member score normalization before static blending",
    )
    p.add_argument("--test_start", default="", help="Override walk-forward test start date YYYY-MM-DD")
    p.add_argument(
        "--test_end",
        default="",
        help="Override walk-forward test end date YYYY-MM-DD. Defaults to provider calendar tail.",
    )
    p.add_argument(
        "--no_auto_test_end",
        action="store_true",
        help="Do not sync runtime config/test end to provider calendar tail.",
    )
    p.add_argument("--test_block", choices=["year", "quarter"], default="year", help="Walk-forward test block size")
    p.add_argument("--valid_days", type=int, default=63, help="Walk-forward validation window in trading days")
    p.add_argument("--embargo_days", type=int, default=None, help="Walk-forward embargo days; defaults to label horizon")
    p.add_argument("--train_lookback_days", type=int, default=None, help="Optional walk-forward train lookback")
    p.add_argument(
        "--recency_half_life_days",
        type=int,
        default=0,
        help="Optional walk-forward exponential sample-weight half-life in observed trading days.",
    )
    p.add_argument(
        "--recency_min_weight",
        type=float,
        default=0.25,
        help="Minimum sample weight when --recency_half_life_days is enabled.",
    )
    p.add_argument("--regime_reweight_feature", default="", help="Pass-through to walk-forward regime sample weighting.")
    p.add_argument(
        "--regime_reweight_thresholds",
        default="-0.04,0.03",
        help="Comma-separated regime thresholds for walk-forward regime sample weighting.",
    )
    p.add_argument(
        "--disable_date_balance_reweight",
        action="store_true",
        help="Pass-through to disable equal-date contribution in regime sample weighting.",
    )
    p.add_argument("--year_balance_reweight", action="store_true", help="Pass-through to walk-forward sample weighting.")
    p.add_argument("--label_tail_reweight", type=float, default=0.0, help="Pass-through to walk-forward sample weighting.")
    p.add_argument("--label_tail_quantile", type=float, default=0.20, help="Pass-through to walk-forward sample weighting.")
    p.add_argument("--sample_max_weight", type=float, default=5.0, help="Pass-through to walk-forward sample weighting.")
    p.add_argument("--out_pred", default="", help="Output pred.pkl path for walk-forward mode")
    p.add_argument("--walkforward_exp_name", default="", help="Optional MLflow experiment name for walk-forward mode")
    p.add_argument(
        "--mlruns_uri",
        default=os.getenv("QLIB_MLRUNS_URI", ""),
        help="Override MLflow tracking directory from the config",
    )
    p.add_argument(
        "--validator",
        default="scripts/validate_us_sharadar_pipeline.py",
        help="Validator script path",
    )
    p.add_argument(
        "--universe_auditor",
        default="scripts/audit_us_universe_integrity.py",
        help="Universe integrity auditor script path",
    )
    p.add_argument(
        "--price_adjustment_validator",
        default="scripts/validate_us_sharadar_price_adjustments.py",
        help="Sharadar/Qlib adjusted-price validator script path",
    )
    p.add_argument(
        "--hedge_feasibility_auditor",
        default="scripts/audit_us_hedge_feasibility.py",
        help="Hedge ETF availability auditor script path",
    )
    p.add_argument("--raw_sep_dir", default="~/.qlib/sharadar/raw/sep", help="Raw Sharadar SEP directory for price adjustment validation")
    p.add_argument("--raw_sfp_dir", default="~/.qlib/sharadar/raw/sfp", help="Raw Sharadar SFP directory for hedge ETF validation")
    p.add_argument(
        "--skip_price_adjustment_integrity",
        action="store_true",
        help="Skip raw Sharadar closeadj vs Qlib close integrity validation before training",
    )
    p.add_argument(
        "--skip_hedge_feasibility",
        action="store_true",
        help="Skip hedge ETF feasibility audit for hedged strategy configs",
    )
    p.add_argument(
        "--hedge_min_history_days",
        type=int,
        default=252,
        help="Minimum qlib close observations required for each hedge ETF candidate",
    )
    p.add_argument(
        "--hedge_max_missing_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed missing qlib close ratio for each hedge ETF candidate",
    )
    p.add_argument(
        "--price_adjustment_max_tickers",
        type=int,
        default=None,
        help="Optional max ticker count for price adjustment validation",
    )
    p.add_argument(
        "--skip_universe_integrity",
        action="store_true",
        help="Skip universe integrity audit before training/validation",
    )
    p.add_argument(
        "--integrity_reference_markets",
        default="sp500,nasdaq100",
        help="Comma-separated reference markets for universe audit",
    )
    p.add_argument(
        "--integrity_min_overlap",
        type=float,
        default=0.60,
        help="Minimum overlap ratio vs each reference market",
    )
    p.add_argument(
        "--integrity_anchors",
        default="AAPL,MSFT,NVDA,AMZN,GOOGL,META|FB,TSLA,JPM,XOM,AVGO,LLY,V,MA,HD,COST",
        help="Comma-separated anchor symbols (supports alias groups with |)",
    )
    p.add_argument("--skip_train", action="store_true", help="Skip training and validate an existing artifact")
    p.add_argument("--skip_data_checks", action="store_true", help="Pass --skip_data_checks to validator")
    p.add_argument(
        "--skip_strategy_weighted_quality",
        action="store_true",
        help="Skip strategy-weighted forward-label release gates",
    )
    p.add_argument(
        "--skip_rebalance_interval_quality",
        action="store_true",
        help="Skip rebalance-to-rebalance forward-return release gates",
    )
    p.add_argument(
        "--skip_active_risk",
        action="store_true",
        help="Skip benchmark-proxy active-risk release gates",
    )
    p.add_argument(
        "--model_quality_mode",
        choices=["strict", "warn", "skip"],
        default="strict",
        help="Generic IC/top-k model-quality gate policy. strict fails release, warn reports only, skip omits it.",
    )
    p.add_argument(
        "--gate_profile",
        choices=["release", "research", "growth", "qqq_release"],
        default="release",
        help="Validator gate profile. qqq_release uses strict QQQ-relative release gates.",
    )
    p.add_argument(
        "--baseline_tickers",
        default="QQQ,SPY,IXIC",
        help="Comma-separated external baseline tickers for validator baseline gates.",
    )
    p.add_argument(
        "--baseline_pkl_map",
        default="",
        help="Comma-separated ticker=return_pkl mappings for baselines missing from qlib, e.g. IXIC=/root/.qlib/qlib_data/us_data/bench_ixic.pkl.",
    )
    p.add_argument(
        "--check_baseline_gates",
        action="store_true",
        help="Pass --check_baseline_gates to validator for non-growth profiles.",
    )
    p.add_argument(
        "--fail_on_baseline_gate_fail",
        action="store_true",
        help="Pass --fail_on_baseline_gate_fail to validator for non-growth profiles.",
    )
    p.add_argument(
        "--skip_baseline_gates",
        action="store_true",
        help="Disable automatic external baseline gates for growth-profile validation.",
    )
    p.add_argument(
        "--collect_all_diagnostics",
        action="store_true",
        help=(
            "Run all requested validator checks without fail-fast validator exit flags. "
            "Use this for research comparisons; release readiness is still reported by the validator."
        ),
    )
    p.add_argument(
        "--training_min_best_iteration",
        type=int,
        default=5,
        help="Minimum accepted one-based best boosting iteration for each walk-forward run.",
    )
    p.add_argument(
        "--strategy_signal_shift",
        type=int,
        default=1,
        help="Trading bars between signal date and execution date for validator portfolio checks.",
    )
    p.add_argument(
        "--year_warmup_days",
        type=int,
        default=5,
        help="Trading-day warm-up for yearly validator backtests before measuring year-slice metrics.",
    )
    p.add_argument(
        "--rolling_mode",
        choices=["independent", "continuous"],
        default="continuous",
        help="Validator rolling mode. continuous slices the full backtest state; independent re-runs each window.",
    )
    p.add_argument(
        "--rolling_warmup_days",
        type=int,
        default=5,
        help="Trading-day warm-up for independent validator rolling windows.",
    )
    p.add_argument(
        "--rolling_ir_metric",
        choices=["strategy", "excess"],
        default="excess",
        help="IR series used by validator rolling gates.",
    )
    p.add_argument(
        "--skip_cold_start_diagnostics",
        action="store_true",
        help="Do not run cold-start yearly/rolling diagnostics in the validator.",
    )
    p.add_argument("--walkforward_manifest", default="", help="Optional walk-forward manifest to pass to validator")
    p.add_argument("--ensemble_manifest", default="", help="Optional ensemble manifest for source-aware validator diagnostics")
    p.add_argument("--ensemble_gate_csv", default="", help="Optional ensemble gate CSV for source-aware validator diagnostics")
    p.add_argument("--ensemble_defensive_label_horizon", type=int, default=60, help="Defensive source horizon for ensemble-aware model-quality diagnostics")
    p.add_argument("--trial_registry", default="", help="Optional JSONL release trial registry path")
    p.add_argument("--trial_id", default="", help="Optional trial ID recorded in --trial_registry")
    p.add_argument("--candidate_name", default="", help="Optional candidate name recorded in --trial_registry")
    p.add_argument("--selection_reason", default="", help="Optional selection reason recorded in --trial_registry")
    p.add_argument("--trial_count", type=int, default=None, help="Optional number of tried candidates for multiple-testing haircut")
    p.add_argument(
        "--allow_pred_beyond_test_segment",
        action="store_true",
        help="Pass --allow_pred_beyond_test_segment to validator for explicit tail-extension runs",
    )
    p.add_argument("--run_id", default="", help="Optional explicit MLflow run ID to validate")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2
    validator_path = Path(args.validator).expanduser().resolve()
    if not validator_path.exists():
        print(f"Validator script not found: {validator_path}")
        return 2
    walkforward_path = Path(args.walkforward_script).expanduser().resolve()
    if args.train_mode == "walkforward" and not args.skip_train and not walkforward_path.exists():
        print(f"Walk-forward trainer script not found: {walkforward_path}")
        return 2
    ensemble_path = Path(args.walkforward_ensemble_script).expanduser().resolve()
    if args.train_mode == "ensemble" and not args.skip_train and not ensemble_path.exists():
        print(f"Walk-forward ensemble trainer script not found: {ensemble_path}")
        return 2
    auditor_path = Path(args.universe_auditor).expanduser().resolve()
    if not args.skip_universe_integrity and not auditor_path.exists():
        print(f"Universe auditor script not found: {auditor_path}")
        return 2
    price_validator_path = Path(args.price_adjustment_validator).expanduser().resolve()
    if not args.skip_price_adjustment_integrity and not price_validator_path.exists():
        print(f"Price adjustment validator script not found: {price_validator_path}")
        return 2
    hedge_auditor_path = Path(args.hedge_feasibility_auditor).expanduser().resolve()

    cfg = _load_yaml(config_path)
    strategy = ((cfg.get("port_analysis_config") or {}).get("strategy") or {})
    strategy_class = str(strategy.get("class") or "")
    strategy_cfg = strategy.get("kwargs") or {}
    strategy_rows = strategy_feasibility_check_rows(strategy_cfg, strategy_class)
    if strategy_rows:
        print("== Strategy Feasibility Preflight ==")
        _print_check_table(strategy_rows)
        if not all(ok for _, ok, _ in strategy_rows):
            print("strategy feasibility preflight failed")
            return 18

    requires_hedge_feasibility = _config_requires_hedge_feasibility(cfg)
    if requires_hedge_feasibility and not args.skip_hedge_feasibility and not hedge_auditor_path.exists():
        print(f"Hedge feasibility auditor script not found: {hedge_auditor_path}")
        return 2
    exp_name, mlruns_uri = _parse_experiment_info(cfg)
    if args.mlruns_uri:
        mlruns_uri = _normalize_mlruns_uri(args.mlruns_uri)
    cfg_test_start, cfg_test_end = _config_test_segment(cfg)
    provider_latest = None if args.no_auto_test_end else _latest_provider_calendar_date(str(args.provider_uri))
    test_start = str(args.test_start or cfg_test_start or "")
    test_end = str(args.test_end or provider_latest or cfg_test_end or "")
    if args.train_mode in {"walkforward", "ensemble"} and (not test_start or not test_end):
        print("Unable to infer walk-forward test_start/test_end; pass --test_start and --test_end")
        return 2
    tmp_dir = Path(tempfile.mkdtemp(prefix="us_sharadar_release_"))
    benchmark_pkl = str(Path(args.benchmark_pkl).expanduser().resolve())
    runtime_config_path = _write_runtime_config(
        cfg,
        source_config=config_path,
        provider_uri=str(args.provider_uri),
        mlruns_uri=mlruns_uri,
        benchmark_pkl=benchmark_pkl,
        tmp_dir=tmp_dir,
        test_start=test_start or None,
        test_end=test_end or None,
        override_label_benchmark_pkl=not bool(args.preserve_label_benchmark_pkl),
    )
    train_config_path = runtime_config_path
    lookup_config_path = config_path if args.skip_train else runtime_config_path

    if not args.skip_universe_integrity:
        reference_markets = _filter_available_reference_markets(
            str(args.integrity_reference_markets),
            str(args.provider_uri),
        )
        missing_reference_markets = sorted(
            set(_parse_csv(str(args.integrity_reference_markets))) - set(_parse_csv(reference_markets))
        )
        if missing_reference_markets:
            print(
                "warning: skipping unavailable integrity reference markets: "
                + ",".join(missing_reference_markets)
            )
        audit_cmd = [
            args.python_bin,
            str(auditor_path),
            "--config",
            str(train_config_path),
            "--provider_uri",
            str(args.provider_uri),
            "--reference_markets",
            reference_markets,
            "--must_include",
            str(args.integrity_anchors),
            "--min_reference_overlap",
            str(args.integrity_min_overlap),
            "--fail_on_overlap_fail",
            "--fail_on_anchor_fail",
        ]
        if args.start:
            audit_cmd += ["--start", args.start]
        if args.end:
            audit_cmd += ["--end", args.end]
        print("== Universe Integrity Audit ==")
        print(" ".join(audit_cmd))
        rc = subprocess.call(audit_cmd)
        if rc != 0:
            print(f"universe integrity audit failed with exit code {rc}")
            return rc

    if not args.skip_price_adjustment_integrity:
        price_cmd = [
            args.python_bin,
            str(price_validator_path),
            "--provider_uri",
            str(args.provider_uri),
            "--raw_sep_dir",
            str(Path(args.raw_sep_dir).expanduser().resolve()),
            "--market",
            str(args.market),
            "--fail_on_error",
        ]
        if args.start:
            price_cmd += ["--start", args.start]
        if args.end:
            price_cmd += ["--end", args.end]
        if args.price_adjustment_max_tickers is not None:
            price_cmd += ["--max_tickers", str(int(args.price_adjustment_max_tickers))]
        print("== Price Adjustment Integrity Audit ==")
        print(" ".join(price_cmd))
        rc = subprocess.call(price_cmd)
        if rc != 0:
            print(f"price adjustment integrity audit failed with exit code {rc}")
            return rc

    if requires_hedge_feasibility and not args.skip_hedge_feasibility:
        hedge_cmd = _build_hedge_feasibility_cmd(
            args,
            hedge_auditor_path=hedge_auditor_path,
            train_config_path=train_config_path,
            test_start=test_start,
            test_end=test_end,
        )
        print("== Hedge Feasibility Audit ==")
        print(" ".join(hedge_cmd))
        rc = subprocess.call(hedge_cmd)
        if rc != 0:
            print(f"hedge feasibility audit failed with exit code {rc}")
            return rc

    run_dir: Optional[Path] = None
    if args.train_mode in {"walkforward", "ensemble"}:
        if args.run_id:
            print("--run_id is only supported with --train_mode qrun")
            return 2
        default_pred_dir = Path(os.getenv("QLIB_SHARADAR_PRED_DIR", "~/.qlib/sharadar/preds")).expanduser()
        default_suffix = (test_end or datetime.now(timezone.utc).strftime("%Y-%m-%d")).replace("-", "")
        out_pred_path = (
            Path(args.out_pred).expanduser().resolve()
            if args.out_pred
            else (default_pred_dir / f"{config_path.stem}_walkforward_{default_suffix}_pred.pkl").resolve()
        )
        manifest_path = (
            Path(args.walkforward_manifest).expanduser().resolve()
            if args.walkforward_manifest
            else out_pred_path.with_suffix(".manifest.json")
        )
        if args.train_mode == "ensemble":
            if (
                not args.skip_train
                and (not str(args.ensemble_primary_config or "").strip() or not str(args.ensemble_defensive_config or "").strip())
            ):
                print("--ensemble_primary_config and --ensemble_defensive_config are required with --train_mode ensemble")
                return 2
            if not args.skip_train:
                wf_exp_name = args.walkforward_exp_name or (
                    f"{exp_name}_release_ensemble_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )
                train_cmd = [
                    args.python_bin,
                    str(ensemble_path),
                    "--primary_config",
                    str(Path(args.ensemble_primary_config).expanduser().resolve()),
                    "--defensive_config",
                    str(Path(args.ensemble_defensive_config).expanduser().resolve()),
                    "--provider_uri",
                    str(args.provider_uri),
                    "--test_start",
                    test_start,
                    "--test_end",
                    test_end,
                    "--test_block",
                    str(args.test_block),
                    "--valid_days",
                    str(int(args.valid_days)),
                    "--primary_name",
                    str(args.ensemble_primary_name),
                    "--defensive_name",
                    str(args.ensemble_defensive_name),
                    "--primary_weight",
                    str(float(args.ensemble_primary_weight)),
                    "--defensive_weight",
                    str(float(args.ensemble_defensive_weight)),
                    "--normalize",
                    str(args.ensemble_normalize),
                    "--tag",
                    wf_exp_name,
                    "--out_pred",
                    str(out_pred_path),
                    "--manifest",
                    str(manifest_path),
                    "--work_dir",
                    str(out_pred_path.parent / f"{out_pred_path.stem}_members"),
                    "--python_bin",
                    str(args.python_bin),
                    "--walkforward_script",
                    str(walkforward_path),
                ]
                if args.mlruns_uri:
                    train_cmd += ["--mlruns_uri", str(mlruns_uri)]
                if not bool(args.preserve_label_benchmark_pkl):
                    train_cmd += [
                        "--benchmark_pkl",
                        benchmark_pkl,
                        "--override_label_benchmark_pkl",
                    ]
                if args.embargo_days is not None:
                    train_cmd += ["--embargo_days", str(int(args.embargo_days))]
                if args.train_lookback_days is not None:
                    train_cmd += ["--train_lookback_days", str(int(args.train_lookback_days))]
                print("== Walk-Forward Ensemble Train ==")
                print(" ".join(train_cmd))
                rc = subprocess.call(train_cmd)
                if rc != 0:
                    print(f"walk-forward ensemble training failed with exit code {rc}")
                    return rc
        elif not args.skip_train:
            wf_exp_name = args.walkforward_exp_name or (
                f"{exp_name}_release_wf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            train_cmd = [
                args.python_bin,
                str(walkforward_path),
                "--config",
                str(train_config_path),
                "--provider_uri",
                str(args.provider_uri),
                "--test_start",
                test_start,
                "--test_end",
                test_end,
                "--test_block",
                str(args.test_block),
                "--valid_days",
                str(int(args.valid_days)),
                "--exp_name",
                wf_exp_name,
                "--out_pred",
                str(out_pred_path),
                "--manifest",
                str(manifest_path),
            ]
            if args.embargo_days is not None:
                train_cmd += ["--embargo_days", str(int(args.embargo_days))]
            if args.train_lookback_days is not None:
                train_cmd += ["--train_lookback_days", str(int(args.train_lookback_days))]
            if int(args.recency_half_life_days) > 0:
                train_cmd += [
                    "--recency_half_life_days",
                    str(int(args.recency_half_life_days)),
                    "--recency_min_weight",
                    str(float(args.recency_min_weight)),
                ]
            if str(args.regime_reweight_feature or "").strip():
                train_cmd += [
                    "--regime_reweight_feature",
                    str(args.regime_reweight_feature),
                    f"--regime_reweight_thresholds={args.regime_reweight_thresholds}",
                    "--label_tail_reweight",
                    str(float(args.label_tail_reweight)),
                    "--label_tail_quantile",
                    str(float(args.label_tail_quantile)),
                    "--sample_max_weight",
                    str(float(args.sample_max_weight)),
                ]
                if bool(args.disable_date_balance_reweight):
                    train_cmd.append("--disable_date_balance_reweight")
                if bool(args.year_balance_reweight):
                    train_cmd.append("--year_balance_reweight")
            print("== Walk-Forward Train ==")
            print(" ".join(train_cmd))
            rc = subprocess.call(train_cmd)
            if rc != 0:
                print(f"walk-forward training failed with exit code {rc}")
                return rc
        pred_path = out_pred_path
        if not pred_path.exists():
            print(f"walk-forward pred.pkl not found: {pred_path}")
            return 2
    else:
        if not args.skip_train:
            train_cmd = [args.qrun_bin, str(train_config_path)]
            print("== Train ==")
            print(" ".join(train_cmd))
            rc = subprocess.call(train_cmd)
            if rc != 0:
                print(f"qrun failed with exit code {rc}")
                return rc

        mlruns_path = _mlruns_local_path(mlruns_uri)
        if mlruns_path is None:
            print(f"qrun run lookup requires a local file MLflow tracking URI, got: {mlruns_uri}")
            return 2

        exp_dir: Optional[Path] = None
        try:
            exp_dir = _find_experiment_dir(mlruns_path, exp_name)
        except FileNotFoundError as e:
            print(f"warning: {e}")
            print("warning: falling back to cross-experiment run lookup")

        if args.run_id:
            try:
                if exp_dir is not None:
                    run_dir = exp_dir / args.run_id
                    if not run_dir.exists():
                        raise FileNotFoundError
                else:
                    run_dir = _find_run_dir_by_id(mlruns_path, args.run_id)
            except Exception:
                try:
                    run_dir = _find_run_dir_by_id(mlruns_path, args.run_id)
                except Exception as e:
                    print(str(e))
                    return 2
        else:
            try:
                if exp_dir is not None:
                    run_dir = _find_latest_matching_run(exp_dir, lookup_config_path)
                else:
                    run_dir = _find_latest_matching_run_any_experiment(mlruns_path, lookup_config_path)
            except Exception as e:
                print(str(e))
                return 2
        pred_path = run_dir / "artifacts" / "pred.pkl"
        if not pred_path.exists():
            print(f"pred.pkl not found: {pred_path}")
            return 2
        manifest_path = (
            Path(args.walkforward_manifest).expanduser().resolve()
            if args.walkforward_manifest
            else pred_path.with_suffix(".manifest.json")
        )

    validate_cmd = [
        args.python_bin,
        str(validator_path),
        "--config",
        str(train_config_path),
        "--provider_uri",
        str(args.provider_uri),
        "--pred",
        str(pred_path),
        "--benchmark_pkl",
        benchmark_pkl,
        "--by_year",
        "--stress_cost_mult",
        "3.0",
        "--stress_deal_price",
        "open",
        "--check_data_quality",
        "--check_training_diagnostics",
        "--training_min_best_iteration",
        str(int(args.training_min_best_iteration)),
        "--pit_staleness_max_p95_days",
        "189",
        "--pit_tail_window_days",
        "252",
        "--require_sf3a_provenance",
        "--require_sf1_pit_provenance",
        "--require_model_feature_provenance",
        "--require_fmp_feature_provenance",
        "--check_gates",
        "--check_rolling",
        "--gate_profile",
        str(args.gate_profile),
        "--strategy_signal_shift",
        str(int(args.strategy_signal_shift)),
        "--year_warmup_days",
        str(int(args.year_warmup_days)),
        "--rolling_mode",
        str(args.rolling_mode),
        "--rolling_warmup_days",
        str(int(args.rolling_warmup_days)),
        "--rolling_ir_metric",
        str(args.rolling_ir_metric),
    ]
    if not bool(args.collect_all_diagnostics):
        validate_cmd += [
            "--fail_on_training_diagnostics_fail",
            "--fail_on_input_check_fail",
            "--fail_on_data_quality_fail",
            "--fail_on_gate_fail",
            "--fail_on_rolling_fail",
            "--fail_on_release_not_ready",
        ]
    if not args.skip_cold_start_diagnostics:
        validate_cmd += [
            "--check_cold_start_years",
            "--check_cold_start_rolling",
        ]
    if args.model_quality_mode != "skip":
        validate_cmd += [
            "--check_model_quality",
            "--check_rebalance_model_quality",
        ]
        if args.ensemble_manifest:
            validate_cmd += [
                "--ensemble_manifest",
                str(Path(args.ensemble_manifest).expanduser().resolve()),
                "--model_quality_use_ensemble_source_horizons",
                "--ensemble_defensive_label_horizon",
                str(int(args.ensemble_defensive_label_horizon)),
            ]
            if args.ensemble_gate_csv:
                validate_cmd += ["--ensemble_gate_csv", str(Path(args.ensemble_gate_csv).expanduser().resolve())]
        if args.model_quality_mode == "strict" and not bool(args.collect_all_diagnostics):
            validate_cmd += ["--fail_on_model_quality_fail"]
    _append_release_quality_gate_args(validate_cmd, args)
    _append_external_baseline_gate_args(validate_cmd, args)
    if args.start:
        validate_cmd += ["--start", args.start]
    if args.end:
        validate_cmd += ["--end", args.end]
    if args.skip_data_checks:
        validate_cmd += ["--skip_data_checks"]
    if args.allow_pred_beyond_test_segment:
        validate_cmd += ["--allow_pred_beyond_test_segment"]
    if manifest_path.exists():
        validate_cmd += ["--walkforward_manifest", str(manifest_path)]
    if args.trial_registry:
        validate_cmd += ["--trial_registry", str(Path(args.trial_registry).expanduser().resolve())]
        if args.trial_id:
            validate_cmd += ["--trial_id", str(args.trial_id)]
        if args.candidate_name:
            validate_cmd += ["--candidate_name", str(args.candidate_name)]
        if args.selection_reason:
            validate_cmd += ["--selection_reason", str(args.selection_reason)]
        if args.trial_count is not None:
            validate_cmd += ["--trial_count", str(int(args.trial_count))]

    print("\n== Validate ==")
    print(f"train_mode={args.train_mode}")
    if run_dir is not None:
        print(f"run_id={run_dir.name}")
    print(f"pred={pred_path}")
    print(f"runtime_config={train_config_path}")
    if manifest_path.exists():
        print(f"walkforward_manifest={manifest_path}")
    print(" ".join(validate_cmd))
    rc = subprocess.call(validate_cmd)
    if rc != 0:
        print(f"release validation failed with exit code {rc}")
        return rc
    if bool(args.collect_all_diagnostics):
        print("release validation diagnostics completed")
    else:
        print("release validation PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
