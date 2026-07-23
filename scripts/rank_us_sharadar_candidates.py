#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.data import Cal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_us_sharadar_pipeline import (
    BASELINE_GATE_PRESETS,
    ROLLING_PRESETS,
    _as_ts,
    _build_rolling_windows,
    _collect_rolling_rows,
    _coerce_ticker_list,
    _evaluate_robustness_gates,
    _evaluate_external_baseline_gates,
    _load_external_baseline_returns,
    _load_yaml,
    _parse_ticker_path_map,
    _run_backtest,
    _safe_get,
    _slice_report,
    _summarize_report,
)
from scripts.us_sharadar_release_checks import release_decision, strategy_feasibility_check_rows


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _safe_float(v) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _print_table(rows: List[Dict[str, object]]):
    if not rows:
        print("(no rows)")
        return
    headers = [
        "rank",
        "candidate",
        "run_id",
        "release",
        "gates",
        "rolling",
        "strategy",
        "baseline",
        "gates_pass",
        "full_excess",
        "full_ir",
        "full_mdd",
        "stress_excess",
        "excess_sharpe",
        "dsr_prob",
        "roll_pass_rate",
        "roll_worst_ex",
        "qqq_latest",
        "spy_latest",
        "ixic_latest",
        "pos_years",
        "worst_year_excess",
        "score",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for i, r in enumerate(rows, start=1):
        line = [
            str(i),
            str(r["candidate"]),
            str(r.get("run_id", "")),
            "PASS" if r.get("release_ready") else "FAIL",
            "PASS" if r["gates_ok"] else "FAIL",
            "PASS" if r["rolling_ok"] else "FAIL",
            "PASS" if r.get("strategy_feasibility_ok", True) else "FAIL",
            "PASS" if r.get("baseline_ok", True) else "FAIL",
            f"{r['gates_pass']}/{r['gates_total']}",
            _fmt(r["full_excess"]),
            _fmt(r["full_ir"]),
            _fmt(r["full_mdd"]),
            _fmt(r["stress_excess"]),
            _fmt(r.get("excess_sharpe")),
            _fmt(r.get("dsr_prob")),
            _fmt(r["rolling_pass_rate"]),
            _fmt(r["rolling_worst_excess"]),
            _fmt(r.get("qqq_latest_rolling")),
            _fmt(r.get("spy_latest_rolling")),
            _fmt(r.get("ixic_latest_rolling")),
            str(r["pos_years"]),
            _fmt(r["worst_year_excess"]),
            _fmt(r["score"]),
        ]
        print(" | ".join(line))


def _compute_score(
    full_excess: Optional[float],
    stress_excess: Optional[float],
    full_ir: Optional[float],
    full_mdd: Optional[float],
    pos_years: int,
    worst_year_excess: Optional[float],
    rolling_pass_rate: Optional[float],
    rolling_worst_excess: Optional[float],
    rolling_worst_mdd_abs: Optional[float],
    dsr_prob: Optional[float] = None,
) -> float:
    ex = full_excess if full_excess is not None else -1.0
    sx = stress_excess if stress_excess is not None else -1.0
    ir = full_ir if full_ir is not None else -1.0
    mdd_penalty = abs(full_mdd) if full_mdd is not None else 1.0
    wye = worst_year_excess if worst_year_excess is not None else -1.0
    roll_pr = rolling_pass_rate if rolling_pass_rate is not None else 0.0
    roll_ex = rolling_worst_excess if rolling_worst_excess is not None else -1.0
    roll_mdd = rolling_worst_mdd_abs if rolling_worst_mdd_abs is not None else 1.0
    dsr = dsr_prob if dsr_prob is not None else 0.0
    return (
        ex
        + 0.5 * sx
        + 0.10 * ir
        + 0.03 * float(pos_years)
        + 0.05 * wye
        - 0.05 * mdd_penalty
        + 0.30 * roll_pr
        + 0.10 * roll_ex
        - 0.05 * roll_mdd
        + 0.10 * dsr
    )


def _baseline_namespace(
    *,
    gate_profile: str,
    tickers: str,
    pkl_map: str,
    stress_cost_mult: float,
    stress_deal_price: str,
    overrides: Optional[dict] = None,
) -> argparse.Namespace:
    preset = dict(BASELINE_GATE_PRESETS[str(gate_profile)])
    overrides = overrides or {}
    for key, value in overrides.items():
        if key in {"enabled", "tickers", "pkl_map"}:
            continue
        arg_key = key if str(key).startswith("baseline_") else f"baseline_{key}"
        if arg_key in preset:
            preset[arg_key] = value
    preset["baseline_tickers"] = str(tickers or preset.get("baseline_tickers") or "")
    preset["baseline_pkl_map"] = str(pkl_map or "")
    preset["stress_cost_mult"] = float(stress_cost_mult)
    preset["stress_deal_price"] = str(stress_deal_price)
    return argparse.Namespace(**preset)


def _cfg_first(cfg: dict, *keys):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return None


def _resolve_rolling_settings(args: argparse.Namespace, rolling_cfg: dict, gate_profile: str) -> Dict[str, object]:
    preset = dict(ROLLING_PRESETS[str(gate_profile)])
    rolling_cfg = rolling_cfg or {}

    def choose(*, cfg_keys: Tuple[str, ...], arg_name: str, preset_name: str, cast):
        cfg_val = _cfg_first(rolling_cfg, *cfg_keys)
        if cfg_val is not None:
            return cast(cfg_val)
        arg_val = getattr(args, arg_name, None)
        if arg_val is not None:
            return cast(arg_val)
        return cast(preset[preset_name])

    mode = str(_cfg_first(rolling_cfg, "rolling_mode", "mode") or getattr(args, "rolling_mode", None) or "independent")
    ir_metric = str(
        _cfg_first(rolling_cfg, "rolling_ir_metric", "ir_metric")
        or getattr(args, "rolling_ir_metric", None)
        or "strategy"
    )
    if mode not in {"independent", "continuous"}:
        raise ValueError(f"invalid rolling mode: {mode}")
    if ir_metric not in {"strategy", "excess"}:
        raise ValueError(f"invalid rolling IR metric: {ir_metric}")

    return {
        "mode": mode,
        "ir_metric": ir_metric,
        "window_days": choose(
            cfg_keys=("rolling_window_days", "window_days"),
            arg_name="rolling_window_days",
            preset_name="rolling_window_days",
            cast=int,
        ),
        "step_days": choose(
            cfg_keys=("rolling_step_days", "step_days"),
            arg_name="rolling_step_days",
            preset_name="rolling_step_days",
            cast=int,
        ),
        "min_days": choose(
            cfg_keys=("rolling_min_days", "min_days"),
            arg_name="rolling_min_days",
            preset_name="rolling_min_days",
            cast=int,
        ),
        "min_excess_ann": choose(
            cfg_keys=("rolling_min_excess_ann", "min_excess_ann"),
            arg_name="rolling_min_excess_ann",
            preset_name="rolling_min_excess_ann",
            cast=float,
        ),
        "min_ir": choose(
            cfg_keys=("rolling_min_ir", "min_ir"),
            arg_name="rolling_min_ir",
            preset_name="rolling_min_ir",
            cast=float,
        ),
        "max_mdd_abs": choose(
            cfg_keys=("rolling_max_mdd_abs", "max_mdd_abs"),
            arg_name="rolling_max_mdd_abs",
            preset_name="rolling_max_mdd_abs",
            cast=float,
        ),
        "max_turnover": choose(
            cfg_keys=("rolling_max_turnover", "max_turnover"),
            arg_name="rolling_max_turnover",
            preset_name="rolling_max_turnover",
            cast=float,
        ),
        "min_pass_rate": choose(
            cfg_keys=("rolling_min_pass_rate", "min_pass_rate"),
            arg_name="rolling_min_pass_rate",
            preset_name="rolling_min_pass_rate",
            cast=float,
        ),
    }


def _flatten_baseline_results(
    comparison_rows: List[Tuple[str, str, Dict[str, float]]],
    rolling_rows: List[Tuple[str, str, Dict[str, float]]],
    check_rows: List[Tuple[str, bool, str]],
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "baseline_ok": bool(check_rows) and all(ok for _, ok, _ in check_rows),
        "baseline_failures": ";".join(name for name, ok, _ in check_rows if not ok),
    }
    for ticker, period, metrics in comparison_rows:
        prefix = str(ticker).lower()
        if str(period) == "full":
            out[f"{prefix}_full_excess"] = _safe_float(metrics.get("excess_ann_return"))
            out[f"{prefix}_full_excess_ir"] = _safe_float(metrics.get("excess_ir"))
        elif str(period).startswith("stress"):
            out[f"{prefix}_stress_excess"] = _safe_float(metrics.get("excess_ann_return"))

    by_ticker: Dict[str, List[Dict[str, float]]] = {}
    for ticker, _window, metrics in rolling_rows:
        by_ticker.setdefault(str(ticker).lower(), []).append(metrics)
    for prefix, rows in by_ticker.items():
        values = [_safe_float(row.get("excess_ann_return")) for row in rows]
        values = [val for val in values if val is not None]
        statuses = [_safe_float(row.get("status")) for row in rows]
        statuses = [val for val in statuses if val is not None]
        out[f"{prefix}_rolling_pass_rate"] = (
            float(sum(1 for val in statuses if val == 1.0)) / float(len(statuses)) if statuses else None
        )
        out[f"{prefix}_worst_rolling"] = min(values) if values else None
        out[f"{prefix}_latest_rolling"] = values[-1] if values else None
    return out


def _candidate_release_decision(row: Dict[str, object], *, require_baseline: bool) -> Dict[str, object]:
    statuses = {
        "strategy_feasibility": bool(row.get("strategy_feasibility_ok", True)),
        "robustness_gates": bool(row.get("gates_ok", False)),
        "rolling": bool(row.get("rolling_ok", False)),
    }
    required = ["strategy_feasibility", "robustness_gates", "rolling"]
    if require_baseline:
        statuses["external_baseline_gates"] = bool(row.get("baseline_ok", False))
        required.append("external_baseline_gates")
    return release_decision(statuses, required_checks=required)


def _select_release_candidate(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    for row in rows:
        if bool(row.get("release_ready", False)):
            return row
    return None


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    ret = pd.Series(returns, dtype=float).replace([math.inf, -math.inf], math.nan).dropna()
    if len(ret) < 3:
        return None
    std = float(ret.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    return float(ret.mean() / std * math.sqrt(periods_per_year))


def _probabilistic_sharpe_ratio(returns: pd.Series, benchmark_daily_sharpe: float = 0.0) -> Optional[float]:
    ret = pd.Series(returns, dtype=float).replace([math.inf, -math.inf], math.nan).dropna()
    n = len(ret)
    if n < 3:
        return None
    std = float(ret.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    sr = float(ret.mean() / std)
    skew = float(ret.skew()) if math.isfinite(float(ret.skew())) else 0.0
    # pandas Series.kurt() returns excess kurtosis; the PSR denominator uses Pearson kurtosis.
    kurt = float(ret.kurt()) + 3.0 if math.isfinite(float(ret.kurt())) else 3.0
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    if not math.isfinite(denom) or denom <= 0:
        return None
    z = (sr - float(benchmark_daily_sharpe)) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(NormalDist().cdf(z))


def _expected_max_daily_sharpe_threshold(n_obs: int, n_trials: int) -> float:
    if n_trials <= 1 or n_obs <= 2:
        return 0.0
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    sr_std = 1.0 / math.sqrt(n_obs - 1)
    z1 = normal.inv_cdf(1.0 - 1.0 / float(n_trials))
    z2 = normal.inv_cdf(1.0 - 1.0 / (float(n_trials) * math.e))
    return float(sr_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2))


def _deflated_sharpe_probability(returns: pd.Series, n_trials: int) -> Optional[float]:
    ret = pd.Series(returns, dtype=float).replace([math.inf, -math.inf], math.nan).dropna()
    if len(ret) < 3:
        return None
    threshold = _expected_max_daily_sharpe_threshold(len(ret), max(1, int(n_trials)))
    return _probabilistic_sharpe_ratio(ret, benchmark_daily_sharpe=threshold)


def _pbo_from_performance_matrix(
    matrix: pd.DataFrame,
    *,
    max_combinations: int,
) -> Dict[str, object]:
    matrix = matrix.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if matrix.shape[0] < 2 or matrix.shape[1] < 4:
        return {"available": False, "reason": f"need >=2 candidates and >=4 complete slices, got {matrix.shape}"}

    n_slices = matrix.shape[1]
    train_size = n_slices // 2
    combos = list(itertools.combinations(range(n_slices), train_size))
    if max_combinations > 0 and len(combos) > max_combinations:
        idx = np.linspace(0, len(combos) - 1, int(max_combinations)).round().astype(int)
        combos = [combos[i] for i in np.unique(idx)]

    lambdas = []
    selected = []
    for combo in combos:
        train_cols = list(combo)
        test_cols = [i for i in range(n_slices) if i not in train_cols]
        train_perf = matrix.iloc[:, train_cols].mean(axis=1)
        best_name = train_perf.idxmax()
        test_perf = matrix.iloc[:, test_cols].mean(axis=1)
        chosen_test = float(test_perf.loc[best_name])
        rank = 1 + int((test_perf < chosen_test).sum())
        w = rank / float(matrix.shape[0] + 1)
        lambdas.append(float(math.log(w / (1.0 - w))))
        selected.append(str(best_name))

    pbo = float(np.mean([x < 0.0 for x in lambdas])) if lambdas else float("nan")
    return {
        "available": True,
        "pbo": pbo,
        "combinations": len(combos),
        "candidates": int(matrix.shape[0]),
        "slices": int(matrix.shape[1]),
        "lambda_median": float(np.median(lambdas)) if lambdas else float("nan"),
        "selected_candidates": selected,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank US Sharadar pipeline candidates with unified robustness gates.")
    p.add_argument("--candidates_yaml", required=True, help="YAML file with candidate list and shared settings")
    p.add_argument("--provider_uri", default=None, help="Override qlib data root")
    p.add_argument("--out_csv", default=None, help="Optional path to write ranking CSV")
    p.add_argument("--out_selection_json", default=None, help="Optional path to write fixed-rule selected release candidate JSON")
    p.add_argument(
        "--gate_profile",
        choices=["release", "research", "growth", "qqq_release"],
        default="release",
        help="Baseline gate preset for external benchmark checks",
    )
    p.add_argument("--check_rolling", action="store_true", help="Include rolling walk-forward checks in ranking")
    p.add_argument("--rolling_window_days", type=int, default=None, help="Rolling window size in trading days")
    p.add_argument("--rolling_step_days", type=int, default=None, help="Rolling step in trading days")
    p.add_argument("--rolling_min_days", type=int, default=None, help="Minimum days for a rolling window")
    p.add_argument("--rolling_min_excess_ann", type=float, default=None, help="Rolling gate: min excess annualized return")
    p.add_argument("--rolling_min_ir", type=float, default=None, help="Rolling gate: min IR")
    p.add_argument("--rolling_max_mdd_abs", type=float, default=None, help="Rolling gate: max absolute MDD")
    p.add_argument("--rolling_max_turnover", type=float, default=None, help="Rolling gate: max average turnover")
    p.add_argument("--rolling_min_pass_rate", type=float, default=None, help="Rolling gate: minimum pass rate")
    p.add_argument(
        "--rolling_mode",
        choices=["independent", "continuous"],
        default=None,
        help="Rolling evaluation mode. Defaults to YAML rolling.mode or independent.",
    )
    p.add_argument(
        "--rolling_ir_metric",
        choices=["strategy", "excess"],
        default=None,
        help="IR series used by rolling gates. Defaults to YAML rolling.ir_metric or strategy.",
    )
    p.add_argument(
        "--skip_provenance_checks",
        action="store_true",
        help="Skip run provenance checks (pred uniqueness, run cleanliness, cmd-config alignment)",
    )
    p.add_argument("--allow_shared_pred", action="store_true", help="Allow multiple candidates to reuse same pred.pkl")
    p.add_argument("--allow_dirty_run", action="store_true", help="Allow dirty code_status snapshots in source runs")
    p.add_argument("--allow_config_mismatch", action="store_true", help="Allow config/run cmd mismatches")
    p.add_argument("--check_pbo", action="store_true", help="Estimate candidate-set probability of backtest overfitting from yearly excess-return slices")
    p.add_argument("--pbo_max_combinations", type=int, default=256, help="Maximum train/test slice combinations for PBO")
    p.add_argument("--max_pbo", type=float, default=0.20, help="Maximum acceptable PBO when --fail_on_pbo_fail is set")
    p.add_argument("--fail_on_pbo_fail", action="store_true", help="Exit non-zero when PBO exceeds --max_pbo")
    p.add_argument("--check_external_baselines", action="store_true", help="Evaluate QQQ/SPY/IXIC-style external baseline gates during ranking")
    p.add_argument("--baseline_tickers", default="", help="Comma-separated external baseline tickers; defaults to preset/global config")
    p.add_argument("--baseline_pkl_map", default="", help="Comma-separated ticker=return_pkl mappings for external baselines")
    p.add_argument("--fail_on_no_release", action="store_true", help="Exit non-zero if no candidate satisfies all required release checks")
    p.add_argument("--fail_on_error", action="store_true", help="Exit non-zero if any candidate evaluation errors")
    return p.parse_args()


def _load_candidates(path: Path) -> Tuple[dict, List[dict]]:
    cfg = _load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    candidates = cfg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates_yaml must contain non-empty `candidates` list")
    return cfg, candidates


def _infer_run_dir_from_pred(pred_path: Path) -> Optional[Path]:
    """
    Infer MLflow run dir from artifact path:
    .../mlruns/<exp_id>/<run_id>/artifacts/pred.pkl
    """
    parts = list(pred_path.parts)
    if "mlruns" not in parts:
        return None
    idx = parts.index("mlruns")
    if len(parts) <= idx + 4:
        return None
    if parts[idx + 3] != "artifacts":
        return None
    run_dir = Path(*parts[: idx + 3])
    return run_dir if run_dir.exists() else None


def _load_cmd_sys_argv(run_dir: Path) -> str:
    fp = run_dir / "params" / "cmd-sys.argv"
    if not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _config_in_cmd(config_path: Path, cmd: str) -> bool:
    if not cmd:
        return False
    cfg_name = config_path.name
    cfg_abs = str(config_path.resolve())
    return cfg_name in cmd or cfg_abs in cmd


def _run_is_dirty(run_dir: Path) -> Optional[bool]:
    fp = run_dir / "artifacts" / "code_status.txt"
    if not fp.exists():
        return None
    try:
        txt = fp.read_text(encoding="utf-8")
    except Exception:
        return None
    patterns = [
        r"Changes not staged for commit",
        r"Untracked files:",
        r"Changes to be committed:",
    ]
    return any(re.search(p, txt) for p in patterns)


def main() -> int:
    args = _parse_args()
    cpath = Path(args.candidates_yaml).expanduser().resolve()
    if not cpath.exists():
        print(f"candidates_yaml not found: {cpath}")
        return 2

    global_cfg, candidates = _load_candidates(cpath)
    provider_uri = args.provider_uri or global_cfg.get("provider_uri") or "/root/.qlib/qlib_data/us_data"
    benchmark_pkl = global_cfg.get("benchmark_pkl")
    if not benchmark_pkl:
        print("candidates_yaml must set benchmark_pkl")
        return 2
    benchmark = pd.read_pickle(Path(benchmark_pkl).expanduser().resolve())
    if not isinstance(benchmark, pd.Series):
        print(f"benchmark_pkl must load as pandas Series: {benchmark_pkl}")
        return 2

    shared_start = global_cfg.get("start")
    shared_end = global_cfg.get("end")
    stress_cost_mult = float(global_cfg.get("stress_cost_mult", 2.0))
    stress_deal_price = str(global_cfg.get("stress_deal_price", "close"))
    gates_cfg = global_cfg.get("gates", {}) or {}
    gate_min_full_excess_ann = float(gates_cfg.get("min_full_excess_ann", 0.03))
    gate_min_full_ir = float(gates_cfg.get("min_full_ir", 0.40))
    gate_max_full_mdd_abs = float(gates_cfg.get("max_full_mdd_abs", 0.35))
    gate_min_stress_excess_ann = float(gates_cfg.get("min_stress_excess_ann", 0.01))
    gate_max_turnover = float(gates_cfg.get("max_turnover", 0.10))
    gate_min_positive_excess_years = int(gates_cfg.get("min_positive_excess_years", 3))
    gate_min_worst_year_excess_ann = float(gates_cfg.get("min_worst_year_excess_ann", -0.20))
    gate_min_year_days = int(gates_cfg.get("min_year_days", 200))
    rolling_cfg = global_cfg.get("rolling", {}) or {}
    check_rolling = bool(args.check_rolling or rolling_cfg.get("enabled", False))
    gate_profile = str(global_cfg.get("gate_profile") or args.gate_profile)
    rolling_settings = _resolve_rolling_settings(args, rolling_cfg, gate_profile)
    rolling_window_days = int(rolling_settings["window_days"])
    rolling_step_days = int(rolling_settings["step_days"])
    rolling_min_days = int(rolling_settings["min_days"])
    rolling_min_excess_ann = float(rolling_settings["min_excess_ann"])
    rolling_min_ir = float(rolling_settings["min_ir"])
    rolling_max_mdd_abs = float(rolling_settings["max_mdd_abs"])
    rolling_max_turnover = float(rolling_settings["max_turnover"])
    rolling_min_pass_rate = float(rolling_settings["min_pass_rate"])
    rolling_mode = str(rolling_settings["mode"])
    rolling_ir_metric = str(rolling_settings["ir_metric"])
    baseline_cfg = global_cfg.get("baselines", {}) or {}
    check_baselines = bool(args.check_external_baselines or baseline_cfg.get("enabled", False))
    baseline_tickers = str(
        args.baseline_tickers
        or baseline_cfg.get("tickers")
        or BASELINE_GATE_PRESETS[gate_profile]["baseline_tickers"]
    )
    baseline_pkl_map = str(args.baseline_pkl_map or baseline_cfg.get("pkl_map") or "")
    baseline_args = _baseline_namespace(
        gate_profile=gate_profile,
        tickers=baseline_tickers,
        pkl_map=baseline_pkl_map,
        stress_cost_mult=stress_cost_mult,
        stress_deal_price=stress_deal_price,
        overrides=baseline_cfg.get("gates", {}) or {},
    )
    baseline_map = _parse_ticker_path_map(baseline_args.baseline_pkl_map)
    provenance_cfg = global_cfg.get("provenance", {}) or {}
    provenance_enabled = not bool(args.skip_provenance_checks)
    require_unique_pred = bool(provenance_cfg.get("require_unique_pred", True)) and not bool(args.allow_shared_pred)
    reject_dirty_run = bool(provenance_cfg.get("reject_dirty_run", True)) and not bool(args.allow_dirty_run)
    require_config_match = bool(provenance_cfg.get("require_config_match", True)) and not bool(args.allow_config_mismatch)

    qlib.init(provider_uri=provider_uri, region=REG_US)
    cal = Cal.calendar(freq="day", future=False)
    cal_end_cap = cal[-2] if len(cal) >= 2 else None

    results: List[Dict[str, object]] = []
    errors: List[str] = []
    seen_pred: Dict[str, str] = {}

    for cand in candidates:
        name = str(cand.get("name") or cand.get("config") or "<unnamed>")
        try:
            config_path = Path(cand["config"]).expanduser().resolve()
            pred_path = Path(cand["pred"]).expanduser().resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"config not found: {config_path}")
            if not pred_path.exists():
                raise FileNotFoundError(f"pred not found: {pred_path}")
            pred_key = str(pred_path)
            if require_unique_pred:
                prev = seen_pred.get(pred_key)
                if prev is not None and prev != name:
                    raise ValueError(
                        f"pred artifact reused: {pred_path} (first={prev}, current={name})"
                    )
                seen_pred[pred_key] = name

            run_id = ""
            if provenance_enabled:
                run_dir = _infer_run_dir_from_pred(pred_path)
                if run_dir is None:
                    raise ValueError(f"cannot infer run dir from pred path: {pred_path}")
                run_id = run_dir.name
                if require_config_match:
                    cmd = _load_cmd_sys_argv(run_dir)
                    if not cmd:
                        raise ValueError(f"missing cmd-sys.argv in run dir: {run_dir}")
                    if not _config_in_cmd(config_path, cmd):
                        raise ValueError(
                            f"config mismatch: {config_path.name} not found in run cmd for run {run_id}"
                        )
                if reject_dirty_run:
                    dirty = _run_is_dirty(run_dir)
                    if dirty is True:
                        raise ValueError(f"dirty run snapshot detected: {run_dir}")

            cfg = _load_yaml(config_path)
            port_cfg = cfg.get("port_analysis_config", {})
            strategy_def = _safe_get(port_cfg, ["strategy"], {}) or {}
            strategy_class = strategy_def.get("class", "WeeklyTopkDropoutStrategy")
            strategy_module = strategy_def.get("module_path", "qlib.contrib.strategy")
            strategy_kwargs = dict(strategy_def.get("kwargs", {}) or {})
            strategy_kwargs.update(cand.get("strategy_overrides", {}) or {})
            strategy_rows = strategy_feasibility_check_rows(strategy_kwargs, str(strategy_class))
            strategy_feasibility_ok = all(ok for _, ok, _ in strategy_rows) if strategy_rows else True
            strategy_feasibility_failures = ";".join(name for name, ok, _ in strategy_rows if not ok)
            backtest_cfg = _safe_get(port_cfg, ["backtest"], {}) or {}
            account = backtest_cfg.get("account", 10000000)
            exchange_kwargs = backtest_cfg.get("exchange_kwargs", {}) or {}

            pred = pd.read_pickle(pred_path)
            if not isinstance(pred, pd.DataFrame):
                raise ValueError(f"pred is not DataFrame: {pred_path}")
            dt = pred.index.get_level_values("datetime")

            start = _as_ts(cand.get("start")) or _as_ts(shared_start) or _as_ts(backtest_cfg.get("start_time")) or dt.min()
            end = _as_ts(cand.get("end")) or _as_ts(shared_end) or _as_ts(backtest_cfg.get("end_time")) or dt.max()
            if cal_end_cap is not None:
                end = min(end, cal_end_cap)
            if end <= start:
                raise ValueError(f"invalid backtest range: {start} -> {end}")

            strategy = {"class": strategy_class, "module_path": strategy_module, "kwargs": strategy_kwargs}
            strategy["kwargs"]["signal"] = pred

            report = _run_backtest(pred, strategy, start, end, benchmark, account=account, exchange_kwargs=exchange_kwargs)
            full = _summarize_report(report)
            excess_returns = report["return"] - report["bench"] - report["cost"]
            excess_sharpe = _annualized_sharpe(excess_returns)
            dsr_prob = _deflated_sharpe_probability(excess_returns, n_trials=max(1, len(candidates)))

            yearly_rows: List[Tuple[str, Dict[str, float]]] = []
            yearly_reports: List[Tuple[str, pd.DataFrame]] = []
            yearly_excess_by_year: Dict[str, float] = {}
            for year in range(start.year, end.year + 1):
                y_start = max(start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(end, pd.Timestamp(f"{year}-12-31"))
                y_cal = Cal.calendar(start_time=y_start, end_time=y_end, freq="day", future=False)
                if len(y_cal) < 2:
                    continue
                report_y = _run_backtest(
                    pred, strategy, y_cal[0], y_cal[-1], benchmark, account=account, exchange_kwargs=exchange_kwargs
                )
                yearly_reports.append((str(year), report_y))
                yearly_metrics = _summarize_report(report_y)
                yearly_rows.append((str(year), yearly_metrics))
                year_excess = _safe_float(yearly_metrics.get("excess_ann_return"))
                if year_excess is not None:
                    yearly_excess_by_year[str(year)] = year_excess

            report_stress = _run_backtest(
                pred,
                strategy,
                start,
                end,
                benchmark,
                account=account,
                exchange_kwargs=exchange_kwargs,
                cost_mult=stress_cost_mult,
                deal_price=stress_deal_price,
            )
            stress = _summarize_report(report_stress)

            gates = _evaluate_robustness_gates(
                full_metrics=full,
                stress_metrics=stress,
                yearly_rows=yearly_rows,
                min_full_excess_ann=gate_min_full_excess_ann,
                min_full_ir=gate_min_full_ir,
                max_full_mdd_abs=gate_max_full_mdd_abs,
                min_stress_excess_ann=gate_min_stress_excess_ann,
                max_turnover=gate_max_turnover,
                min_positive_excess_years=gate_min_positive_excess_years,
                min_worst_year_excess_ann=gate_min_worst_year_excess_ann,
                min_year_days=gate_min_year_days,
                cap_positive_years=str(gate_profile) != "qqq_release",
            )
            gates_total = len(gates)
            gates_pass = sum(1 for _, ok, _ in gates if ok)
            gates_ok = gates_pass == gates_total

            y_excess = []
            for _, m in yearly_rows:
                n_days = _safe_float(m.get("n_days"))
                if n_days is not None and n_days < gate_min_year_days:
                    continue
                ex = _safe_float(m.get("excess_ann_return"))
                if ex is not None:
                    y_excess.append(ex)
            pos_years = sum(1 for x in y_excess if x > 0)
            worst_year_excess = min(y_excess) if y_excess else None

            full_excess = _safe_float(full.get("excess_ann_return"))
            stress_excess = _safe_float(stress.get("excess_ann_return"))
            full_ir = _safe_float(full.get("ir"))
            full_mdd = _safe_float(full.get("mdd"))
            rolling_ok = True
            rolling_total = 0
            rolling_pass = 0
            rolling_pass_rate = None
            rolling_worst_excess = None
            rolling_worst_ir = None
            rolling_worst_mdd_abs = None
            rolling_windows = []
            if check_rolling or check_baselines:
                bt_cal = Cal.calendar(start_time=start, end_time=end, freq="day", future=False)
                rolling_windows = _build_rolling_windows(
                    bt_cal, rolling_window_days, rolling_step_days, rolling_min_days
                )
            if check_rolling:
                rolling_total = len(rolling_windows)
                if rolling_total > 0:
                    if rolling_mode == "continuous":
                        def metrics_by_window(w_start, w_end, _w_days):
                            return _summarize_report(_slice_report(report, w_start, w_end))
                    else:
                        def metrics_by_window(w_start, w_end, _w_days):
                            report_w = _run_backtest(
                                pred,
                                strategy,
                                w_start,
                                w_end,
                                benchmark,
                                account=account,
                                exchange_kwargs=exchange_kwargs,
                            )
                            return _summarize_report(report_w)

                    _, rolling_pass, rolling_worst_excess, rolling_worst_ir, rolling_worst_mdd_abs = _collect_rolling_rows(
                        rolling_windows,
                        metrics_by_window=metrics_by_window,
                        ir_metric=rolling_ir_metric,
                        min_excess_ann=rolling_min_excess_ann,
                        min_ir=rolling_min_ir,
                        max_mdd_abs=rolling_max_mdd_abs,
                        max_turnover=rolling_max_turnover,
                    )
                    rolling_pass_rate = float(rolling_pass) / float(rolling_total)
                    rolling_ok = rolling_pass_rate >= rolling_min_pass_rate

            baseline_metrics: Dict[str, object] = {"baseline_ok": True, "baseline_failures": ""}
            if check_baselines:
                baseline_load_rows: List[Tuple[str, bool, str]] = []
                baseline_returns: Dict[str, pd.Series] = {}
                for ticker in _coerce_ticker_list(baseline_args.baseline_tickers):
                    try:
                        returns, source = _load_external_baseline_returns(
                            ticker,
                            start,
                            end,
                            pkl_map=baseline_map,
                        )
                    except Exception as exc:
                        returns = pd.Series(dtype=float)
                        baseline_load_rows.append((f"baseline_{ticker}_loaded", False, repr(exc)))
                    else:
                        ok = not returns.empty
                        if ok:
                            baseline_returns[ticker] = returns
                            idx = pd.DatetimeIndex(returns.index)
                            detail = f"source={source}, rows={len(returns)}, span={idx.min().date()}->{idx.max().date()}"
                        else:
                            detail = "empty"
                        baseline_load_rows.append((f"baseline_{ticker}_loaded", ok, detail))
                comparison_rows, baseline_rolling_rows, baseline_gate_rows = _evaluate_external_baseline_gates(
                    full_report=report,
                    stress_report=report_stress,
                    yearly_reports=yearly_reports,
                    rolling_windows=rolling_windows,
                    baseline_returns=baseline_returns,
                    args=baseline_args,
                )
                baseline_metrics = _flatten_baseline_results(
                    comparison_rows,
                    baseline_rolling_rows,
                    baseline_load_rows + baseline_gate_rows,
                )

            score = _compute_score(
                full_excess,
                stress_excess,
                full_ir,
                full_mdd,
                pos_years,
                worst_year_excess,
                rolling_pass_rate,
                rolling_worst_excess,
                rolling_worst_mdd_abs,
                dsr_prob,
            )

            row = {
                "candidate": name,
                "run_id": run_id,
                "strategy_feasibility_ok": strategy_feasibility_ok,
                "strategy_feasibility_failures": strategy_feasibility_failures,
                "gates_ok": gates_ok,
                "rolling_ok": rolling_ok,
                "gates_pass": gates_pass,
                "gates_total": gates_total,
                "full_excess": full_excess,
                "full_ir": full_ir,
                "full_mdd": full_mdd,
                "stress_excess": stress_excess,
                "excess_sharpe": excess_sharpe,
                "dsr_prob": dsr_prob,
                "rolling_total": rolling_total,
                "rolling_pass": rolling_pass,
                "rolling_pass_rate": rolling_pass_rate,
                "rolling_worst_excess": rolling_worst_excess,
                "rolling_worst_ir": rolling_worst_ir,
                "rolling_worst_mdd_abs": rolling_worst_mdd_abs,
                "rolling_mode": rolling_mode,
                "rolling_ir_metric": rolling_ir_metric,
                "pos_years": pos_years,
                "worst_year_excess": worst_year_excess,
                "_yearly_excess": yearly_excess_by_year,
                "score": score,
            }
            row.update(baseline_metrics)
            decision = _candidate_release_decision(row, require_baseline=check_baselines)
            row["release_ready"] = bool(decision["release_ready"])
            row["release_missing_or_failed"] = ";".join(decision["missing_or_failed"])
            results.append(row)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    results.sort(
        key=lambda r: (
            0 if r.get("release_ready") else 1,
            0 if r.get("strategy_feasibility_ok", True) else 1,
            0 if r["gates_ok"] else 1,
            0 if r["rolling_ok"] else 1,
            0 if r.get("baseline_ok", True) else 1,
            -(_safe_float(r["rolling_pass_rate"]) if r.get("rolling_pass_rate") is not None else -1.0),
            -(_safe_float(r["full_excess"]) or -999),
            -(_safe_float(r["score"]) or -999),
        )
    )
    _print_table(results)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "candidate",
                    "run_id",
                    "release_ready",
                    "release_missing_or_failed",
                    "strategy_feasibility_ok",
                    "strategy_feasibility_failures",
                    "gates_ok",
                    "rolling_ok",
                    "baseline_ok",
                    "baseline_failures",
                    "gates_pass",
                    "gates_total",
                    "full_excess",
                    "full_ir",
                    "full_mdd",
                    "stress_excess",
                    "excess_sharpe",
                    "dsr_prob",
                    "rolling_total",
                    "rolling_pass",
                    "rolling_pass_rate",
                    "rolling_worst_excess",
                    "rolling_worst_ir",
                    "rolling_worst_mdd_abs",
                    "rolling_mode",
                    "rolling_ir_metric",
                    "qqq_full_excess",
                    "qqq_latest_rolling",
                    "qqq_rolling_pass_rate",
                    "spy_full_excess",
                    "spy_latest_rolling",
                    "spy_rolling_pass_rate",
                    "ixic_full_excess",
                    "ixic_latest_rolling",
                    "ixic_rolling_pass_rate",
                    "pos_years",
                    "worst_year_excess",
                    "score",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if not k.startswith("_")} for row in results])
        print(f"\nSaved CSV: {out_path}")

    selected = _select_release_candidate(results)
    if args.out_selection_json:
        out_path = Path(args.out_selection_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selection_rule": "first sorted candidate with strategy_feasibility, robustness gates, rolling gates, and requested external baseline gates all passing",
            "gate_profile": gate_profile,
            "check_external_baselines": check_baselines,
            "baseline_tickers": _coerce_ticker_list(baseline_args.baseline_tickers),
            "rolling": {
                "enabled": check_rolling,
                "mode": rolling_mode,
                "ir_metric": rolling_ir_metric,
                "window_days": rolling_window_days,
                "step_days": rolling_step_days,
                "min_days": rolling_min_days,
                "min_pass_rate": rolling_min_pass_rate,
            },
            "selected_candidate": selected,
            "no_release_candidate": selected is None,
            "candidates": [{k: v for k, v in row.items() if not k.startswith("_")} for row in results],
            "errors": errors,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"Saved selection JSON: {out_path}")

    pbo_result = None
    if args.check_pbo:
        year_keys = sorted({year for r in results for year in (r.get("_yearly_excess", {}) or {}).keys()})
        matrix = pd.DataFrame(
            {
                str(r["candidate"]): [(r.get("_yearly_excess", {}) or {}).get(year, math.nan) for year in year_keys]
                for r in results
            },
            index=year_keys,
        ).T
        pbo_result = _pbo_from_performance_matrix(matrix, max_combinations=int(args.pbo_max_combinations))
        print("\nPBO:")
        if pbo_result.get("available"):
            print(
                "pbo={pbo:.4f} combinations={combinations} candidates={candidates} slices={slices} lambda_median={lambda_median:.4f}".format(
                    **pbo_result
                )
            )
        else:
            print(f"unavailable: {pbo_result.get('reason')}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        if args.fail_on_error:
            return 3
    if args.fail_on_pbo_fail and pbo_result and pbo_result.get("available"):
        pbo = _safe_float(pbo_result.get("pbo"))
        if pbo is None or pbo > float(args.max_pbo):
            return 4
    if args.fail_on_no_release and selected is None:
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
