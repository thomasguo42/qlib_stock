#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data.data import Cal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_us_sharadar_pipeline import (
    _as_ts,
    _build_rolling_windows,
    _evaluate_robustness_gates,
    _load_yaml,
    _run_backtest,
    _safe_get,
    _summarize_report,
)


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
        "gates",
        "rolling",
        "gates_pass",
        "full_excess",
        "full_ir",
        "full_mdd",
        "stress_excess",
        "roll_pass_rate",
        "roll_worst_ex",
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
            "PASS" if r["gates_ok"] else "FAIL",
            "PASS" if r["rolling_ok"] else "FAIL",
            f"{r['gates_pass']}/{r['gates_total']}",
            _fmt(r["full_excess"]),
            _fmt(r["full_ir"]),
            _fmt(r["full_mdd"]),
            _fmt(r["stress_excess"]),
            _fmt(r["rolling_pass_rate"]),
            _fmt(r["rolling_worst_excess"]),
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
) -> float:
    ex = full_excess if full_excess is not None else -1.0
    sx = stress_excess if stress_excess is not None else -1.0
    ir = full_ir if full_ir is not None else -1.0
    mdd_penalty = abs(full_mdd) if full_mdd is not None else 1.0
    wye = worst_year_excess if worst_year_excess is not None else -1.0
    roll_pr = rolling_pass_rate if rolling_pass_rate is not None else 0.0
    roll_ex = rolling_worst_excess if rolling_worst_excess is not None else -1.0
    roll_mdd = rolling_worst_mdd_abs if rolling_worst_mdd_abs is not None else 1.0
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
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank US Sharadar pipeline candidates with unified robustness gates.")
    p.add_argument("--candidates_yaml", required=True, help="YAML file with candidate list and shared settings")
    p.add_argument("--provider_uri", default=None, help="Override qlib data root")
    p.add_argument("--out_csv", default=None, help="Optional path to write ranking CSV")
    p.add_argument("--check_rolling", action="store_true", help="Include rolling walk-forward checks in ranking")
    p.add_argument("--rolling_window_days", type=int, default=504, help="Rolling window size in trading days")
    p.add_argument("--rolling_step_days", type=int, default=252, help="Rolling step in trading days")
    p.add_argument("--rolling_min_days", type=int, default=252, help="Minimum days for a rolling window")
    p.add_argument("--rolling_min_excess_ann", type=float, default=0.0, help="Rolling gate: min excess annualized return")
    p.add_argument("--rolling_min_ir", type=float, default=0.0, help="Rolling gate: min IR")
    p.add_argument("--rolling_max_mdd_abs", type=float, default=0.35, help="Rolling gate: max absolute MDD")
    p.add_argument("--rolling_max_turnover", type=float, default=0.10, help="Rolling gate: max average turnover")
    p.add_argument("--rolling_min_pass_rate", type=float, default=0.80, help="Rolling gate: minimum pass rate")
    p.add_argument(
        "--skip_provenance_checks",
        action="store_true",
        help="Skip run provenance checks (pred uniqueness, run cleanliness, cmd-config alignment)",
    )
    p.add_argument("--allow_shared_pred", action="store_true", help="Allow multiple candidates to reuse same pred.pkl")
    p.add_argument("--allow_dirty_run", action="store_true", help="Allow dirty code_status snapshots in source runs")
    p.add_argument("--allow_config_mismatch", action="store_true", help="Allow config/run cmd mismatches")
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
    rolling_window_days = int(rolling_cfg.get("window_days", args.rolling_window_days))
    rolling_step_days = int(rolling_cfg.get("step_days", args.rolling_step_days))
    rolling_min_days = int(rolling_cfg.get("min_days", args.rolling_min_days))
    rolling_min_excess_ann = float(rolling_cfg.get("min_excess_ann", args.rolling_min_excess_ann))
    rolling_min_ir = float(rolling_cfg.get("min_ir", args.rolling_min_ir))
    rolling_max_mdd_abs = float(rolling_cfg.get("max_mdd_abs", args.rolling_max_mdd_abs))
    rolling_max_turnover = float(rolling_cfg.get("max_turnover", args.rolling_max_turnover))
    rolling_min_pass_rate = float(rolling_cfg.get("min_pass_rate", args.rolling_min_pass_rate))
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

            yearly_rows: List[Tuple[str, Dict[str, float]]] = []
            for year in range(start.year, end.year + 1):
                y_start = max(start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(end, pd.Timestamp(f"{year}-12-31"))
                y_cal = Cal.calendar(start_time=y_start, end_time=y_end, freq="day", future=False)
                if len(y_cal) < 2:
                    continue
                report_y = _run_backtest(
                    pred, strategy, y_cal[0], y_cal[-1], benchmark, account=account, exchange_kwargs=exchange_kwargs
                )
                yearly_rows.append((str(year), _summarize_report(report_y)))

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
            if check_rolling:
                bt_cal = Cal.calendar(start_time=start, end_time=end, freq="day", future=False)
                rolling_windows = _build_rolling_windows(
                    bt_cal, rolling_window_days, rolling_step_days, rolling_min_days
                )
                rolling_total = len(rolling_windows)
                if rolling_total > 0:
                    for w_start, w_end, _ in rolling_windows:
                        report_w = _run_backtest(
                            pred,
                            strategy,
                            w_start,
                            w_end,
                            benchmark,
                            account=account,
                            exchange_kwargs=exchange_kwargs,
                        )
                        metrics_w = _summarize_report(report_w)
                        excess_w = _safe_float(metrics_w.get("excess_ann_return"))
                        ir_w = _safe_float(metrics_w.get("ir"))
                        mdd_w = _safe_float(metrics_w.get("mdd"))
                        turnover_w = _safe_float(metrics_w.get("avg_turnover"))
                        mdd_abs_w = abs(mdd_w) if mdd_w is not None else None
                        pass_window = (
                            excess_w is not None
                            and excess_w >= rolling_min_excess_ann
                            and ir_w is not None
                            and ir_w >= rolling_min_ir
                            and mdd_abs_w is not None
                            and mdd_abs_w <= rolling_max_mdd_abs
                            and turnover_w is not None
                            and turnover_w <= rolling_max_turnover
                        )
                        if pass_window:
                            rolling_pass += 1
                        if excess_w is not None:
                            rolling_worst_excess = (
                                excess_w if rolling_worst_excess is None else min(rolling_worst_excess, excess_w)
                            )
                        if ir_w is not None:
                            rolling_worst_ir = ir_w if rolling_worst_ir is None else min(rolling_worst_ir, ir_w)
                        if mdd_abs_w is not None:
                            rolling_worst_mdd_abs = (
                                mdd_abs_w if rolling_worst_mdd_abs is None else max(rolling_worst_mdd_abs, mdd_abs_w)
                            )
                    rolling_pass_rate = float(rolling_pass) / float(rolling_total)
                    rolling_ok = rolling_pass_rate >= rolling_min_pass_rate

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
            )

            results.append(
                {
                    "candidate": name,
                    "run_id": run_id,
                    "gates_ok": gates_ok,
                    "rolling_ok": rolling_ok,
                    "gates_pass": gates_pass,
                    "gates_total": gates_total,
                    "full_excess": full_excess,
                    "full_ir": full_ir,
                    "full_mdd": full_mdd,
                    "stress_excess": stress_excess,
                    "rolling_total": rolling_total,
                    "rolling_pass": rolling_pass,
                    "rolling_pass_rate": rolling_pass_rate,
                    "rolling_worst_excess": rolling_worst_excess,
                    "rolling_worst_ir": rolling_worst_ir,
                    "rolling_worst_mdd_abs": rolling_worst_mdd_abs,
                    "pos_years": pos_years,
                    "worst_year_excess": worst_year_excess,
                    "score": score,
                }
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    results.sort(
        key=lambda r: (
            0 if r["gates_ok"] else 1,
            0 if r["rolling_ok"] else 1,
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
                    "gates_ok",
                    "rolling_ok",
                    "gates_pass",
                    "gates_total",
                    "full_excess",
                    "full_ir",
                    "full_mdd",
                    "stress_excess",
                    "rolling_total",
                    "rolling_pass",
                    "rolling_pass_rate",
                    "rolling_worst_excess",
                    "rolling_worst_ir",
                    "rolling_worst_mdd_abs",
                    "pos_years",
                    "worst_year_excess",
                    "score",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV: {out_path}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        if args.fail_on_error:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
