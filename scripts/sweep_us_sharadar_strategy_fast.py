#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import math
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

from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    GATE_PRESETS,
    _as_ts,
    _evaluate_robustness_gates,
    _load_yaml,
    _run_backtest,
    _safe_get,
    _summarize_report,
)


def _safe_float(v) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast strategy sweep for US Sharadar candidates.")
    p.add_argument("--candidates_yaml", required=True, help="YAML with global settings and candidates")
    p.add_argument("--provider_uri", default=None, help="Override qlib provider uri")
    p.add_argument("--out_csv", default=None, help="Optional output csv")
    p.add_argument(
        "--gate_profile",
        choices=["release", "research"],
        default="release",
        help="Gate preset from validate_us_sharadar_pipeline.py",
    )
    p.add_argument(
        "--max_candidates",
        type=int,
        default=None,
        help="Evaluate only first N candidates from yaml",
    )
    return p.parse_args()


def _load_candidates(path: Path) -> Tuple[dict, List[dict]]:
    cfg = _load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    candidates = cfg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates_yaml must contain a non-empty `candidates` list")
    return cfg, candidates


def _print_table(rows: List[Dict[str, object]]):
    if not rows:
        print("(no rows)")
        return
    headers = [
        "rank",
        "candidate",
        "gates",
        "gates_pass",
        "full_excess",
        "full_ir",
        "full_mdd",
        "stress_excess",
        "pos_years",
        "worst_year_excess",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for i, r in enumerate(rows, start=1):
        line = [
            str(i),
            str(r["candidate"]),
            "PASS" if r["gates_ok"] else "FAIL",
            f"{r['gates_pass']}/{r['gates_total']}",
            _fmt(r.get("full_excess")),
            _fmt(r.get("full_ir")),
            _fmt(r.get("full_mdd")),
            _fmt(r.get("stress_excess")),
            str(r.get("pos_years", 0)),
            _fmt(r.get("worst_year_excess")),
        ]
        print(" | ".join(line))


def main() -> int:
    args = _parse_args()
    cpath = Path(args.candidates_yaml).expanduser().resolve()
    if not cpath.exists():
        print(f"candidates_yaml not found: {cpath}")
        return 2

    global_cfg, candidates = _load_candidates(cpath)
    if args.max_candidates is not None and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

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
    gate_defaults = dict(GATE_PRESETS[args.gate_profile])
    gates_cfg = global_cfg.get("gates", {}) or {}
    gate_defaults.update(
        {
            "gate_min_full_excess_ann": float(
                gates_cfg.get("min_full_excess_ann", gate_defaults["gate_min_full_excess_ann"])
            ),
            "gate_min_full_ir": float(gates_cfg.get("min_full_ir", gate_defaults["gate_min_full_ir"])),
            "gate_max_full_mdd_abs": float(
                gates_cfg.get("max_full_mdd_abs", gate_defaults["gate_max_full_mdd_abs"])
            ),
            "gate_min_stress_excess_ann": float(
                gates_cfg.get("min_stress_excess_ann", gate_defaults["gate_min_stress_excess_ann"])
            ),
            "gate_max_turnover": float(gates_cfg.get("max_turnover", gate_defaults["gate_max_turnover"])),
            "gate_min_positive_excess_years": int(
                gates_cfg.get("min_positive_excess_years", gate_defaults["gate_min_positive_excess_years"])
            ),
            "gate_min_worst_year_excess_ann": float(
                gates_cfg.get("min_worst_year_excess_ann", gate_defaults["gate_min_worst_year_excess_ann"])
            ),
            "gate_min_year_days": int(gates_cfg.get("min_year_days", gate_defaults["gate_min_year_days"])),
        }
    )

    qlib.init(provider_uri=provider_uri, region=REG_US)
    cal = Cal.calendar(freq="day", future=False)
    cal_end_cap = cal[-2] if len(cal) >= 2 else None

    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    for cand in candidates:
        name = str(cand.get("name") or cand.get("config") or "<unnamed>")
        try:
            config_path = Path(cand["config"]).expanduser().resolve()
            pred_path = Path(cand["pred"]).expanduser().resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"config not found: {config_path}")
            if not pred_path.exists():
                raise FileNotFoundError(f"pred not found: {pred_path}")

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

            report_full = _run_backtest(
                pred,
                strategy,
                start,
                end,
                benchmark,
                account=account,
                exchange_kwargs=exchange_kwargs,
            )
            full = _summarize_report(report_full)

            yearly_rows: List[Tuple[str, Dict[str, float]]] = []
            report_full = report_full.copy()
            report_full.index = pd.DatetimeIndex(report_full.index)
            for year in range(start.year, end.year + 1):
                y_start = max(start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(end, pd.Timestamp(f"{year}-12-31"))
                report_y = report_full.loc[(report_full.index >= y_start) & (report_full.index <= y_end)]
                if report_y.empty:
                    continue
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
                min_full_excess_ann=gate_defaults["gate_min_full_excess_ann"],
                min_full_ir=gate_defaults["gate_min_full_ir"],
                max_full_mdd_abs=gate_defaults["gate_max_full_mdd_abs"],
                min_stress_excess_ann=gate_defaults["gate_min_stress_excess_ann"],
                max_turnover=gate_defaults["gate_max_turnover"],
                min_positive_excess_years=gate_defaults["gate_min_positive_excess_years"],
                min_worst_year_excess_ann=gate_defaults["gate_min_worst_year_excess_ann"],
                min_year_days=gate_defaults["gate_min_year_days"],
            )
            gates_total = len(gates)
            gates_pass = sum(1 for _, ok, _ in gates if ok)
            gates_ok = gates_pass == gates_total

            year_excess = []
            for _, m in yearly_rows:
                n_days = _safe_float(m.get("n_days"))
                if n_days is not None and n_days < gate_defaults["gate_min_year_days"]:
                    continue
                ex = _safe_float(m.get("excess_ann_return"))
                if ex is not None:
                    year_excess.append(ex)
            pos_years = sum(1 for x in year_excess if x > 0)
            worst_year_excess = min(year_excess) if year_excess else None

            rows.append(
                {
                    "candidate": name,
                    "gates_ok": gates_ok,
                    "gates_pass": gates_pass,
                    "gates_total": gates_total,
                    "full_excess": _safe_float(full.get("excess_ann_return")),
                    "full_ir": _safe_float(full.get("ir")),
                    "full_mdd": _safe_float(full.get("mdd")),
                    "stress_excess": _safe_float(stress.get("excess_ann_return")),
                    "pos_years": pos_years,
                    "worst_year_excess": worst_year_excess,
                }
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    rows.sort(
        key=lambda r: (
            0 if r["gates_ok"] else 1,
            -(r["gates_pass"] / max(1, r["gates_total"])),
            -(_safe_float(r.get("full_excess")) or -999),
            abs(_safe_float(r.get("full_mdd")) or -999),
        )
    )
    _print_table(rows)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "candidate",
                    "gates_ok",
                    "gates_pass",
                    "gates_total",
                    "full_excess",
                    "full_ir",
                    "full_mdd",
                    "stress_excess",
                    "pos_years",
                    "worst_year_excess",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {out_path}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
