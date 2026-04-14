#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.data.data import Cal


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_get(d: dict, keys: Iterable[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_label_horizon(label_expr: str) -> Optional[int]:
    if not isinstance(label_expr, str):
        return None
    # common pattern: Ref($close, -11)/Ref($close, -1) - 1
    import re

    m = re.search(r"Ref\(\$close,\s*-(\d+)\)\s*/\s*Ref\(\$close,\s*-(\d+)\)", label_expr)
    if not m:
        return None
    n1 = int(m.group(1))
    n2 = int(m.group(2))
    if n1 <= n2:
        return None
    return n1 - n2


def _as_ts(val) -> pd.Timestamp:
    return pd.Timestamp(val) if val is not None else None


def _date_span_summary(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return "<missing>"
    return f"{start.date()} -> {end.date()} ({(end - start).days} days)"


def _get_calendar_span(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if start is None or end is None:
        return []
    return Cal.calendar(start_time=start, end_time=end, freq="day", future=False)


def _count_active_instruments(inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], date: pd.Timestamp) -> int:
    total = 0
    for spans in inst_spans.values():
        for s, e in spans:
            if s <= date <= e:
                total += 1
                break
    return total


def _sample_month_starts(calendar: List[pd.Timestamp], max_months: int) -> List[pd.Timestamp]:
    if len(calendar) == 0:
        return []
    ser = pd.Series(calendar)
    grouped = ser.groupby(ser.dt.to_period("M")).first().sort_values()
    if max_months is not None and max_months > 0:
        grouped = grouped.iloc[:max_months]
    return grouped.tolist()


def _sample_instruments(inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], n: int, seed: int) -> List[str]:
    inst = sorted(inst_spans.keys())
    if not inst:
        return []
    if n is None or n <= 0 or n >= len(inst):
        return inst
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inst), size=n, replace=False)
    return [inst[i] for i in sorted(idx)]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join([str(x) for x in col]) for col in df.columns]
    return df


def _missing_ratio(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.isna().mean().sort_values(ascending=False)


def _max_constant_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return 0
    run = 1
    max_run = 1
    for i in range(1, len(vals)):
        if np.isclose(vals[i], vals[i - 1], rtol=1e-10, atol=1e-12):
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    max_run = max(max_run, run)
    return max_run


def _pit_staleness_days(df: pd.DataFrame, field: str) -> pd.Series:
    if df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    series = df[field]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if not isinstance(series.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    wide = series.unstack("instrument")
    res = {}
    for inst in wide.columns:
        vals = wide[inst].values
        res[inst] = _max_constant_run(vals)
    return pd.Series(res)


def _run_backtest(
    pred: pd.DataFrame,
    strategy_cfg: dict,
    start_time,
    end_time,
    benchmark,
    account=10000000,
    exchange_kwargs=None,
    cost_mult=1.0,
    deal_price="close",
):
    from qlib.backtest import backtest as normal_backtest

    merged_exchange_kwargs = {
        "limit_threshold": None,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }
    if isinstance(exchange_kwargs, dict):
        merged_exchange_kwargs.update(exchange_kwargs)
    if deal_price is not None:
        merged_exchange_kwargs["deal_price"] = deal_price
    for key in ("open_cost", "close_cost"):
        val = merged_exchange_kwargs.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            merged_exchange_kwargs[key] = float(val) * float(cost_mult)

    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    backtest_cfg = {
        "start_time": start_time,
        "end_time": end_time,
        "account": account,
        "benchmark": benchmark,
        "exchange_kwargs": merged_exchange_kwargs,
    }
    portfolio_metric_dict, _ = normal_backtest(
        executor=executor_cfg, strategy=strategy_cfg, **backtest_cfg
    )
    report, _ = portfolio_metric_dict["1day"]
    return report


def _summarize_report(report: pd.DataFrame) -> Dict[str, float]:
    strat = risk_analysis(report["return"] - report["cost"], freq="1day")
    bench = risk_analysis(report["bench"], freq="1day")
    excess = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="1day")
    out = {
        "ann_return": float(strat.loc["annualized_return", "risk"]),
        "ir": float(strat.loc["information_ratio", "risk"]),
        "mdd": float(strat.loc["max_drawdown", "risk"]),
        "bench_ann_return": float(bench.loc["annualized_return", "risk"]),
        "excess_ann_return": float(excess.loc["annualized_return", "risk"]),
    }
    if "turnover" in report.columns:
        out["avg_turnover"] = float(report["turnover"].mean())
    if "cost" in report.columns and "return" in report.columns:
        out["avg_cost"] = float(report["cost"].mean())
    out["n_days"] = int(len(report))
    return out


def _format_float(val: Optional[float], digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "n/a"
    return f"{val:.{digits}f}"


def _fmt_opt(val) -> str:
    return "<n/a>" if val is None else str(val)


def _summarize_benchmark(benchmark) -> str:
    if isinstance(benchmark, pd.Series):
        if benchmark.empty:
            return "series(len=0)"
        idx = pd.DatetimeIndex(benchmark.index)
        return f"series(len={len(benchmark)}, span={idx.min().date()}->{idx.max().date()})"
    if isinstance(benchmark, list):
        return "list(" + ",".join([str(x) for x in benchmark]) + ")"
    return str(benchmark)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


def _extract_score_series(pred: pd.DataFrame) -> pd.Series:
    if isinstance(pred, pd.Series):
        return pred
    if isinstance(pred, pd.DataFrame):
        if pred.shape[1] == 0:
            return pd.Series(dtype=float)
        return pred.iloc[:, 0]
    return pd.Series(dtype=float)


def _evaluate_strict_data_quality(
    pred: pd.DataFrame,
    *,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    bt_calendar: List[pd.Timestamp],
    benchmark,
    topk: int,
    min_daily_scores: int,
    min_daily_coverage: float,
    max_nan_score_ratio: float,
    max_missing_close_ratio: float,
    max_missing_volume_ratio: float,
    max_benchmark_nan_ratio: float,
) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    score = _extract_score_series(pred)
    if score.empty:
        checks.append(("pred_has_scores", False, "prediction score series is empty"))
        return checks
    if not isinstance(score.index, pd.MultiIndex) or score.index.nlevels < 2:
        checks.append(("pred_multiindex", False, "prediction score index must be MultiIndex(datetime, instrument)"))
        return checks

    dt_level = "datetime" if "datetime" in score.index.names else 0
    dt_index = pd.DatetimeIndex(score.index.get_level_values(dt_level))
    in_range = (dt_index >= bt_start) & (dt_index <= bt_end)
    score = score[in_range]
    if score.empty:
        checks.append(("pred_non_empty_in_backtest", False, f"no scores in {bt_start.date()}->{bt_end.date()}"))
        return checks

    dup_cnt = int(score.index.duplicated(keep=False).sum())
    checks.append(("pred_no_duplicate_index", dup_cnt == 0, f"duplicate_rows={dup_cnt}"))

    nan_ratio = float(score.isna().mean())
    checks.append(
        (
            "pred_nan_ratio",
            nan_ratio <= max_nan_score_ratio,
            f"{_format_float(nan_ratio)} <= {_format_float(max_nan_score_ratio)}",
        )
    )

    daily_non_na = score.dropna().groupby(level=dt_level).size()
    cal_idx = pd.DatetimeIndex(bt_calendar)
    covered_days = int((daily_non_na.reindex(cal_idx, fill_value=0) >= int(min_daily_scores)).sum())
    total_days = int(len(cal_idx))
    coverage = float(covered_days) / float(total_days) if total_days > 0 else 0.0
    checks.append(
        (
            "pred_daily_coverage",
            coverage >= min_daily_coverage,
            f"{covered_days}/{total_days}={_format_float(coverage)} >= {_format_float(min_daily_coverage)} (min_daily_scores={min_daily_scores})",
        )
    )

    selected_pairs = []
    selected_instruments = set()
    for dt, day_score in score.dropna().groupby(level=dt_level):
        if isinstance(day_score.index, pd.MultiIndex):
            day_inst = day_score.droplevel(dt_level)
        else:
            day_inst = day_score
        top = day_inst.sort_values(ascending=False).head(topk)
        for inst in top.index.tolist():
            selected_pairs.append((pd.Timestamp(dt), inst))
            selected_instruments.add(inst)

    selected_pairs = sorted(set(selected_pairs))
    if not selected_pairs:
        checks.append(("topk_data_quality_pairs", False, "no valid top-k prediction pairs found"))
    else:
        px = D.features(sorted(selected_instruments), ["$close", "$volume"], start_time=bt_start, end_time=bt_end)
        close = px["$close"] if "$close" in px.columns else px.iloc[:, 0]
        volume = px["$volume"] if "$volume" in px.columns else pd.Series(index=close.index, dtype=float)

        if isinstance(close.index, pd.MultiIndex) and "instrument" in close.index.names and "datetime" in close.index.names:
            if close.index.names[0] == "instrument":
                pair_index = pd.MultiIndex.from_tuples(
                    [(inst, dt) for dt, inst in selected_pairs], names=["instrument", "datetime"]
                )
            else:
                pair_index = pd.MultiIndex.from_tuples(selected_pairs, names=["datetime", "instrument"])
        else:
            pair_index = pd.MultiIndex.from_tuples(selected_pairs, names=["datetime", "instrument"])

        close_sel = close.reindex(pair_index)
        volume_sel = volume.reindex(pair_index)
        close_missing = float(close_sel.isna().mean())
        volume_missing = float(volume_sel.isna().mean())
        checks.append(
            (
                "topk_close_missing_ratio",
                close_missing <= max_missing_close_ratio,
                f"{_format_float(close_missing)} <= {_format_float(max_missing_close_ratio)} over {len(pair_index)} top-k pairs",
            )
        )
        checks.append(
            (
                "topk_volume_missing_ratio",
                volume_missing <= max_missing_volume_ratio,
                f"{_format_float(volume_missing)} <= {_format_float(max_missing_volume_ratio)} over {len(pair_index)} top-k pairs",
            )
        )

    if isinstance(benchmark, pd.Series):
        b = benchmark.copy()
        b.index = pd.DatetimeIndex(b.index)
        bench_nan_ratio = float(b.reindex(cal_idx).isna().mean()) if len(cal_idx) > 0 else 0.0
        checks.append(
            (
                "benchmark_nan_ratio",
                bench_nan_ratio <= max_benchmark_nan_ratio,
                f"{_format_float(bench_nan_ratio)} <= {_format_float(max_benchmark_nan_ratio)}",
            )
        )

    return checks


def _evaluate_report_quality(report: pd.DataFrame, *, label: str, max_nan_ratio: float) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    checks.append((f"{label}_report_non_empty", not report.empty, f"rows={len(report)}"))
    if report.empty:
        return checks
    required_cols = ["return", "bench", "cost"]
    for col in required_cols:
        if col not in report.columns:
            checks.append((f"{label}_{col}_present", False, "missing"))
            continue
        s = report[col]
        nan_ratio = float(s.isna().mean())
        inf_count = int(np.isinf(s.fillna(0).values).sum())
        ok = nan_ratio <= max_nan_ratio and inf_count == 0
        checks.append(
            (
                f"{label}_{col}_quality",
                ok,
                f"nan_ratio={_format_float(nan_ratio)} <= {_format_float(max_nan_ratio)}, inf_count={inf_count}",
            )
        )
    return checks


def _evaluate_robustness_gates(
    full_metrics: Dict[str, float],
    stress_metrics: Dict[str, float],
    yearly_rows: List[Tuple[str, Dict[str, float]]],
    *,
    min_full_excess_ann: float,
    min_full_ir: float,
    max_full_mdd_abs: float,
    min_stress_excess_ann: float,
    max_turnover: float,
    min_positive_excess_years: int,
    min_worst_year_excess_ann: float,
    min_year_days: int,
) -> List[Tuple[str, bool, str]]:
    gates: List[Tuple[str, bool, str]] = []

    full_excess = _safe_float(full_metrics.get("excess_ann_return"))
    full_ir = _safe_float(full_metrics.get("ir"))
    full_mdd = _safe_float(full_metrics.get("mdd"))
    full_turnover = _safe_float(full_metrics.get("avg_turnover"))
    stress_excess = _safe_float(stress_metrics.get("excess_ann_return"))

    year_excess = []
    eligible_years = 0
    skipped_short_years = 0
    for period, metrics in yearly_rows:
        if period == "full":
            continue
        n_days_raw = metrics.get("n_days")
        n_days = None
        if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)):
            n_days = int(n_days_raw)
        if n_days is not None and n_days < min_year_days:
            skipped_short_years += 1
            continue
        val = _safe_float(metrics.get("excess_ann_return"))
        if val is not None:
            year_excess.append(val)
            eligible_years += 1

    positive_years = sum(1 for v in year_excess if v > 0)
    worst_year_excess = min(year_excess) if year_excess else None

    ok = full_excess is not None and full_excess >= min_full_excess_ann
    gates.append(
        (
            "full_excess_ann",
            ok,
            f"{_format_float(full_excess)} >= {_format_float(min_full_excess_ann)}",
        )
    )

    ok = full_ir is not None and full_ir >= min_full_ir
    gates.append(("full_ir", ok, f"{_format_float(full_ir)} >= {_format_float(min_full_ir)}"))

    ok = full_mdd is not None and abs(full_mdd) <= max_full_mdd_abs
    gates.append(
        (
            "full_mdd_abs",
            ok,
            f"|{_format_float(full_mdd)}| <= {_format_float(max_full_mdd_abs)}",
        )
    )

    ok = stress_excess is not None and stress_excess >= min_stress_excess_ann
    gates.append(
        (
            "stress_excess_ann",
            ok,
            f"{_format_float(stress_excess)} >= {_format_float(min_stress_excess_ann)}",
        )
    )

    ok = full_turnover is not None and full_turnover <= max_turnover
    gates.append(
        (
            "full_avg_turnover",
            ok,
            f"{_format_float(full_turnover)} <= {_format_float(max_turnover)}",
        )
    )

    ok = positive_years >= min_positive_excess_years
    gates.append(
        (
            "positive_excess_years",
            ok,
            f"{positive_years} >= {min_positive_excess_years} (eligible_years={eligible_years}, min_year_days={min_year_days}, skipped_short_years={skipped_short_years})",
        )
    )

    ok = worst_year_excess is not None and worst_year_excess >= min_worst_year_excess_ann
    gates.append(
        (
            "worst_year_excess_ann",
            ok,
            f"{_format_float(worst_year_excess)} >= {_format_float(min_worst_year_excess_ann)} (eligible_years={eligible_years}, min_year_days={min_year_days}, skipped_short_years={skipped_short_years})",
        )
    )
    return gates


def _print_gate_table(gates: List[Tuple[str, bool, str]]):
    print("gate | status | detail")
    print("--- | --- | ---")
    for name, ok, detail in gates:
        status = "PASS" if ok else "FAIL"
        print(f"{name} | {status} | {detail}")


def _print_check_table(rows: List[Tuple[str, bool, str]]):
    print("check | status | detail")
    print("--- | --- | ---")
    for name, ok, detail in rows:
        print(f"{name} | {'PASS' if ok else 'FAIL'} | {detail}")


def _print_header(title: str):
    print(f"\n== {title} ==")


def _print_kv(label: str, value: str):
    print(f"- {label}: {value}")


def _print_table(rows: List[Tuple[str, Dict[str, float]]]):
    if not rows:
        print("(no data)")
        return
    headers = [
        "period",
        "n_days",
        "ann_return",
        "ir",
        "mdd",
        "bench_ann_return",
        "excess_ann_return",
        "avg_turnover",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for name, metrics in rows:
        n_days_raw = metrics.get("n_days")
        if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)):
            n_days = str(int(n_days_raw))
        else:
            n_days = "n/a"
        row = [
            name,
            n_days,
            _format_float(metrics.get("ann_return")),
            _format_float(metrics.get("ir")),
            _format_float(metrics.get("mdd")),
            _format_float(metrics.get("bench_ann_return")),
            _format_float(metrics.get("excess_ann_return")),
            _format_float(metrics.get("avg_turnover")),
        ]
        print(" | ".join(row))


def _build_rolling_windows(
    calendar: List[pd.Timestamp], window_days: int, step_days: int, min_days: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(calendar) < max(2, min_days):
        return []
    window_days = max(window_days, min_days)
    step_days = max(1, step_days)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start_idx = 0
    while start_idx < len(calendar):
        end_idx = min(start_idx + window_days - 1, len(calendar) - 1)
        n_days = end_idx - start_idx + 1
        if n_days < min_days:
            break
        windows.append((calendar[start_idx], calendar[end_idx], n_days))
        if end_idx == len(calendar) - 1:
            break
        start_idx += step_days

    # Ensure the tail is evaluated even if step/grid misses the final anchor.
    tail_start_idx = max(0, len(calendar) - window_days)
    tail_end_idx = len(calendar) - 1
    tail_days = tail_end_idx - tail_start_idx + 1
    tail_window = (calendar[tail_start_idx], calendar[tail_end_idx], tail_days)
    if tail_days >= min_days and (not windows or windows[-1][1] != tail_window[1]):
        windows.append(tail_window)
    return windows


def _print_rolling_table(rows: List[Tuple[str, Dict[str, str]]]):
    if not rows:
        print("(no rolling windows)")
        return
    headers = [
        "window",
        "status",
        "n_days",
        "ann_return",
        "ir",
        "mdd",
        "bench_ann_return",
        "excess_ann_return",
        "avg_turnover",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for name, metrics in rows:
        row = [
            name,
            metrics.get("status", "n/a"),
            metrics.get("n_days", "n/a"),
            metrics.get("ann_return", "n/a"),
            metrics.get("ir", "n/a"),
            metrics.get("mdd", "n/a"),
            metrics.get("bench_ann_return", "n/a"),
            metrics.get("excess_ann_return", "n/a"),
            metrics.get("avg_turnover", "n/a"),
        ]
        print(" | ".join(row))


GATE_PRESETS = {
    "research": {
        "gate_min_full_excess_ann": 0.03,
        "gate_min_full_ir": 0.40,
        "gate_max_full_mdd_abs": 0.45,
        "gate_min_stress_excess_ann": 0.00,
        "gate_max_turnover": 0.10,
        "gate_min_positive_excess_years": 2,
        "gate_min_worst_year_excess_ann": -0.40,
        "gate_min_year_days": 200,
    },
    "release": {
        "gate_min_full_excess_ann": 0.03,
        "gate_min_full_ir": 0.40,
        "gate_max_full_mdd_abs": 0.35,
        "gate_min_stress_excess_ann": 0.01,
        "gate_max_turnover": 0.10,
        "gate_min_positive_excess_years": 3,
        "gate_min_worst_year_excess_ann": -0.20,
        "gate_min_year_days": 200,
    },
}

ROLLING_PRESETS = {
    "research": {
        "rolling_window_days": 252,
        "rolling_step_days": 63,
        "rolling_min_days": 126,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.50,
        "rolling_max_turnover": 0.12,
        "rolling_min_pass_rate": 0.60,
    },
    "release": {
        "rolling_window_days": 504,
        "rolling_step_days": 252,
        "rolling_min_days": 252,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.35,
        "rolling_max_turnover": 0.10,
        "rolling_min_pass_rate": 0.80,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and stress-check the US Sharadar weekly pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--provider_uri", default=None, help="Override qlib data root")
    parser.add_argument("--pred", default=None, help="Path to pred.pkl for backtest validation")
    parser.add_argument("--benchmark_pkl", default=None, help="Pickled pd.Series of benchmark daily returns")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--by_year", action="store_true", help="Slice backtests by calendar year")
    parser.add_argument("--stress_cost_mult", type=float, default=2.0, help="Cost multiplier for stress test")
    parser.add_argument("--stress_deal_price", default="close", help="Deal price for stress test (close/open)")
    parser.add_argument("--sample_months", type=int, default=24, help="Months to sample for universe stats")
    parser.add_argument("--sample_instruments", type=int, default=200, help="Instrument sample size for data checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--check_start", default=None, help="Override data check start date")
    parser.add_argument("--check_end", default=None, help="Override data check end date")
    parser.add_argument("--skip_data_checks", action="store_true", help="Skip data/universe checks")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip backtest checks")
    parser.add_argument("--check_data_quality", action="store_true", help="Run strict prediction/benchmark data quality checks on backtest inputs")
    parser.add_argument("--data_quality_topk", type=int, default=50, help="Top-k predictions per day to validate against close/volume availability")
    parser.add_argument("--data_quality_min_daily_scores", type=int, default=30, help="Minimum daily non-NaN score rows required for daily coverage")
    parser.add_argument("--data_quality_min_daily_coverage", type=float, default=0.98, help="Minimum ratio of backtest days meeting daily score coverage threshold")
    parser.add_argument("--data_quality_max_nan_score_ratio", type=float, default=0.01, help="Maximum NaN ratio allowed in prediction scores over backtest range")
    parser.add_argument("--data_quality_max_missing_close_ratio", type=float, default=0.01, help="Maximum missing close ratio in top-k prediction pairs")
    parser.add_argument("--data_quality_max_missing_volume_ratio", type=float, default=0.02, help="Maximum missing volume ratio in top-k prediction pairs")
    parser.add_argument("--data_quality_max_benchmark_nan_ratio", type=float, default=0.0, help="Maximum missing ratio for benchmark values over backtest calendar")
    parser.add_argument("--report_max_nan_ratio", type=float, default=0.0, help="Maximum NaN ratio allowed in backtest report core columns")
    parser.add_argument("--fail_on_data_quality_fail", action="store_true", help="Return non-zero exit code if strict data/report quality checks fail")
    parser.add_argument(
        "--gate_profile",
        choices=["release", "research"],
        default="release",
        help="Preset gate profile. release is stricter and intended for deployment decisions.",
    )
    parser.add_argument("--check_gates", action="store_true", help="Evaluate robustness gates on backtest outputs")
    parser.add_argument("--gate_min_full_excess_ann", type=float, default=None, help="Gate: minimum full-period excess annualized return")
    parser.add_argument("--gate_min_full_ir", type=float, default=None, help="Gate: minimum full-period IR")
    parser.add_argument("--gate_max_full_mdd_abs", type=float, default=None, help="Gate: maximum absolute full-period MDD")
    parser.add_argument("--gate_min_stress_excess_ann", type=float, default=None, help="Gate: minimum stress-test excess annualized return")
    parser.add_argument("--gate_max_turnover", type=float, default=None, help="Gate: maximum full-period average turnover")
    parser.add_argument("--gate_min_positive_excess_years", type=int, default=None, help="Gate: minimum number of positive-excess calendar years")
    parser.add_argument("--gate_min_worst_year_excess_ann", type=float, default=None, help="Gate: minimum acceptable worst calendar-year excess annualized return")
    parser.add_argument("--gate_min_year_days", type=int, default=None, help="Gate: minimum trading days for a year slice to be eligible for yearly robustness gates")
    parser.add_argument("--fail_on_gate_fail", action="store_true", help="Return non-zero exit code if any gate fails")
    parser.add_argument("--check_rolling", action="store_true", help="Run rolling walk-forward robustness checks")
    parser.add_argument("--rolling_window_days", type=int, default=None, help="Rolling window size in trading days")
    parser.add_argument("--rolling_step_days", type=int, default=None, help="Rolling step in trading days")
    parser.add_argument("--rolling_min_days", type=int, default=None, help="Minimum trading days required for a rolling window")
    parser.add_argument("--rolling_min_excess_ann", type=float, default=None, help="Rolling gate: minimum annualized excess return")
    parser.add_argument("--rolling_min_ir", type=float, default=None, help="Rolling gate: minimum information ratio")
    parser.add_argument("--rolling_max_mdd_abs", type=float, default=None, help="Rolling gate: maximum absolute max drawdown")
    parser.add_argument("--rolling_max_turnover", type=float, default=None, help="Rolling gate: maximum average turnover")
    parser.add_argument("--rolling_min_pass_rate", type=float, default=None, help="Rolling gate: minimum passing-window ratio")
    parser.add_argument("--fail_on_rolling_fail", action="store_true", help="Return non-zero exit code if rolling pass rate is below threshold")
    args = parser.parse_args()

    gate_defaults = GATE_PRESETS[args.gate_profile]
    for key, val in gate_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    rolling_defaults = ROLLING_PRESETS[args.gate_profile]
    for key, val in rolling_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    return args


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    cfg = _load_yaml(cfg_path)

    # Basic config extraction
    qlib_init = cfg.get("qlib_init", {})
    provider_uri = args.provider_uri or _safe_get(qlib_init, ["provider_uri"], None)
    if provider_uri is None:
        provider_uri = "/root/.qlib/qlib_data/us_data"

    data_handler_config = cfg.get("data_handler_config", {})
    market = cfg.get("market")
    config_benchmark = cfg.get("benchmark", "AAPL")
    port_cfg = cfg.get("port_analysis_config", {})
    strategy_def = _safe_get(port_cfg, ["strategy"], {}) or {}
    strategy_class = strategy_def.get("class", "WeeklyTopkDropoutStrategy")
    strategy_module = strategy_def.get("module_path", "qlib.contrib.strategy")
    strategy_cfg = strategy_def.get("kwargs", {}) or {}
    backtest_cfg = _safe_get(port_cfg, ["backtest"], {}) or {}

    segments = _safe_get(cfg, ["task", "dataset", "kwargs", "segments"], {}) or {}
    train_seg = segments.get("train", [])
    valid_seg = segments.get("valid", [])
    test_seg = segments.get("test", [])

    label_expr = None
    label_cfg = data_handler_config.get("label", [])
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        label_expr = label_cfg[0][0]

    # Print summary
    _print_header("Config Summary")
    _print_kv("config", str(cfg_path))
    _print_kv("gate_profile", str(args.gate_profile))
    _print_kv("provider_uri", provider_uri)
    _print_kv("market", str(market))
    _print_kv("benchmark_config", str(config_benchmark))
    _print_kv("train", _date_span_summary(_as_ts(train_seg[0]) if train_seg else None, _as_ts(train_seg[1]) if len(train_seg) > 1 else None))
    _print_kv("valid", _date_span_summary(_as_ts(valid_seg[0]) if valid_seg else None, _as_ts(valid_seg[1]) if len(valid_seg) > 1 else None))
    _print_kv("test", _date_span_summary(_as_ts(test_seg[0]) if test_seg else None, _as_ts(test_seg[1]) if len(test_seg) > 1 else None))

    label_horizon = _parse_label_horizon(label_expr) if label_expr else None
    _print_kv("label_expr", label_expr or "<missing>")
    _print_kv("label_horizon_days", str(label_horizon) if label_horizon is not None else "<unknown>")

    rebalance_weekday = strategy_cfg.get("rebalance_weekday")
    _print_kv("strategy_class", f"{strategy_module}.{strategy_class}")
    _print_kv("rebalance_weekday", _fmt_opt(rebalance_weekday))
    _print_kv("topk", _fmt_opt(strategy_cfg.get("topk")))
    _print_kv("n_drop", _fmt_opt(strategy_cfg.get("n_drop")))
    _print_kv("hold_thresh", _fmt_opt(strategy_cfg.get("hold_thresh")))

    _print_header("Config Consistency Checks")
    consistency_rows: List[Tuple[str, bool, str]] = []
    train_start = _as_ts(train_seg[0]) if train_seg else None
    train_end = _as_ts(train_seg[1]) if len(train_seg) > 1 else None
    valid_start = _as_ts(valid_seg[0]) if valid_seg else None
    valid_end = _as_ts(valid_seg[1]) if len(valid_seg) > 1 else None
    test_start = _as_ts(test_seg[0]) if test_seg else None
    test_end = _as_ts(test_seg[1]) if len(test_seg) > 1 else None

    seg_ok = all(v is not None for v in [train_start, train_end, valid_start, valid_end, test_start, test_end])
    consistency_rows.append(("segments_present", seg_ok, "train/valid/test each has start+end"))
    if seg_ok:
        temporal_ok = train_start <= train_end < valid_start <= valid_end < test_start <= test_end
        detail = f"{train_start.date()}->{train_end.date()} < {valid_start.date()}->{valid_end.date()} < {test_start.date()}->{test_end.date()}"
        consistency_rows.append(("segments_temporal_order", temporal_ok, detail))

    hold_thresh = strategy_cfg.get("hold_thresh")
    if isinstance(hold_thresh, (int, float)) and label_horizon is not None:
        align_ok = int(hold_thresh) == int(label_horizon)
        consistency_rows.append(
            (
                "hold_horizon_alignment",
                align_ok,
                f"hold_thresh={int(hold_thresh)} vs label_horizon={int(label_horizon)}",
            )
        )

    _print_check_table(consistency_rows)

    # Init qlib
    qlib.init(provider_uri=provider_uri, region=REG_US)

    cal = Cal.calendar(freq="day", future=False)
    if len(cal) > 0:
        last_cal = cal[-1]
        test_end = _as_ts(test_seg[1]) if len(test_seg) > 1 else None
        if test_end is not None and test_end < last_cal:
            _print_kv("warning", f"test end {test_end.date()} < latest calendar {last_cal.date()}")

    # Rebalance weekday holiday check
    if rebalance_weekday is not None and test_seg and len(test_seg) > 1:
        start = _as_ts(test_seg[0])
        end = _as_ts(test_seg[1])
        cal_range = _get_calendar_span(start, end)
        if len(cal_range) > 0:
            weeks = {}
            for dt in cal_range:
                iso = dt.isocalendar()
                key = (iso.year, iso.week)
                weeks.setdefault(key, set()).add(dt.weekday())
            missing_weeks = [k for k, days in weeks.items() if rebalance_weekday not in days]
            _print_kv("weeks_missing_rebalance_day", str(len(missing_weeks)))
            if missing_weeks:
                sample = ", ".join([f"{y}-W{w}" for y, w in missing_weeks[:5]])
                _print_kv("missing_week_samples", sample)

    if not args.skip_data_checks:
        _print_header("Universe & Data Checks")
        # Universe spans with filters
        filter_pipe = data_handler_config.get("filter_pipe", [])
        inst_conf = D.instruments(market, filter_pipe=filter_pipe)
        inst_spans = D.list_instruments(inst_conf, start_time=data_handler_config.get("start_time"), end_time=data_handler_config.get("end_time"), as_list=False)

        check_start = _as_ts(args.check_start) if args.check_start else _as_ts(data_handler_config.get("start_time"))
        check_end = _as_ts(args.check_end) if args.check_end else _as_ts(data_handler_config.get("end_time"))
        cal_range = _get_calendar_span(check_start, check_end)
        sample_dates = _sample_month_starts(cal_range, args.sample_months)

        if sample_dates:
            counts = [
                _count_active_instruments(inst_spans, dt) for dt in sample_dates
            ]
            if counts:
                _print_kv("sample_months", str(len(sample_dates)))
                _print_kv("universe_count_min", str(int(np.min(counts))))
                _print_kv("universe_count_median", str(int(np.median(counts))))
                _print_kv("universe_count_max", str(int(np.max(counts))))

        # Feature availability on sample
        sample_inst = _sample_instruments(inst_spans, args.sample_instruments, args.seed)
        if sample_inst and len(cal_range) > 0:
            sample_start = sample_dates[0] if sample_dates else cal_range[0]
            sample_end = sample_dates[-1] if sample_dates else cal_range[-1]
            pit_fields = data_handler_config.get("pit_fields", [])
            extra_fields = data_handler_config.get("extra_fields", [])
            fields = ["$close", "$volume"]
            if pit_fields:
                fields += [f"P($${str(f).strip().lower()}_{data_handler_config.get('pit_interval', 'q')})" for f in pit_fields[:3]]
            if extra_fields:
                fields += list(extra_fields[:2])

            df = D.features(sample_inst, fields, start_time=sample_start, end_time=sample_end)
            df = _flatten_columns(df)
            missing = _missing_ratio(df)
            _print_kv("sample_instruments", str(len(sample_inst)))
            _print_kv("sample_span", _date_span_summary(sample_start, sample_end))
            if not missing.empty:
                top_missing = missing.head(5)
                _print_kv("missing_ratio_top5", ", ".join([f"{k}={v:.2%}" for k, v in top_missing.items()]))

            # PIT staleness check on first PIT field
            if pit_fields:
                pit_field = f"P($${str(pit_fields[0]).strip().lower()}_{data_handler_config.get('pit_interval', 'q')})"
                if pit_field in df.columns:
                    staleness = _pit_staleness_days(df, pit_field)
                    if not staleness.empty:
                        _print_kv("pit_staleness_median_days", str(int(np.median(staleness))))
                        _print_kv("pit_staleness_p95_days", str(int(np.percentile(staleness, 95))))

    if args.skip_backtest:
        return 0

    if not args.pred:
        _print_header("Backtest Checks")
        _print_kv("note", "pred.pkl not provided; skipping backtest checks")
        return 0

    pred_path = Path(args.pred).expanduser().resolve()
    if not pred_path.exists():
        print(f"pred.pkl not found: {pred_path}")
        return 2

    pred = pd.read_pickle(pred_path)
    if not isinstance(pred, pd.DataFrame):
        print("pred.pkl must be a pandas DataFrame")
        return 2

    # Determine backtest range
    dt_index = pred.index.get_level_values("datetime")
    cfg_bt_start = _as_ts(backtest_cfg.get("start_time"))
    cfg_bt_end = _as_ts(backtest_cfg.get("end_time"))
    if args.start:
        bt_start_raw = pd.Timestamp(args.start)
    elif cfg_bt_start is not None:
        bt_start_raw = cfg_bt_start
    else:
        bt_start_raw = dt_index.min()
    if args.end:
        bt_end_raw = pd.Timestamp(args.end)
    elif cfg_bt_end is not None:
        bt_end_raw = cfg_bt_end
    else:
        bt_end_raw = dt_index.max()

    # Avoid using future calendar index
    cal = Cal.calendar(freq="day", future=False)
    if len(cal) >= 2:
        bt_end_raw = min(bt_end_raw, cal[-2])

    bt_calendar = _get_calendar_span(bt_start_raw, bt_end_raw)
    if len(bt_calendar) < 2:
        print(f"invalid backtest calendar span: start={bt_start_raw.date()} end={bt_end_raw.date()}")
        return 2
    bt_start = bt_calendar[0]
    bt_end = bt_calendar[-1]

    benchmark = config_benchmark
    benchmark_source = "config"
    if args.benchmark_pkl:
        bench_path = Path(args.benchmark_pkl).expanduser().resolve()
        benchmark = pd.read_pickle(bench_path)
        if not isinstance(benchmark, pd.Series):
            print(f"benchmark_pkl must be a pandas Series: {bench_path}")
            return 2
        benchmark_source = f"pkl:{bench_path}"
    _print_kv("benchmark_effective", _summarize_benchmark(benchmark))
    _print_kv("benchmark_source", benchmark_source)

    _print_header("Backtest Input Checks")
    input_rows: List[Tuple[str, bool, str]] = []
    pred_min = dt_index.min()
    pred_max = dt_index.max()
    pred_cover_ok = pred_min <= bt_start and pred_max >= bt_end
    input_rows.append(
        (
            "pred_covers_backtest_range",
            pred_cover_ok,
            f"pred={pred_min.date()}->{pred_max.date()}, bt={bt_start.date()}->{bt_end.date()}",
        )
    )
    pred_in_test_ok = True
    if test_start is not None and test_end is not None:
        pred_in_test_ok = pred_min >= test_start and pred_max <= test_end
        input_rows.append(
            (
                "pred_within_test_segment",
                pred_in_test_ok,
                f"pred={pred_min.date()}->{pred_max.date()}, test={test_start.date()}->{test_end.date()}",
            )
        )
    if isinstance(benchmark, pd.Series):
        bidx = pd.DatetimeIndex(benchmark.index)
        bench_cover_ok = bidx.min() <= bt_start and bidx.max() >= bt_end
        input_rows.append(
            (
                "benchmark_covers_backtest_range",
                bench_cover_ok,
                f"bench={bidx.min().date()}->{bidx.max().date()}, bt={bt_start.date()}->{bt_end.date()}",
            )
        )
    _print_check_table(input_rows)

    if args.check_data_quality:
        _print_header("Strict Data Quality Checks")
        dq_rows = _evaluate_strict_data_quality(
            pred,
            bt_start=bt_start,
            bt_end=bt_end,
            bt_calendar=bt_calendar,
            benchmark=benchmark,
            topk=args.data_quality_topk,
            min_daily_scores=args.data_quality_min_daily_scores,
            min_daily_coverage=args.data_quality_min_daily_coverage,
            max_nan_score_ratio=args.data_quality_max_nan_score_ratio,
            max_missing_close_ratio=args.data_quality_max_missing_close_ratio,
            max_missing_volume_ratio=args.data_quality_max_missing_volume_ratio,
            max_benchmark_nan_ratio=args.data_quality_max_benchmark_nan_ratio,
        )
        _print_check_table(dq_rows)
        dq_ok = all(ok for _, ok, _ in dq_rows)
        _print_kv("data_quality_overall", "PASS" if dq_ok else "FAIL")
        if args.fail_on_data_quality_fail and not dq_ok:
            return 5

    account = backtest_cfg.get("account", 10000000)
    exchange_kwargs = backtest_cfg.get("exchange_kwargs", {}) or {}
    strategy_kwargs = dict(strategy_cfg)
    strategy_kwargs["signal"] = pred
    base_strategy = {"class": strategy_class, "module_path": strategy_module, "kwargs": strategy_kwargs}

    _print_header("Backtest Checks")
    report = _run_backtest(
        pred, base_strategy, bt_start, bt_end, benchmark, account=account, exchange_kwargs=exchange_kwargs
    )
    rows = [(f"full {bt_start.date()}->{bt_end.date()}", _summarize_report(report))]

    if args.check_data_quality:
        full_report_rows = _evaluate_report_quality(report, label="full", max_nan_ratio=args.report_max_nan_ratio)
        _print_header("Backtest Report Quality")
        _print_check_table(full_report_rows)
        full_report_ok = all(ok for _, ok, _ in full_report_rows)
        _print_kv("full_report_quality_overall", "PASS" if full_report_ok else "FAIL")
        if args.fail_on_data_quality_fail and not full_report_ok:
            return 6

    if args.by_year:
        years = range(bt_start.year, bt_end.year + 1)
        for year in years:
            y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
            y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
            y_cal = _get_calendar_span(y_start, y_end)
            if len(y_cal) < 2:
                continue
            report_y = _run_backtest(
                pred, base_strategy, y_cal[0], y_cal[-1], benchmark, account=account, exchange_kwargs=exchange_kwargs
            )
            rows.append((str(year), _summarize_report(report_y)))

    _print_table(rows)

    # Stress test with higher costs / alternate deal price
    stress_rows = []
    stress_report = _run_backtest(
        pred,
        base_strategy,
        bt_start,
        bt_end,
        benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
        cost_mult=args.stress_cost_mult,
        deal_price=args.stress_deal_price,
    )
    stress_rows.append((f"stress x{args.stress_cost_mult} {args.stress_deal_price}", _summarize_report(stress_report)))
    _print_header("Stress Test")
    _print_table(stress_rows)

    if args.check_data_quality:
        stress_report_rows = _evaluate_report_quality(stress_report, label="stress", max_nan_ratio=args.report_max_nan_ratio)
        _print_header("Stress Report Quality")
        _print_check_table(stress_report_rows)
        stress_report_ok = all(ok for _, ok, _ in stress_report_rows)
        _print_kv("stress_report_quality_overall", "PASS" if stress_report_ok else "FAIL")
        if args.fail_on_data_quality_fail and not stress_report_ok:
            return 7

    if args.check_gates:
        gate_rows = list(rows)
        if len(gate_rows) <= 1:
            years = range(bt_start.year, bt_end.year + 1)
            for year in years:
                y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
                y_cal = _get_calendar_span(y_start, y_end)
                if len(y_cal) < 2:
                    continue
                report_y = _run_backtest(
                    pred,
                    base_strategy,
                    y_cal[0],
                    y_cal[-1],
                    benchmark,
                    account=account,
                    exchange_kwargs=exchange_kwargs,
                )
                gate_rows.append((str(year), _summarize_report(report_y)))
        gates = _evaluate_robustness_gates(
            full_metrics=rows[0][1],
            stress_metrics=stress_rows[0][1],
            yearly_rows=gate_rows[1:],
            min_full_excess_ann=args.gate_min_full_excess_ann,
            min_full_ir=args.gate_min_full_ir,
            max_full_mdd_abs=args.gate_max_full_mdd_abs,
            min_stress_excess_ann=args.gate_min_stress_excess_ann,
            max_turnover=args.gate_max_turnover,
            min_positive_excess_years=args.gate_min_positive_excess_years,
            min_worst_year_excess_ann=args.gate_min_worst_year_excess_ann,
            min_year_days=args.gate_min_year_days,
        )
        _print_header("Robustness Gates")
        _print_gate_table(gates)
        overall_ok = all(ok for _, ok, _ in gates)
        _print_kv("gates_overall", "PASS" if overall_ok else "FAIL")
        if args.fail_on_gate_fail and not overall_ok:
            return 3

    if args.check_rolling:
        _print_header("Rolling Walk-Forward Checks")
        rolling_windows = _build_rolling_windows(
            bt_calendar, args.rolling_window_days, args.rolling_step_days, args.rolling_min_days
        )
        if not rolling_windows:
            _print_kv("note", "no rolling windows matched current range/parameters")
        else:
            rolling_rows: List[Tuple[str, Dict[str, str]]] = []
            rolling_pass = 0
            worst_excess = None
            worst_ir = None
            worst_mdd_abs = None
            for w_start, w_end, w_days in rolling_windows:
                report_w = _run_backtest(
                    pred,
                    base_strategy,
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
                    and excess_w >= args.rolling_min_excess_ann
                    and ir_w is not None
                    and ir_w >= args.rolling_min_ir
                    and mdd_abs_w is not None
                    and mdd_abs_w <= args.rolling_max_mdd_abs
                    and turnover_w is not None
                    and turnover_w <= args.rolling_max_turnover
                )
                if pass_window:
                    rolling_pass += 1

                if excess_w is not None:
                    worst_excess = excess_w if worst_excess is None else min(worst_excess, excess_w)
                if ir_w is not None:
                    worst_ir = ir_w if worst_ir is None else min(worst_ir, ir_w)
                if mdd_abs_w is not None:
                    worst_mdd_abs = mdd_abs_w if worst_mdd_abs is None else max(worst_mdd_abs, mdd_abs_w)

                rolling_rows.append(
                    (
                        f"{w_start.date()}->{w_end.date()}",
                        {
                            "status": "PASS" if pass_window else "FAIL",
                            "n_days": str(w_days),
                            "ann_return": _format_float(metrics_w.get("ann_return")),
                            "ir": _format_float(ir_w),
                            "mdd": _format_float(mdd_w),
                            "bench_ann_return": _format_float(metrics_w.get("bench_ann_return")),
                            "excess_ann_return": _format_float(excess_w),
                            "avg_turnover": _format_float(turnover_w),
                        },
                    )
                )

            _print_rolling_table(rolling_rows)
            total_windows = len(rolling_windows)
            pass_rate = float(rolling_pass) / float(total_windows) if total_windows > 0 else 0.0
            _print_kv("rolling_windows_total", str(total_windows))
            _print_kv("rolling_windows_pass", str(rolling_pass))
            _print_kv("rolling_pass_rate", _format_float(pass_rate))
            _print_kv("rolling_worst_excess_ann", _format_float(worst_excess))
            _print_kv("rolling_worst_ir", _format_float(worst_ir))
            _print_kv("rolling_worst_mdd_abs", _format_float(worst_mdd_abs))
            rolling_overall_ok = pass_rate >= args.rolling_min_pass_rate
            _print_kv(
                "rolling_overall",
                f"{'PASS' if rolling_overall_ok else 'FAIL'} (pass_rate={_format_float(pass_rate)} >= threshold={_format_float(args.rolling_min_pass_rate)})",
            )
            if args.fail_on_rolling_fail and not rolling_overall_ok:
                return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
