#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Summarize US Sharadar release-validation failures from validator logs.

The release validator intentionally prints a detailed human-readable report.
This helper turns one or more logs into a compact failure map so candidates can
be compared without manually grepping each diagnostic section.
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CONFIG_PREFIX = "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_"
CONFIG_SUFFIX = "_topk40"

CHECK_STATUS = {"PASS", "FAIL"}
STATUS_KV_SUFFIXES = ("_overall", "_ready")
NUMERIC_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
LEADING_COMPARISON_RE = re.compile(
    r"^\s*(?P<abs>\|)?\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\|?\s*(?:>=|<=|>|<|==|=)"
)
KV_RE = re.compile(r"^-\s+(?P<key>[^:]+):\s*(?P<value>.*)$")
INLINE_KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^,\s]+)")


KEY_COLUMNS = [
    "source_log",
    "candidate",
    "config",
    "gate_profile",
    "benchmark_config",
    "benchmark_source",
    "label_horizon_days",
    "release_ready",
    "release_missing_or_failed",
    "fail_count",
    "fail_checks",
]

TRACKED_CHECKS = [
    "segments_temporal_order",
    "train_valid_embargo",
    "valid_test_embargo",
    "walkforward_valid_test_embargo_min",
    "training_metrics_present",
    "training_best_iteration_min",
    "model_quality_full_mean_ic",
    "model_quality_full_topq_spread",
    "model_quality_full_worst_year_signed_ic",
    "model_quality_recent_mean_ic",
    "rebalance_model_quality_full_mean_ic",
    "strategy_weighted_full_mean_label",
    "strategy_weighted_positive_years",
    "active_risk_mean_core_weight_gap",
    "active_risk_recent_core_weight_gap",
    "active_risk_mean_stock_overlay_weight",
    "active_risk_mean_max_single_stock_weight",
    "active_risk_mean_stock_names",
    "rebalance_interval_full_ann_excess",
    "rebalance_interval_recent_ann_excess",
    "rebalance_interval_positive_excess_rate",
    "rebalance_interval_positive_years",
    "full_excess_ann",
    "full_ir",
    "full_mdd_abs",
    "stress_excess_ann",
    "full_avg_turnover",
    "positive_excess_years",
    "worst_year_excess_ann",
    "baseline_QQQ_missing_ratio",
    "baseline_QQQ_full_excess_ann",
    "baseline_QQQ_stress_excess_ann",
    "baseline_QQQ_full_mdd_abs",
    "baseline_QQQ_mdd_gap",
    "baseline_QQQ_positive_years",
    "baseline_QQQ_yearly_beat_rate",
    "baseline_QQQ_worst_year_excess_ann",
    "baseline_QQQ_rolling_pass_rate",
    "baseline_QQQ_worst_rolling_excess_ann",
    "baseline_QQQ_latest_rolling_excess_ann",
    "baseline_SPY_full_excess_ann",
    "baseline_SPY_stress_excess_ann",
    "baseline_SPY_rolling_pass_rate",
    "baseline_SPY_worst_rolling_excess_ann",
    "baseline_IXIC_full_excess_ann",
    "baseline_IXIC_stress_excess_ann",
    "baseline_IXIC_rolling_pass_rate",
    "baseline_IXIC_worst_rolling_excess_ann",
    "baseline_regime_QQQ_up_days_excess_ann",
    "baseline_regime_QQQ_down_days_excess_ann",
    "baseline_regime_QQQ_63d_strong_excess_ann",
    "baseline_regime_QQQ_63d_weak_excess_ann",
    "baseline_regime_QQQ_vol20_top_quartile_excess_ann",
    "baseline_regime_QQQ_vol20_bottom_quartile_excess_ann",
    "qqq_release_config_benchmark",
    "qqq_release_backtest_benchmark",
    "qqq_release_label_benchmark",
]

TRACKED_KV = [
    "data_checks_overall",
    "training_diagnostics_overall",
    "model_quality_overall",
    "strategy_weighted_quality_overall",
    "data_quality_overall",
    "active_risk_overall",
    "rebalance_interval_quality_overall",
    "full_report_quality_overall",
    "stress_report_quality_overall",
    "external_baseline_gates_overall",
    "gates_overall",
    "rolling_pass_rate",
    "rolling_worst_excess_ann",
    "rolling_worst_ir",
    "rolling_worst_mdd_abs",
    "rolling_overall",
    "cold_start_rolling_pass_rate",
    "cold_start_rolling_worst_excess_ann",
    "cold_start_rolling_worst_ir",
    "cold_start_rolling_worst_mdd_abs",
    "cold_start_rolling_overall",
]

SUMMARY_COLUMNS = KEY_COLUMNS + TRACKED_KV
DETAIL_COLUMNS = [
    *(f"{name}_status" for name in TRACKED_CHECKS),
    *(f"{name}_detail" for name in TRACKED_CHECKS),
]
CSV_COLUMNS = SUMMARY_COLUMNS + TRACKED_CHECKS + DETAIL_COLUMNS
MARKDOWN_COLUMNS = [
    "candidate",
    "release_ready",
    "release_missing_or_failed",
    "baseline_QQQ_full_excess_ann",
    "baseline_QQQ_stress_excess_ann",
    "baseline_QQQ_rolling_pass_rate",
    "baseline_QQQ_worst_rolling_excess_ann",
    "baseline_regime_QQQ_down_days_excess_ann",
    "full_excess_ann",
    "stress_excess_ann",
    "rolling_pass_rate",
    "fail_count",
]


def _normalize_status(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    first = text.split(None, 1)[0].strip().upper()
    return first if first in CHECK_STATUS else ""


def _maybe_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _candidate_from_config(config: str, fallback: str) -> str:
    text = str(config or "").strip()
    stem = Path(text).stem if text else Path(fallback).stem
    if stem.endswith(".runtime"):
        stem = stem[: -len(".runtime")]
    if stem.startswith(CONFIG_PREFIX):
        stem = stem[len(CONFIG_PREFIX) :]
    if stem.endswith(CONFIG_SUFFIX):
        stem = stem[: -len(CONFIG_SUFFIX)]
    return stem


def _extract_metric_value(detail: str) -> Optional[float]:
    text = str(detail or "").strip()
    if not text:
        return None
    leading = LEADING_COMPARISON_RE.search(text)
    if leading:
        value = float(leading.group("value"))
        if leading.group("abs"):
            value = abs(value)
        return value
    match = NUMERIC_RE.search(text)
    return float(match.group(1)) if match else None


def _parse_table_check(line: str) -> Optional[Tuple[str, str, str]]:
    if "|" not in line:
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 3:
        return None
    if parts[1].upper() not in CHECK_STATUS:
        return None
    name = parts[0]
    if not name or name.lower() in {"check", "gate", "---"}:
        return None
    return name, parts[1].upper(), " | ".join(parts[2:]).strip()


def _parse_inline_metrics(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for match in INLINE_KV_RE.finditer(str(value or "")):
        parsed = _maybe_float(match.group("value").strip().strip(","))
        if parsed is not None:
            out[match.group("key")] = parsed
    return out


def _set_check(row: Dict[str, object], name: str, status: str, detail: str) -> None:
    row[f"{name}_status"] = status
    row[f"{name}_detail"] = detail
    value = _extract_metric_value(detail)
    if value is not None:
        row[name] = value


def parse_log(path: Path) -> Dict[str, object]:
    path = path.expanduser().resolve()
    row: Dict[str, object] = {"source_log": str(path)}
    failed_checks: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        kv_match = KV_RE.match(line)
        if kv_match:
            key = kv_match.group("key").strip()
            value = kv_match.group("value").strip()
            if key in {"config", "gate_profile", "benchmark_config", "benchmark_source", "label_horizon_days"}:
                parsed = _maybe_float(value)
                row[key] = parsed if key == "label_horizon_days" and parsed is not None else value
            elif key == "release_ready":
                status = _normalize_status(value) or value
                row[key] = status
                if status == "FAIL":
                    failed_checks.append("release_ready")
            elif key == "release_missing_or_failed":
                row[key] = value
            elif key in TRACKED_KV:
                status = _normalize_status(value)
                row[key] = status or (_maybe_float(value) if _maybe_float(value) is not None else value)
                if status == "FAIL":
                    failed_checks.append(key)
            elif key.startswith(("active_risk_", "rebalance_interval_quality_", "model_quality_", "strategy_weighted_")):
                for subkey, parsed in _parse_inline_metrics(value).items():
                    row[f"{key}_{subkey}"] = parsed
            elif key.endswith(STATUS_KV_SUFFIXES):
                status = _normalize_status(value)
                if status:
                    row[key] = status
                    if status == "FAIL":
                        failed_checks.append(key)
            continue

        parsed_check = _parse_table_check(line)
        if parsed_check is None:
            continue
        name, status, detail = parsed_check
        _set_check(row, name, status, detail)
        if status == "FAIL":
            failed_checks.append(name)

    row["candidate"] = _candidate_from_config(str(row.get("config", "")), path.name)
    row["fail_checks"] = ",".join(dict.fromkeys(failed_checks))
    row["fail_count"] = len([name for name in dict.fromkeys(failed_checks) if name])
    return row


def _collect_logs(paths: Sequence[str], patterns: Sequence[str]) -> List[Path]:
    collected: Dict[str, Path] = {}
    for raw in paths or []:
        if not str(raw).strip():
            continue
        path = Path(raw).expanduser().resolve()
        if path.exists() and path.is_file():
            collected[str(path)] = path
    for pattern in patterns or []:
        if not str(pattern).strip():
            continue
        for match in glob.glob(str(Path(pattern).expanduser()), recursive=True):
            path = Path(match).expanduser().resolve()
            if path.exists() and path.is_file():
                collected[str(path)] = path
    return [collected[key] for key in sorted(collected)]


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_COLUMNS})


def _markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_table(rows, MARKDOWN_COLUMNS), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize release-validator logs into a candidate failure map.")
    parser.add_argument("--log", action="append", default=[], help="Release-validator log file. Repeatable.")
    parser.add_argument("--log_glob", action="append", default=[], help="Glob matching release-validator logs. Repeatable.")
    parser.add_argument("--out_csv", default="", help="Optional CSV output path.")
    parser.add_argument("--out_md", default="", help="Optional Markdown output path.")
    parser.add_argument("--fail_on_release_ready", action="store_true", help="Exit non-zero if any log is release-ready.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logs = _collect_logs(args.log, args.log_glob)
    if not logs:
        print("no logs matched")
        return 2

    rows = [parse_log(path) for path in logs]
    if args.out_csv:
        _write_csv(Path(args.out_csv), rows)
        print(f"csv={Path(args.out_csv).expanduser().resolve()}")
    if args.out_md:
        _write_markdown(Path(args.out_md), rows)
        print(f"markdown={Path(args.out_md).expanduser().resolve()}")
    if not args.out_csv and not args.out_md:
        sys.stdout.write(_markdown_table(rows, MARKDOWN_COLUMNS))

    if args.fail_on_release_ready and any(str(row.get("release_ready", "")).upper() == "PASS" for row in rows):
        return 17
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
