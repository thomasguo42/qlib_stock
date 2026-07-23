#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    _benchmark_forward_return,
    _infer_label_ref_start_days,
    _load_yaml,
    _normalize_datetime_instrument_index,
    _parse_label_horizon,
    _safe_get,
)


def _flatten_label_config(label_cfg) -> Optional[str]:
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        return str(label_cfg[0][0])
    return None


def _handler_class_name(cfg: dict) -> str:
    return str(_safe_get(cfg, ["task", "dataset", "kwargs", "handler", "class"], ""))


def _data_handler_config(cfg: dict) -> dict:
    return cfg.get("data_handler_config") or _safe_get(cfg, ["task", "dataset", "kwargs", "handler", "kwargs"], {}) or {}


def _feature_candidates(cfg: dict, scope: str) -> List[Tuple[str, str, str]]:
    """
    Return [(source, expression, display_name)] for configured feature groups.
    """
    dh = _data_handler_config(cfg)
    handler_class = _handler_class_name(cfg)
    scope = str(scope).lower()
    out: List[Tuple[str, str, str]] = []

    if scope in {"all", "alpha"} and "Alpha158" in handler_class:
        from qlib.contrib.data.loader import Alpha158DL

        fields, names = Alpha158DL.get_feature_config()
        out.extend(("alpha158", str(field), str(name)) for field, name in zip(fields, names))

    if scope in {"all", "pit"}:
        interval = str(dh.get("pit_interval", "q")).lower()
        for raw in dh.get("pit_fields", []) or []:
            field = str(raw).strip().lower()
            if field:
                out.append(("pit", f"P($${field}_{interval})", f"{field.upper()}_{interval.upper()}"))

    if scope in {"all", "extra"}:
        fields = [str(x).strip() for x in (dh.get("extra_fields", []) or []) if str(x).strip()]
        names = [str(x).strip() for x in (dh.get("extra_names", []) or []) if str(x).strip()]
        if len(names) != len(fields):
            names = [f"EXTRA_{i}" for i in range(len(fields))]
        out.extend(("extra", field, name) for field, name in zip(fields, names))

    return out


def _filter_features(
    features: List[Tuple[str, str, str]],
    *,
    pattern: Optional[str],
    max_features: Optional[int],
) -> List[Tuple[str, str, str]]:
    if pattern:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        features = [item for item in features if rx.search(item[1]) or rx.search(item[2]) or rx.search(item[0])]
    if max_features is not None and int(max_features) > 0:
        features = features[: int(max_features)]
    return features


def _sample_month_dates(calendar: Iterable[pd.Timestamp], max_months: int, sample_from: str) -> List[pd.Timestamp]:
    cal = pd.DatetimeIndex(calendar)
    if len(cal) == 0:
        return []
    monthly = pd.Series(cal).groupby(cal.to_period("M")).last().sort_values()
    if max_months is not None and int(max_months) > 0 and len(monthly) > int(max_months):
        if sample_from == "head":
            monthly = monthly.iloc[: int(max_months)]
        elif sample_from == "even":
            idx = np.linspace(0, len(monthly) - 1, int(max_months)).round().astype(int)
            monthly = monthly.iloc[np.unique(idx)]
        else:
            monthly = monthly.iloc[-int(max_months) :]
    return [pd.Timestamp(x) for x in monthly.tolist()]


def _active_instruments(
    inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    date: pd.Timestamp,
) -> List[str]:
    date = pd.Timestamp(date)
    return sorted(
        inst
        for inst, spans in inst_spans.items()
        if any(pd.Timestamp(start) <= date <= pd.Timestamp(end) for start, end in spans)
    )


def _sample_instruments(inst: List[str], n: int, seed: int) -> List[str]:
    if not inst:
        return []
    if n <= 0 or n >= len(inst):
        return sorted(inst)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inst), size=int(n), replace=False)
    return [inst[i] for i in sorted(idx)]


def _normalize_feature_frame(df: pd.DataFrame, display_names: List[str]) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["|".join(str(x) for x in col) for col in out.columns]
    if len(out.columns) == len(display_names):
        out.columns = display_names
    out = out.replace([np.inf, -np.inf], np.nan)
    return _normalize_datetime_instrument_index(out)


def _daily_ic(feature: pd.Series, label: pd.Series, min_daily_count: int) -> pd.Series:
    frame = pd.concat([feature.rename("feature"), label.rename("label")], axis=1, join="inner").dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    vals = {}
    for dt, day in frame.groupby(level="datetime", sort=True):
        if len(day) < int(min_daily_count):
            continue
        if day["feature"].nunique(dropna=True) < 2 or day["label"].nunique(dropna=True) < 2:
            continue
        corr = day["feature"].corr(day["label"], method="spearman")
        if isinstance(corr, (int, float)) and math.isfinite(float(corr)):
            vals[pd.Timestamp(dt)] = float(corr)
    return pd.Series(vals, dtype=float).sort_index()


def _summarize_feature(
    name: str,
    source: str,
    expression: str,
    feature: pd.Series,
    label: pd.Series,
    *,
    min_daily_count: int,
) -> Dict[str, object]:
    aligned = pd.concat([feature.rename("feature"), label.rename("label")], axis=1, join="inner")
    total = int(len(aligned))
    feature_non_na = int(aligned["feature"].notna().sum()) if total else 0
    usable = int(aligned.dropna().shape[0]) if total else 0
    ic = _daily_ic(aligned["feature"], aligned["label"], min_daily_count=min_daily_count)
    years = {}
    if not ic.empty:
        for year, vals in ic.groupby(ic.index.year):
            years[str(int(year))] = float(vals.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    return {
        "source": source,
        "feature": name,
        "expression": expression,
        "rows": total,
        "feature_non_na": feature_non_na,
        "usable_rows": usable,
        "feature_coverage": float(feature_non_na / total) if total else float("nan"),
        "usable_coverage": float(usable / total) if total else float("nan"),
        "ic_days": int(len(ic)),
        "mean_ic": float(ic.mean()) if not ic.empty else float("nan"),
        "icir": float(ic.mean() / ic_std * np.sqrt(252)) if math.isfinite(ic_std) and ic_std > 0 else float("nan"),
        "pos_ic_rate": float((ic > 0).mean()) if not ic.empty else float("nan"),
        "worst_year_mean_ic": min(years.values()) if years else float("nan"),
        "best_year_mean_ic": max(years.values()) if years else float("nan"),
        "year_mean_ic": years,
    }


def _load_benchmark_series(path: Optional[str]) -> Optional[pd.Series]:
    if not path:
        return None
    bench = pd.read_pickle(Path(path).expanduser().resolve())
    if isinstance(bench, pd.DataFrame):
        if bench.shape[1] != 1:
            raise ValueError(f"benchmark_pkl must be a Series or single-column DataFrame: {path}")
        bench = bench.iloc[:, 0]
    if not isinstance(bench, pd.Series):
        raise ValueError(f"benchmark_pkl must be a pandas Series: {path}")
    return bench


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate configured US Sharadar feature coverage and rank-IC.")
    p.add_argument("--config", required=True, help="Workflow config YAML")
    p.add_argument("--provider_uri", default=None, help="Override qlib provider URI")
    p.add_argument("--benchmark_pkl", default=None, help="Override benchmark return Series pickle")
    p.add_argument("--start", default=None, help="Evaluation start date")
    p.add_argument("--end", default=None, help="Evaluation end date")
    p.add_argument("--feature_scope", choices=["all", "alpha", "pit", "extra"], default="extra")
    p.add_argument("--feature_regex", default=None, help="Optional regex over source/name/expression")
    p.add_argument("--max_features", type=int, default=None, help="Optional max feature count after filtering")
    p.add_argument("--sample_months", type=int, default=24, help="Number of month-end dates to sample")
    p.add_argument("--sample_from", choices=["tail", "head", "even"], default="tail", help="Which months to sample")
    p.add_argument("--sample_instruments", type=int, default=300, help="Instrument count sampled from active tail universe")
    p.add_argument("--min_daily_count", type=int, default=30, help="Minimum instruments required for daily rank-IC")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", default=None, help="Optional diagnostics CSV path")
    p.add_argument("--out_json", default=None, help="Optional diagnostics JSON path")
    p.add_argument("--print_top", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2
    cfg = _load_yaml(cfg_path)
    dh = _data_handler_config(cfg)
    qlib_init = dict(cfg.get("qlib_init", {}) or {})
    if args.provider_uri:
        qlib_init["provider_uri"] = args.provider_uri

    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.data.data import Cal

    qlib_init.setdefault("region", REG_US)
    qlib.init(**qlib_init)

    label_expr = _flatten_label_config(dh.get("label", []))
    label_horizon = _parse_label_horizon(label_expr) if label_expr else None
    if label_expr is None or label_horizon is None:
        print(f"Unable to parse label expression/horizon: {label_expr!r}")
        return 2

    features = _filter_features(
        _feature_candidates(cfg, args.feature_scope),
        pattern=args.feature_regex,
        max_features=args.max_features,
    )
    if not features:
        print("No features matched requested scope/filter")
        return 2

    market = cfg.get("market") or dh.get("instruments")
    filter_pipe = dh.get("filter_pipe", [])
    start = pd.Timestamp(args.start or dh.get("start_time"))
    end = pd.Timestamp(args.end or dh.get("end_time"))
    cal = list(Cal.calendar(start_time=start, end_time=end, freq="day", future=False))
    sample_dates = _sample_month_dates(cal, args.sample_months, args.sample_from)
    if not sample_dates:
        print(f"No sample dates in {start.date()}->{end.date()}")
        return 2

    inst_conf = D.instruments(market, filter_pipe=filter_pipe)
    spans = D.list_instruments(inst_conf, start_time=start, end_time=end, as_list=False)
    active_tail = _active_instruments(spans, sample_dates[-1])
    instruments = _sample_instruments(active_tail, int(args.sample_instruments), int(args.seed))
    if not instruments:
        print(f"No active instruments on sample tail {sample_dates[-1].date()}")
        return 2

    fields = [expr for _, expr, _ in features]
    names = [name for _, _, name in features]
    sample_start = sample_dates[0]
    sample_end = sample_dates[-1]
    feature_df = D.features(instruments, fields, start_time=sample_start, end_time=sample_end)
    feature_df = _normalize_feature_frame(feature_df, names)

    raw_label = D.features(instruments, [label_expr], start_time=sample_start, end_time=sample_end)
    label = raw_label.iloc[:, 0] if isinstance(raw_label, pd.DataFrame) else raw_label
    label = _normalize_datetime_instrument_index(label.astype(float))

    benchmark_path = args.benchmark_pkl
    if benchmark_path is None:
        for proc in dh.get("learn_processors", []) or []:
            if isinstance(proc, dict) and str(proc.get("class", "")).split(".")[-1] in {
                "BenchmarkExcessLabel",
                "ResidualForwardReturnLabel",
                "VolScaledExcessLabel",
                "DownsideAdjustedExcessLabel",
                "PortfolioUtilityExcessLabel",
                "DualHorizonPortfolioUtilityLabel",
            }:
                benchmark_path = (proc.get("kwargs", {}) or {}).get("benchmark_pkl")
                break
    bench = _load_benchmark_series(benchmark_path)
    if bench is not None:
        bench_fwd = _benchmark_forward_return(
            bench,
            label_horizon_days=int(label_horizon),
            label_ref_start_days=_infer_label_ref_start_days(dh),
        )
        label_dates = pd.DatetimeIndex(label.index.get_level_values("datetime"))
        label = label - bench_fwd.reindex(label_dates).to_numpy()

    sample_date_set = set(pd.DatetimeIndex(sample_dates).normalize())
    f_dates = pd.DatetimeIndex(feature_df.index.get_level_values("datetime")).normalize()
    l_dates = pd.DatetimeIndex(label.index.get_level_values("datetime")).normalize()
    feature_df = feature_df.loc[f_dates.isin(sample_date_set)]
    label = label.loc[l_dates.isin(sample_date_set)].replace([np.inf, -np.inf], np.nan)

    rows = []
    for source, expr, name in features:
        if name not in feature_df.columns:
            continue
        rows.append(
            _summarize_feature(
                name,
                source,
                expr,
                feature_df[name],
                label,
                min_daily_count=int(args.min_daily_count),
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        print("No feature diagnostics were produced")
        return 2
    out = out.sort_values(["mean_ic", "usable_coverage"], ascending=[False, False])

    print(
        f"features={len(out)} instruments={len(instruments)} sample_dates={len(sample_dates)} "
        f"span={sample_start.date()}->{sample_end.date()} label_horizon={label_horizon}"
    )
    printable = out.drop(columns=["year_mean_ic"], errors="ignore").head(int(args.print_top))
    with pd.option_context("display.max_colwidth", 80, "display.width", 180):
        print(printable.to_string(index=False))

    if args.out_csv:
        csv_path = Path(args.out_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out.drop(columns=["year_mean_ic"], errors="ignore").to_csv(csv_path, index=False)
        print(f"saved_csv={csv_path}")
    if args.out_json:
        json_path = Path(args.out_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "config": str(cfg_path),
            "provider_uri": qlib_init.get("provider_uri"),
            "market": market,
            "feature_scope": args.feature_scope,
            "sample_dates": [str(pd.Timestamp(x).date()) for x in sample_dates],
            "instruments": len(instruments),
            "label_expr": label_expr,
            "label_horizon_days": int(label_horizon),
            "rows": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
