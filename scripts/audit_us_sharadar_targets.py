#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Audit Sharadar target definitions before training release candidates.

This script is intentionally model-free.  It compares target definitions and
simple score fields across horizons so weak or inverted targets are rejected
before a walk-forward training run consumes time.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_us_sharadar_feature_ic import (  # noqa: E402
    _active_instruments,
    _normalize_feature_frame,
    _sample_instruments,
)
from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    _benchmark_forward_return,
    _read_pickle_compat,
    _load_yaml,
    _normalize_datetime_instrument_index,
    _safe_get,
)


DEFAULT_SCORE_SPECS = [
    "RISK_RET_20D=$risk_ret_20d:1.0",
    "RISK_RET_63D=$risk_ret_63d:1.0",
    "RISK_RELRET_SPY_63D=$risk_relret_spy_63d:1.0",
    "BOOK_PX_Q=$book_px_q:1.0",
    "FCF_YIELD_Q=$fcf_yield_q:1.0",
    "EARN_YIELD_Q=$earn_yield_q:1.0",
    "RISK_VOL_20D=$risk_vol_20d:-1.0",
    "RISK_BETA_SPY_63D=$risk_beta_spy_63d:-1.0",
]

DEFAULT_TARGET_KINDS = [
    "raw_return",
    "benchmark_excess",
    "sector_neutral",
    "beta_residual",
    "vol_scaled_excess",
    "downside_adjusted_excess",
    "top_bottom_class",
]

DEFAULT_BUCKETS = [(1, 20), (21, 40), (41, 80), (81, 120), (121, 180), (181, None)]


def _parse_csv_ints(value: str) -> List[int]:
    out: List[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_score_specs(specs: Sequence[str]) -> List[Tuple[str, str, float]]:
    parsed: List[Tuple[str, str, float]] = []
    for raw in specs:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" in text:
            name, expr = text.split("=", 1)
            name = name.strip()
        else:
            expr = text
            name = ""
        weight = 1.0
        expr = expr.strip()
        if ":" in expr:
            maybe_expr, maybe_weight = expr.rsplit(":", 1)
            try:
                weight = float(maybe_weight)
                expr = maybe_expr.strip()
            except ValueError:
                pass
        if not name:
            name = expr.strip().lstrip("$").upper()
        if expr:
            parsed.append((name, expr, float(weight)))
    return parsed


def _read_sector_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    meta = pd.read_csv(path, usecols=lambda c: c in {"ticker", "sector"}, low_memory=False)
    if "ticker" not in meta.columns or "sector" not in meta.columns:
        return {}
    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
    meta["sector"] = meta["sector"].fillna("UNKNOWN").astype(str)
    return meta.drop_duplicates("ticker", keep="last").set_index("ticker")["sector"].to_dict()


def _load_benchmark(path: Path) -> pd.Series:
    obj = _read_pickle_compat(path.expanduser().resolve())
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"benchmark_pkl must be a Series or single-column DataFrame: {path}")
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise ValueError(f"benchmark_pkl must be a pandas Series: {path}")
    out = obj.astype(float).copy()
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def _series_to_price_frame(close: pd.Series) -> pd.DataFrame:
    close = _normalize_datetime_instrument_index(close.astype(float)).sort_index()
    if not isinstance(close.index, pd.MultiIndex):
        raise ValueError("close must have a datetime/instrument MultiIndex")
    return close.unstack("instrument").sort_index()


def _stack_frame(frame: pd.DataFrame, name: str) -> pd.Series:
    out = frame.replace([np.inf, -np.inf], np.nan).stack()
    out.index = out.index.set_names(["datetime", "instrument"])
    out.name = name
    return out.sort_index()


def _forward_return_from_close(
    close: pd.Series,
    *,
    horizon_days: int,
    label_ref_start_days: int = 1,
) -> pd.Series:
    prices = _series_to_price_frame(close)
    start = max(0, int(label_ref_start_days))
    horizon = max(1, int(horizon_days))
    entry = prices.shift(-start)
    exit_ = prices.shift(-(start + horizon))
    return _stack_frame(exit_ / entry - 1.0, f"raw_return_{horizon}d")


def _rolling_beta(
    close: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    prices = _series_to_price_frame(close)
    stock_ret = prices.pct_change()
    bench = benchmark_returns.reindex(stock_ret.index).astype(float)
    stock_mean = stock_ret.rolling(window=int(window), min_periods=int(min_periods)).mean()
    bench_mean = bench.rolling(window=int(window), min_periods=int(min_periods)).mean()
    prod_mean = stock_ret.mul(bench, axis=0).rolling(window=int(window), min_periods=int(min_periods)).mean()
    cov = prod_mean - stock_mean.mul(bench_mean, axis=0)
    var = bench.rolling(window=int(window), min_periods=int(min_periods)).var(ddof=0)
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    return _stack_frame(beta, "beta")


def _rolling_vol(
    close: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    prices = _series_to_price_frame(close)
    vol = prices.pct_change().rolling(window=int(window), min_periods=int(min_periods)).std(ddof=0)
    return _stack_frame(vol, "vol")


def _subtract_by_date(label: pd.Series, values_by_date: pd.Series) -> pd.Series:
    dates = pd.DatetimeIndex(label.index.get_level_values("datetime"))
    out = label.astype(float) - values_by_date.reindex(dates).to_numpy()
    out.name = label.name
    return out


def _neutralize_by_sector(label: pd.Series, sector_map: Dict[str, str]) -> pd.Series:
    if not sector_map:
        return label.copy()
    frame = label.rename("label").to_frame()
    instruments = frame.index.get_level_values("instrument").astype(str).str.upper()
    frame["sector"] = [sector_map.get(inst, "UNKNOWN") for inst in instruments]
    demeaned = frame["label"] - frame.groupby(["datetime", "sector"])["label"].transform("mean")
    demeaned.name = label.name
    return demeaned.sort_index()


def _top_bottom_classification(label: pd.Series, *, top_pct: float = 0.20) -> pd.Series:
    top_pct = min(0.49, max(0.01, float(top_pct)))
    frame = label.rename("label").to_frame().dropna()

    def classify(day: pd.DataFrame) -> pd.Series:
        pct = day["label"].rank(pct=True, method="first")
        out = pd.Series(0.0, index=day.index, dtype=float)
        out.loc[pct >= 1.0 - top_pct] = 1.0
        out.loc[pct <= top_pct] = -1.0
        return out

    result = pd.concat([classify(day) for _, day in frame.groupby(level="datetime", sort=True)])
    result.name = label.name
    return result.sort_index()


def _make_targets(
    close: pd.Series,
    benchmark_returns: pd.Series,
    *,
    horizon_days: int,
    label_ref_start_days: int,
    target_kinds: Iterable[str],
    sector_map: Dict[str, str],
    beta_window: int,
    beta_min_periods: int,
    vol_window: int,
    vol_min_periods: int,
) -> Dict[str, pd.Series]:
    raw = _forward_return_from_close(
        close,
        horizon_days=int(horizon_days),
        label_ref_start_days=int(label_ref_start_days),
    ).replace([np.inf, -np.inf], np.nan)
    bench_fwd = _benchmark_forward_return(
        benchmark_returns,
        label_horizon_days=int(horizon_days),
        label_ref_start_days=int(label_ref_start_days),
    )
    if bench_fwd is None:
        raise ValueError("benchmark forward return could not be computed")
    excess = _subtract_by_date(raw, bench_fwd)
    excess.name = f"benchmark_excess_{horizon_days}d"

    targets: Dict[str, pd.Series] = {}
    requested = set(target_kinds)
    if "raw_return" in requested:
        targets["raw_return"] = raw.rename(f"raw_return_{horizon_days}d")
    if "benchmark_excess" in requested:
        targets["benchmark_excess"] = excess
    if "sector_neutral" in requested:
        targets["sector_neutral"] = _neutralize_by_sector(excess, sector_map).rename(f"sector_neutral_{horizon_days}d")
    if "beta_residual" in requested:
        beta = _rolling_beta(
            close,
            benchmark_returns,
            window=int(beta_window),
            min_periods=int(beta_min_periods),
        )
        dates = pd.DatetimeIndex(raw.index.get_level_values("datetime"))
        beta_component = beta.reindex(raw.index).astype(float) * bench_fwd.reindex(dates).to_numpy()
        targets["beta_residual"] = (raw - beta_component).rename(f"beta_residual_{horizon_days}d")
    if "vol_scaled_excess" in requested:
        vol = _rolling_vol(close, window=int(vol_window), min_periods=int(vol_min_periods)).reindex(excess.index)
        denom = vol * math.sqrt(max(1, int(horizon_days)))
        targets["vol_scaled_excess"] = (excess / denom.replace(0.0, np.nan)).rename(
            f"vol_scaled_excess_{horizon_days}d"
        )
    if "downside_adjusted_excess" in requested:
        downside = (-excess).clip(lower=0.0)
        targets["downside_adjusted_excess"] = (excess - downside).rename(
            f"downside_adjusted_excess_{horizon_days}d"
        )
    if "top_bottom_class" in requested:
        targets["top_bottom_class"] = _top_bottom_classification(excess).rename(f"top_bottom_class_{horizon_days}d")
    return {name: series.replace([np.inf, -np.inf], np.nan).dropna() for name, series in targets.items()}


def _cs_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((numeric - float(numeric.mean())) / std).fillna(0.0).astype(float)


def _composite_score(score_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    if score_df.empty:
        return pd.Series(dtype=float)

    def score_day(day: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=day.index, dtype=float)
        used = 0
        for col, weight in weights.items():
            if col not in day.columns:
                continue
            out = out + float(weight) * _cs_zscore(day[col])
            used += 1
        if used == 0:
            return pd.Series(np.nan, index=day.index, dtype=float)
        return out

    result = pd.concat([score_day(day) for _, day in score_df.groupby(level="datetime", sort=True)])
    result.name = "COMPOSITE"
    return result.sort_index()


def _daily_rank_ic(score: pd.Series, label: pd.Series, *, min_daily_count: int) -> pd.Series:
    frame = pd.concat([score.rename("score"), label.rename("label")], axis=1, join="inner").dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    vals = {}
    for dt, day in frame.groupby(level="datetime", sort=True):
        if len(day) < int(min_daily_count):
            continue
        if day["score"].nunique(dropna=True) < 2 or day["label"].nunique(dropna=True) < 2:
            continue
        ic = day["score"].corr(day["label"], method="spearman")
        if isinstance(ic, (int, float)) and math.isfinite(float(ic)):
            vals[pd.Timestamp(dt)] = float(ic)
    return pd.Series(vals, dtype=float).sort_index()


def _metric_periods(dates: pd.DatetimeIndex, recent_days: int) -> Dict[str, set]:
    unique_dates = pd.DatetimeIndex(sorted(dates.normalize().unique()))
    periods: Dict[str, set] = {"full": set(unique_dates)}
    if int(recent_days) > 0 and len(unique_dates) > 0:
        periods[f"recent_{int(recent_days)}d"] = set(unique_dates[-int(recent_days) :])
    for year in sorted(unique_dates.year.unique()):
        periods[str(int(year))] = set(unique_dates[unique_dates.year == int(year)])
    return periods


def _target_summary(
    target: pd.Series,
    *,
    target_kind: str,
    horizon_days: int,
    recent_days: int,
) -> List[Dict[str, object]]:
    if target.empty:
        return []
    dates = pd.DatetimeIndex(target.index.get_level_values("datetime")).normalize()
    rows = []
    for period, keep_dates in _metric_periods(dates, recent_days).items():
        mask = dates.isin(keep_dates)
        data = target.loc[mask].dropna()
        if data.empty:
            continue
        daily = data.groupby(level="datetime")
        rows.append(
            {
                "target_kind": target_kind,
                "horizon_days": int(horizon_days),
                "period": period,
                "days": int(daily.size().shape[0]),
                "rows": int(len(data)),
                "mean_label": float(data.mean()),
                "median_label": float(data.median()),
                "std_label": float(data.std(ddof=1)) if len(data) > 1 else float("nan"),
                "positive_rate": float((data > 0).mean()),
                "mean_daily_xsec_std": float(daily.std(ddof=0).mean()),
                "mean_daily_count": float(daily.size().mean()),
            }
        )
    return rows


def _ic_summary(
    score: pd.Series,
    target: pd.Series,
    *,
    score_name: str,
    target_kind: str,
    horizon_days: int,
    recent_days: int,
    min_daily_count: int,
) -> Dict[str, object]:
    ic = _daily_rank_ic(score, target, min_daily_count=int(min_daily_count))
    row: Dict[str, object] = {
        "target_kind": target_kind,
        "horizon_days": int(horizon_days),
        "score": score_name,
        "ic_days": int(len(ic)),
    }
    if ic.empty:
        row.update(
            {
                "mean_ic": float("nan"),
                "recent_mean_ic": float("nan"),
                "pos_ic_rate": float("nan"),
                "recent_pos_ic_rate": float("nan"),
                "positive_years": 0,
                "worst_year_mean_ic": float("nan"),
            }
        )
        return row
    recent = ic.tail(int(recent_days)) if int(recent_days) > 0 else ic
    yearly = ic.groupby(ic.index.year).mean()
    row.update(
        {
            "mean_ic": float(ic.mean()),
            "recent_mean_ic": float(recent.mean()) if not recent.empty else float("nan"),
            "pos_ic_rate": float((ic > 0).mean()),
            "recent_pos_ic_rate": float((recent > 0).mean()) if not recent.empty else float("nan"),
            "positive_years": int((yearly > 0).sum()),
            "worst_year_mean_ic": float(yearly.min()) if not yearly.empty else float("nan"),
            "best_year_mean_ic": float(yearly.max()) if not yearly.empty else float("nan"),
        }
    )
    return row


def _bucket_label(start: int, end: Optional[int]) -> str:
    return f"{start:03d}_plus" if end is None else f"{start:03d}_{end:03d}"


def _rank_bucket_daily_means(
    score: pd.Series,
    target: pd.Series,
    *,
    buckets: Sequence[Tuple[int, Optional[int]]],
    min_daily_count: int,
) -> pd.DataFrame:
    frame = pd.concat([score.rename("score"), target.rename("label")], axis=1, join="inner").dropna()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for dt, day in frame.groupby(level="datetime", sort=True):
        if len(day) < int(min_daily_count):
            continue
        ranked = day.sort_values("score", ascending=False)
        for start, end in buckets:
            if end is None:
                bucket = ranked.iloc[int(start) - 1 :]
            else:
                bucket = ranked.iloc[int(start) - 1 : int(end)]
            if bucket.empty:
                continue
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "bucket": _bucket_label(start, end),
                    "rows": int(len(bucket)),
                    "mean_label": float(bucket["label"].mean()),
                    "positive": float(bucket["label"].mean() > 0),
                }
            )
    return pd.DataFrame(rows)


def _rank_bucket_summary(
    score: pd.Series,
    target: pd.Series,
    *,
    score_name: str,
    target_kind: str,
    horizon_days: int,
    recent_days: int,
    buckets: Sequence[Tuple[int, Optional[int]]] = DEFAULT_BUCKETS,
    min_daily_count: int = 30,
) -> List[Dict[str, object]]:
    daily = _rank_bucket_daily_means(
        score,
        target,
        buckets=buckets,
        min_daily_count=int(min_daily_count),
    )
    if daily.empty:
        return []
    dates = pd.DatetimeIndex(daily["datetime"]).normalize()
    rows: List[Dict[str, object]] = []
    for period, keep_dates in _metric_periods(dates, recent_days).items():
        part = daily.loc[dates.isin(keep_dates)]
        if part.empty:
            continue
        for bucket, group in part.groupby("bucket", sort=True):
            rows.append(
                {
                    "target_kind": target_kind,
                    "horizon_days": int(horizon_days),
                    "score": score_name,
                    "period": period,
                    "bucket": bucket,
                    "days": int(group["datetime"].nunique()),
                    "rows": int(group["rows"].sum()),
                    "mean_label": float(group["mean_label"].mean()),
                    "median_daily_label": float(group["mean_label"].median()),
                    "positive_day_rate": float(group["positive"].mean()),
                }
            )
    return rows


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    return value


def _metric_value(row: Optional[pd.Series], key: str, default=float("nan")) -> float:
    if row is None or key not in row:
        return default
    try:
        val = float(row[key])
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def _screen_target_candidates(
    ic_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    recent_days: int,
    top_bucket: str = "001_020",
    min_mean_ic: float = 0.0,
    min_recent_mean_ic: float = 0.0,
    min_recent_pos_ic_rate: float = 0.50,
    min_positive_years: int = 3,
    min_worst_year_mean_ic: float = -0.02,
    min_recent_top_bucket_label: float = 0.0,
    min_recent_top_bucket_positive_day_rate: float = 0.50,
    min_full_top_bucket_label: float = 0.0,
    min_recent_target_positive_rate: float = 0.45,
) -> pd.DataFrame:
    if ic_df.empty:
        return pd.DataFrame()
    bucket_df = bucket_df.copy() if isinstance(bucket_df, pd.DataFrame) else pd.DataFrame()
    target_df = target_df.copy() if isinstance(target_df, pd.DataFrame) else pd.DataFrame()
    recent_period = f"recent_{int(recent_days)}d"
    rows = []
    for _, ic in ic_df.iterrows():
        key_mask = (
            (bucket_df.get("target_kind") == ic.get("target_kind"))
            & (bucket_df.get("horizon_days") == ic.get("horizon_days"))
            & (bucket_df.get("score") == ic.get("score"))
            & (bucket_df.get("bucket") == top_bucket)
        ) if not bucket_df.empty else pd.Series(dtype=bool)
        recent_bucket = None
        full_bucket = None
        if not bucket_df.empty and key_mask.any():
            matches = bucket_df.loc[key_mask]
            rb = matches.loc[matches["period"] == recent_period]
            fb = matches.loc[matches["period"] == "full"]
            if not rb.empty:
                recent_bucket = rb.iloc[0]
            if not fb.empty:
                full_bucket = fb.iloc[0]

        target_mask = (
            (target_df.get("target_kind") == ic.get("target_kind"))
            & (target_df.get("horizon_days") == ic.get("horizon_days"))
            & (target_df.get("period") == recent_period)
        ) if not target_df.empty else pd.Series(dtype=bool)
        recent_target = target_df.loc[target_mask].iloc[0] if not target_df.empty and target_mask.any() else None

        metrics = {
            "mean_ic": _metric_value(ic, "mean_ic"),
            "recent_mean_ic": _metric_value(ic, "recent_mean_ic"),
            "recent_pos_ic_rate": _metric_value(ic, "recent_pos_ic_rate"),
            "positive_years": int(_metric_value(ic, "positive_years", 0.0)),
            "worst_year_mean_ic": _metric_value(ic, "worst_year_mean_ic"),
            "recent_top_bucket_label": _metric_value(recent_bucket, "mean_label"),
            "recent_top_bucket_positive_day_rate": _metric_value(recent_bucket, "positive_day_rate"),
            "full_top_bucket_label": _metric_value(full_bucket, "mean_label"),
            "recent_target_positive_rate": _metric_value(recent_target, "positive_rate"),
        }
        checks = {
            "mean_ic": metrics["mean_ic"] >= float(min_mean_ic),
            "recent_mean_ic": metrics["recent_mean_ic"] >= float(min_recent_mean_ic),
            "recent_pos_ic_rate": metrics["recent_pos_ic_rate"] >= float(min_recent_pos_ic_rate),
            "positive_years": metrics["positive_years"] >= int(min_positive_years),
            "worst_year_mean_ic": metrics["worst_year_mean_ic"] >= float(min_worst_year_mean_ic),
            "recent_top_bucket_label": metrics["recent_top_bucket_label"] >= float(min_recent_top_bucket_label),
            "recent_top_bucket_positive_day_rate": metrics["recent_top_bucket_positive_day_rate"]
            >= float(min_recent_top_bucket_positive_day_rate),
            "full_top_bucket_label": metrics["full_top_bucket_label"] >= float(min_full_top_bucket_label),
            "recent_target_positive_rate": metrics["recent_target_positive_rate"]
            >= float(min_recent_target_positive_rate),
        }
        failures = [name for name, ok in checks.items() if not bool(ok)]
        rows.append(
            {
                "target_kind": ic.get("target_kind"),
                "horizon_days": ic.get("horizon_days"),
                "score": ic.get("score"),
                **metrics,
                "screen_pass": not failures,
                "screen_failures": ";".join(failures),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["screen_pass", "recent_top_bucket_label", "recent_mean_ic", "mean_ic"],
        ascending=[False, False, False, False],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit US Sharadar target/horizon stability before model training.")
    p.add_argument("--config", required=True, help="Workflow config used for market/filter/provider defaults")
    p.add_argument("--provider_uri", default=None, help="Override Qlib provider URI")
    p.add_argument("--benchmark_pkl", default=None, help="Benchmark daily return Series pickle")
    p.add_argument("--market", default=None, help="Override instrument market")
    p.add_argument("--start", default=None, help="Audit start date")
    p.add_argument("--end", default=None, help="Audit end date")
    p.add_argument("--horizons", default="5,10,20,60", help="Comma-separated forward horizons in trading days")
    p.add_argument(
        "--target_kinds",
        default=",".join(DEFAULT_TARGET_KINDS),
        help=f"Comma-separated target kinds: {','.join(DEFAULT_TARGET_KINDS)}",
    )
    p.add_argument(
        "--score_field",
        action="append",
        default=None,
        help="Score spec NAME=EXPR[:WEIGHT]. Repeatable. Defaults to a conservative Sharadar factor set.",
    )
    p.add_argument("--sample_instruments", type=int, default=600, help="Instrument sample from active tail universe; <=0 uses all")
    p.add_argument("--min_daily_count", type=int, default=30)
    p.add_argument("--recent_days", type=int, default=63)
    p.add_argument("--label_ref_start_days", type=int, default=1)
    p.add_argument("--beta_window", type=int, default=63)
    p.add_argument("--beta_min_periods", type=int, default=40)
    p.add_argument("--vol_window", type=int, default=20)
    p.add_argument("--vol_min_periods", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sector_map_csv", default="~/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--out_dir", default="artifacts/research/target_audit")
    p.add_argument("--tag", default=None, help="Output filename tag; defaults to UTC timestamp")
    p.add_argument("--print_top", type=int, default=25)
    p.add_argument("--screen_targets", action="store_true", help="Write target pre-screen results")
    p.add_argument("--fail_on_screen_fail", action="store_true", help="Exit non-zero when no target candidate passes the screen")
    p.add_argument("--screen_top_bucket", default="001_020")
    p.add_argument("--screen_min_mean_ic", type=float, default=0.0)
    p.add_argument("--screen_min_recent_mean_ic", type=float, default=0.0)
    p.add_argument("--screen_min_recent_pos_ic_rate", type=float, default=0.50)
    p.add_argument("--screen_min_positive_years", type=int, default=3)
    p.add_argument("--screen_min_worst_year_mean_ic", type=float, default=-0.02)
    p.add_argument("--screen_min_recent_top_bucket_label", type=float, default=0.0)
    p.add_argument("--screen_min_recent_top_bucket_positive_day_rate", type=float, default=0.50)
    p.add_argument("--screen_min_full_top_bucket_label", type=float, default=0.0)
    p.add_argument("--screen_min_recent_target_positive_rate", type=float, default=0.45)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}")
        return 2
    cfg = _load_yaml(cfg_path)
    dh = cfg.get("data_handler_config") or _safe_get(cfg, ["task", "dataset", "kwargs", "handler", "kwargs"], {}) or {}
    qlib_init = dict(cfg.get("qlib_init", {}) or {})
    if args.provider_uri:
        qlib_init["provider_uri"] = args.provider_uri

    benchmark_pkl = args.benchmark_pkl
    if benchmark_pkl is None:
        for proc in dh.get("learn_processors", []) or []:
            if isinstance(proc, dict) and str(proc.get("class", "")).split(".")[-1] in {
                "BenchmarkExcessLabel",
                "ResidualForwardReturnLabel",
                "VolScaledExcessLabel",
                "DownsideAdjustedExcessLabel",
                "PortfolioUtilityExcessLabel",
                "DualHorizonPortfolioUtilityLabel",
            }:
                benchmark_pkl = (proc.get("kwargs", {}) or {}).get("benchmark_pkl")
                break
    if not benchmark_pkl:
        print("benchmark_pkl is required")
        return 2

    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.data.data import Cal

    qlib_init.setdefault("region", REG_US)
    qlib.init(**qlib_init)

    market = args.market or cfg.get("market") or dh.get("instruments")
    if not market:
        print("market could not be inferred; pass --market")
        return 2
    start = pd.Timestamp(args.start or dh.get("start_time"))
    end = pd.Timestamp(args.end or dh.get("end_time"))
    horizons = _parse_csv_ints(args.horizons)
    target_kinds = _parse_csv_strings(args.target_kinds)
    unknown_targets = sorted(set(target_kinds) - set(DEFAULT_TARGET_KINDS))
    if unknown_targets:
        print(f"unknown target kinds: {unknown_targets}")
        return 2
    if not horizons:
        print("at least one horizon is required")
        return 2

    cal = list(Cal.calendar(start_time=start, end_time=end, freq="day", future=False))
    if not cal:
        print(f"no calendar rows in {start.date()}->{end.date()}")
        return 2
    filter_pipe = dh.get("filter_pipe", [])
    inst_conf = D.instruments(market, filter_pipe=filter_pipe)
    spans = D.list_instruments(inst_conf, start_time=start, end_time=end, as_list=False)
    active_tail = _active_instruments(spans, pd.Timestamp(cal[-1]))
    instruments = _sample_instruments(active_tail, int(args.sample_instruments), int(args.seed))
    if not instruments:
        print(f"no active instruments on {pd.Timestamp(cal[-1]).date()}")
        return 2

    score_specs = _parse_score_specs(args.score_field or DEFAULT_SCORE_SPECS)
    if not score_specs:
        print("no score fields configured")
        return 2
    score_names = [name for name, _, _ in score_specs]
    score_exprs = [expr for _, expr, _ in score_specs]
    score_weights = {name: weight for name, _, weight in score_specs}

    lookback_days = max(int(args.beta_window), int(args.vol_window), 252) * 3
    feature_start = start - pd.Timedelta(days=lookback_days)
    close_raw = D.features(instruments, ["$close"], start_time=feature_start, end_time=end)
    close = close_raw.iloc[:, 0] if isinstance(close_raw, pd.DataFrame) else close_raw
    close = _normalize_datetime_instrument_index(close.astype(float)).sort_index()

    score_raw = D.features(instruments, score_exprs, start_time=start, end_time=end)
    score_df = _normalize_feature_frame(score_raw, score_names)
    composite = _composite_score(score_df, score_weights)
    scores: Dict[str, pd.Series] = {name: score_df[name] for name in score_names if name in score_df.columns}
    if not composite.empty:
        scores["COMPOSITE"] = composite

    benchmark = _load_benchmark(Path(benchmark_pkl))
    sector_map = _read_sector_map(Path(args.sector_map_csv).expanduser() if args.sector_map_csv else None)

    target_rows: List[Dict[str, object]] = []
    ic_rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []
    for horizon in horizons:
        targets = _make_targets(
            close,
            benchmark,
            horizon_days=int(horizon),
            label_ref_start_days=int(args.label_ref_start_days),
            target_kinds=target_kinds,
            sector_map=sector_map,
            beta_window=int(args.beta_window),
            beta_min_periods=int(args.beta_min_periods),
            vol_window=int(args.vol_window),
            vol_min_periods=int(args.vol_min_periods),
        )
        for target_kind, target in targets.items():
            dates = pd.DatetimeIndex(target.index.get_level_values("datetime")).normalize()
            in_window = (dates >= start.normalize()) & (dates <= end.normalize())
            target = target.loc[in_window].dropna()
            target_rows.extend(
                _target_summary(
                    target,
                    target_kind=target_kind,
                    horizon_days=int(horizon),
                    recent_days=int(args.recent_days),
                )
            )
            for score_name, score in scores.items():
                ic_rows.append(
                    _ic_summary(
                        score,
                        target,
                        score_name=score_name,
                        target_kind=target_kind,
                        horizon_days=int(horizon),
                        recent_days=int(args.recent_days),
                        min_daily_count=int(args.min_daily_count),
                    )
                )
                bucket_rows.extend(
                    _rank_bucket_summary(
                        score,
                        target,
                        score_name=score_name,
                        target_kind=target_kind,
                        horizon_days=int(horizon),
                        recent_days=int(args.recent_days),
                        min_daily_count=int(args.min_daily_count),
                    )
                )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_csv = out_dir / f"target_summary_{tag}.csv"
    ic_csv = out_dir / f"feature_ic_{tag}.csv"
    bucket_csv = out_dir / f"rank_buckets_{tag}.csv"
    screen_csv = out_dir / f"target_screen_{tag}.csv"
    json_path = out_dir / f"manifest_{tag}.json"

    target_df = pd.DataFrame(target_rows)
    ic_df = pd.DataFrame(ic_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    screen_df = pd.DataFrame()
    target_df.to_csv(target_csv, index=False)
    ic_df.sort_values(["recent_mean_ic", "mean_ic"], ascending=[False, False]).to_csv(ic_csv, index=False)
    bucket_df.to_csv(bucket_csv, index=False)
    if args.screen_targets or args.fail_on_screen_fail:
        screen_df = _screen_target_candidates(
            ic_df,
            bucket_df,
            target_df,
            recent_days=int(args.recent_days),
            top_bucket=str(args.screen_top_bucket),
            min_mean_ic=float(args.screen_min_mean_ic),
            min_recent_mean_ic=float(args.screen_min_recent_mean_ic),
            min_recent_pos_ic_rate=float(args.screen_min_recent_pos_ic_rate),
            min_positive_years=int(args.screen_min_positive_years),
            min_worst_year_mean_ic=float(args.screen_min_worst_year_mean_ic),
            min_recent_top_bucket_label=float(args.screen_min_recent_top_bucket_label),
            min_recent_top_bucket_positive_day_rate=float(args.screen_min_recent_top_bucket_positive_day_rate),
            min_full_top_bucket_label=float(args.screen_min_full_top_bucket_label),
            min_recent_target_positive_rate=float(args.screen_min_recent_target_positive_rate),
        )
        screen_df.to_csv(screen_csv, index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": str(cfg_path),
        "provider_uri": qlib_init.get("provider_uri"),
        "market": market,
        "benchmark_pkl": str(Path(benchmark_pkl).expanduser().resolve()),
        "start": str(start.date()),
        "end": str(end.date()),
        "horizons": horizons,
        "target_kinds": target_kinds,
        "score_specs": [{"name": n, "expression": e, "weight": w} for n, e, w in score_specs],
        "sample_instruments": len(instruments),
        "target_summary_csv": str(target_csv),
        "feature_ic_csv": str(ic_csv),
        "rank_bucket_csv": str(bucket_csv),
        "target_screen_csv": str(screen_csv) if not screen_df.empty else "",
        "target_screen_pass": int(screen_df["screen_pass"].sum()) if not screen_df.empty else 0,
        "target_screen_total": int(len(screen_df)) if not screen_df.empty else 0,
    }
    json_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

    print(
        f"target_audit instruments={len(instruments)} horizons={horizons} targets={len(target_kinds)} "
        f"scores={len(scores)}"
    )
    print(f"saved_target_summary={target_csv}")
    print(f"saved_feature_ic={ic_csv}")
    print(f"saved_rank_buckets={bucket_csv}")
    if args.screen_targets or args.fail_on_screen_fail:
        print(f"saved_target_screen={screen_csv}")
        print(f"target_screen_pass={int(screen_df['screen_pass'].sum()) if not screen_df.empty else 0}/{len(screen_df)}")
    print(f"saved_manifest={json_path}")

    if not ic_df.empty and int(args.print_top) > 0:
        top = ic_df.sort_values(["recent_mean_ic", "mean_ic"], ascending=[False, False]).head(int(args.print_top))
        with pd.option_context("display.width", 180, "display.max_colwidth", 64):
            print(top.to_string(index=False))
    if args.fail_on_screen_fail and (screen_df.empty or int(screen_df["screen_pass"].sum()) <= 0):
        return 17
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
