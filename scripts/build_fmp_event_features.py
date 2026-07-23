#!/usr/bin/env python
"""Build conservative daily features from FMP event-like raw data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


RAW_ROOT = "/root/.qlib/fmp/raw"
PREPARED_ROOT = "/root/.qlib/fmp/prepared"
PROVENANCE_FILE = "fmp_event_features.json"
DEFAULT_WINDOWS = "10,20,63,126"
DEFAULT_PREFIX = "fmp"
STALE_EVENT_DAYS = 9999.0

PIT_SAFE_DATASETS = ["earnings", "grades", "grades_historical", "price_target_news"]
CURRENT_ONLY_OR_UNPROVEN_DATASETS = [
    "analyst_estimates_annual",
    "analyst_estimates_quarter",
    "price_target_summary",
    "price_target_consensus",
]
RESERVED_SYMBOLS = {"TICKER", "SYMBOL"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def parse_windows(value: str) -> List[int]:
    windows = [int(x) for x in parse_csv_list(value)]
    windows = [w for w in windows if w > 0]
    if not windows:
        raise ValueError("windows must contain at least one positive integer")
    return windows


def canonical_symbol(value: str) -> str:
    return str(value or "").strip().upper()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_payload(payload) -> List[Dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "historical", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        if payload:
            return [payload]
    return []


def load_dataset(raw_root: Path, dataset: str) -> pd.DataFrame:
    rows: List[Dict] = []
    in_dir = raw_root / dataset
    if not in_dir.exists():
        return pd.DataFrame()
    for fp in sorted(in_dir.glob("*.json")):
        payload = _load_json(fp)
        symbol_from_file = canonical_symbol(fp.stem)
        if symbol_from_file in RESERVED_SYMBOLS:
            continue
        for rec in records_from_payload(payload):
            row = dict(rec)
            # Prefer the qlib symbol encoded in the raw filename. FMP may return
            # class shares as BRK-B/BF-B while the qlib universe uses BRK.B/BF.B.
            row["symbol"] = symbol_from_file or canonical_symbol(row.get("symbol"))
            if row["symbol"]:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def raw_symbols(raw_root: Path, datasets: Sequence[str]) -> List[str]:
    symbols = set()
    for dataset in datasets:
        in_dir = raw_root / dataset
        if not in_dir.exists():
            continue
        for fp in in_dir.glob("*.json"):
            symbol = canonical_symbol(fp.stem)
            if symbol and symbol not in RESERVED_SYMBOLS:
                symbols.add(symbol)
    return sorted(symbols)


def grade_score(value) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return np.nan
    if "strong sell" in text:
        return -2.0
    if "strong buy" in text:
        return 2.0
    negative = ("sell", "underperform", "underweight", "reduce", "negative")
    positive = ("buy", "outperform", "overweight", "positive", "accumulate")
    neutral = ("hold", "neutral", "market perform", "sector perform", "equal weight", "in line", "inline")
    if any(x in text for x in negative):
        return -1.0
    if any(x in text for x in positive):
        return 1.0
    if any(x in text for x in neutral):
        return 0.0
    return np.nan


def action_score(value) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    if "upgrade" in text:
        return 1.0
    if "downgrade" in text:
        return -1.0
    return 0.0


def _availability_date(series: pd.Series, lag_days: int) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    parsed = parsed.dt.tz_convert(None).dt.normalize()
    if int(lag_days) != 0:
        parsed = parsed + pd.Timedelta(days=int(lag_days))
    return parsed


def _clip_numeric(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=lower, upper=upper)


def _zero(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype=float)


def _feature(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return _zero(frame.index)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _days_since_event(index: pd.DatetimeIndex, has_event: pd.Series) -> pd.Series:
    event_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    mask = has_event.reindex(index).fillna(False).astype(bool)
    if bool(mask.any()):
        event_dates.loc[mask] = index[mask.to_numpy()]
    latest = event_dates.ffill()
    today = pd.Series(index, index=index, dtype="datetime64[ns]")
    days = (today - latest).dt.days.astype(float)
    return days.where(latest.notna(), STALE_EVENT_DAYS).clip(lower=0.0, upper=STALE_EVENT_DAYS)


def _freshness(days_since: pd.Series, window: int) -> pd.Series:
    days = pd.to_numeric(days_since, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = np.exp(-days.fillna(STALE_EVENT_DAYS).astype(float) / float(max(1, int(window))))
    return pd.Series(out, index=days_since.index, dtype=float).where(days < STALE_EVENT_DAYS, 0.0)


def _directional_alpha_columns(prefix: str, windows: Sequence[int]) -> List[str]:
    cols = [
        f"{prefix}_alpha_earn_surprise_latest",
        f"{prefix}_alpha_rating_bullish",
        f"{prefix}_alpha_rating_bearish_penalty",
        f"{prefix}_alpha_rating_net_change",
        f"{prefix}_alpha_grade_score_latest",
        f"{prefix}_alpha_pt_upside_latest",
        f"{prefix}_alpha_event_freshness",
        f"{prefix}_alpha_event_coverage",
        f"{prefix}_alpha_event_composite",
    ]
    for w in windows:
        cols.extend(
            [
                f"{prefix}_alpha_earn_surprise_{int(w)}d",
                f"{prefix}_alpha_grade_revision_{int(w)}d",
                f"{prefix}_alpha_pt_upside_{int(w)}d",
            ]
        )
    return cols


def add_directional_alpha_features(frame: pd.DataFrame, *, prefix: str, windows: Sequence[int]) -> pd.DataFrame:
    """Add PIT-safe directional FMP event composites.

    Raw FMP event counts are sparse and several have the wrong economic sign as
    standalone predictors.  These derived columns preserve the raw inputs while
    exposing decayed, directionally interpretable signals for model selection.
    """
    out = frame.copy()
    idx = out.index
    window_values = sorted({int(w) for w in windows if int(w) > 0})
    long_window = 63 if 63 in window_values else (max(window_values) if window_values else 1)

    earn_fresh_long = _feature(out, f"{prefix}_earn_count_{long_window}d_freshness")
    grade_fresh_long = _feature(out, f"{prefix}_grade_count_{long_window}d_freshness")
    pt_fresh_long = _feature(out, f"{prefix}_pt_count_{long_window}d_freshness")
    rating_fresh_long = _feature(out, f"{prefix}_rating_{long_window}d_freshness")

    eps_latest = _feature(out, f"{prefix}_eps_surprise_pct_latest")
    rev_latest = _feature(out, f"{prefix}_rev_surprise_pct_latest")
    out[f"{prefix}_alpha_earn_surprise_latest"] = ((0.70 * eps_latest + 0.30 * rev_latest) * earn_fresh_long).astype(
        "float32"
    )

    for w in windows:
        w = int(w)
        eps = _feature(out, f"{prefix}_eps_surprise_pct_{w}d_event_mean")
        rev = _feature(out, f"{prefix}_rev_surprise_pct_{w}d_event_mean")
        earn_fresh = _feature(out, f"{prefix}_earn_count_{w}d_freshness")
        out[f"{prefix}_alpha_earn_surprise_{w}d"] = ((0.70 * eps + 0.30 * rev) * earn_fresh).astype("float32")

        grade_delta = _feature(out, f"{prefix}_grade_delta_{w}d_event_mean")
        grade_up = _feature(out, f"{prefix}_grade_up_{w}d_sum")
        grade_down = _feature(out, f"{prefix}_grade_down_{w}d_sum")
        grade_count = _feature(out, f"{prefix}_grade_count_{w}d_sum").replace(0.0, np.nan)
        grade_balance = ((grade_up - grade_down) / grade_count).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        grade_fresh = _feature(out, f"{prefix}_grade_count_{w}d_freshness")
        out[f"{prefix}_alpha_grade_revision_{w}d"] = ((0.60 * grade_delta + 0.40 * grade_balance) * grade_fresh).astype(
            "float32"
        )

        pt = _feature(out, f"{prefix}_pt_upside_{w}d_event_mean")
        pt_fresh = _feature(out, f"{prefix}_pt_count_{w}d_freshness")
        out[f"{prefix}_alpha_pt_upside_{w}d"] = (pt * pt_fresh).astype("float32")

    bullish = _feature(out, f"{prefix}_rating_bullish_ratio_snapshot")
    bearish = _feature(out, f"{prefix}_rating_bearish_ratio_snapshot")
    net_chg_long = _feature(out, f"{prefix}_rating_net_bullish_{long_window}d_chg")
    out[f"{prefix}_alpha_rating_bullish"] = ((bullish - bearish) * rating_fresh_long).astype("float32")
    out[f"{prefix}_alpha_rating_bearish_penalty"] = (-bearish * rating_fresh_long).astype("float32")
    out[f"{prefix}_alpha_rating_net_change"] = (net_chg_long * rating_fresh_long).astype("float32")
    out[f"{prefix}_alpha_grade_score_latest"] = (
        _feature(out, f"{prefix}_grade_new_score_latest") * grade_fresh_long
    ).astype("float32")
    out[f"{prefix}_alpha_pt_upside_latest"] = (_feature(out, f"{prefix}_pt_upside_latest") * pt_fresh_long).astype(
        "float32"
    )
    out[f"{prefix}_alpha_event_freshness"] = pd.concat(
        [earn_fresh_long, grade_fresh_long, pt_fresh_long, rating_fresh_long], axis=1
    ).max(axis=1).reindex(idx).fillna(0.0).astype("float32")
    coverage = (
        np.log1p(_feature(out, f"{prefix}_earn_count_{long_window}d_sum"))
        + np.log1p(_feature(out, f"{prefix}_grade_count_{long_window}d_sum"))
        + np.log1p(_feature(out, f"{prefix}_pt_count_{long_window}d_sum"))
        + np.log1p(_feature(out, f"{prefix}_rating_total_snapshot"))
    )
    out[f"{prefix}_alpha_event_coverage"] = pd.Series(coverage, index=idx).clip(0.0, 10.0).astype("float32")
    components = [
        _feature(out, f"{prefix}_alpha_earn_surprise_latest").clip(-2.0, 2.0),
        _feature(out, f"{prefix}_alpha_earn_surprise_{long_window}d").clip(-2.0, 2.0),
        _feature(out, f"{prefix}_alpha_rating_bullish").clip(-2.0, 2.0),
        _feature(out, f"{prefix}_alpha_rating_bearish_penalty").clip(-2.0, 2.0),
        _feature(out, f"{prefix}_alpha_grade_revision_{long_window}d").clip(-2.0, 2.0),
        _feature(out, f"{prefix}_alpha_grade_score_latest").clip(-2.0, 2.0),
        0.25 * _feature(out, f"{prefix}_alpha_pt_upside_latest").clip(-2.0, 2.0),
    ]
    out[f"{prefix}_alpha_event_composite"] = pd.concat(components, axis=1).mean(axis=1).fillna(0.0).astype("float32")
    return out


def make_earnings_events(raw_root: Path, lag_days: int, prefix: str) -> pd.DataFrame:
    df = load_dataset(raw_root, "earnings")
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = _availability_date(df["date"], lag_days)
    for col in ("epsActual", "epsEstimated", "revenueActual", "revenueEstimated"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    has_actual = df[["epsActual", "revenueActual"]].notna().any(axis=1)
    df = df[has_actual].dropna(subset=["symbol", "date"])
    if df.empty:
        return pd.DataFrame()
    eps_base = df["epsEstimated"].abs().replace(0, np.nan)
    rev_base = df["revenueEstimated"].abs().replace(0, np.nan)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(canonical_symbol),
            "date": df["date"],
            f"{prefix}_earn_count": 1.0,
            f"{prefix}_eps_surprise_pct": _clip_numeric((df["epsActual"] - df["epsEstimated"]) / eps_base, -5.0, 5.0),
            f"{prefix}_rev_surprise_pct": _clip_numeric((df["revenueActual"] - df["revenueEstimated"]) / rev_base, -5.0, 5.0),
        }
    )
    return out.dropna(subset=["symbol", "date"])


def make_grade_events(raw_root: Path, lag_days: int, prefix: str) -> pd.DataFrame:
    df = load_dataset(raw_root, "grades")
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = _availability_date(df["date"], lag_days)
    old_score = df["previousGrade"].map(grade_score) if "previousGrade" in df.columns else pd.Series(np.nan, index=df.index)
    new_score = df["newGrade"].map(grade_score) if "newGrade" in df.columns else pd.Series(np.nan, index=df.index)
    delta = new_score - old_score
    action = df["action"].map(action_score) if "action" in df.columns else pd.Series(0.0, index=df.index)
    signed_delta = delta.where(delta.notna() & (delta != 0), action)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(canonical_symbol),
            "date": df["date"],
            f"{prefix}_grade_count": 1.0,
            f"{prefix}_grade_up": (signed_delta > 0).astype("float32"),
            f"{prefix}_grade_down": (signed_delta < 0).astype("float32"),
            f"{prefix}_grade_delta": signed_delta.fillna(0.0).astype("float32"),
            f"{prefix}_grade_new_score": new_score.fillna(0.0).astype("float32"),
        }
    )
    return out.dropna(subset=["symbol", "date"])


def make_price_target_events(raw_root: Path, lag_days: int, prefix: str) -> pd.DataFrame:
    df = load_dataset(raw_root, "price_target_news")
    if df.empty or "publishedDate" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = _availability_date(df["publishedDate"], lag_days)
    for col in ("adjPriceTarget", "priceTarget", "priceWhenPosted"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    base = df["priceWhenPosted"].replace(0, np.nan)
    upside = (df["adjPriceTarget"].where(df["adjPriceTarget"].notna(), df["priceTarget"]) / base) - 1.0
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(canonical_symbol),
            "date": df["date"],
            f"{prefix}_pt_count": 1.0,
            f"{prefix}_pt_upside": _clip_numeric(upside, -2.0, 5.0),
            f"{prefix}_pt_target": _clip_numeric(df["priceTarget"], 0.0, 1_000_000.0),
            f"{prefix}_pt_adj_target": _clip_numeric(df["adjPriceTarget"], 0.0, 1_000_000.0),
        }
    )
    return out.dropna(subset=["symbol", "date"])


def make_grade_snapshot(raw_root: Path, lag_days: int, prefix: str) -> pd.DataFrame:
    df = load_dataset(raw_root, "grades_historical")
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = _availability_date(df["date"], lag_days)
    cols = [
        "analystRatingsStrongBuy",
        "analystRatingsBuy",
        "analystRatingsHold",
        "analystRatingsSell",
        "analystRatingsStrongSell",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    total = df[cols].sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(canonical_symbol),
            "date": df["date"],
            f"{prefix}_rating_total": total.fillna(0.0),
            f"{prefix}_rating_bullish_ratio": ((df["analystRatingsStrongBuy"] + df["analystRatingsBuy"]) / total).fillna(0.0),
            f"{prefix}_rating_bearish_ratio": ((df["analystRatingsSell"] + df["analystRatingsStrongSell"]) / total).fillna(0.0),
            f"{prefix}_rating_hold_ratio": (df["analystRatingsHold"] / total).fillna(0.0),
            f"{prefix}_rating_net_bullish": (
                (2 * df["analystRatingsStrongBuy"] + df["analystRatingsBuy"] - df["analystRatingsSell"] - 2 * df["analystRatingsStrongSell"])
                / total
            ).fillna(0.0),
        }
    )
    return out.dropna(subset=["symbol", "date"])


def _date_range(start: str, end: str, events: Sequence[pd.DataFrame]) -> pd.DatetimeIndex:
    if start:
        lo = pd.Timestamp(start)
    else:
        mins = [pd.to_datetime(df["date"], errors="coerce").dropna().min() for df in events if not df.empty and "date" in df.columns]
        lo = pd.Timestamp(min(mins)) if mins else pd.Timestamp.utcnow().normalize()
    if end:
        hi = pd.Timestamp(end)
    else:
        maxs = [pd.to_datetime(df["date"], errors="coerce").dropna().max() for df in events if not df.empty and "date" in df.columns]
        hi = pd.Timestamp(max(maxs)) if maxs else pd.Timestamp.utcnow().normalize()
    if hi < lo:
        raise ValueError(f"end {hi.date()} is earlier than start {lo.date()}")
    return pd.date_range(lo.normalize(), hi.normalize(), freq="D")


def _aggregate_events(df: pd.DataFrame, count_col: str, value_cols: List[str]) -> pd.DataFrame:
    agg = {count_col: "sum"}
    for col in value_cols:
        agg[col] = "sum"
    return df.groupby(["symbol", "date"], as_index=False).agg(agg)


def event_features_for_symbol(
    daily: pd.DataFrame,
    *,
    date_index: pd.DatetimeIndex,
    count_col: str,
    value_cols: List[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    frame = daily.set_index("date").sort_index()
    frame = frame.reindex(date_index)
    out = pd.DataFrame(index=date_index)
    count = pd.to_numeric(frame.get(count_col), errors="coerce").fillna(0.0)
    out[f"{count_col}_daily"] = count.astype("float32")
    days_since = _days_since_event(date_index, count > 0)
    out[f"{count_col}_days_since_latest"] = days_since.astype("float32")
    for w in windows:
        csum = count.rolling(int(w), min_periods=1).sum()
        out[f"{count_col}_{w}d_sum"] = csum.astype("float32")
        out[f"{count_col}_{w}d_freshness"] = _freshness(days_since, int(w)).astype("float32")
    for col in value_cols:
        values = pd.to_numeric(frame.get(col), errors="coerce").fillna(0.0)
        event_mean = (values / count.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        latest = event_mean.ffill().fillna(0.0)
        out[f"{col}_daily_sum"] = values.astype("float32")
        out[f"{col}_latest"] = latest.astype("float32")
        for w in windows:
            csum = count.rolling(int(w), min_periods=1).sum()
            vsum = values.rolling(int(w), min_periods=1).sum()
            out[f"{col}_{w}d_sum"] = vsum.astype("float32")
            out[f"{col}_{w}d_event_mean"] = (vsum / csum.replace(0, np.nan)).fillna(0.0).astype("float32")
    return out


def snapshot_features_for_symbol(
    daily: pd.DataFrame,
    *,
    date_index: pd.DatetimeIndex,
    value_cols: List[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    frame = daily.set_index("date").sort_index().reindex(date_index)
    out = pd.DataFrame(index=date_index)
    snapshot_event = frame.reindex(columns=value_cols).notna().any(axis=1) if value_cols else pd.Series(False, index=date_index)
    days_since = _days_since_event(date_index, snapshot_event)
    snapshot_prefix = "_".join(str(value_cols[0]).split("_")[:2]) if value_cols else "snapshot"
    out[f"{snapshot_prefix}_days_since_snapshot"] = days_since.astype("float32")
    for w in windows:
        out[f"{snapshot_prefix}_{int(w)}d_freshness"] = _freshness(days_since, int(w)).astype("float32")
    for col in value_cols:
        values = pd.to_numeric(frame.get(col), errors="coerce").ffill().fillna(0.0)
        out[f"{col}_snapshot"] = values.astype("float32")
        for w in windows:
            out[f"{col}_{w}d_chg"] = (values - values.shift(int(w))).fillna(0.0).astype("float32")
    return out


def expected_feature_columns(prefix: str, windows: Sequence[int]) -> List[str]:
    cols: List[str] = []

    def event_cols(count_col: str, value_cols: Sequence[str]) -> None:
        cols.append(f"{count_col}_daily")
        cols.append(f"{count_col}_days_since_latest")
        for w in windows:
            cols.append(f"{count_col}_{int(w)}d_sum")
            cols.append(f"{count_col}_{int(w)}d_freshness")
        for col in value_cols:
            cols.append(f"{col}_daily_sum")
            cols.append(f"{col}_latest")
            for w in windows:
                cols.append(f"{col}_{int(w)}d_sum")
                cols.append(f"{col}_{int(w)}d_event_mean")

    event_cols(f"{prefix}_earn_count", [f"{prefix}_eps_surprise_pct", f"{prefix}_rev_surprise_pct"])
    event_cols(
        f"{prefix}_grade_count",
        [f"{prefix}_grade_up", f"{prefix}_grade_down", f"{prefix}_grade_delta", f"{prefix}_grade_new_score"],
    )
    event_cols(f"{prefix}_pt_count", [f"{prefix}_pt_upside", f"{prefix}_pt_target", f"{prefix}_pt_adj_target"])
    for col in [
        f"{prefix}_rating_total",
        f"{prefix}_rating_bullish_ratio",
        f"{prefix}_rating_bearish_ratio",
        f"{prefix}_rating_hold_ratio",
        f"{prefix}_rating_net_bullish",
    ]:
        if col == f"{prefix}_rating_total":
            cols.append(f"{prefix}_rating_days_since_snapshot")
            for w in windows:
                cols.append(f"{prefix}_rating_{int(w)}d_freshness")
        cols.append(f"{col}_snapshot")
        for w in windows:
            cols.append(f"{col}_{int(w)}d_chg")
    cols.extend(_directional_alpha_columns(prefix, windows))
    return cols


def build_feature_frames(
    raw_root: Path,
    *,
    start: str,
    end: str,
    windows: Sequence[int],
    availability_lag_days: int,
    prefix: str,
) -> Dict[str, pd.DataFrame]:
    earnings = make_earnings_events(raw_root, availability_lag_days, prefix)
    grades = make_grade_events(raw_root, availability_lag_days, prefix)
    pt_news = make_price_target_events(raw_root, availability_lag_days, prefix)
    rating_snapshot = make_grade_snapshot(raw_root, availability_lag_days, prefix)

    date_index = _date_range(start, end, [earnings, grades, pt_news, rating_snapshot])
    symbols = sorted(
        set(earnings.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(grades.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(pt_news.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(rating_snapshot.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(raw_symbols(raw_root, ["earnings", "grades", "grades_historical", "price_target_news"]))
    )
    expected_cols = expected_feature_columns(prefix, windows)
    frames: Dict[str, pd.DataFrame] = {}
    sources = [
        (earnings, f"{prefix}_earn_count", [f"{prefix}_eps_surprise_pct", f"{prefix}_rev_surprise_pct"], "event"),
        (grades, f"{prefix}_grade_count", [f"{prefix}_grade_up", f"{prefix}_grade_down", f"{prefix}_grade_delta", f"{prefix}_grade_new_score"], "event"),
        (pt_news, f"{prefix}_pt_count", [f"{prefix}_pt_upside", f"{prefix}_pt_target", f"{prefix}_pt_adj_target"], "event"),
    ]
    snapshot_cols = [
        f"{prefix}_rating_total",
        f"{prefix}_rating_bullish_ratio",
        f"{prefix}_rating_bearish_ratio",
        f"{prefix}_rating_hold_ratio",
        f"{prefix}_rating_net_bullish",
    ]
    for symbol in symbols:
        parts = []
        for df, count_col, value_cols, _kind in sources:
            if df.empty:
                continue
            s = df[df["symbol"] == symbol]
            if s.empty:
                continue
            daily = _aggregate_events(s, count_col, value_cols)
            parts.append(event_features_for_symbol(daily, date_index=date_index, count_col=count_col, value_cols=value_cols, windows=windows))
        if not rating_snapshot.empty:
            snap = rating_snapshot[rating_snapshot["symbol"] == symbol]
            if not snap.empty:
                snap_daily = snap.groupby(["symbol", "date"], as_index=False)[snapshot_cols].last()
                parts.append(snapshot_features_for_symbol(snap_daily, date_index=date_index, value_cols=snapshot_cols, windows=windows))
        if not parts:
            out = pd.DataFrame(0.0, index=date_index, columns=expected_cols)
        else:
            out = pd.concat(parts, axis=1).fillna(0.0)
            out = out.loc[:, ~out.columns.duplicated()]
            out = out.reindex(columns=expected_cols, fill_value=0.0).fillna(0.0).copy()
        out = add_directional_alpha_features(out, prefix=prefix, windows=windows)
        out = out.reindex(columns=expected_cols, fill_value=0.0).fillna(0.0)
        out = out.reset_index().rename(columns={"index": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        frames[symbol] = out
    return frames


def _calendar(provider_uri: Path) -> List[pd.Timestamp]:
    cal_path = provider_uri / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar not found: {cal_path}")
    return [pd.Timestamp(x.strip()) for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _code_to_fname(symbol: str) -> str:
    try:
        from qlib.utils import code_to_fname

        return code_to_fname(symbol).lower()
    except Exception:
        return str(symbol).lower()


def _overwrite_feature_bins(prepared_dir: Path, provider_uri: Path, *, prefix: str, max_workers: int) -> int:
    _ = max_workers
    cal = pd.DatetimeIndex(_calendar(provider_uri))
    features_root = provider_uri / "features"
    written = 0
    for fp in sorted(prepared_dir.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            continue
        mask = (cal >= df["date"].min()) & (cal <= df["date"].max())
        out_index = cal[mask]
        if out_index.empty:
            continue
        start_idx = int(cal.get_loc(out_index[0]))
        df = df.set_index("date").reindex(out_index).fillna(0.0)
        symbol_dir = features_root / _code_to_fname(fp.stem)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for col in df.columns:
            if not str(col).startswith(f"{prefix}_"):
                continue
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32").to_numpy()
            out = np.hstack([start_idx, values]).astype("<f")
            out.tofile(symbol_dir / f"{col.lower()}.day.bin")
            written += 1
    return written


def write_provenance(provider_uri: Path, payload: Mapping) -> Path:
    meta = provider_uri / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    out = meta / PROVENANCE_FILE
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily features from PIT-safe FMP event endpoints.")
    p.add_argument("--raw_root", default=RAW_ROOT)
    p.add_argument("--out_dir", default="")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--windows", default=DEFAULT_WINDOWS)
    p.add_argument("--availability_lag_days", type=int, default=1)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--dump_to_qlib", action="store_true")
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"ERROR: raw_root not found: {raw_root}", file=sys.stderr)
        return 2
    windows = parse_windows(args.windows)
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path(PREPARED_ROOT).expanduser().resolve() / f"fmp_event_features_{stamp}"
    )
    frames = build_feature_frames(
        raw_root,
        start=str(args.start),
        end=str(end),
        windows=windows,
        availability_lag_days=int(args.availability_lag_days),
        prefix=str(args.prefix),
    )
    if not frames:
        print("ERROR: no feature frames were built", file=sys.stderr)
        return 3
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for symbol, frame in frames.items():
            frame.to_csv(out_dir / f"{symbol}.csv", index=False)
    feature_cols = sorted({c for frame in frames.values() for c in frame.columns if c != "date"})
    directional_cols = sorted(c for c in feature_cols if str(c).startswith(f"{args.prefix}_alpha_"))
    written_bins = 0
    provider = Path(args.provider_uri).expanduser().resolve()
    if args.dump_to_qlib:
        if args.dry_run:
            print(f"dry_run: would overwrite {args.prefix}_*.day.bin files from {out_dir}")
        else:
            written_bins = _overwrite_feature_bins(out_dir, provider, prefix=str(args.prefix), max_workers=int(args.max_workers))
            if written_bins <= 0:
                print("ERROR: no FMP feature bins were written", file=sys.stderr)
                return 4
    payload = {
        "created_utc": utc_now_iso(),
        "raw_root": str(raw_root),
        "prepared_dir": str(out_dir),
        "provider_uri": str(provider),
        "start": str(args.start),
        "end": str(end),
        "availability_lag_days": int(args.availability_lag_days),
        "windows": windows,
        "prefix": str(args.prefix),
        "tickers": int(len(frames)),
        "feature_columns": feature_cols,
        "feature_column_count": int(len(feature_cols)),
        "directional_feature_version": 2,
        "directional_feature_columns": directional_cols,
        "dump_to_qlib": bool(args.dump_to_qlib),
        "feature_bins_written": int(written_bins),
        "pit_safe_datasets": list(PIT_SAFE_DATASETS),
        "excluded_datasets": list(CURRENT_ONLY_OR_UNPROVEN_DATASETS),
        "dataset_classification": {
            "pit_safe_event_history": list(PIT_SAFE_DATASETS),
            "excluded_current_only_or_unproven_asof": list(CURRENT_ONLY_OR_UNPROVEN_DATASETS),
        },
        "exclusion_reason": "current-only or insufficient as-of history for historical backtests",
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        prov = write_provenance(provider, payload) if args.dump_to_qlib else out_dir / "fmp_event_features_manifest.json"
        if not args.dump_to_qlib:
            prov.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"prepared_dir={out_dir}")
        print(f"manifest={prov}")
    print(f"tickers={len(frames)} feature_columns={len(feature_cols)} rows={sum(len(f) for f in frames.values())} bins={written_bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
