#!/usr/bin/env python
"""Generate a human-review stock selector report from US Sharadar predictions.

This is decision support, not an order generator.  It ranks current candidates,
adds empirical forward-return bands from historical similar score buckets, and
surfaces risk/context fields so a human can decide what to investigate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    _benchmark_forward_return,
    _infer_label_ref_start_days,
    _load_yaml,
    _normalize_datetime_instrument_index,
    _read_pickle_compat,
)


DEFAULT_RISK_FIELDS = [
    "$marketcap_q",
    "$log_marketcap_q",
    "$risk_beta_qqq_63d",
    "$risk_beta_spy_63d",
    "$risk_vol_20d",
    "$risk_vol_63d",
    "$risk_ret_20d",
    "$risk_ret_63d",
    "$risk_ret_252d",
    "$risk_relret_qqq_63d",
    "$risk_relret_qqq_252d",
    "$fcf_yield_q",
    "$earn_yield_q",
    "$fmp_alpha_earn_surprise_latest",
    "$fmp_alpha_rating_bullish",
    "$fmp_alpha_rating_bearish_penalty",
    "$fmp_alpha_rating_net_change",
    "$fmp_alpha_grade_score_latest",
    "$fmp_alpha_pt_upside_latest",
    "$fmp_alpha_event_freshness",
    "$fmp_alpha_event_coverage",
    "$mkt_qqq_ret_20d_lag1",
    "$mkt_qqq_ret_63d_lag1",
    "$mkt_qqq_dd_126d_lag1",
    "$mkt_qqq_vol_20d_lag1",
    "$mkt_breadth_ret63_pos_lag1",
]


FIELD_RENAMES = {
    "$marketcap_q": "marketcap",
    "$log_marketcap_q": "log_marketcap",
    "$risk_beta_qqq_63d": "beta_qqq_63d",
    "$risk_beta_spy_63d": "beta_spy_63d",
    "$risk_vol_20d": "risk_vol_20d_feature",
    "$risk_vol_63d": "risk_vol_63d_feature",
    "$risk_ret_20d": "ret_20d",
    "$risk_ret_63d": "ret_63d",
    "$risk_ret_252d": "ret_252d",
    "$risk_relret_qqq_63d": "relret_qqq_63d",
    "$risk_relret_qqq_252d": "relret_qqq_252d",
    "$fcf_yield_q": "fcf_yield",
    "$earn_yield_q": "earn_yield",
    "$fmp_alpha_earn_surprise_latest": "fmp_earn_surprise",
    "$fmp_alpha_rating_bullish": "fmp_rating_bullish",
    "$fmp_alpha_rating_bearish_penalty": "fmp_rating_bearish_penalty",
    "$fmp_alpha_rating_net_change": "fmp_rating_net_change",
    "$fmp_alpha_grade_score_latest": "fmp_grade_score",
    "$fmp_alpha_pt_upside_latest": "fmp_pt_upside",
    "$fmp_alpha_event_freshness": "fmp_event_freshness",
    "$fmp_alpha_event_coverage": "fmp_event_coverage",
    "$mkt_qqq_ret_20d_lag1": "mkt_qqq_ret_20d",
    "$mkt_qqq_ret_63d_lag1": "mkt_qqq_ret_63d",
    "$mkt_qqq_dd_126d_lag1": "mkt_qqq_drawdown_126d",
    "$mkt_qqq_vol_20d_lag1": "mkt_qqq_vol_20d",
    "$mkt_breadth_ret63_pos_lag1": "market_breadth_63d",
}


BUCKETS = [
    ("top_1pct", 0.99, 1.00),
    ("top_2_5pct", 0.975, 0.99),
    ("top_5pct", 0.95, 0.975),
    ("top_10pct", 0.90, 0.95),
    ("upper_mid", 0.70, 0.90),
    ("middle", 0.30, 0.70),
    ("lower", 0.00, 0.30),
]


HUMAN_DECISION_COLUMNS = [
    "score_date",
    "symbol",
    "rank",
    "human_status",
    "decision_date",
    "paper_position_size",
    "reject_reason",
    "human_notes",
]


STRICT_ASSESSMENT_ORDER = [
    "Core Candidate",
    "Aggressive Upside",
    "Low Priority / Defensive",
    "Avoid / Watch Only",
    "Needs Review",
]

SELECTOR_ACTION_ORDER = [
    "Best Current Candidates",
    "Speculative / High Upside",
    "Watchlist Only",
    "Avoid",
    "Needs Review",
]

SELECTOR_SOURCE_ORDER = [
    "risk_adjusted_utility",
    "high_risk_upside_proxy",
    "diversified_rerank",
    "base_alpha",
    "excluded_candidate_pool",
]

# Backward-compatible name used by older report consumers.
SELECTOR_CATEGORY_ORDER = SELECTOR_ACTION_ORDER

SELECTOR_CLASSIFIER_VERSION = 3

TIER_ORDER = ["high", "medium", "low"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_int_list(value: str) -> List[int]:
    out = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        item = int(token)
        if item <= 0:
            raise ValueError("horizons must be positive")
        out.append(item)
    deduped = sorted(set(out))
    if not deduped:
        raise ValueError("at least one horizon is required")
    return deduped


def finite_or_none(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id_from_metadata(metadata: Mapping[str, Any]) -> str:
    keys = [
        "score_date",
        "config",
        "pred",
        "provider_uri",
        "benchmark_pkl",
        "selector_schema_version",
        "selector_classifier_version",
        "topn",
        "core_topn",
        "rank_export_n",
        "horizons",
        "primary_horizon",
    ]
    payload = json.dumps({key: metadata.get(key) for key in keys}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(dict(payload)), sort_keys=True) + "\n")


def write_human_decision_template(path: Path, report: pd.DataFrame, score_date: str) -> bool:
    path = Path(path).expanduser().resolve()
    if path.exists():
        return False
    rows = []
    for _, row in report.sort_values("rank").iterrows():
        rows.append(
            {
                "score_date": score_date,
                "symbol": str(row["symbol"]),
                "rank": int(row["rank"]),
                "human_status": "",
                "decision_date": "",
                "paper_position_size": "",
                "reject_reason": "",
                "human_notes": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=HUMAN_DECISION_COLUMNS).to_csv(path, index=False)
    return True


def load_prediction(path: Path) -> pd.Series:
    pred = _read_pickle_compat(path)
    if isinstance(pred, pd.DataFrame):
        if "score" in pred.columns:
            pred = pred["score"]
        else:
            pred = pred.iloc[:, 0]
    if not isinstance(pred, pd.Series):
        raise ValueError("prediction file must contain a pandas Series or DataFrame")
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError("prediction index must be MultiIndex(datetime, instrument)")
    pred = _normalize_datetime_instrument_index(pred)
    if list(pred.index.names[:2]) != ["datetime", "instrument"]:
        raise ValueError("prediction index must contain datetime and instrument levels")
    dates = pd.DatetimeIndex(pred.index.get_level_values("datetime")).normalize()
    instruments = pred.index.get_level_values("instrument").astype(str).str.upper()
    pred.index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    pred = pd.to_numeric(pred, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    pred.name = "score"
    return pred.sort_index()


def select_score_date(pred: pd.Series, as_of: Optional[str]) -> pd.Timestamp:
    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    if len(dates) == 0:
        raise ValueError("prediction file has no dates")
    if as_of is None or str(as_of).strip().lower() in {"", "latest"}:
        return pd.Timestamp(dates.max()).normalize()
    requested = pd.Timestamp(as_of).normalize()
    eligible = dates[dates <= requested]
    if len(eligible) == 0:
        raise ValueError(f"no prediction date <= requested as_of={requested.date()}")
    return pd.Timestamp(eligible.max()).normalize()


def score_bucket(score_percentile: float) -> str:
    pct = finite_or_none(score_percentile)
    if pct is None:
        return "unknown"
    for name, lo, hi in BUCKETS:
        if pct >= lo and pct <= hi:
            return name
    return "unknown"


def bucket_bounds(bucket: str) -> Tuple[float, float]:
    for name, lo, hi in BUCKETS:
        if name == bucket:
            return lo, hi
    return 0.0, 1.0


def rank_scores_for_date(pred: pd.Series, score_date: pd.Timestamp, topn: int) -> pd.DataFrame:
    try:
        scores = pred.xs(pd.Timestamp(score_date).normalize(), level="datetime")
    except KeyError as exc:
        raise ValueError(f"prediction has no scores for {pd.Timestamp(score_date).date()}") from exc
    scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        raise ValueError(f"prediction scores are empty for {pd.Timestamp(score_date).date()}")
    ranks = scores.rank(method="first", ascending=False)
    pct = scores.rank(method="average", pct=True)
    std = float(scores.std(ddof=0))
    z = (scores - float(scores.mean())) / std if std > 1e-12 and math.isfinite(std) else scores * 0.0
    top = scores.sort_values(ascending=False).head(int(topn))
    out = pd.DataFrame(
        {
            "symbol": top.index.astype(str),
            "score": top.astype(float).values,
            "rank": ranks.reindex(top.index).astype(int).values,
            "score_percentile": pct.reindex(top.index).astype(float).values,
            "score_z": z.reindex(top.index).astype(float).values,
        }
    )
    out["score_bucket"] = out["score_percentile"].map(score_bucket)
    return out


def handler_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(
        cfg.get("data_handler_config")
        or (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("handler", {}).get("kwargs", {})
        or {}
    )


def label_expr_for_horizon(horizon: int, ref_start_days: int) -> str:
    ref = max(0, int(ref_start_days))
    start_expr = "$close" if ref == 0 else f"Ref($close, -{ref})"
    return f"Ref($close, -{int(horizon) + ref})/{start_expr} - 1"


def normalize_feature_frame(features: Any, fields: Sequence[str]) -> pd.DataFrame:
    if isinstance(features, pd.Series):
        features = features.to_frame(fields[0] if len(fields) == 1 else "feature")
    if not isinstance(features, pd.DataFrame) or features.empty:
        return pd.DataFrame()
    out = features.copy()
    if isinstance(out.index, pd.MultiIndex) and list(out.index.names[:2]) == ["instrument", "datetime"]:
        out = out.reorder_levels(["datetime", "instrument"]).sort_index()
    if isinstance(out.index, pd.MultiIndex):
        dates = pd.DatetimeIndex(out.index.get_level_values("datetime")).normalize()
        inst = out.index.get_level_values("instrument").astype(str).str.upper()
        out.index = pd.MultiIndex.from_arrays([dates, inst], names=["datetime", "instrument"])
    return out.sort_index()


def load_benchmark(path: Path) -> pd.Series:
    bench = _read_pickle_compat(path)
    if isinstance(bench, pd.DataFrame):
        if bench.shape[1] != 1:
            raise ValueError("benchmark_pkl must be a Series or single-column DataFrame")
        bench = bench.iloc[:, 0]
    if not isinstance(bench, pd.Series):
        raise ValueError("benchmark_pkl must contain a pandas Series")
    bench.index = pd.DatetimeIndex(bench.index).normalize()
    return pd.to_numeric(bench, errors="coerce").sort_index()


def score_percentiles_by_date(pred: pd.Series) -> pd.Series:
    pct = pred.groupby(level="datetime", group_keys=False).rank(method="average", pct=True)
    pct.name = "score_percentile"
    return pct


def summarize_returns(values: pd.Series) -> Dict[str, Any]:
    data = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {
            "sample_count": 0,
            "hit_rate": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "worst": None,
            "best": None,
        }
    return {
        "sample_count": int(len(data)),
        "hit_rate": float((data > 0).mean()),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "p10": float(data.quantile(0.10)),
        "p25": float(data.quantile(0.25)),
        "p75": float(data.quantile(0.75)),
        "p90": float(data.quantile(0.90)),
        "worst": float(data.min()),
        "best": float(data.max()),
    }


def build_calibration_frame(
    pred: pd.Series,
    *,
    score_date: pd.Timestamp,
    cfg: Mapping[str, Any],
    provider_uri: Path,
    benchmark: pd.Series,
    horizons: Sequence[int],
    lookback_days: int,
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.data.data import Cal

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    score_date = pd.Timestamp(score_date).normalize()
    dates = pd.DatetimeIndex(pred.index.get_level_values("datetime"))
    min_date = pd.Timestamp(dates.min()).normalize()
    if int(lookback_days) > 0:
        cal = pd.DatetimeIndex(Cal.calendar(start_time=min_date, end_time=score_date, freq="day", future=False))
        if len(cal) > int(lookback_days):
            min_date = pd.Timestamp(cal[-int(lookback_days)]).normalize()
    max_h = max(int(h) for h in horizons)
    label_ref_start = _infer_label_ref_start_days(handler_config(cfg))
    benchmark_fwd = {
        int(h): _benchmark_forward_return(
            benchmark,
            label_horizon_days=int(h),
            label_ref_start_days=label_ref_start,
        )
        for h in horizons
    }
    usable_end = score_date
    bench_max = benchmark.index.max()
    if pd.notna(bench_max):
        eligible_bench_dates = benchmark.index[benchmark.index <= score_date]
        if len(eligible_bench_dates) > max_h + label_ref_start:
            usable_end = min(usable_end, pd.Timestamp(eligible_bench_dates[-(max_h + label_ref_start + 1)]))

    mask = (dates >= min_date) & (dates <= usable_end)
    hist_pred = pred.loc[mask]
    if hist_pred.empty:
        return pd.DataFrame()
    instruments = sorted(hist_pred.index.get_level_values("instrument").unique().astype(str).tolist())
    start = pd.Timestamp(hist_pred.index.get_level_values("datetime").min()).normalize()
    end = pd.Timestamp(hist_pred.index.get_level_values("datetime").max()).normalize()
    fields = [label_expr_for_horizon(int(h), label_ref_start) for h in horizons]
    raw = D.features(instruments, fields, start_time=start, end_time=end)
    labels = normalize_feature_frame(raw, fields).reindex(hist_pred.index)

    frame = pd.DataFrame({"score": hist_pred, "score_percentile": score_percentiles_by_date(hist_pred)})
    frame["score_bucket"] = frame["score_percentile"].map(score_bucket)
    dt_index = pd.DatetimeIndex(frame.index.get_level_values("datetime")).normalize()
    for h, expr in zip(horizons, fields):
        bench_h = benchmark_fwd[int(h)]
        if bench_h is None:
            continue
        frame[f"raw_return_{int(h)}d"] = pd.to_numeric(labels[expr], errors="coerce")
        frame[f"excess_return_{int(h)}d"] = frame[f"raw_return_{int(h)}d"] - bench_h.reindex(dt_index).to_numpy()
    return frame.replace([np.inf, -np.inf], np.nan)


def summarize_calibration(calibration: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows = []
    if calibration.empty:
        return pd.DataFrame()
    for bucket, lo, hi in BUCKETS:
        part = calibration.loc[calibration["score_bucket"] == bucket]
        for horizon in horizons:
            col = f"excess_return_{int(horizon)}d"
            if col not in part.columns:
                continue
            row = {
                "score_bucket": bucket,
                "percentile_min": lo,
                "percentile_max": hi,
                "horizon_days": int(horizon),
            }
            row.update(summarize_returns(part[col]))
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_for_candidate_bucket(
    calibration_summary: pd.DataFrame,
    bucket: str,
    horizon: int,
) -> Dict[str, Any]:
    if calibration_summary.empty:
        return summarize_returns(pd.Series(dtype=float))
    row = calibration_summary[
        (calibration_summary["score_bucket"] == bucket)
        & (calibration_summary["horizon_days"].astype(int) == int(horizon))
    ]
    if row.empty:
        return summarize_returns(pd.Series(dtype=float))
    return row.iloc[0].to_dict()


def local_calibration_metrics(
    calibration: pd.DataFrame,
    *,
    score_percentile: float,
    horizon: int,
    min_samples: int = 300,
    widths: Sequence[float] = (0.0025, 0.005, 0.01, 0.02, 0.05),
) -> Dict[str, Any]:
    """Estimate candidate-specific calibration from nearby historical score percentiles."""
    pct = finite_or_none(score_percentile)
    col = f"excess_return_{int(horizon)}d"
    if pct is None or calibration.empty or col not in calibration.columns or "score_percentile" not in calibration.columns:
        out = summarize_returns(pd.Series(dtype=float))
        out.update({"window_width": None, "method": "unavailable"})
        return out
    scores = pd.to_numeric(calibration["score_percentile"], errors="coerce")
    selected = pd.DataFrame()
    selected_width = None
    for width in widths:
        mask = (scores >= pct - float(width)) & (scores <= pct + float(width))
        part = calibration.loc[mask, col]
        if pd.to_numeric(part, errors="coerce").notna().sum() >= int(min_samples):
            selected = part.to_frame(name=col)
            selected_width = float(width)
            break
    if selected.empty:
        width = float(widths[-1]) if widths else 0.05
        mask = (scores >= pct - width) & (scores <= pct + width)
        selected = calibration.loc[mask, [col]]
        selected_width = width
    out = summarize_returns(selected[col] if col in selected.columns else pd.Series(dtype=float))
    out.update({"window_width": selected_width, "method": "score_percentile_window"})
    return out


def add_candidate_calibration(
    report: pd.DataFrame,
    calibration: pd.DataFrame,
    horizons: Sequence[int],
    *,
    min_samples: int = 300,
) -> pd.DataFrame:
    out = report.copy()
    metrics = ("sample_count", "hit_rate", "mean", "median", "p10", "p25", "p75", "p90", "worst", "best")
    for i, row in out.iterrows():
        pct = finite_or_none(row.get("score_percentile"))
        for horizon in horizons:
            h = int(horizon)
            local = local_calibration_metrics(
                calibration,
                score_percentile=pct,
                horizon=h,
                min_samples=min_samples,
            )
            local_n = int(local.get("sample_count") or 0)
            use_local = local_n >= int(min_samples)
            for key in metrics:
                out.loc[i, f"local_{h}d_{key}"] = local.get(key)
                fallback = row.get(f"hist_{h}d_{key}")
                out.loc[i, f"calib_{h}d_{key}"] = local.get(key) if use_local else fallback
            out.loc[i, f"local_{h}d_window_width"] = local.get("window_width")
            out.loc[i, f"calib_{h}d_method"] = local.get("method") if use_local else "score_bucket"
    return out


def feature_name_map(cfg: Mapping[str, Any]) -> Dict[str, str]:
    dh = handler_config(cfg)
    fields = [str(x) for x in dh.get("extra_fields", []) or []]
    names = [str(x).upper() for x in dh.get("extra_names", []) or []]
    out = {name: field for name, field in zip(names, fields)}
    for field in fields:
        out.setdefault(str(field).lstrip("$").upper(), str(field))
    return out


def current_feature_snapshot(
    *,
    instruments: Sequence[str],
    score_date: pd.Timestamp,
    provider_uri: Path,
    fields: Sequence[str],
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    raw = D.features(
        sorted(set(str(x).upper() for x in instruments)),
        list(fields),
        start_time=pd.Timestamp(score_date).normalize(),
        end_time=pd.Timestamp(score_date).normalize(),
    )
    feat = normalize_feature_frame(raw, fields)
    if feat.empty:
        return pd.DataFrame(index=pd.Index(instruments, name="symbol"))
    day = feat.xs(pd.Timestamp(score_date).normalize(), level="datetime", drop_level=True)
    day.index = day.index.astype(str).str.upper()
    return day.rename(columns={field: FIELD_RENAMES.get(field, str(field).lstrip("$").lower()) for field in day.columns})


def price_history_metrics(
    *,
    instruments: Sequence[str],
    score_date: pd.Timestamp,
    provider_uri: Path,
    lookback_days: int = 260,
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.utils import get_date_by_shift

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    end = pd.Timestamp(score_date).normalize()
    start = pd.Timestamp(get_date_by_shift(end, -int(lookback_days) + 1, future=True, clip_shift=True)).normalize()
    raw = D.features(
        sorted(set(str(x).upper() for x in instruments)),
        ["$close", "$volume"],
        start_time=start,
        end_time=end,
    )
    frame = normalize_feature_frame(raw, ["$close", "$volume"])
    if frame.empty:
        return pd.DataFrame(index=pd.Index(instruments, name="symbol"))
    close = frame["$close"].unstack("instrument").sort_index()
    volume = frame["$volume"].unstack("instrument").sort_index()
    ret = close.pct_change(fill_method=None)
    rows = []
    for symbol in sorted(set(str(x).upper() for x in instruments)):
        c = pd.to_numeric(close.get(symbol), errors="coerce").dropna()
        v = pd.to_numeric(volume.get(symbol), errors="coerce").reindex(close.index)
        r = pd.to_numeric(ret.get(symbol), errors="coerce").dropna()
        if c.empty:
            rows.append({"symbol": symbol})
            continue
        dollar = (close.get(symbol) * v).dropna()
        row = {
            "symbol": symbol,
            "price": float(c.iloc[-1]),
            "avg_dollar_vol_20d": float(dollar.tail(20).mean()) if len(dollar) else np.nan,
            "vol20_ann": float(r.tail(20).std(ddof=0) * np.sqrt(252)) if len(r.tail(20)) >= 5 else np.nan,
            "vol63_ann": float(r.tail(63).std(ddof=0) * np.sqrt(252)) if len(r.tail(63)) >= 10 else np.nan,
            "drawdown_63d": float(c.iloc[-1] / c.tail(63).max() - 1.0) if len(c.tail(63)) else np.nan,
            "drawdown_126d": float(c.iloc[-1] / c.tail(126).max() - 1.0) if len(c.tail(126)) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol")


def load_ticker_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "exchange"]).set_index("symbol")
    cols = {"ticker", "name", "sector", "industry", "exchange", "scalemarketcap"}
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "exchange"]).set_index("symbol")
    df["symbol"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["symbol"] != ""].drop_duplicates("symbol", keep="last")
    return df.drop(columns=["ticker"], errors="ignore").set_index("symbol")


def make_risk_flags(row: Mapping[str, Any], primary_horizon: int = 20) -> str:
    flags: List[str] = []
    beta = finite_or_none(row.get("beta_qqq_63d"))
    if beta is not None and beta >= 1.40:
        flags.append("high_qqq_beta")
    vol = finite_or_none(row.get("vol20_ann"))
    if vol is not None and vol >= 0.60:
        flags.append("high_vol")
    dd = finite_or_none(row.get("drawdown_63d"))
    if dd is not None and dd <= -0.15:
        flags.append("deep_recent_drawdown")
    adv = finite_or_none(row.get("avg_dollar_vol_20d"))
    if adv is not None and adv < 10_000_000:
        flags.append("low_liquidity")
    event_fresh = finite_or_none(row.get("fmp_event_freshness"))
    if event_fresh is not None and event_fresh <= 0.05:
        flags.append("stale_fmp_event")
    p10 = calibration_value(row, int(primary_horizon), "p10")
    if p10 is None and int(primary_horizon) != 20:
        p10 = finite_or_none(row.get("hist_20d_p10"))
    if p10 is not None and p10 <= -0.10:
        flags.append("wide_20d_downside")
    return ",".join(flags) if flags else "none"


def risk_flag_set(row: Mapping[str, Any]) -> Set[str]:
    raw = str(row.get("risk_flags") or "")
    return {token.strip() for token in raw.split(",") if token.strip() and token.strip() != "none"}


def calibration_value(row: Mapping[str, Any], primary_horizon: int, metric: str) -> Optional[float]:
    h = int(primary_horizon)
    for prefix in ("calib", "local", "hist"):
        val = finite_or_none(row.get(f"{prefix}_{h}d_{metric}"))
        if val is not None:
            return val
    return None


def classify_expected_edge(row: Mapping[str, Any], primary_horizon: int) -> Tuple[str, str]:
    """Classify model edge from rank strength and historical QQQ-excess calibration."""
    h = int(primary_horizon)
    pct = finite_or_none(row.get("score_percentile"))
    z = finite_or_none(row.get("score_z"))
    median = calibration_value(row, h, "median")
    hit = calibration_value(row, h, "hit_rate")

    points = 0
    reasons: List[str] = []
    if pct is not None:
        if pct >= 0.99:
            points += 2
            reasons.append("top 1% score")
        elif pct >= 0.975:
            points += 1
            reasons.append("top 2.5% score")
        elif pct < 0.90:
            points -= 1
            reasons.append("below top-decile score")
    else:
        points -= 1
        reasons.append("missing score percentile")

    if z is not None:
        if z >= 1.5:
            points += 1
            reasons.append("strong score z")
        elif z <= 0.0:
            points -= 1
            reasons.append("weak score z")

    if median is not None:
        if median >= 0.015:
            points += 2
            reasons.append(f"{h}d median excess >= 1.5%")
        elif median >= 0.005:
            points += 1
            reasons.append(f"{h}d median excess positive")
        elif median <= 0.0:
            points -= 2
            reasons.append(f"{h}d median excess non-positive")
    else:
        points -= 1
        reasons.append(f"missing {h}d calibration median")

    if hit is not None:
        if hit >= 0.58:
            points += 1
            reasons.append(f"{h}d hit rate >= 58%")
        elif hit < 0.50:
            points -= 1
            reasons.append(f"{h}d hit rate < 50%")

    if points >= 4:
        return "high", "; ".join(reasons)
    if points >= 2:
        return "medium", "; ".join(reasons)
    return "low", "; ".join(reasons)


def classify_risk(row: Mapping[str, Any], primary_horizon: int) -> Tuple[str, str]:
    """Classify candidate risk from volatility, beta, drawdown, liquidity, and tail loss."""
    h = int(primary_horizon)
    points = 0
    reasons: List[str] = []
    beta = finite_or_none(row.get("beta_qqq_63d"))
    vol20 = finite_or_none(row.get("vol20_ann"))
    vol63 = finite_or_none(row.get("vol63_ann"))
    dd63 = finite_or_none(row.get("drawdown_63d"))
    dd126 = finite_or_none(row.get("drawdown_126d"))
    adv = finite_or_none(row.get("avg_dollar_vol_20d"))
    p10 = calibration_value(row, h, "p10")
    flags = risk_flag_set(row)

    if beta is not None:
        if beta >= 1.50:
            points += 2
            reasons.append("QQQ beta >= 1.50")
        elif beta >= 1.20:
            points += 1
            reasons.append("QQQ beta >= 1.20")
    elif "high_qqq_beta" in flags:
        points += 1
        reasons.append("high beta flag")

    if vol20 is not None:
        if vol20 >= 0.70:
            points += 2
            reasons.append("20d annualized vol >= 70%")
        elif vol20 >= 0.45:
            points += 1
            reasons.append("20d annualized vol >= 45%")
    elif "high_vol" in flags:
        points += 1
        reasons.append("high volatility flag")

    if vol63 is not None and vol63 >= 0.55:
        points += 1
        reasons.append("63d annualized vol >= 55%")
    if dd63 is not None and dd63 <= -0.25:
        points += 2
        reasons.append("63d drawdown <= -25%")
    elif dd63 is not None and dd63 <= -0.15:
        points += 1
        reasons.append("63d drawdown <= -15%")
    if dd126 is not None and dd126 <= -0.30:
        points += 1
        reasons.append("126d drawdown <= -30%")
    if p10 is not None:
        if p10 <= -0.15:
            points += 2
            reasons.append(f"{h}d p10 excess <= -15%")
        elif p10 <= -0.10:
            points += 2
            reasons.append(f"{h}d p10 excess <= -10%")
        elif p10 <= -0.06:
            points += 1
            reasons.append(f"{h}d p10 excess <= -6%")
    elif "wide_20d_downside" in flags:
        points += 1
        reasons.append("wide downside flag")
    if adv is not None:
        if adv < 5_000_000:
            points += 2
            reasons.append("20d ADV < $5M")
        elif adv < 20_000_000:
            points += 1
            reasons.append("20d ADV < $20M")
    elif "low_liquidity" in flags:
        points += 1
        reasons.append("low liquidity flag")

    if points >= 4:
        return "high", "; ".join(reasons) or "multiple elevated risk inputs"
    if points >= 2:
        return "medium", "; ".join(reasons) or "some elevated risk inputs"
    return "low", "; ".join(reasons) or "no major risk flags"


def classify_confidence(row: Mapping[str, Any], primary_horizon: int) -> Tuple[str, str]:
    """Classify report confidence separately from expected edge and risk."""
    h = int(primary_horizon)
    points = 0
    reasons: List[str] = []
    missing_core = 0
    for col in ("score_percentile", f"hist_{h}d_median", f"hist_{h}d_p10", "vol20_ann", "beta_qqq_63d"):
        if finite_or_none(row.get(col)) is None:
            missing_core += 1
    if missing_core:
        points -= missing_core
        reasons.append(f"{missing_core} missing core fields")

    samples = finite_or_none(row.get(f"hist_{h}d_sample_count"))
    if samples is not None:
        if samples >= 1000:
            points += 2
            reasons.append("large calibration sample")
        elif samples >= 300:
            points += 1
            reasons.append("adequate calibration sample")
        elif samples < 100:
            points -= 2
            reasons.append("thin calibration sample")
        else:
            points -= 1
            reasons.append("limited calibration sample")
    else:
        points -= 2
        reasons.append("missing calibration sample")

    coverage = finite_or_none(row.get("fmp_event_coverage"))
    freshness = finite_or_none(row.get("fmp_event_freshness"))
    flags = risk_flag_set(row)
    if coverage is not None:
        if coverage >= 0.50:
            points += 1
            reasons.append("good FMP coverage")
        elif coverage <= 0.05:
            points -= 1
            reasons.append("low FMP coverage")
    if freshness is not None:
        if freshness >= 0.20:
            points += 1
            reasons.append("fresh FMP signal")
        elif freshness <= 0.05:
            points -= 1
            reasons.append("stale FMP signal")
    elif "stale_fmp_event" in flags:
        points -= 1
        reasons.append("stale FMP flag")

    if str(row.get("top_driver") or "").strip():
        points += 1
        reasons.append("driver attribution available")
    if str(row.get("sleeve_contributions") or "").strip():
        points += 1
        reasons.append("sleeve attribution available")

    if missing_core >= 2 or points <= 0:
        return "low", "; ".join(reasons)
    if points >= 4:
        return "high", "; ".join(reasons)
    return "medium", "; ".join(reasons)


def selector_category_for_tiers(edge_tier: str, risk_tier: str, confidence_tier: str) -> str:
    if confidence_tier == "low":
        return "Needs Review"
    if edge_tier == "high" and risk_tier in {"low", "medium"}:
        return "Core Candidate"
    if edge_tier == "high" and risk_tier == "high":
        return "Aggressive Upside"
    if edge_tier == "medium" and risk_tier in {"low", "medium"}:
        return "Low Priority / Defensive"
    return "Avoid / Watch Only"


def clipped(value: Optional[float], lo: float, hi: float, default: float = 0.0) -> float:
    val = finite_or_none(value)
    if val is None:
        return float(default)
    return min(max(float(val), float(lo)), float(hi))


def selector_utility_components(row: Mapping[str, Any], primary_horizon: int) -> Dict[str, float]:
    """Risk-adjusted review-priority score. Higher is better within a candidate pool."""
    h = int(primary_horizon)
    pct = finite_or_none(row.get("score_percentile"))
    z = finite_or_none(row.get("score_z"))
    median = calibration_value(row, h, "median")
    hit = calibration_value(row, h, "hit_rate")
    p10 = calibration_value(row, h, "p10")
    p75 = calibration_value(row, h, "p75")
    beta = finite_or_none(row.get("beta_qqq_63d"))
    vol20 = finite_or_none(row.get("vol20_ann"))
    dd63 = finite_or_none(row.get("drawdown_63d"))
    adv = finite_or_none(row.get("avg_dollar_vol_20d"))
    freshness = finite_or_none(row.get("fmp_event_freshness"))
    coverage = finite_or_none(row.get("fmp_event_coverage"))

    score_component = clipped(z, -3.0, 3.0) * 0.35
    if pct is not None:
        score_component += clipped((pct - 0.90) / 0.10, 0.0, 1.0) * 0.40
    edge_component = clipped(median, -0.03, 0.05) * 35.0
    if hit is not None:
        edge_component += clipped(hit - 0.50, -0.15, 0.20) * 5.0
    upside_component = clipped(p75, 0.0, 0.15) * 6.0

    downside_penalty = abs(min(float(p10), 0.0)) * 5.0 if p10 is not None else 0.25
    beta_penalty = max(float(beta) - 1.15, 0.0) * 0.28 if beta is not None else 0.15
    vol_penalty = max(float(vol20) - 0.35, 0.0) * 0.45 if vol20 is not None else 0.15
    drawdown_penalty = abs(min(float(dd63), 0.0)) * 1.0 if dd63 is not None else 0.0
    liquidity_penalty = 0.0
    if adv is not None:
        if adv < 5_000_000:
            liquidity_penalty = 0.60
        elif adv < 20_000_000:
            liquidity_penalty = 0.30
    fmp_component = 0.0
    if freshness is not None:
        fmp_component += clipped(freshness, 0.0, 1.0) * 0.12
    if coverage is not None:
        fmp_component += clipped(coverage, 0.0, 1.0) * 0.08

    confidence_tier = str(row.get("confidence_tier") or "")
    confidence_multiplier = {"high": 1.0, "medium": 0.85, "low": 0.55}.get(confidence_tier, 0.75)
    gross = score_component + edge_component + upside_component + fmp_component
    penalties = downside_penalty + beta_penalty + vol_penalty + drawdown_penalty + liquidity_penalty
    utility = (gross - penalties) * confidence_multiplier
    return {
        "selector_utility_score": float(utility),
        "utility_score_component": float(score_component),
        "utility_edge_component": float(edge_component),
        "utility_upside_component": float(upside_component),
        "utility_fmp_component": float(fmp_component),
        "utility_downside_penalty": float(downside_penalty),
        "utility_beta_penalty": float(beta_penalty),
        "utility_vol_penalty": float(vol_penalty),
        "utility_drawdown_penalty": float(drawdown_penalty),
        "utility_liquidity_penalty": float(liquidity_penalty),
    }


def add_selector_utility(report: pd.DataFrame, primary_horizon: int) -> pd.DataFrame:
    rows = [selector_utility_components(row, primary_horizon) for row in report.to_dict("records")]
    if not rows:
        return report.copy()
    out = pd.concat([report.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    utility = pd.to_numeric(out["selector_utility_score"], errors="coerce")
    out["selector_utility_percentile"] = utility.rank(method="average", pct=True)
    try:
        decile = pd.qcut(utility.rank(method="first"), q=min(10, int(utility.notna().sum())), labels=False, duplicates="drop")
        out["selector_utility_decile"] = (decile.astype("float64") + 1).where(decile.notna(), np.nan)
    except ValueError:
        out["selector_utility_decile"] = np.nan
    return out


def add_diversified_selector_rank(
    report: pd.DataFrame,
    *,
    topn: int,
    sector_cap: Optional[int] = None,
    high_risk_cap: Optional[int] = None,
) -> pd.DataFrame:
    out = report.copy().reset_index(drop=True)
    if out.empty:
        return out
    if "model_rank" not in out.columns:
        out["model_rank"] = out.get("rank", pd.Series(np.arange(1, len(out) + 1))).astype(int)
    for col in ("selector_utility_score", "score_percentile", "score"):
        if col not in out.columns:
            out[col] = 0.0
    n = min(int(topn), len(out))
    sector_cap = int(sector_cap or max(3, math.ceil(n * 0.36)))
    high_risk_cap = int(high_risk_cap or max(3, math.ceil(n * 0.68)))
    ordered = out.sort_values(
        ["selector_utility_score", "score_percentile", "score"],
        ascending=[False, False, False],
        na_position="last",
    )
    selected: List[int] = []
    deferred: List[int] = []
    sector_counts: Dict[str, int] = {}
    high_risk_count = 0
    for idx, row in ordered.iterrows():
        sector = str(row.get("sector") or "UNKNOWN")
        risk = str(row.get("risk_tier") or "")
        sector_blocked = sector_counts.get(sector, 0) >= sector_cap
        risk_blocked = risk == "high" and high_risk_count >= high_risk_cap
        if len(selected) < n and not sector_blocked and not risk_blocked:
            selected.append(int(idx))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if risk == "high":
                high_risk_count += 1
        else:
            deferred.append(int(idx))
        if len(selected) >= n:
            break
    if len(selected) < n:
        for idx in list(ordered.index):
            i = int(idx)
            if i not in selected:
                selected.append(i)
            if len(selected) >= n:
                break
    remaining = [int(idx) for idx in ordered.index if int(idx) not in selected]
    full_order = selected + remaining
    rank_map = {idx: rank for rank, idx in enumerate(full_order, start=1)}
    out["selector_rank"] = out.index.map(rank_map).astype(int)
    out["in_selector_watchlist"] = out["selector_rank"] <= n
    out["sector_cap_used"] = sector_cap
    out["high_risk_cap_used"] = high_risk_cap
    return out


def selector_action_for_row(row: Mapping[str, Any], *, topn: int) -> Tuple[str, str]:
    confidence = str(row.get("confidence_tier") or "")
    strict = str(row.get("strict_assessment") or "")
    risk = str(row.get("risk_tier") or "")
    edge = str(row.get("expected_edge_tier") or "")
    rank = finite_or_none(row.get("selector_rank"))
    pct = finite_or_none(row.get("selector_utility_percentile"))
    in_watchlist = bool(row.get("in_selector_watchlist"))
    if confidence == "low":
        return "Needs Review", "low confidence or incomplete core data"
    if not in_watchlist:
        return "Avoid", "outside diversified selector watchlist"
    best_cutoff = max(4, math.ceil(int(topn) * 0.25))
    speculative_cutoff = max(best_cutoff + 4, math.ceil(int(topn) * 0.60))
    if strict == "Core Candidate":
        return "Best Current Candidates", "strict assessment already core"
    if rank is not None and rank <= best_cutoff and risk != "high":
        return "Best Current Candidates", "top diversified utility with acceptable risk"
    if rank is not None and rank <= best_cutoff and risk == "high":
        return "Speculative / High Upside", "top diversified utility but high risk"
    if strict == "Aggressive Upside":
        return "Speculative / High Upside", "strict assessment is high-edge/high-risk"
    if rank is not None and rank <= speculative_cutoff:
        return "Speculative / High Upside" if risk == "high" or edge == "medium" else "Watchlist Only", "upper diversified utility tier"
    if pct is not None and pct >= 0.45:
        return "Watchlist Only", "middle utility tier for human review"
    return "Avoid", "low utility relative to current alternatives"


def selector_source_for_row(row: Mapping[str, Any], *, topn: int) -> Tuple[str, str]:
    strict = str(row.get("strict_assessment") or "")
    risk = str(row.get("risk_tier") or "")
    edge = str(row.get("expected_edge_tier") or "")
    action = str(row.get("selector_action") or row.get("selector_category") or "")
    in_watchlist = bool(row.get("in_selector_watchlist"))
    model_rank = finite_or_none(row.get("model_rank"))
    selector_rank = finite_or_none(row.get("selector_rank"))
    utility_pct = finite_or_none(row.get("selector_utility_percentile"))

    if strict == "Aggressive Upside" and risk == "high" and edge == "high":
        return (
            "high_risk_upside_proxy",
            "high-edge/high-risk calibrated slice; promote only if current diagnostics support this source",
        )
    if in_watchlist and (action == "Best Current Candidates" or (utility_pct is not None and utility_pct >= 0.60)):
        return "risk_adjusted_utility", "selected by risk-adjusted utility within the current candidate pool"
    if model_rank is not None and selector_rank is not None and abs(float(model_rank) - float(selector_rank)) >= 5:
        direction = "promoted" if float(selector_rank) < float(model_rank) else "deferred"
        return "diversified_rerank", f"{direction} versus base model rank by utility and diversification rules"
    if not in_watchlist:
        return "excluded_candidate_pool", "outside the diversified selector watchlist"
    return "base_alpha", "carried forward from the base model rank with no large selector rerank"


def add_selector_categories(report: pd.DataFrame, primary_horizon: int, topn: Optional[int] = None) -> pd.DataFrame:
    out = report.copy()
    rows = []
    for row in out.to_dict("records"):
        edge_tier, edge_reason = classify_expected_edge(row, primary_horizon)
        risk_tier, risk_reason = classify_risk(row, primary_horizon)
        confidence_tier, confidence_reason = classify_confidence(row, primary_horizon)
        strict = selector_category_for_tiers(edge_tier, risk_tier, confidence_tier)
        rows.append(
            {
                "expected_edge_tier": edge_tier,
                "risk_tier": risk_tier,
                "confidence_tier": confidence_tier,
                "strict_assessment": strict,
                "strict_assessment_reason": (
                    f"edge={edge_tier}: {edge_reason or 'not enough positive evidence'}; "
                    f"risk={risk_tier}: {risk_reason or 'not enough risk evidence'}; "
                    f"confidence={confidence_tier}: {confidence_reason or 'standard coverage'}"
                ),
            }
        )
    if rows:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out = add_selector_utility(out, primary_horizon)
    out = add_diversified_selector_rank(out, topn=int(topn or len(out) or 1))
    action_rows = []
    for row in out.to_dict("records"):
        action, reason = selector_action_for_row(row, topn=int(topn or len(out) or 1))
        action_rows.append(
            {
                "selector_action": action,
                "selector_action_reason": reason,
                # Backward-compatible alias. New code should prefer selector_action.
                "selector_category": action,
                "category_reason": reason,
            }
        )
    if action_rows:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(action_rows)], axis=1)
    source_rows = []
    for row in out.to_dict("records"):
        source, reason = selector_source_for_row(row, topn=int(topn or len(out) or 1))
        source_rows.append({"selector_source": source, "selector_source_reason": reason})
    if source_rows:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(source_rows)], axis=1)
    return out


def mark_core_watchlist(report: pd.DataFrame, core_topn: int = 5) -> pd.DataFrame:
    out = report.copy().reset_index(drop=True)
    out["in_core_watchlist"] = False
    out["core_rank"] = np.nan
    out["core_role"] = "top_25_context"
    out["core_reason"] = ""
    if out.empty or int(core_topn) <= 0:
        return out

    n = min(int(core_topn), len(out))
    actionable = out[
        ~out.get("selector_action", pd.Series("", index=out.index)).astype(str).isin({"Avoid", "Needs Review"})
    ].copy()
    if len(actionable) < n:
        pool = out.copy()
    else:
        pool = actionable
    action_order = {name: idx for idx, name in enumerate(SELECTOR_ACTION_ORDER)}
    action_values = pool.get("selector_action", pd.Series("", index=pool.index))
    rank_values = pool.get("selector_rank", pool.get("rank", pd.Series(np.arange(1, len(pool) + 1), index=pool.index)))
    pool["_core_action_order"] = action_values.map(lambda value: action_order.get(str(value), 999))
    pool["_core_rank_key"] = pd.to_numeric(rank_values, errors="coerce")
    pool["_core_utility_key"] = pd.to_numeric(pool.get("selector_utility_score"), errors="coerce")
    selected_idx = (
        pool.sort_values(
            ["_core_action_order", "_core_rank_key", "_core_utility_key"],
            ascending=[True, True, False],
            na_position="last",
        )
        .head(n)
        .index
        .tolist()
    )
    for core_rank, idx in enumerate(selected_idx, start=1):
        action = str(out.loc[idx].get("selector_action") or "")
        risk = str(out.loc[idx].get("risk_tier") or "")
        edge = str(out.loc[idx].get("expected_edge_tier") or "")
        out.loc[idx, "in_core_watchlist"] = True
        out.loc[idx, "core_rank"] = int(core_rank)
        out.loc[idx, "core_role"] = "core_5"
        if action == "Best Current Candidates":
            reason = "highest-priority current candidate after utility, calibration, and risk checks"
        elif action == "Speculative / High Upside":
            reason = "included for upside review despite elevated risk controls"
        elif action == "Watchlist Only":
            reason = "included from the broader watchlist by selector rank"
        else:
            reason = "included because fewer actionable names were available"
        if edge or risk:
            reason = f"{reason}; edge={edge or 'unknown'}, risk={risk or 'unknown'}"
        out.loc[idx, "core_reason"] = reason
    return out


def infer_regime_state(row: Mapping[str, Any], model_kwargs: Mapping[str, Any]) -> str:
    trend = finite_or_none(row.get("MKT_QQQ_RET_63D_LAG1"))
    fast = finite_or_none(row.get("MKT_QQQ_RET_20D_LAG1"))
    drawdown = finite_or_none(row.get("MKT_QQQ_DD_126D_LAG1"))
    vol = finite_or_none(row.get("MKT_QQQ_VOL_20D_LAG1"))
    breadth = finite_or_none(row.get("MKT_BREADTH_RET63_POS_LAG1"))
    default = str(model_kwargs.get("default_state", "chop"))
    if trend is None or fast is None or drawdown is None:
        return default
    risk_off_threshold = float(model_kwargs.get("risk_off_threshold", -0.04))
    risk_on_threshold = float(model_kwargs.get("risk_on_threshold", 0.03))
    recovery_fast_threshold = float(model_kwargs.get("recovery_fast_threshold", 0.02))
    fading_fast_threshold = float(model_kwargs.get("fading_fast_threshold", -0.02))
    drawdown_limit = float(model_kwargs.get("drawdown_limit", -0.10))
    drawdown_warning = float(model_kwargs.get("drawdown_warning", -0.06))
    high_vol_threshold = model_kwargs.get("high_vol_threshold", 0.35)
    breadth_threshold = model_kwargs.get("breadth_threshold", 0.45)
    risk_off = trend <= risk_off_threshold or drawdown <= drawdown_limit
    if breadth is not None and breadth_threshold is not None:
        risk_off = risk_off or (breadth < float(breadth_threshold) and trend < 0.0)
    risk_on = trend >= risk_on_threshold and drawdown > drawdown_warning
    if high_vol_threshold is not None and vol is not None:
        risk_on = risk_on and vol <= float(high_vol_threshold)
    fading = (trend > 0.0 and fast <= fading_fast_threshold) or (drawdown <= drawdown_warning and not risk_off)
    recovery = risk_off and fast >= recovery_fast_threshold
    states = set((model_kwargs.get("state_weights") or {}).keys())
    if risk_on and "risk_on" in states:
        return "risk_on"
    if fading and "fading" in states:
        return "fading"
    if risk_off and "risk_off" in states:
        return "risk_off"
    if recovery and "recovery" in states:
        return "recovery"
    return default


def zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((numeric - float(numeric.mean())) / std).fillna(0.0)


def deterministic_sleeve_attribution(
    *,
    cfg: Mapping[str, Any],
    score_date: pd.Timestamp,
    all_symbols: Sequence[str],
    top_symbols: Sequence[str],
    provider_uri: Path,
) -> pd.DataFrame:
    model = ((cfg.get("task") or {}).get("model") or {})
    if str(model.get("class")) != "RegimeSleeveScoreModel":
        return pd.DataFrame(index=pd.Index(top_symbols, name="symbol"))
    kwargs = model.get("kwargs", {}) or {}
    sleeves = kwargs.get("sleeves", {}) or {}
    state_weights = kwargs.get("state_weights", {}) or {}
    if not sleeves or not state_weights:
        return pd.DataFrame(index=pd.Index(top_symbols, name="symbol"))

    fmap = feature_name_map(cfg)
    required_names = set()
    for weights in sleeves.values():
        required_names.update(str(name).upper() for name in weights)
    state_features = [
        kwargs.get("trend_feature", "MKT_QQQ_RET_63D_LAG1"),
        kwargs.get("fast_trend_feature", "MKT_QQQ_RET_20D_LAG1"),
        kwargs.get("drawdown_feature", "MKT_QQQ_DD_126D_LAG1"),
        kwargs.get("vol_feature", "MKT_QQQ_VOL_20D_LAG1"),
        kwargs.get("breadth_feature", "MKT_BREADTH_RET63_POS_LAG1"),
    ]
    required_names.update(str(x).upper() for x in state_features if x)
    expr_by_name = {name: fmap[name] for name in required_names if name in fmap}
    if not expr_by_name:
        return pd.DataFrame(index=pd.Index(top_symbols, name="symbol"))

    snap = current_feature_snapshot(
        instruments=all_symbols,
        score_date=score_date,
        provider_uri=provider_uri,
        fields=list(dict.fromkeys(expr_by_name.values())),
    )
    if snap.empty:
        return pd.DataFrame(index=pd.Index(top_symbols, name="symbol"))
    reverse = {expr: name for name, expr in expr_by_name.items()}
    snap = snap.rename(columns={FIELD_RENAMES.get(expr, str(expr).lstrip("$").lower()): reverse.get(expr, expr) for expr in expr_by_name.values()})
    # If renaming through FIELD_RENAMES missed a raw expression, keep direct names too.
    for expr, name in reverse.items():
        if expr in snap.columns and name not in snap.columns:
            snap[name] = snap[expr]

    normalized = pd.DataFrame(index=snap.index)
    for name in required_names:
        if name in snap.columns:
            normalized[name] = zscore_series(snap[name])

    rows = []
    for symbol in [str(x).upper() for x in top_symbols]:
        if symbol not in normalized.index:
            rows.append({"symbol": symbol})
            continue
        raw_row = {name: snap.loc[symbol, name] for name in snap.columns if name in required_names}
        state = infer_regime_state(raw_row, kwargs)
        weights = dict(state_weights.get(state) or state_weights.get(str(kwargs.get("default_state", "chop"))) or {})
        total_state_weight = sum(float(v) for v in weights.values() if float(v) >= 0)
        contributions: Dict[str, float] = {}
        for sleeve, feature_weights in sleeves.items():
            sleeve_score = 0.0
            for feature, weight in (feature_weights or {}).items():
                feature_name = str(feature).upper()
                if feature_name in normalized.columns:
                    sleeve_score += float(weight) * float(normalized.loc[symbol, feature_name])
            state_weight = float(weights.get(sleeve, 0.0)) / total_state_weight if total_state_weight > 0 else 0.0
            contributions[str(sleeve)] = float(state_weight * sleeve_score)
        top_driver = ""
        if contributions:
            top_driver = max(contributions.items(), key=lambda item: abs(item[1]))[0]
        rows.append(
            {
                "symbol": symbol,
                "regime_state": state,
                "top_driver": top_driver,
                "sleeve_contributions": json.dumps(json_safe(contributions), sort_keys=True),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def format_pct(value: Any, digits: int = 1) -> str:
    val = finite_or_none(value)
    if val is None:
        return ""
    return f"{val * 100:.{digits}f}%"


def format_float(value: Any, digits: int = 2) -> str:
    val = finite_or_none(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def build_markdown_report(
    *,
    report: pd.DataFrame,
    calibration: pd.DataFrame,
    metadata: Mapping[str, Any],
    primary_horizon: int,
) -> str:
    h = int(primary_horizon)
    lines = [
        "# US Stock Selector Report",
        "",
        f"- Generated UTC: {metadata.get('generated_utc')}",
        f"- Score date: {metadata.get('score_date')}",
        f"- Prediction file: `{metadata.get('pred')}`",
        f"- Config: `{metadata.get('config')}`",
        f"- Top names: {metadata.get('topn')}",
        f"- Core names: {metadata.get('core_topn', 5)}",
        f"- Primary horizon: {h} trading days",
        "",
        "This report is a human-review watchlist. Expected edge is model-calibrated QQQ excess, not a guaranteed return.",
        "",
        "## Core 5 Shortlist",
        "",
    ]
    core = pd.DataFrame()
    if not report.empty and "in_core_watchlist" in report.columns:
        core = report.loc[report["in_core_watchlist"].astype(bool)].copy()
    if core.empty:
        lines.append("No core shortlist rows were available.")
    else:
        core = core.sort_values(["core_rank", "rank"], na_position="last")
        lines.extend(
            [
                "| Core rank | Selector rank | Symbol | Action | Edge | Risk | Utility | Reason |",
                "| ---: | ---: | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for _, row in core.iterrows():
            lines.append(
                "| {core_rank} | {rank} | {symbol} | {action} | {edge} | {risk} | {utility} | {reason} |".format(
                    core_rank="" if finite_or_none(row.get("core_rank")) is None else int(row.get("core_rank")),
                    rank=int(row["rank"]),
                    symbol=row["symbol"],
                    action=row.get("selector_action", row.get("selector_category", "")),
                    edge=row.get("expected_edge_tier", ""),
                    risk=row.get("risk_tier", ""),
                    utility=format_float(row.get("selector_utility_score")),
                    reason=row.get("core_reason", ""),
                )
            )
    lines.extend(
        [
            "",
        "## Actionable Watchlist",
        "",
        ]
    )
    if report.empty:
        lines.append("No watchlist rows were available.")
    else:
        category_values = report.get(
            "selector_action",
            report.get(
                "selector_category",
                pd.Series(["Uncategorized"] * len(report), index=report.index),
            ),
        ).fillna("Uncategorized")
        strict_values = report.get(
            "strict_assessment",
            pd.Series(["Uncategorized"] * len(report), index=report.index),
        ).fillna("Uncategorized")
        category_counts = category_values.value_counts().to_dict()
        ordered_categories = [cat for cat in SELECTOR_ACTION_ORDER if cat in category_counts]
        ordered_categories.extend(sorted(cat for cat in category_counts if cat not in ordered_categories))
        lines.extend(["| Selector action | Count |", "| --- | ---: |"])
        for category in ordered_categories:
            lines.append(f"| {category} | {int(category_counts.get(category, 0))} |")
        strict_counts = strict_values.value_counts().to_dict()
        if strict_counts:
            lines.extend(["", "| Strict assessment | Count |", "| --- | ---: |"])
            ordered_strict = [cat for cat in STRICT_ASSESSMENT_ORDER if cat in strict_counts]
            ordered_strict.extend(sorted(cat for cat in strict_counts if cat not in ordered_strict))
            for category in ordered_strict:
                lines.append(f"| {category} | {int(strict_counts.get(category, 0))} |")
        for category in ordered_categories:
            part = report.loc[category_values == category].sort_values("rank")
            if part.empty:
                continue
            lines.extend(
                [
                    "",
                    f"### {category}",
                    "",
                    (
                        "| Selector rank | Model rank | Symbol | Sector | Score pct | Utility | Action | Source | Strict | "
                        "Edge | Risk | Confidence | "
                        f"Calib {h}d hit | Calib {h}d median excess | Calib {h}d p10 | "
                        "Beta QQQ | Vol20 | Risk flags | Top driver |"
                    ),
                    "| ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for _, row in part.iterrows():
                lines.append(
                    "| {rank} | {model_rank} | {symbol} | {sector} | {pct} | {utility} | {action} | {source} | {strict} | "
                    "{edge} | {risk} | {confidence} | {hit} | {median} | {p10} | {beta} | {vol} | {flags} | {driver} |".format(
                        rank=int(row["rank"]),
                        model_rank="" if finite_or_none(row.get("model_rank")) is None else int(row.get("model_rank")),
                        symbol=row["symbol"],
                        sector=str(row.get("sector") or ""),
                        pct=format_pct(row.get("score_percentile")),
                        utility=format_float(row.get("selector_utility_score")),
                        action=row.get("selector_action", row.get("selector_category", "")),
                        source=row.get("selector_source", ""),
                        strict=row.get("strict_assessment", ""),
                        edge=row.get("expected_edge_tier", ""),
                        risk=row.get("risk_tier", ""),
                        confidence=row.get("confidence_tier", ""),
                        hit=format_pct(calibration_value(row, h, "hit_rate")),
                        median=format_pct(calibration_value(row, h, "median")),
                        p10=format_pct(calibration_value(row, h, "p10")),
                        beta=format_float(row.get("beta_qqq_63d")),
                        vol=format_pct(row.get("vol20_ann")),
                        flags=row.get("risk_flags", ""),
                        driver=row.get("top_driver", ""),
                    )
                )
    lines.extend(["", "## Calibration", ""])
    if calibration.empty:
        lines.append("No calibration rows were available.")
    else:
        cal = calibration[calibration["horizon_days"].astype(int) == h].copy()
        lines.extend(
            [
                "| Bucket | Samples | Hit rate | Median excess | p10 | p25 | p75 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in cal.iterrows():
            lines.append(
                "| {bucket} | {n} | {hit} | {median} | {p10} | {p25} | {p75} |".format(
                    bucket=row["score_bucket"],
                    n=int(row.get("sample_count") or 0),
                    hit=format_pct(row.get("hit_rate")),
                    median=format_pct(row.get("median")),
                    p10=format_pct(row.get("p10")),
                    p25=format_pct(row.get("p25")),
                    p75=format_pct(row.get("p75")),
                )
            )
    lines.extend(["", "## Candidate Notes", ""])
    for _, row in report.iterrows():
        lines.append(
            "- **{rank}. {symbol}**: action `{action}`, source `{source}`, strict `{strict}`, model rank `{model_rank}`, "
            "bucket `{bucket}`, sector `{sector}`, driver `{driver}`, flags `{flags}`. "
            "Calibrated {h}d median excess {median}, p10 {p10}, hit rate {hit}. "
            "Action reason: {action_reason}. Source reason: {source_reason}. Strict reason: {strict_reason}".format(
                rank=int(row["rank"]),
                symbol=row["symbol"],
                action=row.get("selector_action", row.get("selector_category", "")),
                source=row.get("selector_source", ""),
                strict=row.get("strict_assessment", ""),
                model_rank="" if finite_or_none(row.get("model_rank")) is None else int(row.get("model_rank")),
                bucket=row.get("score_bucket", ""),
                sector=str(row.get("sector") or ""),
                driver=row.get("top_driver", ""),
                flags=row.get("risk_flags", ""),
                h=h,
                median=format_pct(calibration_value(row, h, "median")),
                p10=format_pct(calibration_value(row, h, "p10")),
                hit=format_pct(calibration_value(row, h, "hit_rate")),
                action_reason=row.get("selector_action_reason", row.get("category_reason", "")),
                source_reason=row.get("selector_source_reason", ""),
                strict_reason=row.get("strict_assessment_reason", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    cfg_path: Path,
    pred_path: Path,
    provider_uri: Path,
    benchmark_pkl: Path,
    as_of: Optional[str],
    topn: int,
    core_topn: int,
    rank_export_n: int,
    horizons: Sequence[int],
    primary_horizon: int,
    calibration_lookback_days: int,
    tickers_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = _load_yaml(cfg_path)
    pred = load_prediction(pred_path)
    score_date = select_score_date(pred, as_of)
    export_n = max(int(topn), int(rank_export_n))
    rankings = rank_scores_for_date(pred, score_date, topn=export_n)
    ranked_pool = rankings.head(int(export_n)).copy()
    symbols = ranked_pool["symbol"].astype(str).str.upper().tolist()
    all_symbols = pred.xs(score_date, level="datetime").index.astype(str).str.upper().tolist()
    benchmark = load_benchmark(benchmark_pkl)

    calibration = build_calibration_frame(
        pred,
        score_date=score_date,
        cfg=cfg,
        provider_uri=provider_uri,
        benchmark=benchmark,
        horizons=horizons,
        lookback_days=calibration_lookback_days,
    )
    calibration_summary = summarize_calibration(calibration, horizons=horizons)

    risk = current_feature_snapshot(
        instruments=symbols,
        score_date=score_date,
        provider_uri=provider_uri,
        fields=DEFAULT_RISK_FIELDS,
    )
    price = price_history_metrics(instruments=symbols, score_date=score_date, provider_uri=provider_uri)
    meta = load_ticker_metadata(tickers_csv)
    attribution = deterministic_sleeve_attribution(
        cfg=cfg,
        score_date=score_date,
        all_symbols=all_symbols,
        top_symbols=symbols,
        provider_uri=provider_uri,
    )

    report_pool = ranked_pool.set_index("symbol")
    for frame in (meta, risk, price, attribution):
        if frame is not None and not frame.empty:
            report_pool = report_pool.join(frame, how="left")
    report_pool = report_pool.reset_index()
    report_pool = report_pool.rename(columns={"index": "symbol"})
    report_pool["model_rank"] = report_pool["rank"].astype(int)
    for _, bucket_row in report_pool[["symbol", "score_bucket"]].iterrows():
        bucket = str(bucket_row["score_bucket"])
        symbol = str(bucket_row["symbol"])
        idx = report_pool.index[report_pool["symbol"] == symbol]
        if len(idx) == 0:
            continue
        i = idx[0]
        for horizon in horizons:
            metrics = metrics_for_candidate_bucket(calibration_summary, bucket, int(horizon))
            prefix = f"hist_{int(horizon)}d"
            for key in ("sample_count", "hit_rate", "mean", "median", "p10", "p25", "p75", "p90", "worst", "best"):
                report_pool.loc[i, f"{prefix}_{key}"] = metrics.get(key)
    report_pool = add_candidate_calibration(report_pool, calibration, horizons)
    report_pool["risk_flags"] = [
        make_risk_flags(row, primary_horizon=int(primary_horizon)) for row in report_pool.to_dict("records")
    ]
    report_pool = add_selector_categories(report_pool, primary_horizon=int(primary_horizon), topn=int(topn))
    report_pool = report_pool.sort_values("selector_rank").reset_index(drop=True)
    report_pool["candidate_pool_rank"] = np.arange(1, len(report_pool) + 1)
    report = report_pool.loc[report_pool["in_selector_watchlist"]].copy().head(int(topn))
    report["rank"] = report["selector_rank"].astype(int)
    report = report.sort_values("selector_rank").reset_index(drop=True)
    report = mark_core_watchlist(report, core_topn=int(core_topn))
    rankings = report_pool.sort_values("model_rank").reset_index(drop=True)

    metadata = {
        "generated_utc": utc_now(),
        "selector_schema_version": 5,
        "selector_classifier_version": SELECTOR_CLASSIFIER_VERSION,
        "selector_action_order": SELECTOR_ACTION_ORDER,
        "selector_source_order": SELECTOR_SOURCE_ORDER,
        "strict_assessment_order": STRICT_ASSESSMENT_ORDER,
        "selector_category_order": SELECTOR_CATEGORY_ORDER,
        "config": str(cfg_path),
        "pred": str(pred_path),
        "provider_uri": str(provider_uri),
        "benchmark_pkl": str(benchmark_pkl),
        "score_date": str(score_date.date()),
        "requested_as_of": as_of or "latest",
        "topn": int(topn),
        "core_topn": int(core_topn),
        "core_symbols": report.loc[report["in_core_watchlist"].astype(bool), "symbol"].astype(str).tolist(),
        "rank_export_n": int(export_n),
        "selector_candidate_pool_n": int(len(report_pool)),
        "rankings_csv_role": "candidate_pool_diagnostics",
        "official_cadence": "weekly",
        "horizons": [int(h) for h in horizons],
        "primary_horizon": int(primary_horizon),
        "calibration_lookback_days": int(calibration_lookback_days),
        "prediction_date_min": str(pd.Timestamp(pred.index.get_level_values("datetime").min()).date()),
        "prediction_date_max": str(pd.Timestamp(pred.index.get_level_values("datetime").max()).date()),
    }
    return report, calibration_summary, rankings, metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a human-review US stock selector report.")
    p.add_argument("--config", required=True, help="Workflow YAML config")
    p.add_argument("--pred", required=True, help="Prediction pickle with score column")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--benchmark_pkl", default="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
    p.add_argument("--as_of", default="latest", help="Score date to use; default latest prediction date")
    p.add_argument("--topn", type=int, default=25)
    p.add_argument("--core_topn", type=int, default=5, help="Emit a first-review shortlist from the watchlist")
    p.add_argument("--rank_export_n", type=int, default=100, help="Freeze top-N rankings for later diagnostics")
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--primary_horizon", type=int, default=20)
    p.add_argument("--calibration_lookback_days", type=int, default=756)
    p.add_argument("--tickers_csv", default="/root/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--registry_out", default="artifacts/stock_selector/registry.jsonl")
    p.add_argument("--no_registry", action="store_true", help="Do not append this frozen run to the registry")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    benchmark_pkl = Path(args.benchmark_pkl).expanduser().resolve()
    tickers_csv = Path(args.tickers_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    horizons = parse_int_list(args.horizons)
    if int(args.primary_horizon) not in horizons:
        horizons = sorted(set([*horizons, int(args.primary_horizon)]))
    for path, label in ((cfg_path, "config"), (pred_path, "pred"), (provider_uri, "provider_uri"), (benchmark_pkl, "benchmark_pkl")):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    report, calibration, rankings, metadata = build_report(
        cfg_path=cfg_path,
        pred_path=pred_path,
        provider_uri=provider_uri,
        benchmark_pkl=benchmark_pkl,
        as_of=args.as_of,
        topn=int(args.topn),
        core_topn=int(args.core_topn),
        rank_export_n=int(args.rank_export_n),
        horizons=horizons,
        primary_horizon=int(args.primary_horizon),
        calibration_lookback_days=int(args.calibration_lookback_days),
        tickers_csv=tickers_csv,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    watchlist_csv = out_dir / "watchlist.csv"
    core_watchlist_csv = out_dir / "core_watchlist.csv"
    calibration_csv = out_dir / "calibration.csv"
    watchlist_json = out_dir / "watchlist.json"
    core_watchlist_json = out_dir / "core_watchlist.json"
    watchlist_md = out_dir / "watchlist.md"
    core_watchlist_md = out_dir / "core_watchlist.md"
    rankings_csv = out_dir / "rankings.csv"
    candidate_diagnostics_csv = out_dir / "candidate_diagnostics.csv"
    human_csv = out_dir / "human_decisions.csv"
    manifest_json = out_dir / "run_manifest.json"
    core_report = report.loc[report["in_core_watchlist"].astype(bool)].copy()
    core_report = core_report.sort_values(["core_rank", "rank"], na_position="last").reset_index(drop=True)
    report.to_csv(watchlist_csv, index=False)
    core_report.to_csv(core_watchlist_csv, index=False)
    calibration.to_csv(calibration_csv, index=False)
    rankings.to_csv(rankings_csv, index=False)
    rankings.to_csv(candidate_diagnostics_csv, index=False)
    human_template_created = write_human_decision_template(human_csv, report, str(metadata["score_date"]))
    watchlist_json.write_text(
        json.dumps(
            {"metadata": json_safe(metadata), "watchlist": json_safe(report.to_dict("records"))},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    core_watchlist_json.write_text(
        json.dumps(
            {"metadata": json_safe(metadata), "core_watchlist": json_safe(core_report.to_dict("records"))},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    watchlist_md.write_text(
        build_markdown_report(
            report=report,
            calibration=calibration,
            metadata=metadata,
            primary_horizon=int(args.primary_horizon),
        ),
        encoding="utf-8",
    )
    core_watchlist_md.write_text(
        build_markdown_report(
            report=core_report,
            calibration=calibration,
            metadata={**metadata, "topn": len(core_report), "core_topn": len(core_report)},
            primary_horizon=int(args.primary_horizon),
        ),
        encoding="utf-8",
    )

    file_hashes = {
        "watchlist_csv": file_sha256(watchlist_csv),
        "core_watchlist_csv": file_sha256(core_watchlist_csv),
        "watchlist_md": file_sha256(watchlist_md),
        "core_watchlist_md": file_sha256(core_watchlist_md),
        "watchlist_json": file_sha256(watchlist_json),
        "core_watchlist_json": file_sha256(core_watchlist_json),
        "calibration_csv": file_sha256(calibration_csv),
        "rankings_csv": file_sha256(rankings_csv),
        "candidate_diagnostics_csv": file_sha256(candidate_diagnostics_csv),
        "human_decisions_csv": file_sha256(human_csv),
    }
    run_id = run_id_from_metadata(metadata)
    metadata["run_id"] = run_id
    manifest = {
        "run_id": run_id,
        "metadata": metadata,
        "files": {
            "watchlist_csv": str(watchlist_csv.resolve()),
            "core_watchlist_csv": str(core_watchlist_csv.resolve()),
            "watchlist_md": str(watchlist_md.resolve()),
            "core_watchlist_md": str(core_watchlist_md.resolve()),
            "watchlist_json": str(watchlist_json.resolve()),
            "core_watchlist_json": str(core_watchlist_json.resolve()),
            "calibration_csv": str(calibration_csv.resolve()),
            "rankings_csv": str(rankings_csv.resolve()),
            "candidate_diagnostics_csv": str(candidate_diagnostics_csv.resolve()),
            "human_decisions_csv": str(human_csv.resolve()),
        },
        "file_hashes": file_hashes,
        "human_template_created": bool(human_template_created),
    }
    manifest_json.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry_entry = {
        "run_id": run_id,
        "created_utc": metadata["generated_utc"],
        "score_date": metadata["score_date"],
        "topn": metadata["topn"],
        "rank_export_n": metadata["rank_export_n"],
        "horizons": metadata["horizons"],
        "primary_horizon": metadata["primary_horizon"],
        "manifest_json": str(manifest_json.resolve()),
        "out_dir": str(out_dir.resolve()),
        "watchlist_csv": str(watchlist_csv.resolve()),
        "core_watchlist_csv": str(core_watchlist_csv.resolve()),
        "rankings_csv": str(rankings_csv.resolve()),
        "candidate_diagnostics_csv": str(candidate_diagnostics_csv.resolve()),
        "calibration_csv": str(calibration_csv.resolve()),
        "human_decisions_csv": str(human_csv.resolve()),
        "file_hashes": file_hashes,
    }
    if not args.no_registry:
        append_jsonl(Path(args.registry_out), registry_entry)

    print(f"score_date={metadata['score_date']} topn={len(report)} core_topn={len(core_report)}")
    print(f"run_id={run_id}")
    print(f"watchlist_csv={watchlist_csv}")
    print(f"core_watchlist_csv={core_watchlist_csv}")
    print(f"watchlist_md={watchlist_md}")
    print(f"core_watchlist_md={core_watchlist_md}")
    print(f"watchlist_json={watchlist_json}")
    print(f"core_watchlist_json={core_watchlist_json}")
    print(f"calibration_csv={calibration_csv}")
    print(f"rankings_csv={rankings_csv}")
    print(f"candidate_diagnostics_csv={candidate_diagnostics_csv}")
    print(f"human_decisions_csv={human_csv}")
    print(f"manifest_json={manifest_json}")
    if not args.no_registry:
        print(f"registry_out={Path(args.registry_out).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
