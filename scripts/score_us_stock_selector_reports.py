#!/usr/bin/env python
"""Score frozen US stock selector reports after outcomes are available."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_us_stock_selector_report import (  # noqa: E402
    DEFAULT_RISK_FIELDS,
    FIELD_RENAMES,
    current_feature_snapshot,
    finite_or_none,
    json_safe,
    load_prediction,
    load_ticker_metadata,
    parse_int_list,
    rank_scores_for_date,
    summarize_returns,
)
from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    _benchmark_forward_return,
    _infer_label_ref_start_days,
    _load_yaml,
    _read_pickle_compat,
)


CONTROL_FEATURES = ["$log_marketcap_q", "$risk_vol_20d"]

SELECTOR_SEGMENT_COLUMNS = {
    "selector_source": "source",
    "selector_action": "action",
    "selector_category": "category",
    "strict_assessment": "strict",
    "expected_edge_tier": "edge",
    "risk_tier": "risk",
    "confidence_tier": "confidence",
    "selector_utility_decile": "utility_decile",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def unique_manifest_paths(registry_rows: Iterable[Mapping[str, Any]], explicit: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for raw in list(explicit or []) + [str(row.get("manifest_json") or "") for row in registry_rows]:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        paths.append(path)
        seen.add(key)
    return paths


def load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    files = obj.get("files") or {}
    metadata = obj.get("metadata") or {}
    if not files.get("watchlist_csv") or not metadata.get("score_date"):
        raise ValueError(f"manifest missing required files/metadata: {path}")
    return obj


def provider_calendar(provider_uri: Path) -> pd.DatetimeIndex:
    cal_path = Path(provider_uri).expanduser().resolve() / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"qlib calendar not found: {cal_path}")
    dates = [pd.Timestamp(line.strip()).normalize() for line in cal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not dates:
        raise ValueError(f"qlib calendar is empty: {cal_path}")
    return pd.DatetimeIndex(sorted(set(dates)))


def horizon_dates(
    calendar: pd.DatetimeIndex,
    score_date: pd.Timestamp,
    horizon: int,
    *,
    label_ref_start_days: int = 1,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    score_date = pd.Timestamp(score_date).normalize()
    cal = pd.DatetimeIndex(calendar).sort_values()
    pos = int(np.searchsorted(cal.values, np.datetime64(score_date), side="left"))
    if pos >= len(cal) or pd.Timestamp(cal[pos]).normalize() != score_date:
        return None
    entry_pos = pos + max(0, int(label_ref_start_days))
    exit_pos = entry_pos + int(horizon)
    if entry_pos >= len(cal) or exit_pos >= len(cal):
        return None
    return pd.Timestamp(cal[entry_pos]).normalize(), pd.Timestamp(cal[exit_pos]).normalize()


def completed_horizons(
    calendar: pd.DatetimeIndex,
    score_date: pd.Timestamp,
    horizons: Sequence[int],
    *,
    label_ref_start_days: int,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    max_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp(calendar.max()).normalize()
    out: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for horizon in horizons:
        dates = horizon_dates(calendar, score_date, int(horizon), label_ref_start_days=label_ref_start_days)
        if dates is None:
            continue
        _, exit_date = dates
        if exit_date <= max_date:
            out[int(horizon)] = dates
    return out


def normalize_close_frame(raw: Any) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        if "$close" in raw.columns:
            series = raw["$close"]
        elif raw.shape[1] == 1:
            series = raw.iloc[:, 0]
        else:
            return pd.DataFrame()
    elif isinstance(raw, pd.Series):
        series = raw
    else:
        return pd.DataFrame()
    if not isinstance(series.index, pd.MultiIndex):
        return pd.DataFrame()
    if list(series.index.names[:2]) == ["instrument", "datetime"]:
        series = series.reorder_levels(["datetime", "instrument"])
    dates = pd.DatetimeIndex(series.index.get_level_values("datetime")).normalize()
    inst = series.index.get_level_values("instrument").astype(str).str.upper()
    series.index = pd.MultiIndex.from_arrays([dates, inst], names=["datetime", "instrument"])
    return pd.to_numeric(series, errors="coerce").unstack("instrument").sort_index()


def forward_returns_from_close(
    close: pd.DataFrame,
    horizon_map: Mapping[int, Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[int, pd.Series]:
    out: Dict[int, pd.Series] = {}
    for horizon, (entry_date, exit_date) in horizon_map.items():
        if entry_date not in close.index or exit_date not in close.index:
            out[int(horizon)] = pd.Series(dtype=float)
            continue
        entry = pd.to_numeric(close.loc[entry_date], errors="coerce")
        exit_ = pd.to_numeric(close.loc[exit_date], errors="coerce")
        ret = exit_ / entry - 1.0
        ret = ret.where((entry > 0) & (exit_ > 0)).replace([np.inf, -np.inf], np.nan)
        ret.name = f"raw_return_{int(horizon)}d"
        out[int(horizon)] = ret
    return out


def load_benchmark_return(path: Path) -> pd.Series:
    obj = _read_pickle_compat(Path(path))
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"benchmark must be a Series or single-column DataFrame: {path}")
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise ValueError(f"benchmark must be a pandas Series: {path}")
    obj.index = pd.DatetimeIndex(obj.index).normalize()
    return pd.to_numeric(obj, errors="coerce").sort_index()


def benchmark_forward_values(
    benchmark: pd.Series,
    score_date: pd.Timestamp,
    horizons: Sequence[int],
    *,
    label_ref_start_days: int,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for horizon in horizons:
        fwd = _benchmark_forward_return(
            benchmark,
            label_horizon_days=int(horizon),
            label_ref_start_days=int(label_ref_start_days),
        )
        val = None if fwd is None else finite_or_none(fwd.reindex([pd.Timestamp(score_date).normalize()]).iloc[0])
        out[int(horizon)] = float(val) if val is not None else float("nan")
    return out


def add_quantile_bucket(values: pd.Series, labels: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() < 2:
        return pd.Series("unknown", index=values.index)
    try:
        bucket = pd.qcut(numeric.rank(method="first"), q=min(int(labels), int(numeric.notna().sum())), labels=False, duplicates="drop")
    except ValueError:
        return pd.Series("unknown", index=values.index)
    return bucket.astype("Int64").astype(str).where(bucket.notna(), "unknown")


def build_control_universe(
    *,
    pred: pd.Series,
    score_date: pd.Timestamp,
    provider_uri: Path,
    tickers_csv: Path,
) -> pd.DataFrame:
    scores = pred.xs(pd.Timestamp(score_date).normalize(), level="datetime")
    rankings = rank_scores_for_date(pred, score_date, topn=len(scores))
    symbols = rankings["symbol"].astype(str).str.upper().tolist()
    meta = load_ticker_metadata(tickers_csv)
    features = current_feature_snapshot(
        instruments=symbols,
        score_date=score_date,
        provider_uri=provider_uri,
        fields=CONTROL_FEATURES,
    ).rename(
        columns={
            FIELD_RENAMES.get("$log_marketcap_q", "log_marketcap"): "log_marketcap",
            FIELD_RENAMES.get("$risk_vol_20d", "risk_vol_20d_feature"): "risk_vol_20d_feature",
        }
    )
    universe = rankings.set_index("symbol")
    for frame in (meta, features):
        if frame is not None and not frame.empty:
            universe = universe.join(frame, how="left")
    universe["sector"] = universe.get("sector", pd.Series(index=universe.index, dtype=object)).fillna("UNKNOWN")
    universe["marketcap_bucket"] = add_quantile_bucket(universe.get("log_marketcap", pd.Series(index=universe.index)))
    universe["vol_bucket"] = add_quantile_bucket(universe.get("risk_vol_20d_feature", pd.Series(index=universe.index)))
    return universe.reset_index()


def matched_control_symbols(
    symbol: str,
    universe: pd.DataFrame,
    selected_symbols: Iterable[str],
    *,
    n_controls: int = 10,
) -> List[str]:
    symbol = str(symbol).upper()
    selected = {str(x).upper() for x in selected_symbols}
    if "symbol" not in universe.columns or symbol not in set(universe["symbol"].astype(str).str.upper()):
        return []
    indexed = universe.copy()
    indexed["symbol"] = indexed["symbol"].astype(str).str.upper()
    indexed = indexed.set_index("symbol", drop=False)
    row = indexed.loc[symbol]
    candidates = indexed.loc[~indexed.index.isin(selected)].copy()
    if candidates.empty:
        return []
    same_sector = candidates[candidates["sector"].astype(str) == str(row.get("sector", "UNKNOWN"))]
    if len(same_sector) >= min(max(1, int(n_controls)), 3):
        candidates = same_sector

    cols = [col for col in ("log_marketcap", "risk_vol_20d_feature") if col in candidates.columns and col in row.index]
    if not cols:
        return candidates["symbol"].head(int(n_controls)).tolist()
    distances = pd.Series(0.0, index=candidates.index, dtype=float)
    for col in cols:
        values = pd.to_numeric(candidates[col], errors="coerce")
        center = finite_or_none(row.get(col))
        if center is None:
            continue
        scale = float(values.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        distances = distances + ((values - center) / scale).fillna(0.0).abs()
    ordered = distances.sort_values().index.tolist()
    return [str(x) for x in ordered[: int(n_controls)]]


def load_human_decisions(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df.drop_duplicates("symbol", keep="last")


def summary_row(
    *,
    run_id: str,
    score_date: str,
    horizon: int,
    group: str,
    values: pd.Series,
    metric: str = "qqq_excess_return",
    expected_median: Optional[float] = None,
) -> Dict[str, Any]:
    stats = summarize_returns(values)
    row = {
        "run_id": run_id,
        "score_date": score_date,
        "horizon_days": int(horizon),
        "group": group,
        "metric": metric,
    }
    row.update(stats)
    row["expected_median"] = expected_median
    median = finite_or_none(stats.get("median"))
    row["median_minus_expected"] = median - expected_median if median is not None and expected_median is not None else None
    return row


def slug_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    out: List[str] = []
    last_underscore = False
    for char in text:
        if char.isalnum():
            out.append(char)
            last_underscore = False
        elif not last_underscore:
            out.append("_")
            last_underscore = True
    return "".join(out).strip("_") or "unknown"


def watchlist_segment_summary_rows(
    *,
    run_id: str,
    score_date: str,
    horizon: int,
    outcomes: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if outcomes.empty:
        return rows
    for column, prefix in SELECTOR_SEGMENT_COLUMNS.items():
        if column not in outcomes.columns:
            continue
        frame = outcomes.copy()
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        frame = frame[frame[column] != ""]
        if frame.empty:
            continue
        for value, part in frame.groupby(column, sort=True):
            expected = None
            if "expected_bucket_median" in part.columns:
                expected = finite_or_none(pd.to_numeric(part["expected_bucket_median"], errors="coerce").mean())
            rows.append(
                summary_row(
                    run_id=run_id,
                    score_date=score_date,
                    horizon=int(horizon),
                    group=f"{prefix}_{slug_label(value)}",
                    values=part["qqq_excess_return"],
                    expected_median=expected,
                )
            )
    return rows


def score_manifest(
    manifest_path: Path,
    *,
    provider_uri_override: Optional[Path],
    benchmark_pkl_override: Optional[Path],
    tickers_csv: Path,
    horizons: Sequence[int],
    as_of: Optional[pd.Timestamp],
    n_controls: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    metadata = manifest.get("metadata") or {}
    files = manifest.get("files") or {}
    run_id = str(manifest.get("run_id") or metadata.get("run_id") or Path(manifest_path).parent.name)
    score_date = pd.Timestamp(metadata["score_date"]).normalize()
    cfg_path = Path(metadata["config"]).expanduser().resolve()
    pred_path = Path(metadata["pred"]).expanduser().resolve()
    provider_uri = provider_uri_override or Path(metadata.get("provider_uri") or "/root/.qlib/qlib_data/us_data").expanduser().resolve()
    benchmark_pkl = benchmark_pkl_override or Path(metadata.get("benchmark_pkl") or "/root/.qlib/qlib_data/us_data/bench_qqq.pkl").expanduser().resolve()
    watchlist = pd.read_csv(files["watchlist_csv"])
    watchlist["symbol"] = watchlist["symbol"].astype(str).str.upper().str.strip()
    pred = load_prediction(pred_path)
    cfg = _load_yaml(cfg_path)
    label_ref_start = _infer_label_ref_start_days(cfg.get("data_handler_config", {}))
    cal = provider_calendar(provider_uri)
    horizon_map = completed_horizons(
        cal,
        score_date,
        horizons,
        label_ref_start_days=label_ref_start,
        as_of=as_of,
    )
    if not horizon_map:
        return pd.DataFrame(), pd.DataFrame(), {
            "run_id": run_id,
            "score_date": str(score_date.date()),
            "status": "NO_COMPLETED_HORIZONS",
        }

    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    universe = build_control_universe(
        pred=pred,
        score_date=score_date,
        provider_uri=provider_uri,
        tickers_csv=tickers_csv,
    )
    symbols = sorted(
        set(universe["symbol"].astype(str).str.upper())
        | set(watchlist["symbol"].astype(str).str.upper())
        | {"SPY"}
    )
    start = score_date
    end = max(exit_date for _, exit_date in horizon_map.values())
    raw_close = D.features(symbols, ["$close"], start_time=start, end_time=end)
    close = normalize_close_frame(raw_close)
    returns_by_h = forward_returns_from_close(close, horizon_map)
    benchmark = load_benchmark_return(benchmark_pkl)
    qqq_returns = benchmark_forward_values(
        benchmark,
        score_date,
        list(horizon_map),
        label_ref_start_days=label_ref_start,
    )
    spy_returns = {
        int(h): finite_or_none(returns_by_h.get(int(h), pd.Series(dtype=float)).get("SPY"))
        for h in horizon_map
    }
    human = load_human_decisions(Path(files.get("human_decisions_csv", "")))
    human_cols = [col for col in human.columns if col not in set(watchlist.columns) and col != "symbol"]
    if not human.empty:
        watchlist = watchlist.merge(human[["symbol", *human_cols]], on="symbol", how="left")

    selected_symbols = watchlist["symbol"].astype(str).str.upper().tolist()
    outcome_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for horizon, (entry_date, exit_date) in horizon_map.items():
        ret = returns_by_h.get(int(horizon), pd.Series(dtype=float))
        qqq_ret = finite_or_none(qqq_returns.get(int(horizon)))
        spy_ret = finite_or_none(spy_returns.get(int(horizon)))
        sector_returns = (
            universe.assign(raw_return=universe["symbol"].map(ret))
            .groupby("sector")["raw_return"]
            .mean()
            .to_dict()
        )
        for _, row in watchlist.iterrows():
            symbol = str(row["symbol"]).upper()
            raw = finite_or_none(ret.get(symbol))
            controls = matched_control_symbols(symbol, universe, selected_symbols, n_controls=n_controls)
            control_values = pd.to_numeric(ret.reindex(controls), errors="coerce").dropna()
            control_ret = finite_or_none(control_values.mean()) if not control_values.empty else None
            sector = str(row.get("sector") or "UNKNOWN")
            sector_ret = finite_or_none(sector_returns.get(sector))
            out = {
                "run_id": run_id,
                "score_date": str(score_date.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "horizon_days": int(horizon),
                "symbol": symbol,
                "rank": int(row.get("rank")),
                "score": finite_or_none(row.get("score")),
                "score_percentile": finite_or_none(row.get("score_percentile")),
                "score_bucket": row.get("score_bucket"),
                "model_rank": finite_or_none(row.get("model_rank")),
                "selector_rank": finite_or_none(row.get("selector_rank")),
                "selector_utility_score": finite_or_none(row.get("selector_utility_score")),
                "selector_utility_percentile": finite_or_none(row.get("selector_utility_percentile")),
                "selector_utility_decile": finite_or_none(row.get("selector_utility_decile")),
                "selector_source": row.get("selector_source"),
                "selector_action": row.get("selector_action"),
                "selector_category": row.get("selector_category"),
                "strict_assessment": row.get("strict_assessment"),
                "expected_edge_tier": row.get("expected_edge_tier"),
                "risk_tier": row.get("risk_tier"),
                "confidence_tier": row.get("confidence_tier"),
                "category_reason": row.get("category_reason"),
                "selector_source_reason": row.get("selector_source_reason"),
                "selector_action_reason": row.get("selector_action_reason"),
                "strict_assessment_reason": row.get("strict_assessment_reason"),
                "sector": sector,
                "raw_return": raw,
                "qqq_return": qqq_ret,
                "spy_return": spy_ret,
                "qqq_excess_return": raw - qqq_ret if raw is not None and qqq_ret is not None else None,
                "spy_excess_return": raw - spy_ret if raw is not None and spy_ret is not None else None,
                "sector_return": sector_ret,
                "sector_excess_return": raw - sector_ret if raw is not None and sector_ret is not None else None,
                "matched_control_return": control_ret,
                "matched_control_excess_return": raw - control_ret if raw is not None and control_ret is not None else None,
                "matched_control_count": int(control_values.notna().sum()),
                "matched_controls": ",".join(controls),
            }
            for col in human_cols:
                out[col] = row.get(col)
            expected_col = f"hist_{int(horizon)}d_median"
            out["expected_bucket_median"] = finite_or_none(row.get(expected_col))
            outcome_rows.append(out)

        outcomes_h = pd.DataFrame([row for row in outcome_rows if int(row["horizon_days"]) == int(horizon)])
        summary_rows.append(
            summary_row(
                run_id=run_id,
                score_date=str(score_date.date()),
                horizon=int(horizon),
                group="watchlist_top25",
                values=outcomes_h["qqq_excess_return"],
                expected_median=finite_or_none(outcomes_h["expected_bucket_median"].mean()),
            )
        )
        summary_rows.append(
            summary_row(
                run_id=run_id,
                score_date=str(score_date.date()),
                horizon=int(horizon),
                group="watchlist_vs_matched_controls",
                values=outcomes_h["matched_control_excess_return"],
                metric="matched_control_excess_return",
            )
        )
        summary_rows.extend(
            watchlist_segment_summary_rows(
                run_id=run_id,
                score_date=str(score_date.date()),
                horizon=int(horizon),
                outcomes=outcomes_h,
            )
        )
        if "selector_rank" in outcomes_h.columns:
            ranks = pd.to_numeric(outcomes_h["selector_rank"], errors="coerce")
            for n in (5, 10, 25):
                values = outcomes_h.loc[ranks <= n, "qqq_excess_return"]
                if not values.empty:
                    summary_rows.append(
                        summary_row(
                            run_id=run_id,
                            score_date=str(score_date.date()),
                            horizon=int(horizon),
                            group=f"selector_rank_top{n}",
                            values=values,
                        )
                    )

        for n in (5, 10, 25, 100):
            group_symbols = universe.sort_values("rank").head(n)["symbol"].tolist()
            values = ret.reindex(group_symbols) - qqq_ret if qqq_ret is not None else pd.Series(dtype=float)
            summary_rows.append(
                summary_row(
                    run_id=run_id,
                    score_date=str(score_date.date()),
                    horizon=int(horizon),
                    group=f"rank_top{n}",
                    values=values,
                )
            )
        for bucket in ("top_1pct", "top_5pct", "middle", "lower"):
            group_symbols = universe.loc[universe["score_bucket"] == bucket, "symbol"].tolist()
            values = ret.reindex(group_symbols) - qqq_ret if qqq_ret is not None else pd.Series(dtype=float)
            summary_rows.append(
                summary_row(
                    run_id=run_id,
                    score_date=str(score_date.date()),
                    horizon=int(horizon),
                    group=f"bucket_{bucket}",
                    values=values,
                )
            )
        if "human_status" in outcomes_h.columns:
            for status, part in outcomes_h.dropna(subset=["human_status"]).groupby("human_status"):
                text = str(status).strip()
                if text:
                    summary_rows.append(
                        summary_row(
                            run_id=run_id,
                            score_date=str(score_date.date()),
                            horizon=int(horizon),
                            group=f"human_{text}",
                            values=part["qqq_excess_return"],
                        )
                    )

    outcome_df = pd.DataFrame(outcome_rows)
    summary_df = pd.DataFrame(summary_rows)
    info = {
        "run_id": run_id,
        "score_date": str(score_date.date()),
        "status": "SCORED",
        "completed_horizons": sorted(int(h) for h in horizon_map),
        "entry_exit_dates": {
            str(h): {"entry_date": str(dates[0].date()), "exit_date": str(dates[1].date())}
            for h, dates in horizon_map.items()
        },
    }
    return outcome_df, summary_df, info


def build_markdown_summary(summary: pd.DataFrame, infos: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Selector Performance",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Scored runs: {sum(1 for info in infos if info.get('status') == 'SCORED')}",
        f"- Pending/no completed horizons: {sum(1 for info in infos if info.get('status') != 'SCORED')}",
        "",
    ]
    if summary.empty:
        lines.append("No completed selector horizons were available.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Score date | Horizon | Group | Samples | Hit rate | Median | p10 | p75 | Median vs expected |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    base_groups = {"watchlist_top25", "watchlist_vs_matched_controls", "rank_top5", "rank_top10", "rank_top25", "rank_top100"}
    groups = summary["group"].astype(str)
    focus = summary[
        groups.isin(base_groups)
        | groups.str.startswith("action_")
        | groups.str.startswith("source_")
        | groups.str.startswith("category_")
        | groups.str.startswith("strict_")
        | groups.str.startswith("edge_")
        | groups.str.startswith("risk_")
        | groups.str.startswith("confidence_")
        | groups.str.startswith("utility_decile_")
        | groups.str.startswith("selector_rank_top")
    ]
    for _, row in focus.sort_values(["score_date", "horizon_days", "group"]).iterrows():
        lines.append(
            "| {date} | {h} | {group} | {n} | {hit} | {median} | {p10} | {p75} | {gap} |".format(
                date=row.get("score_date"),
                h=int(row.get("horizon_days")),
                group=row.get("group"),
                n=int(row.get("sample_count") or 0),
                hit="" if finite_or_none(row.get("hit_rate")) is None else f"{float(row['hit_rate']) * 100:.1f}%",
                median="" if finite_or_none(row.get("median")) is None else f"{float(row['median']) * 100:.2f}%",
                p10="" if finite_or_none(row.get("p10")) is None else f"{float(row['p10']) * 100:.2f}%",
                p75="" if finite_or_none(row.get("p75")) is None else f"{float(row['p75']) * 100:.2f}%",
                gap="" if finite_or_none(row.get("median_minus_expected")) is None else f"{float(row['median_minus_expected']) * 100:.2f}%",
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score frozen stock selector reports after outcomes are available.")
    p.add_argument("--registry", default="artifacts/stock_selector/registry.jsonl")
    p.add_argument("--manifest", action="append", default=[], help="Explicit run_manifest.json path; repeatable")
    p.add_argument("--provider_uri", default="", help="Optional provider override")
    p.add_argument("--benchmark_pkl", default="", help="Optional QQQ benchmark return pickle override")
    p.add_argument("--tickers_csv", default="/root/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--as_of", default="", help="Only score exits on or before this date; default latest local calendar")
    p.add_argument("--n_controls", type=int, default=10)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def parse_as_of(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text or text.lower() in {"latest", "local_latest", "max"}:
        return None
    return pd.Timestamp(text).normalize()


def main() -> int:
    args = parse_args()
    registry_rows = read_jsonl(Path(args.registry))
    manifest_paths = unique_manifest_paths(registry_rows, args.manifest)
    if not manifest_paths:
        print("No selector manifests found.", file=sys.stderr)
        return 2
    provider_override = Path(args.provider_uri).expanduser().resolve() if args.provider_uri else None
    benchmark_override = Path(args.benchmark_pkl).expanduser().resolve() if args.benchmark_pkl else None
    tickers_csv = Path(args.tickers_csv).expanduser().resolve()
    horizons = parse_int_list(args.horizons)
    as_of = parse_as_of(args.as_of)

    all_outcomes: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    infos: List[Dict[str, Any]] = []
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            infos.append({"manifest_json": str(manifest_path), "status": "MISSING_MANIFEST"})
            continue
        outcomes, summary, info = score_manifest(
            manifest_path,
            provider_uri_override=provider_override,
            benchmark_pkl_override=benchmark_override,
            tickers_csv=tickers_csv,
            horizons=horizons,
            as_of=as_of,
            n_controls=int(args.n_controls),
        )
        info["manifest_json"] = str(manifest_path)
        infos.append(info)
        if not outcomes.empty:
            all_outcomes.append(outcomes)
        if not summary.empty:
            all_summaries.append(summary)

    outcomes_df = pd.concat(all_outcomes, ignore_index=True) if all_outcomes else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_csv = out_dir / "selector_outcomes.csv"
    performance_csv = out_dir / "selector_performance.csv"
    performance_json = out_dir / "selector_performance.json"
    performance_md = out_dir / "selector_performance.md"
    outcomes_df.to_csv(outcomes_csv, index=False)
    summary_df.to_csv(performance_csv, index=False)
    performance_json.write_text(
        json.dumps(
            {
                "generated_utc": utc_now(),
                "infos": json_safe(infos),
                "summary": json_safe(summary_df.to_dict("records")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    performance_md.write_text(build_markdown_summary(summary_df, infos), encoding="utf-8")
    print(f"runs={len(infos)} scored={sum(1 for info in infos if info.get('status') == 'SCORED')}")
    print(f"outcomes_csv={outcomes_csv}")
    print(f"performance_csv={performance_csv}")
    print(f"performance_md={performance_md}")
    print(f"performance_json={performance_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
