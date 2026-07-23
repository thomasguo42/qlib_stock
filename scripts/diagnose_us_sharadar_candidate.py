#!/usr/bin/env python
import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_us_sharadar_pipeline import (  # noqa: E402
    _benchmark_forward_return,
    _best_metric_iteration,
    _infer_label_ref_start_days,
    _parse_label_horizon,
    _read_pickle_compat,
)


DIRECT_RISK_ALPHA_NAMES = {
    "RISK_VOL_20D",
    "RISK_VOL_63D",
    "RISK_DVOL_20D",
    "RISK_BETA_SPY_63D",
    "RISK_BETA_SPY_252D",
}

MOMENTUM_NAMES = {
    "RISK_RET_20D",
    "RISK_RET_63D",
    "RISK_RET_252D",
    "RISK_RELRET_SPY_63D",
    "RISK_RELRET_SPY_252D",
}

QUALITY_VALUE_NAMES = {
    "ROE_Q",
    "ROA_Q",
    "EBITDA_MARGIN_Q",
    "FCF_MARGIN_Q",
    "LEVERAGE_Q",
    "CASH_ASSETS_Q",
    "CAPEX_ASSETS_Q",
    "ASSET_TURN_Q",
    "DIV_YIELD_PX_Q",
    "EARN_YIELD_Q",
    "BOOK_PX_Q",
    "FCF_YIELD_Q",
}

SIZE_NAMES = {
    "MARKETCAP_Q",
    "LOG_MARKETCAP_Q",
}

DEFAULT_ATTRIBUTION_WINDOWS = [
    ("2023_h1", "2023-01-04", "2023-07-06"),
    ("2024_h1", "2024-01-05", "2024-07-08"),
    ("2025_apr_oct", "2025-04-09", "2025-10-08"),
    ("2026_ytd", "2026-01-09", "2026-05-04"),
]


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML structure: {path}")
    return data


def _processor_class(proc: Dict) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def _handler_config(cfg: Dict) -> Dict:
    return cfg.get("data_handler_config") or cfg["task"]["dataset"]["kwargs"]["handler"].get("kwargs", {})


def _model_summary(cfg: Dict) -> Dict[str, object]:
    model = cfg.get("task", {}).get("model", {}) or {}
    kwargs = model.get("kwargs", {}) or {}
    dh = _handler_config(cfg)
    processors = dh.get("learn_processors", []) or []
    infer_processors = dh.get("infer_processors", []) or []
    learn_classes = [_processor_class(p) for p in processors if isinstance(p, dict)]
    infer_classes = [_processor_class(p) for p in infer_processors if isinstance(p, dict)]
    fields = [str(x) for x in dh.get("extra_fields", []) or []]
    names = [str(x) for x in dh.get("extra_names", []) or []]
    if not names and fields:
        names = [field.lstrip("$").upper() for field in fields]
    return {
        "model_class": model.get("class"),
        "objective": kwargs.get("objective", kwargs.get("loss")),
        "uses_rank_label_processor": "CSRankNorm" in learn_classes,
        "uses_true_ranker": str(model.get("class")) == "LGBRankerModel",
        "uses_selective_feature_norm": "SelectiveCSZScoreNorm" in infer_classes,
        "uses_plain_cszscore": "CSZScoreNorm" in infer_classes,
        "extra_field_count": len(fields),
        "market_field_count": sum(field.startswith("$mkt_") for field in fields),
        "sf3a_field_count": sum("inst13f" in field.lower() for field in fields),
        "regime_interaction_count": sum(field.startswith("$regime_") for field in fields),
        "direct_risk_alpha_field_count": sum(name in DIRECT_RISK_ALPHA_NAMES for name in names),
        "momentum_field_count": sum(name in MOMENTUM_NAMES for name in names),
        "quality_value_field_count": sum(name in QUALITY_VALUE_NAMES for name in names),
        "size_field_count": sum(name in SIZE_NAMES for name in names),
    }


def _best_iterations(manifest_path: Optional[Path], mlruns_root: Path) -> List[Dict[str, object]]:
    if manifest_path is None or not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest.get("used_runs", []) or []:
        run_id = str(item.get("run_id", ""))
        matches = list(mlruns_root.glob(f"*/{run_id}"))
        best_iter = None
        best_value = None
        metric_points = 0
        metric_name = None
        if matches:
            best = _best_metric_iteration(matches[0])
            if best is not None:
                metric_name, best_iter, best_value, metric_points, _ = best
        rows.append(
            {
                "run_id": run_id,
                "test": item.get("test"),
                "metric_name": metric_name,
                "metric_points": metric_points,
                "best_iteration": best_iter,
                "best_valid_metric": best_value,
            }
        )
    return rows


def _weekly_signal_dates(pred: pd.Series) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values("datetime").unique()))
    trade_dates = []
    seen = set()
    for dt in dates:
        iso = pd.Timestamp(dt).isocalendar()
        key = (int(iso.year), int(iso.week))
        if key not in seen:
            seen.add(key)
            trade_dates.append(pd.Timestamp(dt))
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    return pd.DatetimeIndex([prev[dt] for dt in trade_dates if dt in prev])


def _topk_rows(pred: pd.Series, topk: int) -> pd.DataFrame:
    rows = []
    for dt in _weekly_signal_dates(pred):
        if dt not in pred.index.get_level_values("datetime"):
            continue
        scores = pred.xs(dt, level="datetime").dropna().sort_values(ascending=False).head(int(topk))
        rows.extend((dt, inst, rank, float(score)) for rank, (inst, score) in enumerate(scores.items(), start=1))
    return pd.DataFrame(rows, columns=["datetime", "instrument", "rank", "score"])


def _feature_group(name: str) -> str:
    upper = str(name).upper()
    if upper in DIRECT_RISK_ALPHA_NAMES:
        return "direct_risk_alpha"
    if upper in MOMENTUM_NAMES:
        return "momentum"
    if upper in QUALITY_VALUE_NAMES:
        return "quality_value"
    if upper.startswith("MKT_"):
        return "market_regime"
    if upper.startswith("REGIME_"):
        return "regime_interaction"
    if upper.startswith("INST13F_"):
        return "sf3a"
    if upper.startswith("ALPHA158_"):
        return "alpha158"
    return "other"


def _handler_uses_alpha158(cfg: Dict) -> bool:
    handler = cfg.get("task", {}).get("dataset", {}).get("kwargs", {}).get("handler", {}) or {}
    return "Alpha158" in str(handler.get("class", ""))


def _extra_name_for_feature(feature: str, cfg: Dict) -> str:
    match = re.fullmatch(r"Column_(\d+)", str(feature))
    if not match:
        return str(feature)
    col = int(match.group(1))
    offset = 158 if _handler_uses_alpha158(cfg) else 0
    extra_names = [str(x) for x in _handler_config(cfg).get("extra_names", []) or []]
    idx = col - offset
    if 0 <= idx < len(extra_names):
        return extra_names[idx]
    return f"ALPHA158_{col}" if offset and col < offset else str(feature)


def _model_feature_importance(model_pkl: Path, cfg: Dict, *, topn: int = 20) -> Dict[str, object]:
    if not model_pkl.exists():
        return {"error": f"missing model_pkl: {model_pkl}"}
    try:
        with model_pkl.open("rb") as f:
            model = pickle.load(f)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    booster = getattr(model, "model", None)
    if booster is None or not hasattr(booster, "feature_name") or not hasattr(booster, "feature_importance"):
        return {"error": "model does not expose a LightGBM booster"}
    raw_names = list(booster.feature_name())
    names = [_extra_name_for_feature(name, cfg) for name in raw_names]
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    total = float(np.nansum(gains))
    if not math.isfinite(total) or total <= 0:
        return {"total_gain": 0.0, "top_features": [], "group_gain_share": {}}
    rows = []
    group_gain: Dict[str, float] = {}
    for name, gain in zip(names, gains):
        share = float(gain / total)
        group = _feature_group(name)
        group_gain[group] = group_gain.get(group, 0.0) + share
        rows.append({"feature": name, "group": group, "gain_share": share})
    rows.sort(key=lambda row: row["gain_share"], reverse=True)
    return {
        "total_gain": total,
        "top_features": rows[: int(topn)],
        "group_gain_share": dict(sorted(group_gain.items(), key=lambda item: item[1], reverse=True)),
    }


def _weekly_topk_overlap(rows: pd.DataFrame) -> Optional[float]:
    if rows.empty:
        return None
    overlaps = []
    prev = None
    for _, group in rows.sort_values(["datetime", "rank"]).groupby("datetime", sort=True):
        cur = set(group["instrument"].astype(str))
        if prev is not None and cur:
            overlaps.append(len(prev & cur) / float(len(cur)))
        prev = cur
    return float(np.mean(overlaps)) if overlaps else None


def _parse_attribution_windows(values: Iterable[str]) -> List[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    for raw in values or []:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            name, start, end = text.split(":", 2)
            out.append((name.strip(), pd.Timestamp(start), pd.Timestamp(end)))
        except Exception as exc:
            raise ValueError(f"invalid attribution window '{text}', expected name:YYYY-MM-DD:YYYY-MM-DD") from exc
    return out


def _failed_windows_from_validation_log(path: Path) -> List[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if not path.exists():
        raise FileNotFoundError(f"validation log not found: {path}")
    windows = []
    seen = set()
    pattern = re.compile(r"(?P<start>20\d\d-\d\d-\d\d)->(?P<end>20\d\d-\d\d-\d\d)(?:\s+\(\d+d\))?\s*\|\s*FAIL\b")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        start = pd.Timestamp(match.group("start"))
        end = pd.Timestamp(match.group("end"))
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        windows.append((f"failed_rolling_{len(windows) + 1}", start, end))
    return windows


def _annualized_return_from_daily_returns(returns: pd.Series) -> Optional[float]:
    data = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return None
    total = float((1.0 + data).prod() - 1.0)
    if len(data) <= 0:
        return None
    return float((1.0 + total) ** (252.0 / float(len(data))) - 1.0)


def _regime_state_share(frame: pd.DataFrame) -> Dict[str, float]:
    required = {"mkt_qqq_ret_63d_lag1", "mkt_qqq_ret_20d_lag1", "mkt_qqq_dd_126d_lag1"}
    if not required.issubset(frame.columns):
        return {}
    trend = pd.to_numeric(frame["mkt_qqq_ret_63d_lag1"], errors="coerce")
    fast = pd.to_numeric(frame["mkt_qqq_ret_20d_lag1"], errors="coerce")
    drawdown = pd.to_numeric(frame["mkt_qqq_dd_126d_lag1"], errors="coerce")
    breadth = (
        pd.to_numeric(frame["mkt_breadth_ret63_pos_lag1"], errors="coerce")
        if "mkt_breadth_ret63_pos_lag1" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    state = pd.Series("chop", index=frame.index, dtype=object)
    risk_off = (trend <= -0.04) | (drawdown <= -0.10) | ((breadth < 0.45) & (trend < 0.0))
    risk_on = (trend >= 0.03) & (drawdown > -0.06)
    fading = ((trend > 0.0) & (fast <= -0.02)) | ((drawdown <= -0.06) & ~risk_off)
    recovery = risk_off & (fast >= 0.02)
    state.loc[risk_on.fillna(False)] = "risk_on"
    state.loc[fading.fillna(False)] = "fading"
    state.loc[risk_off.fillna(False)] = "risk_off"
    state.loc[recovery.fillna(False)] = "recovery"
    return _json_float_dict(state.value_counts(normalize=True).to_dict())


def _window_attribution(
    top: pd.DataFrame,
    *,
    feature_cols: List[str],
    exposure_cols: List[str],
    market_cols: Optional[List[str]] = None,
    benchmark_returns: pd.Series,
    windows: List[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, start, end in windows:
        mask = (top["datetime"] >= start) & (top["datetime"] <= end)
        part = top.loc[mask].copy()
        bench_part = benchmark_returns.loc[
            (pd.DatetimeIndex(benchmark_returns.index) >= start) & (pd.DatetimeIndex(benchmark_returns.index) <= end)
        ]
        if part.empty:
            out[name] = {
                "start": str(start.date()),
                "end": str(end.date()),
                "topk_rows": 0,
            }
            continue
        row = {
            "start": str(start.date()),
            "end": str(end.date()),
            "topk_rows": int(len(part)),
            "signal_dates": int(part["datetime"].nunique()),
            "mean_excess_label": float(part["excess_label"].mean()),
            "median_excess_label": float(part["excess_label"].median()),
            "positive_excess_label_rate": float((part["excess_label"] > 0).mean()),
            "benchmark_ann_return": _annualized_return_from_daily_returns(bench_part),
        }
        if "score" in part.columns:
            row["mean_score"] = float(pd.to_numeric(part["score"], errors="coerce").mean())
        feature_means = part[exposure_cols].mean(numeric_only=True).to_dict() if exposure_cols else {}
        row["feature_percentile_mean"] = _json_float_dict(feature_means)
        row["feature_percentile_delta"] = _json_float_dict({col: val - 0.5 for col, val in feature_means.items()})
        if market_cols:
            cols = [col for col in market_cols if col in part.columns]
            row["market_state_mean"] = _json_float_dict(part[cols].mean(numeric_only=True).to_dict()) if cols else {}
            row["regime_state_share"] = _regime_state_share(part)
        if "sector" in part.columns:
            row["top_sector_share"] = _json_float_dict(
                part["sector"].fillna("UNKNOWN").value_counts(normalize=True).head(8).to_dict()
            )
        out[name] = row
    return out


def _label_expr_from_config(cfg: Dict) -> str:
    label = _handler_config(cfg).get("label", [])
    try:
        return str(label[0][0])
    except Exception:
        return "Ref($close, -11)/Ref($close, -1) - 1"


def _strategy_kwargs(cfg: Dict) -> Dict:
    return cfg.get("port_analysis_config", {}).get("strategy", {}).get("kwargs", {}) or {}


def _feature_control_maps(cfg: Dict) -> tuple[Dict[str, float], Dict[str, float]]:
    kwargs = _strategy_kwargs(cfg)

    def normalize(raw, *, lower=None, upper=None) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out = {}
        for field, value in raw.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            if lower is not None:
                v = max(float(lower), v)
            if upper is not None:
                v = min(float(upper), v)
            out[str(field)] = v
        return out

    return (
        normalize(kwargs.get("feature_score_weights")),
        normalize(kwargs.get("feature_min_percentiles"), lower=0.0, upper=1.0),
    )


def _normalize_feature_panel(feat, fields: List[str]) -> pd.DataFrame:
    if isinstance(feat, pd.Series):
        feat = feat.to_frame(fields[0] if len(fields) == 1 else "feature")
    if not isinstance(feat, pd.DataFrame) or feat.empty:
        return pd.DataFrame()
    out = feat.copy()
    if len(fields) == 1 and fields[0] not in out.columns and out.shape[1] == 1:
        out = out.rename(columns={out.columns[0]: fields[0]})
    if isinstance(out.index, pd.MultiIndex) and list(out.index.names) == ["instrument", "datetime"]:
        out = out.reorder_levels(["datetime", "instrument"]).sort_index()
    cols = [field for field in fields if field in out.columns]
    return out[cols].sort_index() if cols else pd.DataFrame()


def _cs_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((numeric - float(numeric.mean())) / std).fillna(0.0).astype(float)


def _demote_masked_scores(scores: pd.Series, mask: pd.Series) -> pd.Series:
    adjusted = scores.copy()
    mask = mask.reindex(adjusted.index).fillna(False).astype(bool)
    if adjusted.empty or not bool(mask.any()):
        return adjusted
    finite = adjusted.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return adjusted
    spread = float(finite.max() - finite.min())
    if not math.isfinite(spread):
        spread = 1.0
    floor = float(finite.min() - max(1.0, spread))
    step = max(1e-9, max(1.0, spread) * 1e-9)
    for inst in adjusted.index[mask.to_numpy()]:
        adjusted.loc[inst] = floor
        floor -= step
    return adjusted


def _apply_strategy_feature_controls(pred: pd.Series, feat: pd.DataFrame, cfg: Dict) -> pd.Series:
    weights, mins = _feature_control_maps(cfg)
    if pred.empty or (not weights and not mins) or feat.empty:
        return pred
    panel = pd.DataFrame({"score": pred}).join(feat.reindex(pred.index), how="left")

    def adjust_day(group: pd.DataFrame) -> pd.Series:
        base = pd.to_numeric(group["score"], errors="coerce")
        adjusted = _cs_zscore(base) if weights else base.astype(float).copy()
        for field, weight in weights.items():
            if field in group.columns:
                adjusted = adjusted + float(weight) * _cs_zscore(group[field])
        for field, min_pct in mins.items():
            if field in group.columns:
                pct = pd.to_numeric(group[field], errors="coerce").rank(pct=True)
                adjusted = _demote_masked_scores(adjusted, pct.isna() | (pct < float(min_pct)))
        return adjusted.where(base.notna(), np.nan)

    adjusted = pd.concat([adjust_day(group) for _, group in panel.groupby(level="datetime", sort=True)]).sort_index()
    adjusted.name = pred.name
    return adjusted.reindex(pred.index)


def _topk_diagnostics(
    pred_path: Path,
    *,
    cfg: Dict,
    provider_uri: Path,
    benchmark_pkl: Path,
    tickers_csv: Optional[Path],
    topk: int,
    start: str,
    end: str,
    attribution_windows: Optional[List[tuple[str, pd.Timestamp, pd.Timestamp]]] = None,
) -> Dict[str, object]:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    pred_obj = _read_pickle_compat(pred_path)
    pred = pred_obj.iloc[:, 0] if isinstance(pred_obj, pd.DataFrame) else pred_obj
    pred = pred.sort_index()
    pred_dates = pd.DatetimeIndex(pred.index.get_level_values("datetime"))
    pred_in_range = pred[(pred_dates >= pd.Timestamp(start)) & (pred_dates <= pd.Timestamp(end))]
    if pred_in_range.empty:
        return {"topk_rows": 0}

    label_expr = _label_expr_from_config(cfg)
    label_horizon_days = _parse_label_horizon(label_expr) or 10
    label_ref_start_days = _infer_label_ref_start_days(_handler_config(cfg))
    weights, mins = _feature_control_maps(cfg)
    control_fields = list(dict.fromkeys([*weights.keys(), *mins.keys()]))
    fields = [
        "$risk_beta_spy_63d",
        "$risk_vol_20d",
        "$risk_ret_20d",
        "$risk_ret_252d",
        "$fcf_yield_q",
        "$earn_yield_q",
        "$book_px_q",
        "$div_yield_px_q",
        "$log_marketcap_q",
        "$mkt_qqq_ret_20d_lag1",
        "$mkt_qqq_ret_63d_lag1",
        "$mkt_qqq_vol_20d_lag1",
        "$mkt_qqq_dd_126d_lag1",
        "$mkt_breadth_ret63_pos_lag1",
        "$mkt_breadth_ret63_above_spy_lag1",
        "$mkt_dispersion_ret63_lag1",
        label_expr,
    ]
    fields = list(dict.fromkeys(fields + control_fields))
    instruments = sorted(pred_in_range.index.get_level_values("instrument").unique().tolist())
    feat = D.features(
        instruments,
        fields,
        start_time=pred_in_range.index.get_level_values("datetime").min(),
        end_time=pred_in_range.index.get_level_values("datetime").max(),
    )
    feat = _normalize_feature_panel(feat, fields)
    if control_fields:
        pred_in_range = _apply_strategy_feature_controls(pred_in_range, feat.reindex(columns=control_fields), cfg)
    rows = _topk_rows(pred_in_range, topk=topk)
    if rows.empty:
        return {"topk_rows": 0}
    feature_cols = [
        "risk_beta_spy_63d",
        "risk_vol_20d",
        "risk_ret_20d",
        "risk_ret_252d",
        "fcf_yield_q",
        "earn_yield_q",
        "book_px_q",
        "div_yield_px_q",
        "log_marketcap_q",
    ]
    feat = feat.rename(
        columns={
            "$risk_beta_spy_63d": "risk_beta_spy_63d",
            "$risk_vol_20d": "risk_vol_20d",
            "$risk_ret_20d": "risk_ret_20d",
            "$risk_ret_252d": "risk_ret_252d",
            "$fcf_yield_q": "fcf_yield_q",
            "$earn_yield_q": "earn_yield_q",
            "$book_px_q": "book_px_q",
            "$div_yield_px_q": "div_yield_px_q",
            "$log_marketcap_q": "log_marketcap_q",
            "$mkt_qqq_ret_20d_lag1": "mkt_qqq_ret_20d_lag1",
            "$mkt_qqq_ret_63d_lag1": "mkt_qqq_ret_63d_lag1",
            "$mkt_qqq_vol_20d_lag1": "mkt_qqq_vol_20d_lag1",
            "$mkt_qqq_dd_126d_lag1": "mkt_qqq_dd_126d_lag1",
            "$mkt_breadth_ret63_pos_lag1": "mkt_breadth_ret63_pos_lag1",
            "$mkt_breadth_ret63_above_spy_lag1": "mkt_breadth_ret63_above_spy_lag1",
            "$mkt_dispersion_ret63_lag1": "mkt_dispersion_ret63_lag1",
            label_expr: "raw_label",
        }
    )
    bench = _read_pickle_compat(benchmark_pkl)
    if isinstance(bench, pd.DataFrame):
        bench = bench.iloc[:, 0]
    fwd = _benchmark_forward_return(
        bench,
        label_horizon_days=label_horizon_days,
        label_ref_start_days=label_ref_start_days,
    )
    dts = pd.DatetimeIndex(feat.index.get_level_values("datetime"))
    feat["excess_label"] = feat["raw_label"].astype(float) - fwd.reindex(dts).to_numpy()
    ranked = feat[feature_cols].groupby(level="datetime").rank(pct=True)
    ranked.columns = [f"{col}_pct" for col in feature_cols]
    feat = feat.join(ranked)
    top = rows.join(feat.reindex(pd.MultiIndex.from_frame(rows[["datetime", "instrument"]])).reset_index(drop=True))
    top["year"] = top["datetime"].dt.year
    label_by_year = top.groupby("year")["excess_label"].agg(["count", "mean", "median"]).to_dict("index")
    exposure_cols = [f"{col}_pct" for col in feature_cols]
    exposure_by_year = top.groupby("year")[exposure_cols].mean().to_dict("index")
    exposure_delta_by_year = {
        year: {col: val - 0.5 for col, val in row.items()}
        for year, row in top.groupby("year")[exposure_cols].mean().to_dict("index").items()
    }

    sector_by_year = {}
    if tickers_csv is not None and tickers_csv.exists():
        meta = pd.read_csv(tickers_csv, usecols=lambda c: c in {"ticker", "sector", "scalemarketcap"}, low_memory=False)
        meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
        meta = meta.drop_duplicates("ticker", keep="last").set_index("ticker")
        top_meta = top.join(meta, on="instrument")
        for year, group in top_meta.groupby("year"):
            sector_by_year[str(int(year))] = (
                group["sector"].fillna("UNKNOWN").value_counts(normalize=True).head(8).to_dict()
            )
        top = top_meta

    windows = attribution_windows or [
        (name, pd.Timestamp(start_dt), pd.Timestamp(end_dt)) for name, start_dt, end_dt in DEFAULT_ATTRIBUTION_WINDOWS
    ]
    market_cols = [
        "mkt_qqq_ret_20d_lag1",
        "mkt_qqq_ret_63d_lag1",
        "mkt_qqq_vol_20d_lag1",
        "mkt_qqq_dd_126d_lag1",
        "mkt_breadth_ret63_pos_lag1",
        "mkt_breadth_ret63_above_spy_lag1",
        "mkt_dispersion_ret63_lag1",
    ]

    return {
        "topk_rows": int(len(top)),
        "label_horizon_days": int(label_horizon_days),
        "selection_feature_score_weights": weights,
        "selection_feature_min_percentiles": mins,
        "weekly_topk_overlap": _weekly_topk_overlap(rows),
        "topk_excess_label_by_year": _json_float_dict(label_by_year),
        "topk_feature_percentile_by_year": _json_float_dict(exposure_by_year),
        "topk_feature_percentile_delta_by_year": _json_float_dict(exposure_delta_by_year),
        "topk_sector_share_by_year": _json_float_dict(sector_by_year),
        "window_attribution": _window_attribution(
            top,
            feature_cols=feature_cols,
            exposure_cols=exposure_cols,
            market_cols=market_cols,
            benchmark_returns=bench,
            windows=windows,
        ),
    }


def _json_float_dict(obj):
    if isinstance(obj, dict):
        return {str(k): _json_float_dict(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a US Sharadar candidate for known release-readiness failure modes.")
    p.add_argument("--config", required=True)
    p.add_argument("--pred", default="")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--benchmark_pkl", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--model_pkl", default="")
    p.add_argument("--mlruns_root", default="/workspace/qlib/mlruns")
    p.add_argument("--tickers_csv", default="~/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument(
        "--attribution_window",
        action="append",
        default=[],
        help="Named slice as name:YYYY-MM-DD:YYYY-MM-DD. Repeatable. Defaults to known weak windows.",
    )
    p.add_argument(
        "--validation_log",
        default="",
        help="Optional validate_us_sharadar_pipeline log; failed rolling windows are added to attribution.",
    )
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    payload: Dict[str, object] = {
        "config": str(cfg_path),
        "model": _model_summary(cfg),
    }
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    payload["training"] = _best_iterations(manifest, Path(args.mlruns_root).expanduser().resolve())
    if args.model_pkl:
        payload["feature_importance"] = _model_feature_importance(
            Path(args.model_pkl).expanduser().resolve(),
            cfg,
        )
    if args.pred:
        benchmark = args.benchmark_pkl or _handler_config(cfg).get("learn_processors", [{}])[1].get("kwargs", {}).get("benchmark_pkl", "")
        attribution_windows = _parse_attribution_windows(args.attribution_window) if args.attribution_window else None
        failed_windows = (
            _failed_windows_from_validation_log(Path(args.validation_log).expanduser().resolve())
            if args.validation_log
            else []
        )
        if failed_windows:
            attribution_windows = (attribution_windows or []) + failed_windows
        payload["topk"] = _topk_diagnostics(
            Path(args.pred).expanduser().resolve(),
            cfg=cfg,
            provider_uri=Path(args.provider_uri).expanduser().resolve(),
            benchmark_pkl=Path(benchmark).expanduser().resolve(),
            tickers_csv=Path(args.tickers_csv).expanduser().resolve() if args.tickers_csv else None,
            topk=int(args.topk),
            start=args.start,
            end=args.end,
            attribution_windows=attribution_windows,
        )

    text = json.dumps(_json_float_dict(payload), indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
