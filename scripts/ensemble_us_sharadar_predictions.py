#!/usr/bin/env python
"""Build a release-testable ensemble pred.pkl from two US Sharadar prediction artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _read_pickle_compat(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if not str(getattr(e, "name", "")).startswith("numpy._core"):
            raise
        import numpy as np

        sys.modules.setdefault("numpy._core", np.core)
        for name in ("numeric", "multiarray", "umath"):
            sys.modules.setdefault(f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}"))
        return pd.read_pickle(path)


def load_prediction(path: Path) -> pd.DataFrame:
    pred = _read_pickle_compat(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"prediction artifact must be a DataFrame or Series: {path}")
    if not isinstance(pred.index, pd.MultiIndex) or set(pred.index.names) != {"datetime", "instrument"}:
        raise ValueError("prediction index must be a MultiIndex with datetime and instrument levels")
    if "score" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "score"})
    out = pred[["score"]].copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["score"])
    dates = pd.DatetimeIndex(out.index.get_level_values("datetime")).normalize()
    inst = out.index.get_level_values("instrument").astype(str).str.upper()
    out.index = pd.MultiIndex.from_arrays([dates, inst], names=["datetime", "instrument"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def daily_score_std(pred: pd.DataFrame) -> pd.Series:
    if pred.empty:
        return pd.Series(dtype=float, name="primary_std")
    stat = pred["score"].groupby(level="datetime").std(ddof=0)
    stat.name = "primary_std"
    return stat.sort_index()


def load_return_series(path: Optional[Path]) -> pd.Series:
    if path is None:
        return pd.Series(dtype=float)
    obj = _read_pickle_compat(path)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"benchmark return pkl must be a Series or single-column DataFrame: {path}")
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise TypeError(f"benchmark return pkl must be a pandas Series: {path}")
    ret = pd.to_numeric(obj, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ret.index = pd.DatetimeIndex(ret.index).normalize()
    return ret[~ret.index.duplicated(keep="last")].sort_index().dropna()


def confidence_gate_frame(
    primary: pd.DataFrame,
    *,
    min_history_days: int = 252,
    min_std_ratio: Optional[float] = 0.25,
    min_abs_std: Optional[float] = None,
) -> pd.DataFrame:
    """Flag dates where the primary model's score dispersion has collapsed.

    The reference is shifted by one date, so the current date never contributes
    to its own release/fallback decision.
    """
    stat = daily_score_std(primary)
    if stat.empty:
        return pd.DataFrame(columns=["primary_std", "reference_std", "std_ratio", "fallback"])
    min_periods = max(1, int(min_history_days))
    reference = stat.shift(1).expanding(min_periods=min_periods).median()
    fallback = pd.Series(False, index=stat.index)
    if min_std_ratio is not None:
        ratio_threshold = reference * float(min_std_ratio)
        fallback = fallback | (reference.notna() & (stat < ratio_threshold))
    if min_abs_std is not None:
        fallback = fallback | (stat < float(min_abs_std))
    out = pd.DataFrame(
        {
            "primary_std": stat,
            "reference_std": reference,
            "std_ratio": stat / reference.replace(0.0, np.nan),
            "fallback": fallback.astype(bool),
        }
    )
    return out


def regime_confidence_gate_frame(
    primary: pd.DataFrame,
    *,
    benchmark_returns: Optional[pd.Series] = None,
    min_history_days: int = 252,
    min_std_ratio: Optional[float] = 0.25,
    min_abs_std: Optional[float] = None,
    trend_window: int = 63,
    min_trend_return: Optional[float] = -0.08,
    drawdown_window: int = 126,
    max_drawdown: Optional[float] = -0.12,
    vol_window: int = 20,
    max_ann_vol: Optional[float] = 0.35,
) -> pd.DataFrame:
    gate = confidence_gate_frame(
        primary,
        min_history_days=min_history_days,
        min_std_ratio=min_std_ratio,
        min_abs_std=min_abs_std,
    )
    if gate.empty:
        return gate
    gate = gate.copy()
    gate["confidence_fallback"] = gate["fallback"].astype(bool)
    if benchmark_returns is None or benchmark_returns.empty:
        gate["trend_return"] = np.nan
        gate["drawdown"] = np.nan
        gate["ann_vol"] = np.nan
        gate["regime_fallback"] = False
        return gate

    ret = pd.to_numeric(benchmark_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ret.index = pd.DatetimeIndex(ret.index).normalize()
    ret = ret[~ret.index.duplicated(keep="last")].sort_index()
    equity = (1.0 + ret).cumprod()
    shifted_equity = equity.shift(1)
    shifted_ret = ret.shift(1)

    trend = shifted_equity / shifted_equity.shift(max(1, int(trend_window))) - 1.0
    rolling_peak = shifted_equity.rolling(max(2, int(drawdown_window)), min_periods=max(2, min(20, int(drawdown_window)))).max()
    drawdown = shifted_equity / rolling_peak - 1.0
    ann_vol = shifted_ret.rolling(max(2, int(vol_window)), min_periods=max(2, min(10, int(vol_window)))).std(ddof=0) * np.sqrt(252.0)

    gate["trend_return"] = trend.reindex(gate.index)
    gate["drawdown"] = drawdown.reindex(gate.index)
    gate["ann_vol"] = ann_vol.reindex(gate.index)
    regime = pd.Series(False, index=gate.index)
    if min_trend_return is not None:
        regime = regime | (gate["trend_return"].notna() & (gate["trend_return"] <= float(min_trend_return)))
    if max_drawdown is not None:
        regime = regime | (gate["drawdown"].notna() & (gate["drawdown"] <= float(max_drawdown)))
    if max_ann_vol is not None:
        regime = regime | (gate["ann_vol"].notna() & (gate["ann_vol"] >= float(max_ann_vol)))
    gate["regime_fallback"] = regime.astype(bool)
    gate["fallback"] = (gate["confidence_fallback"] | gate["regime_fallback"]).astype(bool)
    return gate


def _normalize_scores(pred: pd.DataFrame, method: str) -> pd.Series:
    score = pred["score"].astype(float)
    if method == "none":
        return score

    def zscore(s: pd.Series) -> pd.Series:
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            return pd.Series(0.0, index=s.index)
        return (s - float(s.mean())) / std

    if method == "zscore":
        return score.groupby(level="datetime", group_keys=False).apply(zscore)
    if method == "rank_zscore":
        ranked = score.groupby(level="datetime", group_keys=False).rank(method="average", pct=True)
        return ranked.groupby(level="datetime", group_keys=False).apply(zscore)
    raise ValueError(f"unknown normalization method: {method}")


def combine_predictions(
    primary: pd.DataFrame,
    defensive: pd.DataFrame,
    *,
    gate: Optional[pd.DataFrame] = None,
    normal_primary_weight: float = 1.0,
    normal_defensive_weight: float = 0.0,
    fallback_primary_weight: float = 0.0,
    fallback_defensive_weight: float = 1.0,
    normalize: str = "none",
) -> pd.DataFrame:
    if float(normal_primary_weight) + float(normal_defensive_weight) <= 0:
        raise ValueError("normal weights must have positive total")
    if float(fallback_primary_weight) + float(fallback_defensive_weight) <= 0:
        raise ValueError("fallback weights must have positive total")

    p = _normalize_scores(primary, normalize)
    d = _normalize_scores(defensive, normalize)
    idx = p.index.union(d.index)
    p = p.reindex(idx)
    d = d.reindex(idx)
    dates = pd.DatetimeIndex(idx.get_level_values("datetime")).normalize()
    if gate is None or gate.empty:
        fallback_by_date = pd.Series(False, index=pd.DatetimeIndex(sorted(set(dates))))
    else:
        fallback_by_date = gate["fallback"].astype(bool)
        fallback_by_date.index = pd.DatetimeIndex(fallback_by_date.index).normalize()
    use_fallback = pd.Series(dates, index=idx).map(fallback_by_date).fillna(False).astype(bool)

    wp = pd.Series(np.where(use_fallback, float(fallback_primary_weight), float(normal_primary_weight)), index=idx)
    wd = pd.Series(np.where(use_fallback, float(fallback_defensive_weight), float(normal_defensive_weight)), index=idx)
    p_ok = p.notna()
    d_ok = d.notna()
    denom = wp.where(p_ok, 0.0) + wd.where(d_ok, 0.0)
    score = (wp * p.fillna(0.0) + wd * d.fillna(0.0)) / denom.replace(0.0, np.nan)
    out = score.dropna().to_frame("score").sort_index()
    out.index = pd.MultiIndex.from_tuples(out.index, names=["datetime", "instrument"])
    return out


def _date_summary(pred: pd.DataFrame) -> Dict[str, object]:
    if pred.empty:
        return {"rows": 0, "dates": 0}
    dates = pd.DatetimeIndex(pred.index.get_level_values("datetime")).normalize()
    return {
        "rows": int(len(pred)),
        "dates": int(dates.nunique()),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend two pred.pkl files with an optional confidence fallback gate.")
    p.add_argument("--primary_pred", required=True, help="High-return or preferred pred.pkl")
    p.add_argument("--defensive_pred", required=True, help="Defensive fallback pred.pkl")
    p.add_argument("--out", required=True, help="Output ensemble pred.pkl")
    p.add_argument("--method", choices=["confidence_gate", "regime_confidence_gate", "static_blend"], default="confidence_gate")
    p.add_argument("--normal_primary_weight", type=float, default=1.0)
    p.add_argument("--normal_defensive_weight", type=float, default=0.0)
    p.add_argument("--fallback_primary_weight", type=float, default=0.0)
    p.add_argument("--fallback_defensive_weight", type=float, default=1.0)
    p.add_argument("--normalize", choices=["none", "zscore", "rank_zscore"], default="none")
    p.add_argument("--min_history_days", type=int, default=252)
    p.add_argument("--min_std_ratio", type=float, default=0.25)
    p.add_argument("--min_abs_std", type=float, default=None)
    p.add_argument("--regime_benchmark_pkl", default="", help="Optional benchmark return Series pkl for regime_confidence_gate")
    p.add_argument("--regime_trend_window", type=int, default=63)
    p.add_argument("--regime_min_trend_return", type=float, default=-0.08)
    p.add_argument("--regime_drawdown_window", type=int, default=126)
    p.add_argument("--regime_max_drawdown", type=float, default=-0.12)
    p.add_argument("--regime_vol_window", type=int, default=20)
    p.add_argument("--regime_max_ann_vol", type=float, default=0.35)
    p.add_argument("--manifest_out", default="")
    p.add_argument("--gate_report_csv", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    primary_path = Path(args.primary_pred).expanduser().resolve()
    defensive_path = Path(args.defensive_pred).expanduser().resolve()
    if not primary_path.exists():
        print(f"primary prediction not found: {primary_path}", file=sys.stderr)
        return 2
    if not defensive_path.exists():
        print(f"defensive prediction not found: {defensive_path}", file=sys.stderr)
        return 2

    primary = load_prediction(primary_path)
    defensive = load_prediction(defensive_path)
    gate = None
    if args.method == "confidence_gate":
        gate = confidence_gate_frame(
            primary,
            min_history_days=int(args.min_history_days),
            min_std_ratio=float(args.min_std_ratio) if args.min_std_ratio is not None else None,
            min_abs_std=args.min_abs_std,
        )
    elif args.method == "regime_confidence_gate":
        regime_benchmark = (
            load_return_series(Path(args.regime_benchmark_pkl).expanduser().resolve())
            if str(args.regime_benchmark_pkl or "").strip()
            else pd.Series(dtype=float)
        )
        gate = regime_confidence_gate_frame(
            primary,
            benchmark_returns=regime_benchmark,
            min_history_days=int(args.min_history_days),
            min_std_ratio=float(args.min_std_ratio) if args.min_std_ratio is not None else None,
            min_abs_std=args.min_abs_std,
            trend_window=int(args.regime_trend_window),
            min_trend_return=float(args.regime_min_trend_return) if args.regime_min_trend_return is not None else None,
            drawdown_window=int(args.regime_drawdown_window),
            max_drawdown=float(args.regime_max_drawdown) if args.regime_max_drawdown is not None else None,
            vol_window=int(args.regime_vol_window),
            max_ann_vol=float(args.regime_max_ann_vol) if args.regime_max_ann_vol is not None else None,
        )
    combined = combine_predictions(
        primary,
        defensive,
        gate=gate,
        normal_primary_weight=float(args.normal_primary_weight),
        normal_defensive_weight=float(args.normal_defensive_weight),
        fallback_primary_weight=float(args.fallback_primary_weight),
        fallback_defensive_weight=float(args.fallback_defensive_weight),
        normalize=str(args.normalize),
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(out_path)

    manifest = {
        "method": str(args.method),
        "primary_pred": str(primary_path),
        "defensive_pred": str(defensive_path),
        "out": str(out_path),
        "normalize": str(args.normalize),
        "weights": {
            "normal_primary_weight": float(args.normal_primary_weight),
            "normal_defensive_weight": float(args.normal_defensive_weight),
            "fallback_primary_weight": float(args.fallback_primary_weight),
            "fallback_defensive_weight": float(args.fallback_defensive_weight),
        },
        "confidence_gate": {
            "min_history_days": int(args.min_history_days),
            "min_std_ratio": float(args.min_std_ratio) if args.min_std_ratio is not None else None,
            "min_abs_std": float(args.min_abs_std) if args.min_abs_std is not None else None,
        },
        "regime_gate": {
            "benchmark_pkl": str(Path(args.regime_benchmark_pkl).expanduser().resolve()) if str(args.regime_benchmark_pkl or "").strip() else "",
            "trend_window": int(args.regime_trend_window),
            "min_trend_return": float(args.regime_min_trend_return) if args.regime_min_trend_return is not None else None,
            "drawdown_window": int(args.regime_drawdown_window),
            "max_drawdown": float(args.regime_max_drawdown) if args.regime_max_drawdown is not None else None,
            "vol_window": int(args.regime_vol_window),
            "max_ann_vol": float(args.regime_max_ann_vol) if args.regime_max_ann_vol is not None else None,
        },
        "primary": _date_summary(primary),
        "defensive": _date_summary(defensive),
        "combined": _date_summary(combined),
    }
    if gate is not None and not gate.empty:
        fallback_dates = gate.index[gate["fallback"].astype(bool)].tolist()
        manifest["confidence_gate"].update(
            {
                "fallback_dates": int(len(fallback_dates)),
                "fallback_date_ratio": float(len(fallback_dates) / len(gate)),
                "fallback_start": str(pd.Timestamp(fallback_dates[0]).date()) if fallback_dates else None,
                "fallback_end": str(pd.Timestamp(fallback_dates[-1]).date()) if fallback_dates else None,
            }
        )
        if "confidence_fallback" in gate.columns:
            manifest["confidence_gate"]["confidence_fallback_dates"] = int(gate["confidence_fallback"].astype(bool).sum())
        if "regime_fallback" in gate.columns:
            manifest["regime_gate"]["regime_fallback_dates"] = int(gate["regime_fallback"].astype(bool).sum())
        if args.gate_report_csv:
            gate_path = Path(args.gate_report_csv).expanduser().resolve()
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate.to_csv(gate_path)
            manifest["confidence_gate"]["gate_report_csv"] = str(gate_path)

    if args.manifest_out:
        manifest_path = Path(args.manifest_out).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "saved={out} rows={rows} dates={dates} fallback_dates={fallback}".format(
            out=out_path,
            rows=manifest["combined"]["rows"],
            dates=manifest["combined"]["dates"],
            fallback=manifest.get("confidence_gate", {}).get("fallback_dates", 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
