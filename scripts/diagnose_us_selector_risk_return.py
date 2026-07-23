#!/usr/bin/env python
"""Diagnose whether selector predictions produce return edge by risk tier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_us_stock_selector_report import (  # noqa: E402
    build_calibration_frame,
    finite_or_none,
    load_benchmark,
    load_prediction,
    normalize_feature_frame,
    summarize_returns,
)
from scripts.validate_us_sharadar_pipeline import _load_yaml  # noqa: E402


RISK_FIELDS = ["$risk_beta_qqq_63d", "$risk_vol_20d"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def score_band(score_percentile: Any) -> str:
    pct = finite_or_none(score_percentile)
    if pct is None:
        return "unknown"
    if pct >= 0.99:
        return "top_1pct"
    if pct >= 0.975:
        return "top_2_5pct"
    if pct >= 0.95:
        return "top_5pct"
    if pct >= 0.90:
        return "top_10pct"
    if pct >= 0.70:
        return "upper_mid"
    if pct >= 0.30:
        return "middle"
    return "lower"


def risk_tier(
    beta: Any,
    vol: Any,
    *,
    low_beta_max: float = 1.05,
    low_vol_max: float = 0.35,
    medium_beta_max: float = 1.50,
    medium_vol_max: float = 0.60,
) -> str:
    b = finite_or_none(beta)
    v = finite_or_none(vol)
    if b is None or v is None:
        return "unknown"
    if b <= low_beta_max and v <= low_vol_max:
        return "low"
    if b <= medium_beta_max and v <= medium_vol_max:
        return "medium"
    return "high"


def load_risk_features(
    *,
    frame: pd.DataFrame,
    provider_uri: Path,
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    instruments = sorted(frame.index.get_level_values("instrument").unique().astype(str))
    start = pd.Timestamp(frame.index.get_level_values("datetime").min()).normalize()
    end = pd.Timestamp(frame.index.get_level_values("datetime").max()).normalize()
    raw = D.features(instruments, RISK_FIELDS, start_time=start, end_time=end)
    risk = normalize_feature_frame(raw, RISK_FIELDS).reindex(frame.index)
    return risk.rename(columns={"$risk_beta_qqq_63d": "beta_qqq_63d", "$risk_vol_20d": "vol_20d"})


def summarize_group(group: pd.DataFrame, return_col: str) -> Dict[str, Any]:
    stats = summarize_returns(pd.to_numeric(group[return_col], errors="coerce"))
    stats["avg_beta_qqq_63d"] = finite_or_none(pd.to_numeric(group.get("beta_qqq_63d"), errors="coerce").mean())
    stats["avg_vol_20d"] = finite_or_none(pd.to_numeric(group.get("vol_20d"), errors="coerce").mean())
    stats["avg_score_percentile"] = finite_or_none(pd.to_numeric(group.get("score_percentile"), errors="coerce").mean())
    return stats


def build_summary(
    frame: pd.DataFrame,
    *,
    horizon: int,
    group_cols: Sequence[str],
    min_samples: int,
) -> pd.DataFrame:
    return_col = f"excess_return_{int(horizon)}d"
    rows = []
    for keys, group in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(summarize_group(group, return_col))
        if int(row.get("sample_count") or 0) >= int(min_samples):
            rows.append(row)
    return pd.DataFrame(rows)


def add_period_summary(
    frame: pd.DataFrame,
    *,
    horizon: int,
    min_samples: int,
) -> pd.DataFrame:
    return_col = f"excess_return_{int(horizon)}d"
    rows = []
    for (score, risk, quarter), group in frame.groupby(["score_band", "risk_tier", "quarter"], dropna=False):
        stats = summarize_returns(pd.to_numeric(group[return_col], errors="coerce"))
        if int(stats.get("sample_count") or 0) < int(min_samples):
            continue
        rows.append({"score_band": score, "risk_tier": risk, "quarter": quarter, **stats})
    return pd.DataFrame(rows)


def format_pct(value: Any, digits: int = 2) -> str:
    val = finite_or_none(value)
    return "" if val is None else f"{val * 100:.{digits}f}%"


def build_markdown(
    *,
    metadata: Mapping[str, Any],
    score_summary: pd.DataFrame,
    score_risk_summary: pd.DataFrame,
    quarterly_summary: pd.DataFrame,
    horizon: int,
) -> str:
    lines = [
        "# Selector Risk/Return Diagnostics",
        "",
        f"- Generated UTC: {metadata.get('generated_utc')}",
        f"- Prediction file: `{metadata.get('pred')}`",
        f"- Score date: {metadata.get('score_date')}",
        f"- Horizon: {int(horizon)} trading days QQQ excess",
        "",
        "## Score Buckets",
        "",
        "| Score bucket | Samples | Hit rate | Median | Mean | p10 | p75 | Avg beta | Avg vol |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not score_summary.empty:
        for _, row in score_summary.iterrows():
            lines.append(
                "| {bucket} | {n} | {hit} | {median} | {mean} | {p10} | {p75} | {beta} | {vol} |".format(
                    bucket=row.get("score_band"),
                    n=int(row.get("sample_count") or 0),
                    hit=format_pct(row.get("hit_rate")),
                    median=format_pct(row.get("median")),
                    mean=format_pct(row.get("mean")),
                    p10=format_pct(row.get("p10")),
                    p75=format_pct(row.get("p75")),
                    beta="" if finite_or_none(row.get("avg_beta_qqq_63d")) is None else f"{float(row['avg_beta_qqq_63d']):.2f}",
                    vol=format_pct(row.get("avg_vol_20d")),
                )
            )
    lines.extend(
        [
            "",
            "## Score X Risk",
            "",
            "| Score bucket | Risk tier | Samples | Hit rate | Median | Mean | p10 | p75 | Avg beta | Avg vol |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not score_risk_summary.empty:
        for _, row in score_risk_summary.iterrows():
            lines.append(
                "| {bucket} | {risk} | {n} | {hit} | {median} | {mean} | {p10} | {p75} | {beta} | {vol} |".format(
                    bucket=row.get("score_band"),
                    risk=row.get("risk_tier"),
                    n=int(row.get("sample_count") or 0),
                    hit=format_pct(row.get("hit_rate")),
                    median=format_pct(row.get("median")),
                    mean=format_pct(row.get("mean")),
                    p10=format_pct(row.get("p10")),
                    p75=format_pct(row.get("p75")),
                    beta="" if finite_or_none(row.get("avg_beta_qqq_63d")) is None else f"{float(row['avg_beta_qqq_63d']):.2f}",
                    vol=format_pct(row.get("avg_vol_20d")),
                )
            )
    lines.extend(["", "## Rolling Stability", ""])
    if quarterly_summary.empty:
        lines.append("No quarterly rows met the sample threshold.")
    else:
        roll = (
            quarterly_summary.groupby(["score_band", "risk_tier"])
            .agg(
                quarters=("quarter", "count"),
                positive_median_rate=("median", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
                worst_median=("median", "min"),
                latest_quarter=("quarter", "max"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "| Score bucket | Risk tier | Quarters | Positive median rate | Worst median | Latest quarter |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in roll.iterrows():
            lines.append(
                "| {bucket} | {risk} | {q} | {pos} | {worst} | {latest} |".format(
                    bucket=row.get("score_band"),
                    risk=row.get("risk_tier"),
                    q=int(row.get("quarters") or 0),
                    pos=format_pct(row.get("positive_median_rate")),
                    worst=format_pct(row.get("worst_median")),
                    latest=row.get("latest_quarter"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def run_diagnostics(
    *,
    config: Path,
    pred: Path,
    provider_uri: Path,
    benchmark_pkl: Path,
    score_date: str,
    horizon: int,
    lookback_days: int,
    min_samples: int,
) -> Dict[str, Any]:
    cfg = _load_yaml(config)
    predictions = load_prediction(pred)
    benchmark = load_benchmark(benchmark_pkl)
    as_of = pd.Timestamp(score_date).normalize() if score_date and str(score_date).lower() != "latest" else pd.Timestamp(
        predictions.index.get_level_values("datetime").max()
    ).normalize()
    frame = build_calibration_frame(
        predictions,
        score_date=as_of,
        cfg=cfg,
        provider_uri=provider_uri,
        benchmark=benchmark,
        horizons=[int(horizon)],
        lookback_days=int(lookback_days),
    )
    risk = load_risk_features(frame=frame, provider_uri=provider_uri)
    frame = frame.join(risk)
    frame["score_band"] = frame["score_percentile"].map(score_band)
    frame["risk_tier"] = [
        risk_tier(beta, vol) for beta, vol in zip(frame["beta_qqq_63d"], frame["vol_20d"])
    ]
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime")).normalize()
    frame["quarter"] = pd.PeriodIndex(dates, freq="Q").astype(str)

    score_summary = build_summary(frame, horizon=horizon, group_cols=["score_band"], min_samples=min_samples)
    score_risk_summary = build_summary(
        frame,
        horizon=horizon,
        group_cols=["score_band", "risk_tier"],
        min_samples=min_samples,
    )
    quarterly_summary = add_period_summary(frame, horizon=horizon, min_samples=max(20, min_samples // 10))
    metadata = {
        "generated_utc": utc_now(),
        "config": str(config),
        "pred": str(pred),
        "provider_uri": str(provider_uri),
        "benchmark_pkl": str(benchmark_pkl),
        "score_date": str(as_of.date()),
        "horizon": int(horizon),
        "lookback_days": int(lookback_days),
        "rows": int(len(frame)),
    }
    return {
        "metadata": metadata,
        "score_summary": score_summary,
        "score_risk_summary": score_risk_summary,
        "quarterly_summary": quarterly_summary,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose selector return edge by score bucket and risk tier.")
    p.add_argument("--config", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--benchmark_pkl", default="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
    p.add_argument("--score_date", default="latest")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--lookback_days", type=int, default=756)
    p.add_argument("--min_samples", type=int, default=300)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_diagnostics(
        config=Path(args.config).expanduser().resolve(),
        pred=Path(args.pred).expanduser().resolve(),
        provider_uri=Path(args.provider_uri).expanduser().resolve(),
        benchmark_pkl=Path(args.benchmark_pkl).expanduser().resolve(),
        score_date=str(args.score_date),
        horizon=int(args.horizon),
        lookback_days=int(args.lookback_days),
        min_samples=int(args.min_samples),
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    score_csv = out_dir / "score_bucket_summary.csv"
    score_risk_csv = out_dir / "score_risk_summary.csv"
    quarterly_csv = out_dir / "quarterly_score_risk_summary.csv"
    md_path = out_dir / "risk_return_diagnostics.md"
    json_path = out_dir / "risk_return_diagnostics.json"
    result["score_summary"].to_csv(score_csv, index=False)
    result["score_risk_summary"].to_csv(score_risk_csv, index=False)
    result["quarterly_summary"].to_csv(quarterly_csv, index=False)
    md_path.write_text(
        build_markdown(
            metadata=result["metadata"],
            score_summary=result["score_summary"],
            score_risk_summary=result["score_risk_summary"],
            quarterly_summary=result["quarterly_summary"],
            horizon=int(args.horizon),
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "metadata": result["metadata"],
                "files": {
                    "score_bucket_summary_csv": str(score_csv),
                    "score_risk_summary_csv": str(score_risk_csv),
                    "quarterly_score_risk_summary_csv": str(quarterly_csv),
                    "markdown": str(md_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"rows={result['metadata']['rows']}")
    print(f"score_bucket_summary_csv={score_csv}")
    print(f"score_risk_summary_csv={score_risk_csv}")
    print(f"quarterly_score_risk_summary_csv={quarterly_csv}")
    print(f"markdown={md_path}")
    print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
