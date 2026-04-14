#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _detect_value_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in exclude]
    out = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            out.append(col)
    return out


def _resolve_inputs(in_path: Path, input_pattern: str) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        files = sorted([p for p in in_path.glob(input_pattern) if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No input files matched pattern '{input_pattern}' in {in_path}")
        return files
    raise FileNotFoundError(f"Input path not found: {in_path}")


def _parse_windows(raw: str) -> List[int]:
    windows = [int(x) for x in _parse_csv_list(raw)]
    windows = [w for w in windows if w > 0]
    if not windows:
        raise ValueError("windows must contain at least one positive integer")
    return windows


def _build_features_for_ticker(
    s: pd.DataFrame,
    ticker: str,
    date_col: str,
    ticker_col: str,
    value_cols: List[str],
    windows: List[int],
    prefix: str,
    aggregation_mode: str,
    winsor_clip: float,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> pd.DataFrame:
    if range_end < range_start:
        raise ValueError(f"range_end {range_end} < range_start {range_start}")

    range_start = pd.Timestamp(range_start)
    range_end = pd.Timestamp(range_end)

    s = s.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])

    s = s.set_index(date_col).sort_index()
    # Snapshot tables should carry-forward the last known observation into the start of the resample window.
    # Without this, limiting output to a narrow window can incorrectly reset snapshots to 0 at range_start.
    if aggregation_mode == "snapshot" and not s.empty:
        try:
            last = s.loc[:range_start].tail(1)
            if not last.empty and range_start not in s.index:
                s.loc[range_start] = last.iloc[0]
                s = s.sort_index()
        except Exception:
            pass

    idx_dates = pd.date_range(range_start, range_end, freq="D")
    s = s.reindex(idx_dates)
    s.index.name = "date"
    s[ticker_col] = ticker

    if aggregation_mode == "snapshot":
        # Snapshot tables (e.g., 13F holdings) should carry forward the last observation.
        s[value_cols] = s[value_cols].ffill().fillna(0.0)
    else:
        # Event tables should be zero on non-event days.
        s[value_cols] = s[value_cols].fillna(0.0)

    s["event_count"] = s["event_count"].fillna(0).astype("int32")

    if winsor_clip and winsor_clip > 0:
        clipv = float(winsor_clip)
        for col in value_cols:
            s[col] = s[col].clip(lower=-clipv, upper=clipv)

    out = pd.DataFrame(index=s.index)
    for col in value_cols:
        base = s[col].astype("float32")
        out[f"{prefix}_{col}_daily"] = base
        for w in windows:
            if aggregation_mode == "snapshot":
                lag = base.shift(w)
                pct = (base / (lag + 1e-12) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                out[f"{prefix}_{col}_{w}d_mean"] = base.rolling(w, min_periods=1).mean().astype("float32")
                out[f"{prefix}_{col}_{w}d_chg"] = (base - lag).fillna(0.0).astype("float32")
                out[f"{prefix}_{col}_{w}d_pct"] = pct.astype("float32")
            else:
                out[f"{prefix}_{col}_{w}d_sum"] = base.rolling(w, min_periods=1).sum().astype("float32")
                out[f"{prefix}_{col}_{w}d_mean"] = base.rolling(w, min_periods=1).mean().astype("float32")

    ec = s["event_count"].astype("float32")
    out[f"{prefix}_count_daily"] = ec
    for w in windows:
        out[f"{prefix}_count_{w}d_sum"] = ec.rolling(w, min_periods=1).sum().astype("float32")

    out = out.reset_index().rename(columns={"date": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize Sharadar event tables into daily per-ticker feature CSVs.")
    p.add_argument("--input", required=True, help="Input table CSV path OR directory of per-ticker CSVs")
    p.add_argument("--input_pattern", default="*.csv", help="Glob pattern when --input is a directory")
    p.add_argument("--out_dir", required=True, help="Output directory of per-ticker feature CSVs")
    p.add_argument("--ticker_col", default="ticker", help="Ticker column name")
    p.add_argument("--date_col", default="date", help="Event date column name")
    p.add_argument(
        "--value_cols",
        default="",
        help="Comma-separated numeric value columns. If empty, auto-detect numeric-like columns.",
    )
    p.add_argument(
        "--windows",
        default="5,20,63",
        help="Comma-separated rolling windows in calendar days",
    )
    p.add_argument(
        "--aggregation_mode",
        default="event",
        choices=["event", "snapshot"],
        help="event: zero-fill non-event days; snapshot: forward-fill reported values",
    )
    p.add_argument("--prefix", default="event", help="Feature name prefix")
    p.add_argument("--start", default="", help="Optional start date filter YYYY-MM-DD")
    p.add_argument("--end", default="", help="Optional end date filter YYYY-MM-DD")
    p.add_argument(
        "--resample_start",
        default="",
        help="Optional daily output start date YYYY-MM-DD (defaults to --start or first observed date)",
    )
    p.add_argument(
        "--resample_end",
        default="",
        help="Optional daily output end date YYYY-MM-DD (defaults to --end or last observed date)",
    )
    p.add_argument(
        "--winsor_clip",
        type=float,
        default=0.0,
        help="Optional absolute clip value for raw daily numeric values before rolling (0 to disable)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    input_files = _resolve_inputs(in_path, args.input_pattern)

    value_cols = _parse_csv_list(args.value_cols)
    if not value_cols:
        for fp in input_files:
            df_probe = pd.read_csv(fp, low_memory=False)
            if df_probe.empty:
                continue
            probe_exclude = [args.ticker_col, args.date_col]
            value_cols = _detect_value_columns(df_probe, exclude=probe_exclude)
            if value_cols:
                break
    if not value_cols:
        raise ValueError("No numeric-like value columns found; pass --value_cols explicitly")
    logger.info(f"Using value_cols: {value_cols}")

    windows = _parse_windows(args.windows)

    total_rows = 0
    total_files = 0
    written_tickers = 0
    logger.info(f"Preparing event features from {len(input_files)} input file(s)")

    for fidx, fp in enumerate(input_files, 1):
        df = pd.read_csv(fp, low_memory=False)
        if df.empty:
            continue

        if args.ticker_col not in df.columns and in_path.is_dir():
            df[args.ticker_col] = fp.stem.upper()
        if args.ticker_col not in df.columns:
            raise ValueError(f"ticker_col '{args.ticker_col}' not found in {fp}")
        if args.date_col not in df.columns:
            raise ValueError(f"date_col '{args.date_col}' not found in {fp}")

        for col in value_cols:
            if col not in df.columns:
                df[col] = np.nan

        df[args.ticker_col] = df[args.ticker_col].astype(str).str.upper().str.strip()
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.dropna(subset=[args.ticker_col, args.date_col])
        df = df[df[args.ticker_col] != ""]

        if args.start:
            df = df[df[args.date_col] >= pd.Timestamp(args.start)]
        if args.end:
            df = df[df[args.date_col] <= pd.Timestamp(args.end)]
        if df.empty:
            continue

        df = _coerce_numeric(df, value_cols)
        df = df.dropna(subset=value_cols, how="all")
        if df.empty:
            continue

        daily = df.groupby([args.ticker_col, args.date_col], as_index=False)[value_cols].sum()
        count_df = (
            df.groupby([args.ticker_col, args.date_col], as_index=False)
            .size()
            .rename(columns={"size": "event_count"})
        )
        daily = daily.merge(count_df, on=[args.ticker_col, args.date_col], how="left")
        daily["event_count"] = daily["event_count"].fillna(0).astype("int32")

        tickers = sorted(daily[args.ticker_col].unique())
        for ticker in tickers:
            s = daily[daily[args.ticker_col] == ticker].copy().sort_values(args.date_col)
            if s.empty:
                continue

            range_start = pd.Timestamp(args.resample_start) if args.resample_start else (
                pd.Timestamp(args.start) if args.start else pd.Timestamp(s[args.date_col].min())
            )
            range_end = pd.Timestamp(args.resample_end) if args.resample_end else (
                pd.Timestamp(args.end) if args.end else pd.Timestamp(s[args.date_col].max())
            )
            out = _build_features_for_ticker(
                s=s,
                ticker=ticker,
                date_col=args.date_col,
                ticker_col=args.ticker_col,
                value_cols=value_cols,
                windows=windows,
                prefix=args.prefix,
                aggregation_mode=args.aggregation_mode,
                winsor_clip=args.winsor_clip,
                range_start=range_start,
                range_end=range_end,
            )
            out_path = out_dir / f"{ticker}.csv"
            out.to_csv(out_path, index=False)
            total_rows += len(out)
            written_tickers += 1

        total_files += 1
        if fidx % 200 == 0:
            logger.info(f"Progress: files {fidx}/{len(input_files)}")

    if written_tickers == 0:
        raise ValueError("No output was written. Check date/value column filters and input data.")
    logger.info(f"Done. files={total_files}, tickers={written_tickers}, rows={total_rows}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
