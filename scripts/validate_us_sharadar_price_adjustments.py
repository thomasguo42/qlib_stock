#!/usr/bin/env python
"""Validate Qlib SEP prices against raw Sharadar adjusted closes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_market_tickers(
    provider_uri: Path,
    market: str,
    max_tickers: Optional[int],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> List[str]:
    inst_path = provider_uri / "instruments" / f"{market}.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"instrument file not found: {inst_path}")
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if start is not None or end is not None:
        inst_start = pd.to_datetime(df["start"], errors="coerce")
        inst_end = pd.to_datetime(df["end"], errors="coerce")
        left = pd.Timestamp.min if start is None else pd.Timestamp(start)
        right = pd.Timestamp.max if end is None else pd.Timestamp(end)
        overlap = (inst_end.isna() | (inst_end >= left)) & (inst_start.isna() | (inst_start <= right))
        df = df[overlap]
    tickers = sorted(df["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist())
    if max_tickers is not None and int(max_tickers) > 0:
        tickers = tickers[: int(max_tickers)]
    return tickers


def _read_raw_closeadj(path: Path, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, usecols=lambda c: c in {"date", "closeadj"}, low_memory=False)
    if df.empty or "date" not in df.columns or "closeadj" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["closeadj"] = pd.to_numeric(df["closeadj"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return pd.Series(df["closeadj"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), dtype=float)


def _normalize_qlib_close(close: pd.Series) -> pd.Series:
    out = close.copy()
    if isinstance(out.index, pd.MultiIndex):
        if "datetime" in out.index.names and "instrument" in out.index.names:
            out = out.reorder_levels(["datetime", "instrument"]).sort_index()
        else:
            out.index = out.index.set_names(["datetime", "instrument"])
            out = out.sort_index()
    return out.astype(float)


def _compare_symbol(
    symbol: str,
    raw_closeadj: pd.Series,
    qlib_close: pd.Series,
    *,
    max_relative_error: float,
    max_return_error: float,
    max_scale_change: float,
    jump_threshold: float,
    raw_jump_threshold: float,
) -> Dict[str, object]:
    common = raw_closeadj.rename("raw").to_frame().join(qlib_close.rename("qlib"), how="inner").dropna()
    if common.empty:
        return {
            "symbol": symbol,
            "common_days": 0,
            "bad_price_days": 0,
            "suspicious_jump_days": 0,
            "max_relative_error": math.nan,
            "max_qlib_abs_return": math.nan,
            "max_raw_abs_return": math.nan,
            "examples": [],
        }
    rel_err = (common["qlib"] / common["raw"] - 1.0).replace([np.inf, -np.inf], np.nan).abs()
    raw_ret = common["raw"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    qlib_ret = common["qlib"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return_err = (qlib_ret - raw_ret).abs().replace([np.inf, -np.inf], np.nan)
    scale = (common["qlib"] / common["raw"]).replace([np.inf, -np.inf], np.nan)
    scale_change = scale.pct_change(fill_method=None).abs().replace([np.inf, -np.inf], np.nan)
    jump_mask = (qlib_ret.abs() > float(jump_threshold)) & (raw_ret.abs() < float(raw_jump_threshold))
    return_mask = return_err > float(max_return_error)
    scale_jump_mask = (rel_err > float(max_relative_error)) & (scale_change > float(max_scale_change))
    failure_mask = return_mask | scale_jump_mask | jump_mask
    examples = []
    for dt in common.index[failure_mask][:5]:
        examples.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "raw_closeadj": float(common.loc[dt, "raw"]),
                "qlib_close": float(common.loc[dt, "qlib"]),
                "relative_error": float(rel_err.loc[dt]) if pd.notna(rel_err.loc[dt]) else math.nan,
                "return_abs_error": float(return_err.loc[dt]) if pd.notna(return_err.loc[dt]) else math.nan,
                "scale_change": float(scale_change.loc[dt]) if pd.notna(scale_change.loc[dt]) else math.nan,
                "raw_return": float(raw_ret.loc[dt]) if pd.notna(raw_ret.loc[dt]) else math.nan,
                "qlib_return": float(qlib_ret.loc[dt]) if pd.notna(qlib_ret.loc[dt]) else math.nan,
            }
        )
    return {
        "symbol": symbol,
        "common_days": int(len(common)),
        "bad_price_days": int((return_mask | scale_jump_mask).sum()),
        "suspicious_jump_days": int(jump_mask.sum()),
        "max_relative_error": float(rel_err.max()) if rel_err.notna().any() else math.nan,
        "max_return_error": float(return_err.max()) if return_err.notna().any() else math.nan,
        "max_scale_change": float(scale_change.max()) if scale_change.notna().any() else math.nan,
        "max_qlib_abs_return": float(qlib_ret.abs().max()) if qlib_ret.notna().any() else math.nan,
        "max_raw_abs_return": float(raw_ret.abs().max()) if raw_ret.notna().any() else math.nan,
        "examples": examples,
    }


def _load_qlib_close(
    symbols: List[str],
    *,
    provider_uri: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.Series:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_US)
    df = D.features(symbols, ["$close"], start_time=start, end_time=end, freq="day")
    close = df["$close"] if "$close" in df.columns else df.iloc[:, 0]
    return _normalize_qlib_close(close)


def validate_prices(
    *,
    provider_uri: Path,
    raw_sep_dir: Path,
    market: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_tickers: Optional[int],
    max_relative_error: float,
    max_return_error: float,
    max_scale_change: float,
    jump_threshold: float,
    raw_jump_threshold: float,
) -> Dict[str, object]:
    symbols = _read_market_tickers(provider_uri, market, max_tickers, start=start, end=end)
    qlib_close = _load_qlib_close(symbols, provider_uri=provider_uri, start=start, end=end)
    rows = []
    for symbol in symbols:
        raw = _read_raw_closeadj(raw_sep_dir / f"{symbol}.csv", start, end)
        try:
            q = qlib_close.xs(symbol, level="instrument")
        except Exception:
            q = pd.Series(dtype=float)
        rows.append(
            _compare_symbol(
                symbol,
                raw,
                q,
                max_relative_error=max_relative_error,
                max_return_error=max_return_error,
                max_scale_change=max_scale_change,
                jump_threshold=jump_threshold,
                raw_jump_threshold=raw_jump_threshold,
            )
        )
    bad = [r for r in rows if int(r["bad_price_days"]) > 0 or int(r["suspicious_jump_days"]) > 0]
    missing = [r for r in rows if int(r["common_days"]) == 0]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "provider_uri": str(provider_uri),
        "raw_sep_dir": str(raw_sep_dir),
        "market": market,
        "start": str(start.date()) if start is not None else None,
        "end": str(end.date()) if end is not None else None,
        "symbols_checked": int(len(symbols)),
        "symbols_with_failures": int(len(bad)),
        "symbols_without_common_days": int(len(missing)),
        "bad_price_days_total": int(sum(int(r["bad_price_days"]) for r in rows)),
        "suspicious_jump_days_total": int(sum(int(r["suspicious_jump_days"]) for r in rows)),
        "max_relative_error": float(np.nanmax([r["max_relative_error"] for r in rows])) if rows else math.nan,
        "max_return_error": float(np.nanmax([r["max_return_error"] for r in rows])) if rows else math.nan,
        "max_scale_change": float(np.nanmax([r["max_scale_change"] for r in rows])) if rows else math.nan,
        "failures": sorted(bad, key=lambda r: (int(r["suspicious_jump_days"]), int(r["bad_price_days"]), float(r["max_relative_error"] or 0.0)), reverse=True),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Qlib US prices against raw Sharadar closeadj.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--raw_sep_dir", default="~/.qlib/sharadar/raw/sep")
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--max_tickers", type=int, default=None)
    p.add_argument("--max_relative_error", type=float, default=1e-4)
    p.add_argument("--max_return_error", type=float, default=1e-3)
    p.add_argument("--max_scale_change", type=float, default=1e-3)
    p.add_argument("--jump_threshold", type=float, default=0.50)
    p.add_argument("--raw_jump_threshold", type=float, default=0.35)
    p.add_argument("--out_json", default="")
    p.add_argument("--fail_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = validate_prices(
        provider_uri=Path(args.provider_uri).expanduser().resolve(),
        raw_sep_dir=Path(args.raw_sep_dir).expanduser().resolve(),
        market=str(args.market),
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        max_tickers=args.max_tickers,
        max_relative_error=float(args.max_relative_error),
        max_return_error=float(args.max_return_error),
        max_scale_change=float(args.max_scale_change),
        jump_threshold=float(args.jump_threshold),
        raw_jump_threshold=float(args.raw_jump_threshold),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved_json={out}")
    print(
        "price_adjustment_check "
        f"symbols={result['symbols_checked']} failures={result['symbols_with_failures']} "
        f"missing_common_days={result['symbols_without_common_days']} "
        f"bad_price_days={result['bad_price_days_total']} suspicious_jumps={result['suspicious_jump_days_total']} "
        f"max_relative_error={result['max_relative_error']} "
        f"max_return_error={result['max_return_error']} max_scale_change={result['max_scale_change']}"
    )
    if result["failures"]:
        print(json.dumps(result["failures"][:10], indent=2, sort_keys=True))
    if args.fail_on_error and (result["symbols_with_failures"] or result["symbols_without_common_days"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
