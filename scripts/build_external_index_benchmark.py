#!/usr/bin/env python
"""Build external index benchmark return pickles for validation baselines."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


TICKER_ALIASES: Dict[str, str] = {
    "ICIC": "IXIC",
    "^IXIC": "IXIC",
    "NASDAQ": "IXIC",
    "NASDAQCOMPOSITE": "IXIC",
}

NASDAQ_SYMBOLS: Dict[str, str] = {
    "IXIC": "COMP",
}


def canonical_ticker(ticker: str) -> str:
    raw = str(ticker).strip().upper()
    return TICKER_ALIASES.get(raw, raw)


def _clean_price(value) -> float:
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def returns_from_close(close: pd.Series, *, name: str) -> pd.Series:
    px = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    px.index = pd.DatetimeIndex(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index().dropna()
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret.name = name
    return ret


def load_nasdaq_index_close(ticker: str, *, start: str, end: str, timeout: int = 30) -> pd.Series:
    canonical = canonical_ticker(ticker)
    symbol = NASDAQ_SYMBOLS.get(canonical, canonical)
    url = (
        "https://api.nasdaq.com/api/quote/"
        f"{symbol}/historical?assetclass=index&fromdate={start}&todate={end}&limit=9999"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows") or [])
    if not rows:
        return pd.Series(dtype=float, name=canonical)
    frame = pd.DataFrame(rows)
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"Nasdaq payload missing date/close fields for {canonical}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    close = frame["close"].map(_clean_price)
    out = pd.Series(close.to_numpy(dtype=float), index=dates, name=canonical)
    out = out[~out.index.isna()]
    return out.sort_index()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build external index benchmark return pkl.")
    p.add_argument("--ticker", default="IXIC")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--source", choices=["nasdaq"], default="nasdaq")
    p.add_argument("--out", default="/root/.qlib/qlib_data/us_data/bench_ixic.pkl")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ticker = canonical_ticker(args.ticker)
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    if args.source != "nasdaq":
        print(f"unsupported source: {args.source}", file=sys.stderr)
        return 2
    close = load_nasdaq_index_close(ticker, start=str(args.start), end=str(end))
    returns = returns_from_close(close, name=f"bench_{ticker.lower()}")
    if returns.empty:
        print(f"ERROR: no benchmark returns built for {ticker}", file=sys.stderr)
        return 3
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    returns.to_pickle(out)
    idx = pd.DatetimeIndex(returns.index)
    print(f"saved={out} ticker={ticker} rows={len(returns)} span={idx.min().date()}->{idx.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
