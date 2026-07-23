#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TICKER_ALIASES = {
    "ICIC": "IXIC",
    "^IXIC": "IXIC",
    "NASDAQ": "IXIC",
    "NASDAQCOMPOSITE": "IXIC",
}


def _canonical_ticker(ticker: str) -> str:
    raw = str(ticker).strip().upper()
    return TICKER_ALIASES.get(raw, raw)


def _parse_tickers(value: str) -> list[str]:
    out = []
    seen = set()
    for item in str(value or "").replace(",", "\n").splitlines():
        ticker = _canonical_ticker(item)
        if not ticker or ticker in seen:
            continue
        out.append(ticker)
        seen.add(ticker)
    return out


def _extract_close_series(features, *, field: str = "$close") -> pd.Series:
    if isinstance(features, pd.Series):
        close = features.copy()
    elif isinstance(features, pd.DataFrame) and not features.empty:
        close = features[field].copy() if field in features.columns else features.iloc[:, 0].copy()
    else:
        return pd.Series(dtype=float)
    close = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if isinstance(close.index, pd.MultiIndex):
        names = list(close.index.names)
        if "datetime" in names:
            close.index = pd.DatetimeIndex(close.index.get_level_values("datetime")).normalize()
        else:
            close.index = pd.DatetimeIndex(close.index.get_level_values(0)).normalize()
        close = close.groupby(level=0).last()
    else:
        close.index = pd.DatetimeIndex(close.index).normalize()
    return close.sort_index()


def _returns_from_close(close: pd.Series, *, name: str) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float, name=name)
    px = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    px.index = pd.DatetimeIndex(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret.name = str(name)
    return ret


def build_qlib_ticker_benchmark(
    *,
    provider_uri: str,
    ticker: str,
    start_time: str,
    end_time: str,
    field: str = "$close",
) -> pd.Series:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(Path(provider_uri).expanduser().resolve()), region=REG_US)
    symbol = _canonical_ticker(ticker)
    features = D.features([symbol], [field], start_time=start_time, end_time=end_time)
    close = _extract_close_series(features, field=field)
    return _returns_from_close(close, name=f"bench_{symbol.lower()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build qlib benchmark return pkl files for one or more tickers.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--ticker", default="QQQ")
    p.add_argument("--tickers", default="", help="Optional comma-separated tickers; overrides --ticker")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--field", default="$close")
    p.add_argument("--out", default="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
    p.add_argument("--out_dir", default="", help="Output directory for --tickers mode; defaults to provider_uri")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    provider = Path(args.provider_uri).expanduser().resolve()
    if not provider.exists():
        print(f"ERROR: provider not found: {provider}", file=sys.stderr)
        return 2
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    tickers = _parse_tickers(args.tickers) if str(args.tickers or "").strip() else [_canonical_ticker(args.ticker)]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else provider
    failures = 0
    for ticker in tickers:
        returns = build_qlib_ticker_benchmark(
            provider_uri=str(provider),
            ticker=ticker,
            start_time=args.start,
            end_time=end,
            field=args.field,
        )
        if returns.empty:
            print(f"ERROR: no benchmark returns built for {ticker}", file=sys.stderr)
            failures += 1
            continue
        out = (
            Path(args.out).expanduser().resolve()
            if len(tickers) == 1 and not str(args.tickers or "").strip()
            else (out_dir / f"bench_{ticker.lower()}.pkl").resolve()
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        returns.to_pickle(out)
        idx = pd.DatetimeIndex(returns.index)
        print(f"saved={out} ticker={ticker} rows={len(returns)} span={idx.min().date()}->{idx.max().date()}")
    if failures:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
