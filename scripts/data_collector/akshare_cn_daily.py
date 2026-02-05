#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Download CN A-share daily data from Akshare and prepare CSVs for Qlib dump.

This script:
1) Reads a CSI500 instruments file (point-in-time membership history).
2) Expands to the unique stock list overlapping a date window.
3) Downloads raw daily OHLCV + amount from Akshare.
4) Computes vwap and an adjustment factor (qfq_close / raw_close).
5) Saves one CSV per symbol (e.g., SH600000.csv).

NOTE:
- Akshare does not provide official adjustment factors. The factor here is
  derived from qfq prices and therefore uses full-history adjustment.
  This is NOT strictly point-in-time and may introduce look-ahead. If you
  need strict point-in-time adjustments, use a corporate-action data source.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import akshare as ak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CN daily data from Akshare for Qlib.")
    parser.add_argument("--instruments_path", required=True, help="Path to csi500 instruments file (tab-separated).")
    parser.add_argument("--start_date", required=True, help="Start date, e.g. 20210201.")
    parser.add_argument("--end_date", required=True, help="End date, e.g. 20260131.")
    parser.add_argument("--out_dir", required=True, help="Output directory for per-symbol CSV files.")
    parser.add_argument("--max_workers", type=int, default=4, help="Max concurrent downloads.")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds).")
    parser.add_argument(
        "--source",
        choices=["em", "tx"],
        default="em",
        help="Data source: em (Eastmoney, stock_zh_a_hist) or tx (Tencent, stock_zh_a_hist_tx).",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per symbol on network errors.")
    parser.add_argument("--retry_delay", type=float, default=1.0, help="Base delay between retries (seconds).")
    parser.add_argument(
        "--factor_mode",
        choices=["qfq", "hfq", "none"],
        default="qfq",
        help="Adjustment mode for factor calculation (factor = adj_close/raw_close).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols (for testing).")
    parser.add_argument("--force", action="store_true", help="Re-download even if output CSV exists.")
    return parser.parse_args()


def load_symbols(instruments_path: Path, start_date: str, end_date: str) -> List[str]:
    df = pd.read_csv(
        instruments_path,
        sep="\t",
        header=None,
        names=["symbol", "start_date", "end_date"],
        dtype={"symbol": str},
    )
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df["start_date"] <= end_dt) & (df["end_date"] >= start_dt)
    symbols = df.loc[mask, "symbol"].dropna().unique().tolist()
    symbols = sorted(set(symbols))
    return symbols


def _rename_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=rename_map)
    return df


def _fetch_em(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    code = symbol[2:] if symbol.startswith(("SH", "SZ")) else symbol
    return ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )


def _fetch_tx(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    code = symbol.lower()
    if code.startswith("sh") or code.startswith("sz"):
        pass
    else:
        prefix = "sh" if symbol.upper().startswith("SH") else "sz"
        code = f"{prefix}{symbol[2:]}"
    return ak.stock_zh_a_hist_tx(
        symbol=code,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )


def fetch_symbol_data(
    symbol: str, start_date: str, end_date: str, factor_mode: str, source: str
) -> Optional[pd.DataFrame]:
    if source == "em":
        raw = _fetch_em(symbol, start_date, end_date, adjust="")
    else:
        raw = _fetch_tx(symbol, start_date, end_date, adjust="")
    if raw is None or raw.empty:
        return None

    if source == "em":
        raw = _rename_raw_columns(raw)
        raw = raw[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        # EM volume is in hands (100 shares). Convert to shares.
        raw["volume"] = raw["volume"] * 100.0
    else:
        # TX columns: date, open, close, high, low, amount (amount == volume in hands)
        raw = raw.rename(columns={"amount": "volume"})
        raw = raw[["date", "open", "high", "low", "close", "volume"]].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw["volume"] = raw["volume"] * 100.0  # hands -> shares
        raw["amount"] = raw["close"] * raw["volume"]

    factor = None
    if factor_mode != "none":
        if source == "em":
            adj = _fetch_em(symbol, start_date, end_date, adjust=factor_mode)
        else:
            adj = _fetch_tx(symbol, start_date, end_date, adjust=factor_mode)
        if adj is not None and not adj.empty:
            if source == "em":
                adj = _rename_raw_columns(adj)
                adj = adj[["date", "close"]].copy()
                adj["date"] = pd.to_datetime(adj["date"])
            else:
                adj = adj[["date", "close"]].copy()
                adj["date"] = pd.to_datetime(adj["date"])
            factor = adj.merge(raw[["date", "close"]], on="date", how="inner", suffixes=("_adj", "_raw"))
            factor["factor"] = factor["close_adj"] / factor["close_raw"]
            factor = factor[["date", "factor"]]

    df = raw
    if factor is not None:
        df = df.merge(factor, on="date", how="left")
    else:
        df["factor"] = 1.0

    # vwap
    df["vwap"] = df["amount"] / df["volume"]
    df.loc[df["volume"].isna() | (df["volume"] <= 0), "vwap"] = df["close"]

    # clean
    df["factor"] = df["factor"].replace([pd.NA, pd.NaT], 1.0).astype(float)
    df = df.sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def save_symbol_csv(df: pd.DataFrame, out_dir: Path, symbol: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.upper()}.csv"
    df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    instruments_path = Path(args.instruments_path).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    symbols = load_symbols(instruments_path, args.start_date, args.end_date)
    if args.limit is not None:
        symbols = symbols[: args.limit]
    if not symbols:
        raise SystemExit("No symbols found for the given window.")

    def _task(sym: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        out_path = out_dir / f"{sym.upper()}.csv"
        if out_path.exists() and not args.force:
            return sym, None, "skip"
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                df = fetch_symbol_data(sym, args.start_date, args.end_date, args.factor_mode, args.source)
                return sym, df, None
            except Exception as e:
                last_err = e
                time.sleep(args.retry_delay * attempt)
        return sym, None, str(last_err)

    errors = {}
    err_path = out_dir / "errors.txt"
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_task, sym) for sym in symbols]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            sym, df, err = fut.result()
            if err == "skip":
                continue
            if err is not None:
                errors[sym] = err
                with err_path.open("a") as fp:
                    fp.write(f"{sym}\t{err}\n")
                continue
            if df is None or df.empty:
                errors[sym] = "empty"
                with err_path.open("a") as fp:
                    fp.write(f"{sym}\tempty\n")
                continue
            save_symbol_csv(df, out_dir, sym)
            time.sleep(args.delay)

    if errors:
        print(f"Completed with {len(errors)} errors. See {err_path}")
    else:
        print("Completed without errors.")


if __name__ == "__main__":
    main()
