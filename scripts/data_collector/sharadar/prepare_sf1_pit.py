#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger


DEFAULT_FIELDS: List[str] = [
    "assets",
    "liabilities",
    "equity",
    "revenue",
    "netinc",
    "ebit",
    "ebitda",
    "cashneq",
    "debt",
    "fcf",
    "capex",
    "workingcapital",
    "currentratio",
    "grossmargin",
    "netmargin",
    "roe",
    "roa",
    "roic",
    "eps",
    "epsdil",
    "bvps",
    "shareswa",
    "shareswadil",
    "divyield",
    "dps",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Sharadar SF1 into PIT-normalized CSVs per symbol.")
    parser.add_argument("--sf1", required=True, help="Path to SF1 CSV (e.g. ~/.qlib/sharadar/raw/sf1_MRQ.csv)")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for per-ticker normalized PIT CSVs",
    )
    parser.add_argument("--dimension", default="MRQ", help="SF1 dimension to keep (e.g. MRQ, MRY)")
    parser.add_argument(
        "--date_col",
        default="datekey",
        help="Publication date column to use for PIT (e.g. datekey, reportperiod, lastupdated)",
    )
    parser.add_argument(
        "--period_col",
        default="calendardate",
        help="Period end column to derive quarterly/annual period (e.g. calendardate, reportperiod)",
    )
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated list of SF1 fields to keep (default: curated set).",
    )
    parser.add_argument(
        "--exclude_fields",
        default="",
        help="Comma-separated list of SF1 fields to drop.",
    )
    parser.add_argument(
        "--date_offset_days",
        type=int,
        default=45,
        help="Offset in days to approximate filing delay (e.g. 45 for quarterly). Use 0 to disable.",
    )
    parser.add_argument("--chunksize", type=int, default=200000)
    return parser.parse_args()


def _as_int_date(series: pd.Series, offset_days: int = 0) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if offset_days:
        dt = dt + pd.to_timedelta(offset_days, unit="D")
    return (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("Int32")


def _as_period_int(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    quarter = ((dt.dt.month - 1) // 3 + 1).astype("Int32")
    return (dt.dt.year * 100 + quarter).astype("Int32")


def _parse_field_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    args = _parse_args()
    sf1_path = Path(args.sf1).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sf1_path.exists():
        raise FileNotFoundError(f"SF1 not found: {sf1_path}")

    fields = _parse_field_list(args.fields)
    if not fields:
        fields = list(DEFAULT_FIELDS)
    excludes = set(_parse_field_list(args.exclude_fields))
    if excludes:
        fields = [f for f in fields if f not in excludes]

    # read header to validate fields
    header = pd.read_csv(sf1_path, nrows=0)
    available = set(header.columns)
    missing = [f for f in fields if f not in available]
    if missing:
        logger.warning(f"Missing fields in SF1: {missing}")
    fields = [f for f in fields if f in available]
    if not fields:
        raise ValueError("No valid fields to extract.")

    usecols = ["ticker", "dimension", args.date_col, args.period_col] + fields
    logger.info(f"Reading SF1 with fields={len(fields)} chunksize={args.chunksize}")

    total_rows = 0
    for chunk in pd.read_csv(sf1_path, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        if "dimension" in chunk.columns:
            chunk = chunk[chunk["dimension"].astype(str).str.upper() == args.dimension.upper()]
        if chunk.empty:
            continue

        chunk = chunk.copy()
        chunk["date"] = _as_int_date(chunk[args.date_col], offset_days=args.date_offset_days)
        chunk["period"] = _as_period_int(chunk[args.period_col])
        chunk = chunk.dropna(subset=["date", "period", "ticker"])
        if chunk.empty:
            continue

        id_vars = ["ticker", "date", "period"]
        long = chunk[id_vars + fields].melt(id_vars=id_vars, var_name="field", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        if long.empty:
            continue

        for ticker, df_t in long.groupby("ticker"):
            out_path = out_dir / f"{str(ticker).upper()}.csv"
            header_needed = not out_path.exists()
            df_t.to_csv(out_path, mode="a", header=header_needed, index=False)
            total_rows += len(df_t)

    logger.info(f"Done. Wrote PIT normalized rows: {total_rows}")


if __name__ == "__main__":
    main()
