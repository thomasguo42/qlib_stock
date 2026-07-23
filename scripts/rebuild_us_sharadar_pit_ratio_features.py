#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROVENANCE_FILE = "sharadar_sf1_ratio_features.json"
RATIO_FIELDS = [
    "roe_q",
    "roa_q",
    "ebitda_margin_q",
    "fcf_margin_q",
    "leverage_q",
    "cash_assets_q",
    "capex_assets_q",
    "asset_turn_q",
    "div_yield_px_q",
    "earn_yield_q",
    "book_px_q",
    "fcf_yield_q",
    "marketcap_q",
    "log_marketcap_q",
]
SF1_FIELDS = [
    "assets",
    "equity",
    "revenue",
    "netinc",
    "ebitda",
    "cashneq",
    "debt",
    "fcf",
    "capex",
    "dps",
    "eps",
    "bvps",
    "shareswa",
    "marketcap",
]


def _calendar(provider_uri: Path) -> List[pd.Timestamp]:
    cal_path = provider_uri / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar not found: {cal_path}")
    return [pd.Timestamp(x.strip()) for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_market_tickers(provider_uri: Path, market: str, max_tickers: Optional[int]) -> List[str]:
    inst_path = provider_uri / "instruments" / f"{market}.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"instrument file not found: {inst_path}")
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    tickers = sorted(df["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist())
    if max_tickers is not None and int(max_tickers) > 0:
        tickers = tickers[: int(max_tickers)]
    return tickers


def _code_to_fname(symbol: str) -> str:
    try:
        from qlib.utils import code_to_fname

        return code_to_fname(symbol).lower()
    except Exception:
        return str(symbol).lower()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / (den.astype(float) + 1e-12)
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def _daily_field_frame(
    sf1_ticker: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    date_col: str,
    fields: Iterable[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=calendar)
    if sf1_ticker.empty:
        for field in fields:
            out[field] = np.nan
        return out.astype("float32")

    s = sf1_ticker.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col]).sort_values([date_col, "calendardate"])
    s = s.drop_duplicates(date_col, keep="last").set_index(date_col)
    union_index = calendar.union(pd.DatetimeIndex(s.index)).sort_values()
    for field in fields:
        raw = pd.to_numeric(s[field], errors="coerce") if field in s.columns else pd.Series(index=s.index, dtype=float)
        out[field] = raw.reindex(union_index).ffill().reindex(calendar).astype("float32")
    return out


def _compute_ratio_frame(
    sf1_ticker: pd.DataFrame,
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    date_col: str = "datekey",
) -> pd.DataFrame:
    f = _daily_field_frame(sf1_ticker, calendar, date_col=date_col, fields=SF1_FIELDS)
    close = pd.Series(close, index=calendar, dtype=float).replace(0, np.nan).astype("float32")
    out = pd.DataFrame(index=calendar)
    out["roe_q"] = _safe_div(f["netinc"], f["equity"])
    out["roa_q"] = _safe_div(f["netinc"], f["assets"])
    out["ebitda_margin_q"] = _safe_div(f["ebitda"], f["revenue"])
    out["fcf_margin_q"] = _safe_div(f["fcf"], f["revenue"])
    out["leverage_q"] = _safe_div(f["debt"], f["assets"])
    out["cash_assets_q"] = _safe_div(f["cashneq"], f["assets"])
    out["capex_assets_q"] = _safe_div(f["capex"], f["assets"])
    out["asset_turn_q"] = _safe_div(f["revenue"], f["assets"])
    out["div_yield_px_q"] = _safe_div(f["dps"], close)
    out["earn_yield_q"] = _safe_div(f["eps"], close)
    out["book_px_q"] = _safe_div(f["bvps"], close)
    out["fcf_yield_q"] = _safe_div(_safe_div(f["fcf"], f["shareswa"]), close)
    out["marketcap_q"] = pd.to_numeric(f["marketcap"], errors="coerce").astype("float32")
    out["log_marketcap_q"] = np.log1p(out["marketcap_q"].clip(lower=0)).astype("float32")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "date"
    return out[RATIO_FIELDS].astype("float32")


def _overwrite_feature_bins(prepared_dir: Path, provider_uri: Path, *, max_workers: int) -> int:
    _ = max_workers
    cal = _calendar(provider_uri)
    cal_index = pd.DatetimeIndex(cal)
    features_root = provider_uri / "features"
    written = 0

    for fp in sorted(prepared_dir.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            continue
        start = df["date"].min()
        end = df["date"].max()
        mask = (cal_index >= start) & (cal_index <= end)
        out_index = cal_index[mask]
        if out_index.empty:
            continue
        start_idx = int(cal_index.get_loc(out_index[0]))
        df = df.set_index("date").reindex(out_index)

        symbol_dir = features_root / _code_to_fname(fp.stem)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for col in RATIO_FIELDS:
            values = pd.to_numeric(df[col], errors="coerce").astype("float32").to_numpy()
            out = np.hstack([start_idx, values]).astype("<f")
            out.tofile(symbol_dir / f"{col}.day.bin")
            written += 1
    return written


def _write_provenance(provider_uri: Path, payload: Dict) -> Path:
    meta_dir = provider_uri / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / PROVENANCE_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute clean Sharadar SF1 PIT ratio features into daily qlib bins.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--raw_sf1", default="~/.qlib/sharadar/raw/sf1_MRQ.csv")
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--dimension", default="MRQ")
    p.add_argument("--date_col", default="datekey")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--out_dir", default="")
    p.add_argument("--dump_to_qlib", action="store_true")
    p.add_argument("--max_tickers", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    provider = Path(args.provider_uri).expanduser().resolve()
    raw_sf1 = Path(args.raw_sf1).expanduser().resolve()
    if not raw_sf1.exists():
        print(f"raw SF1 not found: {raw_sf1}", file=sys.stderr)
        return 2

    cal = pd.DatetimeIndex(_calendar(provider))
    if args.start:
        cal = cal[cal >= pd.Timestamp(args.start)]
    if args.end:
        cal = cal[cal <= pd.Timestamp(args.end)]
    if len(cal) == 0:
        print("empty calendar after start/end filters", file=sys.stderr)
        return 2

    tickers = _read_market_tickers(provider, args.market, args.max_tickers)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path("~/.qlib/sharadar/prepared").expanduser().resolve() / f"sf1_ratio_features_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["ticker", "dimension", "calendardate", args.date_col] + SF1_FIELDS
    sf1 = pd.read_csv(raw_sf1, usecols=usecols, low_memory=False)
    sf1 = sf1[sf1["dimension"].astype(str).str.upper() == str(args.dimension).upper()].copy()
    sf1["ticker"] = sf1["ticker"].astype(str).str.upper().str.strip()
    sf1 = sf1[sf1["ticker"].isin(set(tickers))]

    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region=REG_US)
    close_df = D.features(tickers, ["$close"], start_time=cal[0], end_time=cal[-1])
    close = close_df.iloc[:, 0] if isinstance(close_df, pd.DataFrame) else close_df
    close = close.reorder_levels(["datetime", "instrument"]).sort_index()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider_uri": str(provider),
                    "raw_sf1": str(raw_sf1),
                    "market": args.market,
                    "tickers": len(tickers),
                    "calendar_days": len(cal),
                    "out_dir": str(out_dir),
                    "ratio_fields": RATIO_FIELDS,
                },
                indent=2,
            )
        )
        return 0

    written_csv = 0
    for i, ticker in enumerate(tickers, start=1):
        sf1_t = sf1[sf1["ticker"] == ticker]
        try:
            close_t = close.xs(ticker, level="instrument").reindex(cal)
        except Exception:
            close_t = pd.Series(index=cal, dtype=float)
        ratio = _compute_ratio_frame(sf1_t, close_t, cal, date_col=args.date_col)
        ratio = ratio.reset_index()
        ratio["date"] = ratio["date"].dt.strftime("%Y-%m-%d")
        ratio.to_csv(out_dir / f"{ticker}.csv", index=False)
        written_csv += 1
        if i % 100 == 0:
            print(f"prepared_tickers={i}/{len(tickers)}")

    written_bins = 0
    if args.dump_to_qlib:
        written_bins = _overwrite_feature_bins(out_dir, provider, max_workers=int(args.max_workers))
        print(f"feature_bins_written={written_bins}")
        if written_bins <= 0:
            print("ERROR: no ratio feature bins were written", file=sys.stderr)
            return 3

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "table": "SF1",
        "mode": "pit_ratio_features",
        "raw_sf1": str(raw_sf1),
        "provider_uri": str(provider),
        "market": str(args.market),
        "dimension": str(args.dimension).upper(),
        "date_col": str(args.date_col),
        "date_offset_days": 0,
        "calendar_start": str(cal[0].date()),
        "calendar_end": str(cal[-1].date()),
        "tickers": int(len(tickers)),
        "prepared_dir": str(out_dir),
        "prepared_csv_files": int(written_csv),
        "dump_to_qlib": bool(args.dump_to_qlib),
        "feature_bins_written": int(written_bins),
        "ratio_fields": list(RATIO_FIELDS),
    }
    prov = _write_provenance(provider, payload)
    print(f"prepared_csv_files={written_csv}")
    print(f"provenance: {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
