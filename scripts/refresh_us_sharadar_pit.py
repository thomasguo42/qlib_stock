#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROVENANCE_FILE = "sharadar_sf1_pit.json"


def _load_sharadar_collector() -> type:
    here = Path(__file__).resolve()
    collector_path = here.parent / "data_collector" / "sharadar" / "collector.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("sharadar_collector", collector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load collector: {collector_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "SharadarCollector")


def _write_tickers_csv_from_instruments(instruments_tsv: Path, out_csv: Path, max_tickers: Optional[int]) -> int:
    df = pd.read_csv(instruments_tsv, sep="\t", header=None, names=["ticker", "start", "end"])
    tickers = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if max_tickers is not None:
        tickers = tickers[: int(max_tickers)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(out_csv, index=False)
    return len(tickers)


def _write_provenance(provider_uri: Path, payload: Dict) -> Path:
    meta_dir = provider_uri / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / PROVENANCE_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _sf1_ticker_coverage(raw_sf1: Path, expected_tickers: int) -> Tuple[int, float]:
    if expected_tickers <= 0:
        return 0, 0.0
    if not raw_sf1.exists() or raw_sf1.stat().st_size == 0:
        return 0, 1.0
    try:
        df = pd.read_csv(raw_sf1, low_memory=False)
    except Exception:
        return 0, 1.0
    cols = {str(c).lower(): c for c in df.columns}
    ticker_col = cols.get("ticker")
    if ticker_col is None:
        return 0, 1.0
    covered = int(df[ticker_col].astype(str).str.upper().str.strip().replace({"": np.nan}).dropna().nunique())
    missing_ratio = max(0.0, 1.0 - (float(covered) / float(expected_tickers)))
    return covered, missing_ratio


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh Sharadar SF1 PIT fundamentals and optionally overwrite qlib PIT bins.")
    p.add_argument("--provider_uri", default=os.getenv("QLIB_PROVIDER_URI", "/root/.qlib/qlib_data/us_data"))
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--api_key", default="", help="Optional; defaults to env NDL_API_KEY")
    p.add_argument("--out_dir", default="~/.qlib/sharadar")
    p.add_argument("--max_tickers", type=int, default=None)
    p.add_argument("--dimension", default="MRQ")
    p.add_argument("--date_col", default="datekey")
    p.add_argument("--period_col", default="calendardate")
    p.add_argument("--date_offset_days", type=int, default=0, help="Use 0 with datekey; use a lag only for period-date inputs.")
    p.add_argument("--fields", default="")
    p.add_argument("--exclude_fields", default="")
    p.add_argument("--dump_to_qlib", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Overwrite PIT files when dumping to qlib")
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument(
        "--max_missing_sf1_ticker_ratio",
        type=float,
        default=0.05,
        help="Allowed SF1 ticker coverage miss ratio after download",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    api_key = args.api_key.strip() or os.getenv("NDL_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: provide --api_key or set NDL_API_KEY", file=sys.stderr)
        return 2

    provider = Path(args.provider_uri).expanduser().resolve()
    inst_path = provider / "instruments" / f"{args.market}.txt"
    if not inst_path.exists():
        print(f"ERROR: market instruments not found: {inst_path}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser().resolve()
    tickers_csv = out_root / "universe" / f"{args.market}_tickers.csv"
    n_tickers = _write_tickers_csv_from_instruments(inst_path, tickers_csv, args.max_tickers)
    print(f"market={args.market} tickers={n_tickers}")

    raw_sf1 = out_root / "raw" / f"sf1_{args.dimension.upper()}.csv"
    pit_dir = out_root / "pit_normalized" / f"{args.dimension.lower()}_{stamp}"
    repo_root = Path(__file__).resolve().parents[1]

    if not args.dry_run:
        SharadarCollector = _load_sharadar_collector()
        collector = SharadarCollector(api_key=api_key, out_dir=str(out_root))
        collector.download_sf1(
            tickers_file=str(tickers_csv),
            dimension=str(args.dimension).upper(),
            out_file=str(raw_sf1),
        )
    else:
        print(f"dry_run: would download SF1 to {raw_sf1}")

    sf1_covered_tickers = 0
    sf1_missing_ratio = 0.0
    if not args.dry_run:
        sf1_covered_tickers, sf1_missing_ratio = _sf1_ticker_coverage(raw_sf1, n_tickers)
        print(f"sf1_ticker_coverage={sf1_covered_tickers}/{n_tickers} missing_ratio={sf1_missing_ratio:.2%}")
        if sf1_missing_ratio > float(args.max_missing_sf1_ticker_ratio):
            print(
                "ERROR: SF1 ticker coverage miss ratio exceeds threshold: "
                f"{sf1_missing_ratio:.2%} > {float(args.max_missing_sf1_ticker_ratio):.2%}",
                file=sys.stderr,
            )
            return 3

    prepare_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "data_collector" / "sharadar" / "prepare_sf1_pit.py"),
        "--sf1",
        str(raw_sf1),
        "--out_dir",
        str(pit_dir),
        "--dimension",
        str(args.dimension).upper(),
        "--date_col",
        str(args.date_col),
        "--period_col",
        str(args.period_col),
        "--date_offset_days",
        str(int(args.date_offset_days)),
    ]
    if args.fields:
        prepare_cmd += ["--fields", args.fields]
    if args.exclude_fields:
        prepare_cmd += ["--exclude_fields", args.exclude_fields]
    _run(prepare_cmd, dry_run=args.dry_run)

    if args.dump_to_qlib:
        dump_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "dump_pit.py"),
            "--csv_path",
            str(pit_dir),
            "--qlib_dir",
            str(provider),
            "--freq",
            "quarterly",
            "--max_workers",
            str(int(args.max_workers)),
            "dump",
            "--interval",
            "quarterly",
            "--overwrite",
            str(bool(args.overwrite)),
        ]
        _run(dump_cmd, dry_run=args.dry_run)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "table": "SF1",
        "dimension": str(args.dimension).upper(),
        "date_col": str(args.date_col),
        "period_col": str(args.period_col),
        "date_offset_days": int(args.date_offset_days),
        "raw_sf1": str(raw_sf1),
        "pit_normalized_dir": str(pit_dir),
        "provider_uri": str(provider),
        "market": str(args.market),
        "dump_to_qlib": bool(args.dump_to_qlib),
        "overwrite": bool(args.overwrite),
        "sf1_covered_tickers": int(sf1_covered_tickers),
        "sf1_expected_tickers": int(n_tickers),
        "sf1_missing_ticker_ratio": float(sf1_missing_ratio),
        "max_missing_sf1_ticker_ratio": float(args.max_missing_sf1_ticker_ratio),
    }
    if not args.dry_run:
        prov = _write_provenance(provider, payload)
        print(f"provenance: {prov}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
