#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from update_us_sharadar_data import (  # noqa: E402
    _benchmark_tickers_from_lines,
    _build_bench_etf_basket,
    _resolve_benchmark_tickers_file,
    _write_sfp_qlib_files,
)


def _run(cmd, *, dry_run: bool) -> None:
    print("+", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Import Sharadar SFP benchmark ETFs into an existing qlib provider.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--raw_sfp_dir", default="~/.qlib/sharadar/raw/sfp")
    p.add_argument("--benchmark_tickers_file", default="")
    p.add_argument("--out_dir", default="~/.qlib/sharadar")
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    provider = Path(args.provider_uri).expanduser().resolve()
    raw_sfp_dir = Path(args.raw_sfp_dir).expanduser().resolve()
    if not provider.exists():
        print(f"ERROR: provider not found: {provider}", file=sys.stderr)
        return 2
    if not raw_sfp_dir.exists():
        print(f"ERROR: raw SFP dir not found: {raw_sfp_dir}", file=sys.stderr)
        return 2

    tickers_file = _resolve_benchmark_tickers_file(args.benchmark_tickers_file)
    lines = tickers_file.read_text(encoding="utf-8").splitlines() if tickers_file.exists() else []
    tickers = _benchmark_tickers_from_lines(lines)
    if not tickers:
        print(f"ERROR: benchmark ticker list missing or empty: {tickers_file}", file=sys.stderr)
        return 2

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    prepared_dir = Path(args.out_dir).expanduser().resolve() / "prepared" / f"qlib_sfp_benchmark_import_{stamp}"
    written = _write_sfp_qlib_files(raw_sfp_dir, tickers, prepared_dir)
    print(f"sfp_benchmark_files_written={written} prepared_dir={prepared_dir}")
    if written != len(tickers):
        missing = sorted(set(tickers) - {p.stem.upper() for p in prepared_dir.glob('*.csv')})
        print(f"ERROR: missing prepared SFP benchmark tickers: {','.join(missing)}", file=sys.stderr)
        return 3

    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "dump_bin.py"),
            "dump_update",
            "--data_path",
            str(prepared_dir),
            "--qlib_dir",
            str(provider),
            "--freq",
            "day",
            "--date_field_name",
            "date",
            "--symbol_field_name",
            "symbol",
            "--file_suffix",
            ".csv",
            "--exclude_fields",
            "symbol",
            "--max_workers",
            str(int(args.max_workers)),
        ],
        dry_run=args.dry_run,
    )

    bench = _build_bench_etf_basket(raw_sfp_dir, tickers)
    out_bench = provider / "bench_etf_basket.pkl"
    print(f"bench_etf_basket: start={bench.index.min().date()} end={bench.index.max().date()} rows={len(bench)}")
    if not args.dry_run:
        bench.to_pickle(out_bench)
        print(f"saved: {out_bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
