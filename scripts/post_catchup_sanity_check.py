#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Sanity checks after Sharadar catch-up.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--sep_dir", default="~/.qlib/sharadar/raw/sep")
    p.add_argument("--check_sep_coverage", action="store_true", help="Verify active tickers have SEP through calendar end")
    args = p.parse_args()

    provider = Path(args.provider_uri).expanduser().resolve()
    cal_path = provider / "calendars" / "day.txt"
    if not cal_path.exists():
        print(f"ERROR: calendar missing: {cal_path}")
        return 2
    cal_lines = cal_path.read_text(encoding="utf-8").strip().splitlines()
    if not cal_lines:
        print(f"ERROR: calendar empty: {cal_path}")
        return 2
    cal_end = cal_lines[-1]
    print(f"calendar_end {cal_end}")

    inst_path = provider / "instruments" / f"{args.market}.txt"
    if not inst_path.exists():
        print(f"ERROR: instruments missing: {inst_path}")
        return 2
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if df.empty:
        print(f"ERROR: instruments empty: {inst_path}")
        return 2

    print(f"market {args.market}")
    print(f"market_rows {len(df)} tickers {df['ticker'].nunique()}")
    print(f"market_max_end {df['end'].max()}")
    sub = df[df["ticker"].isin(["CMA", "DAY", "MPW"])]
    if not sub.empty:
        print("CMA/DAY/MPW:")
        print(sub.to_string(index=False))

    active = df[df["end"] == cal_end]["ticker"].astype(str).str.upper().tolist()
    print(f"active_at_calendar_end {len(active)}")

    bench_path = provider / "bench_etf_basket.pkl"
    if bench_path.exists():
        s = pd.read_pickle(bench_path).sort_index()
        if len(s) > 0:
            print(f"bench_end {s.index.max().strftime('%Y-%m-%d')} rows {len(s)}")
            print("bench_tail:")
            print(s.tail(3).to_string())
    else:
        print(f"WARN: benchmark missing: {bench_path}")

    if args.check_sep_coverage:
        sep_dir = Path(args.sep_dir).expanduser().resolve()
        if not sep_dir.exists():
            print(f"ERROR: SEP dir missing: {sep_dir}")
            return 2
        bad = []
        for t in active:
            fp = sep_dir / f"{t}.csv"
            if not fp.exists():
                bad.append((t, "MISSING"))
                continue
            try:
                s = pd.read_csv(fp, usecols=["date"], low_memory=False)["date"]
                mx = pd.to_datetime(s, errors="coerce").max()
                mx = "" if pd.isna(mx) else pd.Timestamp(mx).strftime("%Y-%m-%d")
            except Exception as e:
                bad.append((t, f"ERR:{e}"))
                continue
            if mx != cal_end:
                bad.append((t, mx))

        print(f"sep_coverage_bad {len(bad)}")
        if bad:
            print("sep_coverage_bad_sample:")
            for t, mx in bad[:30]:
                print(f"  {t} {mx}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

