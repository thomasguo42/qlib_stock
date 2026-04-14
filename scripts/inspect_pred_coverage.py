#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect pred.pkl date coverage.")
    p.add_argument("--pred", required=True, help="Path to pred.pkl")
    args = p.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    if not pred_path.exists():
        print(f"ERROR: pred not found: {pred_path}")
        return 2

    df = pd.read_pickle(pred_path)
    idx = df.index
    if hasattr(idx, "nlevels") and idx.nlevels >= 1:
        dt = idx.get_level_values(0)
    else:
        dt = idx
    dt = pd.to_datetime(dt, errors="coerce")
    dt = dt.dropna()
    if len(dt) == 0:
        print("ERROR: could not parse any datetimes from pred index")
        return 2

    uniq = pd.Index(sorted(dt.unique()))
    print(f"pred {pred_path}")
    print(f"pred_rows {len(df)}")
    print(f"pred_start {pd.Timestamp(dt.min()).strftime('%Y-%m-%d')}")
    print(f"pred_end {pd.Timestamp(dt.max()).strftime('%Y-%m-%d')}")
    print(f"pred_unique_dates {dt.nunique()}")
    print("last_5_dates " + ", ".join([pd.Timestamp(x).strftime("%Y-%m-%d") for x in uniq[-5:]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

