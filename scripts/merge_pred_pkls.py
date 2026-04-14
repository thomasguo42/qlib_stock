#!/usr/bin/env python3
"""
Merge two Qlib prediction pickles (pred.pkl) into one.

Use case:
- Extend an existing historical pred.pkl with a newer "tail" pred.pkl.
- If there is overlap, the tail overwrites the base (keep="last").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pred must be a pandas DataFrame: {path}")
    if list(df.columns) != ["score"]:
        raise ValueError(f"unexpected pred columns in {path}: {list(df.columns)} (expected ['score'])")
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["datetime", "instrument"]:
        raise ValueError(
            f"unexpected pred index in {path}: names={getattr(df.index, 'names', None)} (expected MultiIndex datetime/instrument)"
        )
    return df


def _span(df: pd.DataFrame) -> str:
    dt = df.index.get_level_values("datetime")
    return f"{dt.min().date()}->{dt.max().date()} rows={len(df)}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge pred.pkl files (tail overwrites base on overlap).")
    p.add_argument("--base", required=True, help="Base pred.pkl path (historical)")
    p.add_argument("--tail", required=True, help="Tail pred.pkl path (newer predictions)")
    p.add_argument("--out", required=True, help="Output pred.pkl path")
    p.add_argument("--out_manifest", default=None, help="Optional output manifest JSON path")
    args = p.parse_args(argv)

    base_path = Path(args.base).expanduser().resolve()
    tail_path = Path(args.tail).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not tail_path.exists():
        raise FileNotFoundError(tail_path)

    base = _load_pred(base_path)
    tail = _load_pred(tail_path)

    merged = pd.concat([base, tail], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()

    merged.to_pickle(out_path)

    print(f"base:  {base_path} ({_span(base)})")
    print(f"tail:  {tail_path} ({_span(tail)})")
    print(f"saved: {out_path} ({_span(merged)})")

    if args.out_manifest:
        mpath = Path(args.out_manifest).expanduser().resolve()
        mpath.parent.mkdir(parents=True, exist_ok=True)
        dt = merged.index.get_level_values("datetime")
        manifest = {
            "type": "merged_pred",
            "base_pred": str(base_path),
            "tail_pred": str(tail_path),
            "rows": int(len(merged)),
            "span": [str(dt.min().date()), str(dt.max().date())],
        }
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"saved: {mpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
