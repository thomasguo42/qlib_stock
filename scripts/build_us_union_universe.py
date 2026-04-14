#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Span:
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _read_instrument_file(path: Path) -> Dict[str, Span]:
    out: Dict[str, Span] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            sym = parts[0].strip().upper()
            if not sym:
                continue
            out[sym] = Span(start=_parse_ts(parts[1]), end=_parse_ts(parts[2]))
    return out


def _merge_span(a: Span, b: Span) -> Span:
    # Use union range from source universes, then optionally clamp with all.txt bounds.
    return Span(start=min(a.start, b.start), end=max(a.end, b.end))


def _fmt_ts(x: pd.Timestamp) -> str:
    return x.strftime("%Y-%m-%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a union US instrument universe from existing market files.")
    parser.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data", help="Qlib provider URI")
    parser.add_argument(
        "--source_markets",
        default="pit_mrq_large,sp500,nasdaq100",
        help="Comma-separated instrument files (without .txt) under provider_uri/instruments",
    )
    parser.add_argument("--out_market", required=True, help="Output instrument file name (without .txt)")
    parser.add_argument(
        "--use_all_bounds",
        action="store_true",
        help="Clamp output spans by all.txt bounds and skip symbols absent from all.txt",
    )
    parser.add_argument(
        "--prefer_primary_span",
        action="store_true",
        help="Use the first source market span as authoritative when a symbol exists there",
    )
    parser.add_argument(
        "--force_symbols",
        default="",
        help="Comma-separated symbols or alias groups (e.g. META|FB) to force include when available",
    )
    parser.add_argument("--min_start", default=None, help="Optional floor for start date (YYYY-MM-DD)")
    parser.add_argument("--max_end", default=None, help="Optional cap for end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    provider_uri = Path(args.provider_uri).expanduser().resolve()
    inst_dir = provider_uri / "instruments"
    if not inst_dir.exists():
        print(f"instruments directory not found: {inst_dir}")
        return 2

    source_names = _parse_csv(args.source_markets)
    if not source_names:
        print("source_markets is empty")
        return 2

    source_maps: Dict[str, Dict[str, Span]] = {}
    for name in source_names:
        p = inst_dir / f"{name}.txt"
        if not p.exists():
            print(f"source file missing: {p}")
            return 2
        source_maps[name] = _read_instrument_file(p)

    all_map: Optional[Dict[str, Span]] = None
    if args.use_all_bounds:
        all_file = inst_dir / "all.txt"
        if not all_file.exists():
            print(f"--use_all_bounds requested but missing: {all_file}")
            return 2
        all_map = _read_instrument_file(all_file)

    merged: Dict[str, Span] = {}
    for name, sym_map in source_maps.items():
        _ = name
        for sym, span in sym_map.items():
            if sym in merged:
                merged[sym] = _merge_span(merged[sym], span)
            else:
                merged[sym] = Span(start=span.start, end=span.end)

    # Optional guardrail: trust the first source list's span where available.
    # This is useful when benchmark membership files have stale end dates.
    if args.prefer_primary_span and source_names:
        primary = source_maps[source_names[0]]
        for sym, span in primary.items():
            merged[sym] = Span(start=span.start, end=span.end)

    forced_added = 0
    for token in _parse_csv(args.force_symbols):
        choices = [x.strip().upper() for x in token.split("|") if x.strip()]
        if not choices:
            continue
        selected = None
        if all_map is not None:
            # Prefer an already-present symbol that is also valid in all.txt.
            for sym in choices:
                if sym in merged and sym in all_map:
                    selected = sym
                    break
            # Fall back to first symbol available in all.txt and add it.
            if selected is None:
                for sym in choices:
                    if sym in all_map:
                        if sym not in merged:
                            merged[sym] = Span(start=all_map[sym].start, end=all_map[sym].end)
                            forced_added += 1
                        selected = sym
                        break
        else:
            # Without all.txt bounds, just ensure one choice is present in merged.
            for sym in choices:
                if sym in merged:
                    selected = sym
                    break

    dropped_not_in_all = 0
    dropped_invalid = 0
    min_start = _parse_ts(args.min_start) if args.min_start else None
    max_end = _parse_ts(args.max_end) if args.max_end else None

    final: Dict[str, Span] = {}
    for sym, span in merged.items():
        s = span.start
        e = span.end
        if all_map is not None:
            all_span = all_map.get(sym)
            if all_span is None:
                dropped_not_in_all += 1
                continue
            s = max(s, all_span.start)
            e = min(e, all_span.end)
        if min_start is not None:
            s = max(s, min_start)
        if max_end is not None:
            e = min(e, max_end)
        if s > e:
            dropped_invalid += 1
            continue
        final[sym] = Span(start=s, end=e)

    out_file = inst_dir / f"{args.out_market}.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for sym in sorted(final):
            span = final[sym]
            f.write(f"{sym}\t{_fmt_ts(span.start)}\t{_fmt_ts(span.end)}\n")

    print("== Union Universe Built ==")
    print(f"- provider_uri: {provider_uri}")
    print(f"- source_markets: {','.join(source_names)}")
    print(f"- out_file: {out_file}")
    print(f"- symbols_out: {len(final)}")
    print(f"- prefer_primary_span: {bool(args.prefer_primary_span)}")
    print(f"- forced_added: {forced_added}")
    print(f"- dropped_not_in_all: {dropped_not_in_all}")
    print(f"- dropped_invalid_span: {dropped_invalid}")
    if min_start is not None or max_end is not None:
        print(f"- clamp_range: {args.min_start or '<none>'} -> {args.max_end or '<none>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
