#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml

import qlib
from qlib.constant import REG_US
from qlib.data import D


DEFAULT_ANCHORS = "AAPL,MSFT,NVDA,AMZN,GOOGL,META|FB,TSLA,JPM,XOM,AVGO,LLY,V,MA,HD,COST"


def _parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_get(d: dict, keys: Iterable[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_symbols_from_file(path: Path) -> Set[str]:
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            head = s.split("\t", 1)[0].split(",", 1)[0].strip().upper()
            if head and head not in {"SYMBOL", "TICKER"}:
                out.add(head)
    return out


def _load_market_symbols(market: str, start: Optional[str], end: Optional[str]) -> Set[str]:
    inst_conf = D.instruments(market)
    spans = D.list_instruments(inst_conf, start_time=start, end_time=end, as_list=False)
    return set(spans.keys())


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _format_rows(rows: List[Tuple[str, int, int, int, float]]) -> str:
    lines = []
    header = "reference | base_count | ref_count | overlap | ref_coverage"
    sep = "--- | ---: | ---: | ---: | ---:"
    lines.append(header)
    lines.append(sep)
    for ref_name, base_n, ref_n, ov, cov in rows:
        lines.append(f"{ref_name} | {base_n} | {ref_n} | {ov} | {_format_pct(cov)}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit US universe integrity against reference universes.")
    parser.add_argument("--config", default=None, help="Optional pipeline YAML config for defaults")
    parser.add_argument("--provider_uri", default=None, help="Override qlib provider URI")
    parser.add_argument("--market", default=None, help="Target market to audit")
    parser.add_argument("--start", default=None, help="Optional start date for list_instruments")
    parser.add_argument("--end", default=None, help="Optional end date for list_instruments")
    parser.add_argument(
        "--reference_markets",
        default="sp500,nasdaq100",
        help="Comma-separated qlib reference markets",
    )
    parser.add_argument(
        "--reference_files",
        default="",
        help="Comma-separated instrument files (tab/csv first column is symbol)",
    )
    parser.add_argument(
        "--must_include",
        default=DEFAULT_ANCHORS,
        help="Comma-separated anchor symbols required in the base universe",
    )
    parser.add_argument(
        "--min_reference_overlap",
        type=float,
        default=0.60,
        help="Minimum overlap ratio against each reference universe",
    )
    parser.add_argument(
        "--show_missing_limit",
        type=int,
        default=20,
        help="Max missing symbols shown per reference",
    )
    parser.add_argument("--fail_on_overlap_fail", action="store_true", help="Exit non-zero if any overlap is below threshold")
    parser.add_argument("--fail_on_anchor_fail", action="store_true", help="Exit non-zero if any anchor symbol is missing")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    cfg = {}
    cfg_path = None
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}")
            return 2
        cfg = _load_yaml(cfg_path)

    provider_uri = args.provider_uri
    if provider_uri is None:
        provider_uri = _safe_get(cfg, ["qlib_init", "provider_uri"], "/root/.qlib/qlib_data/us_data")

    market = args.market or _safe_get(cfg, ["market"], None)
    if not market:
        print("Missing target market. Provide --market or --config with `market`.")
        return 2

    start = args.start or _safe_get(cfg, ["data_handler_config", "start_time"], None)
    end = args.end or _safe_get(cfg, ["data_handler_config", "end_time"], None)

    qlib.init(provider_uri=provider_uri, region=REG_US)
    base_symbols = _load_market_symbols(market, start, end)
    base_n = len(base_symbols)

    print("== Universe Integrity Audit ==")
    if cfg_path is not None:
        print(f"- config: {cfg_path}")
    print(f"- provider_uri: {provider_uri}")
    print(f"- market: {market}")
    print(f"- span: {start} -> {end}")
    print(f"- base_count: {base_n}")

    if base_n == 0:
        print("Base universe is empty.")
        return 3

    ref_rows: List[Tuple[str, int, int, int, float]] = []
    overlap_fail = False

    for ref_market in _parse_csv(args.reference_markets):
        ref_syms = _load_market_symbols(ref_market, start, end)
        ref_n = len(ref_syms)
        ov = len(base_symbols & ref_syms)
        cov = float(ov) / float(ref_n) if ref_n > 0 else 0.0
        ref_rows.append((f"market:{ref_market}", base_n, ref_n, ov, cov))
        if cov < args.min_reference_overlap:
            overlap_fail = True
        missing = sorted(ref_syms - base_symbols)
        if missing:
            limit = max(0, int(args.show_missing_limit))
            sample = ", ".join(missing[:limit])
            print(f"- missing_from_{ref_market}_sample({min(len(missing), limit)}/{len(missing)}): {sample}")

    for fp in _parse_csv(args.reference_files):
        path = Path(fp).expanduser().resolve()
        if not path.exists():
            print(f"- warning: reference file missing: {path}")
            continue
        ref_syms = _read_symbols_from_file(path)
        ref_n = len(ref_syms)
        ov = len(base_symbols & ref_syms)
        cov = float(ov) / float(ref_n) if ref_n > 0 else 0.0
        ref_rows.append((f"file:{path.name}", base_n, ref_n, ov, cov))
        if cov < args.min_reference_overlap:
            overlap_fail = True
        missing = sorted(ref_syms - base_symbols)
        if missing:
            limit = max(0, int(args.show_missing_limit))
            sample = ", ".join(missing[:limit])
            print(f"- missing_from_{path.name}_sample({min(len(missing), limit)}/{len(missing)}): {sample}")

    if ref_rows:
        print("\n" + _format_rows(ref_rows))
        print(f"\n- min_reference_overlap_threshold: {_format_pct(args.min_reference_overlap)}")

    anchors = [x.upper() for x in _parse_csv(args.must_include)]
    missing_anchors = []
    for group in anchors:
        choices = [x.strip() for x in group.split("|") if x.strip()]
        if not choices:
            continue
        if not any(x in base_symbols for x in choices):
            missing_anchors.append(group)

    print(f"- anchors_total: {len(anchors)}")
    print(f"- anchors_missing: {len(missing_anchors)}")
    if missing_anchors:
        print(f"- anchors_missing_list: {', '.join(missing_anchors)}")

    if args.fail_on_overlap_fail and overlap_fail:
        print("FAIL: reference overlap check below threshold.")
        return 4
    if args.fail_on_anchor_fail and missing_anchors:
        print("FAIL: required anchor symbols are missing.")
        return 5

    if overlap_fail or missing_anchors:
        print("RESULT: WARN")
    else:
        print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
