#!/usr/bin/env python
"""Shared release-readiness checks for US Sharadar pipeline scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def coerce_ticker_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace(",", "\n").splitlines()
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        ticker = str(item).strip().upper()
        if not ticker or ticker.startswith("#") or ticker in seen:
            continue
        out.append(ticker)
        seen.add(ticker)
    return out


def strategy_benchmark_tickers(strategy_cfg: Dict) -> List[str]:
    tickers = coerce_ticker_list(strategy_cfg.get("benchmark_tickers"))
    path_val = str(strategy_cfg.get("benchmark_tickers_file") or "").strip()
    if path_val:
        path = Path(path_val).expanduser()
        if path.exists():
            tickers.extend(coerce_ticker_list(path.read_text(encoding="utf-8").splitlines()))
    topn = _safe_int(strategy_cfg.get("benchmark_topn"))
    out = coerce_ticker_list(tickers)
    if topn is not None and topn > 0:
        out = out[:topn]
    return out


def _safe_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value) -> Optional[int]:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _fmt(value) -> str:
    val = _safe_float(value)
    return "n/a" if val is None else f"{val:.4f}"


def benchmark_core_capacity(strategy_cfg: Dict) -> Optional[float]:
    """Return the explicit benchmark sleeve capacity implied by per-name caps.

    This only applies when the strategy uses an explicit benchmark ticker list,
    for example a single ``QQQ`` ETF core. Market-cap proxy cores require
    point-in-time data, so their effective capacity is validated by the active
    risk checks instead.
    """
    tickers = strategy_benchmark_tickers(strategy_cfg)
    if not tickers:
        return None
    n_tickers = len(tickers)
    max_benchmark_weight = _safe_float(strategy_cfg.get("benchmark_max_weight"))
    max_weight = _safe_float(strategy_cfg.get("max_weight"))
    per_name_cap = max_benchmark_weight if max_benchmark_weight is not None else max_weight
    if per_name_cap is None:
        return 1.0
    return float(max(0.0, per_name_cap) * n_tickers)


def strategy_feasibility_check_rows(
    strategy_cfg: Dict,
    strategy_class: str,
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    is_benchmark_aware = "BenchmarkAware" in str(strategy_class)
    if not is_benchmark_aware:
        return rows

    topk = _safe_int(strategy_cfg.get("topk"))
    rows.append(
        (
            "strategy_topk_positive",
            topk is not None and topk > 0,
            f"topk={topk}",
        )
    )

    core = _safe_float(strategy_cfg.get("benchmark_core_weight", 0.0))
    rows.append(
        (
            "strategy_benchmark_core_weight_range",
            core is not None and 0.0 <= core <= 1.0,
            f"benchmark_core_weight={_fmt(core)}",
        )
    )

    max_weight = _safe_float(strategy_cfg.get("max_weight"))
    if max_weight is not None:
        rows.append(
            (
                "strategy_max_weight_positive",
                max_weight > 0.0,
                f"max_weight={_fmt(max_weight)}",
            )
        )

    max_benchmark_weight = _safe_float(strategy_cfg.get("benchmark_max_weight"))
    if max_benchmark_weight is not None:
        rows.append(
            (
                "strategy_benchmark_max_weight_positive",
                max_benchmark_weight > 0.0,
                f"benchmark_max_weight={_fmt(max_benchmark_weight)}",
            )
        )

    tickers = strategy_benchmark_tickers(strategy_cfg)
    if tickers:
        capacity = benchmark_core_capacity(strategy_cfg)
        rows.append(
            (
                "strategy_explicit_benchmark_tickers_present",
                len(tickers) > 0,
                f"tickers={tickers}",
            )
        )
        if core is not None and capacity is not None:
            rows.append(
                (
                    "strategy_benchmark_core_capacity",
                    core <= capacity + 1e-12,
                    "requested_core={requested} <= capacity={capacity} "
                    "(tickers={n}, max_weight={max_weight}, benchmark_max_weight={benchmark_max_weight})".format(
                        requested=_fmt(core),
                        capacity=_fmt(capacity),
                        n=len(tickers),
                        max_weight=_fmt(max_weight),
                        benchmark_max_weight=_fmt(max_benchmark_weight),
                    ),
                )
            )

        max_holdings = _safe_int(strategy_cfg.get("max_holdings"))
        if max_holdings is not None and topk is not None and topk > 0:
            min_needed = topk + len(tickers)
            rows.append(
                (
                    "strategy_max_holdings_covers_alpha_and_core",
                    max_holdings >= min_needed,
                    f"max_holdings={max_holdings} >= topk+benchmark_tickers={min_needed}",
                )
            )

    max_active_weight = _safe_float(strategy_cfg.get("max_active_weight"))
    if max_active_weight is not None:
        rows.append(
            (
                "strategy_max_active_weight_positive",
                max_active_weight > 0.0,
                f"max_active_weight={_fmt(max_active_weight)}",
            )
        )

    return rows


def release_decision(
    statuses: Dict[str, Optional[bool]],
    *,
    required_checks: Iterable[str],
) -> Dict[str, object]:
    required = list(required_checks)
    missing_or_failed = [name for name in required if statuses.get(name) is not True]
    return {
        "release_ready": not missing_or_failed,
        "required_checks": required,
        "missing_or_failed": missing_or_failed,
    }
