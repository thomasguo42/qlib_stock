# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def normalize_long_weights(weights: pd.Series) -> pd.Series:
    """Return non-negative weights normalized to sum to one."""
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    out = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out > 0].astype(float)
    total = float(out.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(dtype=float)
    return out / total


def rank_score_weights(scores: pd.Series, *, topn: int, method: str = "rank", temperature: float = 1.0) -> pd.Series:
    """Convert a score series to long-only alpha weights."""
    if scores is None or scores.empty or int(topn) <= 0:
        return pd.Series(dtype=float)
    top = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False).head(int(topn))
    if top.empty:
        return pd.Series(dtype=float)
    method = str(method or "rank").lower()
    if method == "equal":
        return pd.Series(1.0 / len(top), index=top.index, dtype=float)
    if method == "rank":
        ranks = np.arange(len(top), 0, -1, dtype=float)
        return pd.Series(ranks / ranks.sum(), index=top.index, dtype=float)

    std = float(top.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(1.0 / len(top), index=top.index, dtype=float)
    z = (top - float(top.mean())) / std
    if method == "softmax":
        temp = max(float(temperature), 1e-6)
        raw = np.exp((z / temp) - float((z / temp).max()))
        return pd.Series(raw / raw.sum(), index=top.index, dtype=float)
    if method == "zscore":
        z = z - float(z.min())
        return normalize_long_weights(z)
    return pd.Series(1.0 / len(top), index=top.index, dtype=float)


def benchmark_weights_from_marketcap(marketcap: pd.Series, *, topn: Optional[int] = None) -> pd.Series:
    """Build a point-in-time cap-weighted benchmark proxy."""
    bench = normalize_long_weights(marketcap)
    if topn is not None and int(topn) > 0:
        bench = bench.sort_values(ascending=False).head(int(topn))
        bench = normalize_long_weights(bench)
    return bench


def cap_and_redistribute(weights: pd.Series, upper_bounds: pd.Series, *, max_iter: int = 25) -> pd.Series:
    """Cap weights and redistribute excess to names with remaining capacity."""
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    idx = weights.index.union(upper_bounds.index)
    raw = pd.to_numeric(weights, errors="coerce").reindex(idx).fillna(0.0)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(dtype=float)
    weights = raw / total
    upper = pd.to_numeric(upper_bounds, errors="coerce").reindex(idx).fillna(1.0).clip(lower=0.0)
    if float(upper.sum()) < 1.0 - 1e-10:
        return weights
    out = weights.clip(upper=upper)
    for _ in range(int(max_iter)):
        deficit = 1.0 - float(out.sum())
        if deficit <= 1e-10:
            break
        room = (upper - out).clip(lower=0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        out = out + room / room_sum * deficit
        out = out.clip(upper=upper)
    return normalize_long_weights(out)


def _prune_long_weights(
    weights: pd.Series,
    *,
    min_position_weight: float = 0.0,
    max_positions: Optional[int] = None,
) -> pd.Series:
    out = normalize_long_weights(weights)
    if out.empty:
        return out
    if max_positions is not None and int(max_positions) > 0 and len(out) > int(max_positions):
        out = out.sort_values(ascending=False).head(int(max_positions))
    if min_position_weight is not None and float(min_position_weight) > 0:
        pruned = out[out >= float(min_position_weight)]
        if pruned.empty:
            pruned = out.sort_values(ascending=False).head(1)
        out = pruned
    return normalize_long_weights(out)


def limit_turnover_toward_target(
    target_weights: pd.Series,
    current_weights: Optional[pd.Series],
    *,
    max_turnover: Optional[float] = None,
    min_trade_weight: float = 0.0,
    min_position_weight: float = 0.0,
    max_positions: Optional[int] = None,
) -> pd.Series:
    """
    Move from current weights toward target weights while bounding one-way turnover.

    ``max_turnover`` is measured as ``0.5 * sum(abs(target - current))`` on
    normalized long-only stock weights. ``min_trade_weight`` keeps tiny weight
    changes at the current weight before applying the turnover cap.
    ``max_positions`` and ``min_position_weight`` prevent turnover-limited
    portfolios from accumulating tiny residual positions indefinitely.
    """
    target = normalize_long_weights(target_weights)
    if target.empty or current_weights is None:
        return _prune_long_weights(
            target,
            min_position_weight=min_position_weight,
            max_positions=max_positions,
        )
    current = normalize_long_weights(current_weights)
    if current.empty:
        return _prune_long_weights(
            target,
            min_position_weight=min_position_weight,
            max_positions=max_positions,
        )

    idx = target.index.union(current.index)
    adjusted = target.reindex(idx).fillna(0.0).astype(float)
    current = current.reindex(idx).fillna(0.0).astype(float)

    if min_trade_weight is not None and float(min_trade_weight) > 0:
        delta = adjusted - current
        small = delta.abs() < float(min_trade_weight)
        if bool(small.any()):
            adjusted.loc[small] = current.loc[small]
            adjusted = normalize_long_weights(adjusted).reindex(idx).fillna(0.0)

    if max_turnover is not None:
        cap = max(0.0, float(max_turnover))
        turnover = float(0.5 * (adjusted - current).abs().sum())
        if np.isfinite(turnover) and turnover > cap + 1e-12:
            scale = cap / turnover if turnover > 0 else 0.0
            adjusted = current + (adjusted - current) * scale
    return _prune_long_weights(
        adjusted,
        min_position_weight=min_position_weight,
        max_positions=max_positions,
    )


def build_benchmark_aware_weights(
    scores: pd.Series,
    benchmark_weights: pd.Series,
    *,
    topk: int,
    benchmark_core_weight: float = 0.40,
    alpha_weighting: str = "rank",
    alpha_temperature: float = 1.0,
    max_weight: Optional[float] = None,
    max_benchmark_weight: Optional[float] = None,
    max_active_weight: Optional[float] = None,
) -> pd.Series:
    """
    Blend a benchmark proxy core with an alpha top-k sleeve.

    This is intentionally long-only and simple: it reduces benchmark drift without
    depending on a full external risk model.
    """
    bench = normalize_long_weights(benchmark_weights)
    alpha = rank_score_weights(scores, topn=int(topk), method=alpha_weighting, temperature=alpha_temperature)
    if bench.empty:
        target = alpha
    elif alpha.empty:
        target = bench
    else:
        core = float(np.clip(float(benchmark_core_weight), 0.0, 1.0))
        idx = bench.index.union(alpha.index)
        target = core * bench.reindex(idx).fillna(0.0) + (1.0 - core) * alpha.reindex(idx).fillna(0.0)
    target = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(target.sum())
    if not np.isfinite(total) or total <= 0:
        return target
    target = target / total

    if max_weight is not None and max_benchmark_weight is not None and not bench.empty:
        bench_names = target.index.intersection(bench.index)
        stock_names = target.index.difference(bench_names)
        stock_total = float(target.reindex(stock_names).fillna(0.0).sum())
        if stock_total > 0 and len(stock_names) > 0:
            stock_cap = max(0.0, float(max_weight))
            stock_capacity = stock_cap * float(len(stock_names))
            if stock_capacity >= stock_total - 1e-12:
                stock_upper = pd.Series(stock_cap / stock_total, index=stock_names, dtype=float)
                stock_weights = cap_and_redistribute(target.reindex(stock_names).fillna(0.0) / stock_total, stock_upper)
                target.loc[stock_names] = stock_weights.reindex(stock_names).fillna(0.0) * stock_total
            else:
                target.loc[stock_names] = target.reindex(stock_names).fillna(0.0).clip(upper=stock_cap)
                residual = 1.0 - float(target.reindex(stock_names).fillna(0.0).sum())
                bench_part = normalize_long_weights(target.reindex(bench_names).fillna(0.0))
                if not bench_part.empty:
                    target.loc[bench_names] = bench_part.reindex(bench_names).fillna(0.0) * residual

    upper = pd.Series(1.0, index=target.index, dtype=float)
    if max_weight is not None and not (max_benchmark_weight is not None and not bench.empty):
        upper = upper.clip(upper=max(0.0, float(max_weight)))
    elif max_weight is not None:
        non_benchmark_names = target.index.difference(bench.index)
        upper.loc[non_benchmark_names] = max(0.0, float(max_weight))
    if max_benchmark_weight is not None and not bench.empty:
        bench_names = target.index.intersection(bench.index)
        upper.loc[bench_names] = max(0.0, float(max_benchmark_weight))
    if max_active_weight is not None and not bench.empty:
        bench_aligned = bench.reindex(target.index).fillna(0.0)
        upper = np.minimum(upper, bench_aligned + max(0.0, float(max_active_weight)))
        upper = pd.Series(upper, index=target.index, dtype=float)
    return cap_and_redistribute(target, upper)


def active_weight_metrics(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    *,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Summarize benchmark drift for one rebalance date."""
    port = normalize_long_weights(portfolio_weights)
    bench = normalize_long_weights(benchmark_weights)
    idx = port.index.union(bench.index)
    p = port.reindex(idx).fillna(0.0)
    b = bench.reindex(idx).fillna(0.0)
    diff = p - b
    held = p[p > 0].index
    out = {
        "active_share": float(0.5 * diff.abs().sum()),
        "max_abs_active_weight": float(diff.abs().max()) if len(diff) else float("nan"),
        "benchmark_weight_held": float(b.reindex(held).sum()) if len(held) else 0.0,
        "portfolio_names": float((p > 0).sum()),
        "benchmark_names": float((b > 0).sum()),
    }
    if sector_map:
        sectors = pd.Series(
            [sector_map.get(str(inst).upper().strip(), "__UNKNOWN__") for inst in idx],
            index=idx,
        )
        sector_diff = diff.groupby(sectors).sum()
        out["max_abs_sector_active_weight"] = float(sector_diff.abs().max()) if not sector_diff.empty else float("nan")
    return out
