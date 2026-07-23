#!/usr/bin/env python
"""Sweep portfolio-construction parameters for an existing US Sharadar prediction."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_us_sharadar_pipeline as vsp
from scripts.us_sharadar_release_checks import strategy_feasibility_check_rows


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value or "").split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in str(value or "").split(",") if part.strip()]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def _strategy_variant(
    strategy_cfg: dict,
    *,
    benchmark_core_weight: float,
    topk: int,
    max_sector_weight: float | None = None,
    max_weight: float | None = None,
    benchmark_max_weight: float | None = None,
    max_turnover: float | None = None,
    max_active_weight: float | None = None,
) -> dict:
    out = copy.deepcopy(strategy_cfg)
    out["benchmark_core_weight"] = float(benchmark_core_weight)
    out["topk"] = int(topk)
    out["max_holdings"] = max(int(out.get("max_holdings") or 0), int(topk) + int(out.get("benchmark_topn") or 0))
    if max_sector_weight is not None:
        out["max_sector_weight"] = float(max_sector_weight)
    if max_weight is not None:
        out["max_weight"] = float(max_weight)
    if benchmark_max_weight is not None:
        out["benchmark_max_weight"] = float(benchmark_max_weight)
    if max_turnover is not None:
        out["max_turnover"] = float(max_turnover)
    if max_active_weight is not None:
        out["max_active_weight"] = float(max_active_weight)
    return out


def _default_backtest_bounds(config: dict, pred: pd.DataFrame, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dt_index = pd.DatetimeIndex(pred.index.get_level_values("datetime"))
    backtest_cfg = ((config.get("port_analysis_config") or {}).get("backtest") or {})
    bt_start = pd.Timestamp(start or backtest_cfg.get("start_time") or dt_index.min())
    bt_end = pd.Timestamp(end or backtest_cfg.get("end_time") or dt_index.max())
    return bt_start, bt_end


def _row_for_variant(
    *,
    pred: pd.DataFrame,
    config: dict,
    benchmark: pd.Series,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    bt_calendar: List[pd.Timestamp],
    strategy_class: str,
    strategy_cfg: dict,
    recent_rebalances: int,
    args: argparse.Namespace,
) -> dict:
    interval_frame = vsp._rebalance_interval_quality_frame(
        pred,
        benchmark=benchmark,
        bt_start=bt_start,
        bt_end=bt_end,
        bt_calendar=bt_calendar,
        strategy_cfg=strategy_cfg,
        strategy_class=strategy_class,
        rebalance_weekday=strategy_cfg.get("rebalance_weekday"),
        args=args,
    )
    full_interval = vsp._rebalance_interval_quality_metrics(interval_frame)
    recent_interval = vsp._rebalance_interval_quality_metrics(
        interval_frame,
        recent_rebalances=int(recent_rebalances),
    )

    active_metrics = vsp._active_risk_metrics_for_predictions(
        pred,
        bt_start=bt_start,
        bt_end=bt_end,
        bt_calendar=bt_calendar,
        strategy_cfg=strategy_cfg,
        strategy_class=strategy_class,
        rebalance_weekday=strategy_cfg.get("rebalance_weekday"),
        args=args,
    )
    full_active = vsp._summarize_active_risk(active_metrics)

    data_handler_config = config.get("data_handler_config") or vsp._safe_get(
        config,
        ["task", "dataset", "kwargs", "handler", "kwargs"],
        {},
    )
    label_expr = None
    label_cfg = data_handler_config.get("label", [])
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        label_expr = label_cfg[0][0]
    label_horizon = vsp._parse_label_horizon(label_expr) if label_expr else None
    label_ref_start = vsp._infer_label_ref_start_days(data_handler_config)
    model_metrics = {}
    recent_model_metrics = {}
    if label_expr and label_horizon is not None:
        model_frame = vsp._load_prediction_label_frame(
            pred,
            label_expr=label_expr,
            benchmark=benchmark,
            bt_start=bt_start,
            bt_end=bt_end,
            label_horizon_days=int(label_horizon),
            label_ref_start_days=label_ref_start,
            strategy_cfg=strategy_cfg,
        )
        model_metrics = vsp._model_quality_metrics(model_frame, topk=int(strategy_cfg.get("topk", 40)), min_daily_count=30)
        recent_model_metrics = vsp._model_quality_metrics(
            model_frame,
            topk=int(strategy_cfg.get("topk", 40)),
            min_daily_count=30,
            recent_days=63,
        )

    feasibility_rows = strategy_feasibility_check_rows(strategy_cfg, strategy_class)
    return {
        "strategy_feasible": bool(all(ok for _, ok, _ in feasibility_rows)),
        "strategy_feasibility_failures": ";".join(name for name, ok, _ in feasibility_rows if not ok),
        "benchmark_core_weight": float(strategy_cfg.get("benchmark_core_weight", 0.0)),
        "topk": int(strategy_cfg.get("topk", 0)),
        "max_holdings": int(strategy_cfg.get("max_holdings", 0)),
        "max_sector_weight": strategy_cfg.get("max_sector_weight"),
        "max_weight": strategy_cfg.get("max_weight"),
        "benchmark_max_weight": strategy_cfg.get("benchmark_max_weight"),
        "full_interval_ann_excess": full_interval.get("ann_excess_return"),
        "recent_interval_ann_excess": recent_interval.get("ann_excess_return"),
        "full_interval_positive_rate": full_interval.get("positive_excess_rate"),
        "recent_interval_positive_rate": recent_interval.get("positive_excess_rate"),
        "active_share": full_active.get("mean_active_share"),
        "sector_active": full_active.get("mean_max_abs_sector_active_weight"),
        "max_abs_active_weight": full_active.get("mean_max_abs_active_weight"),
        "portfolio_names": full_active.get("mean_portfolio_names"),
        "mean_ic": model_metrics.get("mean_ic"),
        "topk_mean_label": model_metrics.get("topk_mean_label"),
        "recent_mean_ic": recent_model_metrics.get("mean_ic"),
        "recent_topk_mean_label": recent_model_metrics.get("topk_mean_label"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep strategy parameters for an existing US Sharadar pred.pkl.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    parser.add_argument("--benchmark_pkl", default="/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--core_weights", default="0.60,0.65,0.67,0.70")
    parser.add_argument("--topk_values", default="20,40")
    parser.add_argument("--max_sector_weight", type=float, default=None)
    parser.add_argument("--max_weight", type=float, default=None)
    parser.add_argument("--benchmark_max_weight", type=float, default=None)
    parser.add_argument("--max_turnover", type=float, default=None)
    parser.add_argument("--max_active_weight", type=float, default=None)
    parser.add_argument("--recent_rebalances", type=int, default=13)
    parser.add_argument("--active_risk_benchmark_topn", type=int, default=7)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--strategy_signal_shift", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import qlib

    qlib.init(provider_uri=args.provider_uri, region="us")
    config = _load_yaml(Path(args.config).expanduser().resolve())
    pred = vsp._read_pickle_compat(Path(args.pred).expanduser().resolve())
    benchmark = vsp._read_pickle_compat(Path(args.benchmark_pkl).expanduser().resolve())
    if not isinstance(pred, pd.DataFrame):
        raise TypeError("pred must be a pandas DataFrame")
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark_pkl must contain a pandas Series")

    bt_start_raw, bt_end_raw = _default_backtest_bounds(config, pred, args.start, args.end)
    bt_calendar = vsp._get_calendar_span(bt_start_raw, bt_end_raw)
    if len(bt_calendar) < 2:
        raise ValueError(f"Invalid backtest calendar: {bt_start_raw}->{bt_end_raw}")
    bt_start, bt_end = bt_calendar[0], bt_calendar[-1]

    strategy = ((config.get("port_analysis_config") or {}).get("strategy") or {})
    strategy_class = str(strategy.get("class") or "")
    base_strategy_cfg = strategy.get("kwargs") or {}
    helper_args = argparse.Namespace(
        strategy_signal_shift=int(args.strategy_signal_shift),
        rebalance_interval_marketcap_field=None,
        active_risk_marketcap_field="$marketcap_q",
        rebalance_interval_price_field="",
        rebalance_interval_deal_price=str((((config.get("port_analysis_config") or {}).get("backtest") or {}).get("exchange_kwargs") or {}).get("deal_price", "close")),
        rebalance_interval_topk=0,
        active_risk_topk=0,
        active_risk_benchmark_topn=int(args.active_risk_benchmark_topn),
    )

    rows = []
    for core in _parse_float_list(args.core_weights):
        for topk in _parse_int_list(args.topk_values):
            strategy_cfg = _strategy_variant(
                base_strategy_cfg,
                benchmark_core_weight=core,
                topk=topk,
                max_sector_weight=args.max_sector_weight,
                max_weight=args.max_weight,
                benchmark_max_weight=args.benchmark_max_weight,
                max_turnover=args.max_turnover,
                max_active_weight=args.max_active_weight,
            )
            rows.append(
                _row_for_variant(
                    pred=pred,
                    config=config,
                    benchmark=benchmark,
                    bt_start=bt_start,
                    bt_end=bt_end,
                    bt_calendar=bt_calendar,
                    strategy_class=strategy_class,
                    strategy_cfg=strategy_cfg,
                    recent_rebalances=int(args.recent_rebalances),
                    args=helper_args,
                )
            )

    out = pd.DataFrame(rows).sort_values(
        ["strategy_feasible", "full_interval_ann_excess", "recent_interval_ann_excess"],
        ascending=[False, False, False],
    )
    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(out_path)
    else:
        print(out.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
