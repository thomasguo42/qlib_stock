#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest as normal_backtest
from qlib.data.data import Cal


def _load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if not isinstance(pred, pd.DataFrame):
        raise ValueError("pred.pkl must be a DataFrame")
    return pred


def _get_time_range(pred: pd.DataFrame, start_time, end_time):
    dt = pred.index.get_level_values("datetime")
    start = pd.Timestamp(start_time) if start_time else dt.min()
    end = pd.Timestamp(end_time) if end_time else dt.max()
    # avoid running into calendar_index+1 out of range when future calendar is unavailable
    cal = Cal.calendar(freq="day", future=False)
    if len(cal) >= 2:
        end = min(end, cal[-2])
    return start, end


def _run_backtest(pred, strategy_cfg, start_time, end_time, benchmark):
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    backtest_cfg = {
        "start_time": start_time,
        "end_time": end_time,
        "account": 10000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }
    portfolio_metric_dict, _ = normal_backtest(
        executor=executor_cfg, strategy=strategy_cfg, **backtest_cfg
    )
    report, _ = portfolio_metric_dict["1day"]
    return report


def _print_metrics(name, report):
    strat = risk_analysis(report["return"] - report["cost"], freq="1day")
    bench = risk_analysis(report["bench"], freq="1day")
    excess = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="1day")
    print(f"\n== {name} ==")
    print("Strategy (with cost):")
    print(strat)
    print("Benchmark (buy & hold):")
    print(bench)
    print("Excess (strategy - benchmark - cost):")
    print(excess)


def main():
    parser = argparse.ArgumentParser(description="Backtest using saved pred.pkl and print metrics.")
    parser.add_argument("--pred", required=True, help="Path to pred.pkl")
    parser.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    parser.add_argument("--benchmark", default="AAPL")
    parser.add_argument(
        "--benchmark_list",
        default=None,
        help="Comma-separated tickers for equal-weight benchmark (e.g., SPY,QQQ,IWM)",
    )
    parser.add_argument(
        "--benchmark_pkl",
        default=None,
        help="Path to pickled pd.Series of benchmark daily returns",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n_drop", type=int, default=10)
    parser.add_argument("--rebalance_weekday", type=int, default=0)
    parser.add_argument("--opt_method", default="gmv", help="optimizer method: inv/gmv/mvo/rp")
    parser.add_argument("--opt_delta", type=float, default=0.2, help="turnover limit for optimizer")
    parser.add_argument("--opt_alpha", type=float, default=0.0, help="l2 regularization for optimizer")
    parser.add_argument("--skip_optimized", action="store_true", help="Skip optimized strategy backtest")
    parser.add_argument("--include_weighted", action="store_true", help="Evaluate score-weighted strategy")
    parser.add_argument("--weighting", default="rank", help="weighting: equal|rank|zscore|softmax")
    parser.add_argument("--score_clip", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_weight", type=float, default=0.05)
    parser.add_argument("--liquidity_window", type=int, default=20)
    parser.add_argument("--min_avg_dollar_vol", type=float, default=None)
    parser.add_argument("--liquidity_buffer", type=int, default=3)
    parser.add_argument("--vol_window", type=int, default=None)
    parser.add_argument("--vol_scale", action="store_true")
    args = parser.parse_args()

    qlib.init(provider_uri=args.provider_uri, region=REG_US)
    pred = _load_pred(Path(args.pred).expanduser().resolve())
    start, end = _get_time_range(pred, args.start, args.end)

    if args.benchmark_pkl:
        bench_path = Path(args.benchmark_pkl).expanduser().resolve()
        benchmark = pd.read_pickle(bench_path)
        if not isinstance(benchmark, pd.Series):
            raise ValueError("benchmark_pkl must be a pickled pandas Series of daily returns")
    elif args.benchmark_list:
        benchmark = [t.strip().upper() for t in args.benchmark_list.split(",") if t.strip()]
        if not benchmark:
            raise ValueError("benchmark_list is empty after parsing")
    else:
        benchmark = args.benchmark

    # Strategy 1: weekly topk dropout
    s1 = {
        "class": "WeeklyTopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": pred,
            "topk": args.topk,
            "n_drop": args.n_drop,
            "hold_thresh": 5,
            "only_tradable": True,
            "rebalance_weekday": args.rebalance_weekday,
        },
    }
    report1 = _run_backtest(pred, s1, start, end, benchmark)
    _print_metrics("WeeklyTopkDropout", report1)

    if not args.skip_optimized:
        # Strategy 2: weekly optimized topk
        s2 = {
            "class": "WeeklyOptimizedTopkStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": pred,
                "topk": args.topk,
                "rebalance_weekday": args.rebalance_weekday,
                "cov_window": 60,
                "min_history": 20,
                "max_weight": 0.05,
                "use_score_as_return": True,
                "optimizer_kwargs": {
                    "method": args.opt_method,
                    "delta": args.opt_delta,
                    "alpha": args.opt_alpha,
                    "lamb": 0.1,
                },
            },
        }
        report2 = _run_backtest(pred, s2, start, end, benchmark)
        _print_metrics("WeeklyOptimizedTopk", report2)

    if args.include_weighted:
        s3 = {
            "class": "WeeklyScoreWeightedStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": pred,
                "topk": args.topk,
                "rebalance_weekday": args.rebalance_weekday,
                "weighting": args.weighting,
                "score_clip": args.score_clip,
                "temperature": args.temperature,
                "max_weight": args.max_weight,
                "liquidity_window": args.liquidity_window,
                "min_avg_dollar_vol": args.min_avg_dollar_vol,
                "liquidity_buffer": args.liquidity_buffer,
                "vol_window": args.vol_window,
                "vol_scale": args.vol_scale,
            },
        }
        report3 = _run_backtest(pred, s3, start, end, benchmark)
        _print_metrics("WeeklyScoreWeighted", report3)


if __name__ == "__main__":
    main()
