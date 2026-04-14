#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Generate a concrete "what to trade" plan (buy/sell + share amounts) from a saved pred.pkl.

This is intended for manual execution: you run it before the rebalance day and it prints/saves
an order list for that trade date.

Notes:
- The weekly strategies in this repo use previous trading day's scores (shift=1). So the trade
  date will consume scores from the prior trading day.
- This script DOES NOT execute trades. It only outputs recommended orders.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from ruamel.yaml import YAML

import qlib
from qlib.constant import REG_US
from qlib.backtest.account import Account
from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange
from qlib.backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager
from qlib.utils import init_instance_by_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _load_pred(path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if not isinstance(pred, pd.DataFrame):
        raise ValueError("pred.pkl must be a pandas DataFrame (or Series)")
    if not isinstance(pred.index, pd.MultiIndex) or set(pred.index.names) != {"datetime", "instrument"}:
        raise ValueError(
            "pred.pkl must be indexed by a MultiIndex with names {'datetime','instrument'}"
        )
    if "score" not in pred.columns:
        # fall back to first column if needed
        pred = pred.rename(columns={pred.columns[0]: "score"})
    return pred[["score"]]


def _pred_date_span(pred: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dts = pred.index.get_level_values("datetime")
    return pd.Timestamp(dts.min()), pd.Timestamp(dts.max())


def _load_positions_csv(path: Path) -> Dict[str, Any]:
    """
    Expected columns (case-insensitive):
    - symbol/instrument/ticker
    - shares/amount
    Optional:
    - price
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("instrument") or cols.get("ticker")
    amt_col = cols.get("shares") or cols.get("amount")
    price_col = cols.get("price")
    if sym_col is None or amt_col is None:
        raise ValueError(
            f"positions_csv must have columns symbol(or instrument/ticker) and shares(or amount): {path}"
        )
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip().upper()
        if not sym:
            continue
        amt = float(row[amt_col])
        if amt <= 0:
            continue
        if price_col is None or pd.isna(row[price_col]):
            out[sym] = {"amount": amt}
        else:
            out[sym] = {"amount": amt, "price": float(row[price_col])}
    return out


def _extract_strategy_and_exchange_kwargs(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    port_cfg = cfg.get("port_analysis_config", {}) or {}
    strategy_def = port_cfg.get("strategy", {}) or {}
    backtest_def = port_cfg.get("backtest", {}) or {}
    exchange_kwargs = backtest_def.get("exchange_kwargs", {}) or {}
    if not isinstance(strategy_def, dict) or not strategy_def:
        raise ValueError("config missing port_analysis_config.strategy")
    if "class" not in strategy_def:
        raise ValueError("config port_analysis_config.strategy missing 'class'")
    return strategy_def, exchange_kwargs


def _build_infras(
    *,
    trade_date: pd.Timestamp,
    exchange_kwargs: Dict[str, Any],
    init_cash: float,
    position_dict: Dict[str, Any],
    benchmark: str = "AAPL",
) -> Tuple[LevelInfrastructure, CommonInfrastructure, TradeCalendarManager, Exchange, Account]:
    trade_calendar = TradeCalendarManager(freq="day", start_time=trade_date, end_time=trade_date)

    exchange = Exchange(
        freq="day",
        start_time=trade_date,
        end_time=trade_date,
        codes="all",
        **exchange_kwargs,
    )
    account = Account(
        init_cash=float(init_cash),
        position_dict=position_dict,
        freq="day",
        benchmark_config={"benchmark": benchmark, "start_time": trade_date, "end_time": trade_date},
        port_metr_enabled=False,
    )
    level_infra = LevelInfrastructure(trade_calendar=trade_calendar)
    common_infra = CommonInfrastructure(trade_account=account, trade_exchange=exchange)
    return level_infra, common_infra, trade_calendar, exchange, account


def _format_side(direction: int) -> str:
    if direction == Order.BUY:
        return "BUY"
    if direction == Order.SELL:
        return "SELL"
    return str(direction)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a manual trade plan (orders) from a pred.pkl using the strategy/backtest settings in config."
    )
    p.add_argument("--config", required=True, help="Workflow YAML config (used for strategy + exchange kwargs)")
    p.add_argument("--pred", required=True, help="Path to pred.pkl (MultiIndex: datetime,instrument; col: score)")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data", help="Qlib provider uri")
    p.add_argument("--trade_date", required=True, help="Trade date (YYYY-MM-DD) you want to execute on")
    p.add_argument("--capital", type=float, default=100000.0, help="Account capital (USD)")
    p.add_argument("--positions_csv", default="", help="Optional current holdings CSV (symbol,shares[,price])")
    p.add_argument("--benchmark", default="AAPL", help="Benchmark ticker (only used to init Account)")
    p.add_argument("--out_csv", default="", help="Optional output CSV path for the order list")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}")
        return 2
    if not pred_path.exists():
        print(f"pred not found: {pred_path}")
        return 2

    cfg = _load_yaml(cfg_path)
    pred = _load_pred(pred_path)
    pred_min_dt, pred_max_dt = _pred_date_span(pred)

    trade_date = pd.Timestamp(args.trade_date).normalize()
    if trade_date < pred_min_dt.normalize() or trade_date > pred_max_dt.normalize():
        print(
            "trade_date is outside pred range: "
            f"trade_date={trade_date.date()} pred=[{pred_min_dt.date()}..{pred_max_dt.date()}]"
        )
        return 2

    qlib.init(provider_uri=args.provider_uri, region=REG_US)

    strategy_def, exchange_kwargs = _extract_strategy_and_exchange_kwargs(cfg)
    strat_kwargs = dict(strategy_def.get("kwargs", {}) or {})
    # Replace placeholder and force using the provided pred.
    strat_kwargs["signal"] = pred
    strat_cfg = {
        "class": strategy_def["class"],
        "module_path": strategy_def.get("module_path", "qlib.contrib.strategy"),
        "kwargs": strat_kwargs,
    }

    position_dict: Dict[str, Any] = {}
    if args.positions_csv:
        position_dict = _load_positions_csv(Path(args.positions_csv).expanduser().resolve())

    level_infra, common_infra, trade_calendar, exchange, _account = _build_infras(
        trade_date=trade_date,
        exchange_kwargs=exchange_kwargs,
        init_cash=float(args.capital),
        position_dict=position_dict,
        benchmark=args.benchmark,
    )

    strategy = init_instance_by_config(
        strat_cfg,
        level_infra=level_infra,
        common_infra=common_infra,
    )

    # For weekly strategies, scores are read from the previous trading day (shift=1).
    pred_start_time, _pred_end_time = trade_calendar.get_step_time(0, shift=1)
    pred_date = pd.Timestamp(pred_start_time).normalize()

    td = strategy.generate_trade_decision()
    orders = list(getattr(td, "order_list", getattr(td, "get_decision", lambda: [])()))
    if not orders:
        print("No orders generated.")
        print(f"- trade_date: {trade_date.date()}")
        print(f"- score_date_used_by_strategy (shift=1): {pred_date.date()}")
        print("This usually means trade_date is not a rebalance day for the strategy settings.")
        return 0

    rows = []
    for o in orders:
        if not isinstance(o, Order):
            continue
        side = _format_side(o.direction)
        dir_enum = OrderDir.BUY if o.direction == Order.BUY else OrderDir.SELL
        px = exchange.get_deal_price(
            stock_id=o.stock_id, start_time=o.start_time, end_time=o.end_time, direction=dir_enum
        )
        notional = float(o.amount) * float(px) if px is not None else float("nan")
        rows.append(
            {
                "trade_date": str(trade_date.date()),
                "score_date": str(pred_date.date()),
                "side": side,
                "symbol": str(o.stock_id),
                "shares": float(o.amount),
                "est_price": float(px) if px is not None else float("nan"),
                "est_notional": notional,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["side", "est_notional"], ascending=[True, False])
    buy_notional = float(out_df.loc[out_df["side"] == "BUY", "est_notional"].sum())
    sell_notional = float(out_df.loc[out_df["side"] == "SELL", "est_notional"].sum())
    invested_frac = buy_notional / float(args.capital) if float(args.capital) > 0 else float("nan")

    print("== Manual Trade Plan ==")
    print(f"- trade_date: {trade_date.date()}")
    print(f"- score_date_used_by_strategy (shift=1): {pred_date.date()}")
    print(
        f"- orders: {len(out_df)} (BUY notional≈{buy_notional:,.2f}, SELL notional≈{sell_notional:,.2f}, invested≈{invested_frac:.2%})"
    )
    print("")
    print(out_df.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nsaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
