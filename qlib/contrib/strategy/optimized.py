# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.utils import get_date_by_shift, get_pre_trading_date

from .order_generator import OrderGenWInteract, OrderGenWOInteract
from .signal_strategy import WeightStrategyBase
from .optimizer import PortfolioOptimizer


class OptimizedTopkStrategy(WeightStrategyBase):
    """
    Top-k selection + portfolio optimization weights.
    """

    def __init__(
        self,
        *,
        topk: int,
        cov_window: int = 60,
        min_history: int = 20,
        max_weight: Optional[float] = None,
        use_score_as_return: bool = True,
        optimizer_kwargs: Optional[dict] = None,
        risk_degree: float = 0.95,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        super().__init__(
            order_generator_cls_or_obj=order_generator_cls_or_obj,
            **kwargs,
        )
        self.topk = topk
        self.cov_window = cov_window
        self.min_history = min_history
        self.max_weight = max_weight
        self.use_score_as_return = use_score_as_return
        self.risk_degree = risk_degree

        if optimizer_kwargs is None:
            # robust defaults: stable risk-based weights + mild turnover control
            optimizer_kwargs = {"method": "gmv", "delta": 0.2}
        elif "method" not in optimizer_kwargs:
            optimizer_kwargs = dict(optimizer_kwargs)
            optimizer_kwargs["method"] = "gmv"
        self.optimizer = PortfolioOptimizer(**optimizer_kwargs)

    def get_risk_degree(self, trade_step=None):
        return self.risk_degree

    def _get_close_history(self, instruments: Iterable[str], start_time, end_time) -> pd.DataFrame:
        close = D.features(list(instruments), ["$close"], start_time=start_time, end_time=end_time)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return pd.DataFrame()
        idx_names = list(close.index.names)
        if "instrument" in idx_names:
            inst_level = idx_names.index("instrument")
        else:
            inst_level = 0
        close = close.unstack(inst_level)
        return close.sort_index()

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        if score.empty:
            return {}

        topk_scores = score.sort_values(ascending=False).iloc[: self.topk]
        instruments = topk_scores.index.tolist()
        if len(instruments) == 0:
            return {}

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        hist_start = get_date_by_shift(pre_date, -self.cov_window, future=True, clip_shift=True)

        close = self._get_close_history(instruments, hist_start, pre_date)
        if close.empty:
            weight = np.repeat(1.0 / len(instruments), len(instruments))
            return {inst: float(w) for inst, w in zip(instruments, weight)}

        ret = close.pct_change().dropna(how="all")
        if ret.empty:
            weight = np.repeat(1.0 / len(instruments), len(instruments))
            return {inst: float(w) for inst, w in zip(instruments, weight)}

        valid = ret.count() >= self.min_history
        ret = ret.loc[:, valid]
        if ret.shape[1] < 2:
            weight = np.repeat(1.0 / len(instruments), len(instruments))
            return {inst: float(w) for inst, w in zip(instruments, weight)}

        cov = ret.cov()
        exp_ret = None
        if self.use_score_as_return:
            exp_ret = topk_scores.reindex(cov.index).fillna(topk_scores.min())

        w0 = None
        if current is not None and hasattr(current, "get_stock_weight_dict"):
            try:
                w0_series = pd.Series(current.get_stock_weight_dict(only_stock=False))
                w0_series = w0_series.reindex(cov.index).fillna(0.0)
                if w0_series.sum() > 0:
                    w0 = w0_series / w0_series.sum()
            except Exception:
                w0 = None

        weight = self.optimizer(cov, r=exp_ret, w0=w0)
        if isinstance(weight, np.ndarray):
            weight = pd.Series(weight, index=cov.index)

        if self.max_weight is not None:
            weight = weight.clip(upper=self.max_weight)
            if weight.sum() > 0:
                weight = weight / weight.sum()

        return {inst: float(w) for inst, w in weight.items() if w > 0}


class WeeklyOptimizedTopkStrategy(OptimizedTopkStrategy):
    """
    OptimizedTopkStrategy that only trades on a specific weekday.

    rebalance_weekday: 0=Monday, 4=Friday
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        if trade_start_time.weekday() != self.rebalance_weekday:
            from qlib.backtest.decision import TradeDecisionWO

            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)
