# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import numpy as np
import pandas as pd

from qlib.backtest.decision import TradeDecisionWO
from qlib.data import D
from qlib.utils import get_date_by_shift, get_pre_trading_date

from .signal_strategy import TopkDropoutStrategy, WeightStrategyBase


class WeeklyTopkDropoutStrategy(TopkDropoutStrategy):
    """
    TopkDropoutStrategy that only trades on a specific weekday.

    rebalance_weekday: 0=Monday, 4=Friday
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            # Avoid out-of-range access on the last calendar step.
            return TradeDecisionWO([], self)
        if trade_start_time.weekday() != self.rebalance_weekday:
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)


class ScoreWeightedStrategy(WeightStrategyBase):
    """
    Score-weighted top-k strategy with optional liquidity and volatility scaling.

    weighting: equal | rank | zscore | softmax
    """

    def __init__(
        self,
        *,
        topk: int,
        weighting: str = "rank",
        score_clip: float = 3.0,
        temperature: float = 1.0,
        max_weight: Optional[float] = None,
        liquidity_window: int = 20,
        min_avg_dollar_vol: Optional[float] = None,
        liquidity_buffer: int = 3,
        vol_window: Optional[int] = None,
        vol_scale: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self.weighting = weighting
        self.score_clip = score_clip
        self.temperature = max(temperature, 1e-6)
        self.max_weight = max_weight
        self.liquidity_window = liquidity_window
        self.min_avg_dollar_vol = min_avg_dollar_vol
        self.liquidity_buffer = max(liquidity_buffer, 1)
        self.vol_window = vol_window
        self.vol_scale = vol_scale

    def _score_to_weight(self, scores: pd.Series) -> pd.Series:
        n = len(scores)
        if n == 0:
            return scores
        if self.weighting == "equal":
            w = np.ones(n) / n
            return pd.Series(w, index=scores.index)
        if self.weighting == "rank":
            ranks = np.arange(n, 0, -1, dtype=float)
            w = ranks / ranks.sum()
            return pd.Series(w, index=scores.index)

        s = scores.astype(float)
        std = s.std()
        if std == 0 or np.isnan(std):
            w = np.ones(n) / n
            return pd.Series(w, index=scores.index)
        z = (s - s.mean()) / std
        if self.score_clip is not None:
            z = z.clip(lower=-self.score_clip, upper=self.score_clip)

        if self.weighting == "zscore":
            z = z - z.min()
            if z.sum() <= 0:
                w = np.ones(n) / n
            else:
                w = (z / z.sum()).values
            return pd.Series(w, index=scores.index)

        if self.weighting == "softmax":
            z = (z / self.temperature).values
            z = z - np.max(z)
            exp_z = np.exp(z)
            if exp_z.sum() <= 0:
                w = np.ones(n) / n
            else:
                w = exp_z / exp_z.sum()
            return pd.Series(w, index=scores.index)

        # fallback
        w = np.ones(n) / n
        return pd.Series(w, index=scores.index)

    def _filter_by_liquidity(self, scores: pd.Series, trade_start_time) -> pd.Series:
        if self.min_avg_dollar_vol is None or self.liquidity_window <= 0:
            return scores

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        start = get_date_by_shift(
            pre_date, -self.liquidity_window + 1, future=True, clip_shift=True
        )
        instruments = scores.index.tolist()
        if not instruments:
            return scores

        close = D.features(instruments, ["$close"], start_time=start, end_time=pre_date)
        volume = D.features(instruments, ["$volume"], start_time=start, end_time=pre_date)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        dollar_vol = (close * volume).dropna()
        if dollar_vol.empty:
            return scores

        avg_dv = dollar_vol.groupby(level="instrument").mean()
        liquid = avg_dv[avg_dv >= self.min_avg_dollar_vol].index
        filtered = scores.reindex(liquid).dropna()

        # if filtering is too strict, fall back to unfiltered
        if len(filtered) < min(self.topk, 5):
            return scores
        return filtered

    def _apply_vol_scale(self, weights: pd.Series, trade_start_time) -> pd.Series:
        if not self.vol_scale or not self.vol_window or self.vol_window <= 1:
            return weights

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        start = get_date_by_shift(
            pre_date, -self.vol_window + 1, future=True, clip_shift=True
        )
        instruments = weights.index.tolist()
        if not instruments:
            return weights

        close = D.features(instruments, ["$close"], start_time=start, end_time=pre_date)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return weights
        close = close.unstack("instrument").sort_index()
        ret = close.pct_change().dropna(how="all")
        if ret.empty:
            return weights
        vol = ret.std().reindex(weights.index).fillna(ret.std().mean())
        scaled = weights / vol.replace(0, np.nan)
        if scaled.sum() <= 0:
            return weights
        return scaled / scaled.sum()

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        if score.empty:
            return {}

        sorted_scores = score.sort_values(ascending=False)
        cand_n = min(len(sorted_scores), max(self.topk, 1) * self.liquidity_buffer)
        candidates = sorted_scores.iloc[:cand_n]
        candidates = self._filter_by_liquidity(candidates, trade_start_time)

        top = candidates.sort_values(ascending=False).iloc[: self.topk]
        if top.empty:
            return {}

        weights = self._score_to_weight(top)
        weights = self._apply_vol_scale(weights, trade_start_time)

        if self.max_weight is not None:
            weights = weights.clip(upper=self.max_weight)
            if weights.sum() > 0:
                weights = weights / weights.sum()

        return {inst: float(w) for inst, w in weights.items() if w > 0}


class WeeklyScoreWeightedStrategy(ScoreWeightedStrategy):
    """
    ScoreWeightedStrategy that only trades on a specific weekday.

    rebalance_weekday: 0=Monday, 4=Friday
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            # Avoid out-of-range access on the last calendar step.
            return TradeDecisionWO([], self)
        if trade_start_time.weekday() != self.rebalance_weekday:
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)
