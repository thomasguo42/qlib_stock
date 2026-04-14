# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple

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

    def _is_rebalance_day(self, trade_start_time) -> bool:
        # Rebalance on the specified weekday; if it's a holiday, use the next trading day.
        cal = getattr(self.trade_calendar, "_calendar", None)
        if cal is None or len(cal) == 0:
            return trade_start_time.weekday() == self.rebalance_weekday
        week_start = trade_start_time.normalize() - pd.Timedelta(days=trade_start_time.weekday())
        target_date = week_start + pd.Timedelta(days=self.rebalance_weekday)
        target = np.datetime64(target_date)
        idx = np.searchsorted(cal, target, side="left")
        if idx >= len(cal):
            return False
        candidate = pd.Timestamp(cal[idx]).normalize()
        return candidate == trade_start_time.normalize()

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            # Avoid out-of-range access on the last calendar step.
            return TradeDecisionWO([], self)
        if not self._is_rebalance_day(trade_start_time):
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)


class RiskManagedTopkDropoutStrategy(TopkDropoutStrategy):
    """
    TopkDropoutStrategy with adaptive risk control.

    The strategy scales portfolio exposure using realized volatility and drawdown
    of the candidate basket (plus an optional market trend filter) to reduce
    tail risk and regime instability.
    """

    def __init__(
        self,
        *,
        dynamic_risk: bool = True,
        risk_window: int = 20,
        risk_target_ann: float = 0.14,
        risk_floor: float = 0.15,
        risk_ceiling: float = 0.95,
        risk_smoothing: float = 0.35,
        risk_smoothing_up: Optional[float] = None,
        risk_smoothing_down: Optional[float] = None,
        max_risk_step_up: Optional[float] = None,
        max_risk_step_down: Optional[float] = None,
        candidate_buffer: int = 2,
        drawdown_window: int = 126,
        drawdown_limit: float = 0.10,
        drawdown_penalty: float = 0.55,
        market_index: Optional[str] = None,
        market_trend_window: int = 63,
        market_trend_thresh: float = -0.04,
        market_trend_penalty: float = 0.65,
        market_trend_boost_thresh: Optional[float] = None,
        market_trend_boost: float = 1.0,
        market_bull_floor_thresh: Optional[float] = None,
        market_bull_risk_floor: Optional[float] = None,
        include_current_positions: bool = False,
        market_drawdown_window: int = 126,
        market_drawdown_limit: Optional[float] = None,
        market_drawdown_penalty: float = 1.0,
        crash_guard: bool = False,
        crash_drawdown_limit: Optional[float] = None,
        crash_return_lookback: int = 5,
        crash_return_limit: Optional[float] = None,
        crash_penalty: float = 1.0,
        crash_cooldown_steps: int = 0,
        min_history: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dynamic_risk = bool(dynamic_risk)
        self.risk_window = max(2, int(risk_window))
        self.risk_target_ann = float(risk_target_ann)
        self.risk_floor = float(risk_floor)
        self.risk_ceiling = float(risk_ceiling)
        self.risk_smoothing = float(np.clip(float(risk_smoothing), 0.0, 1.0))
        self.risk_smoothing_up = (
            self.risk_smoothing
            if risk_smoothing_up is None
            else float(np.clip(float(risk_smoothing_up), 0.0, 1.0))
        )
        self.risk_smoothing_down = (
            self.risk_smoothing
            if risk_smoothing_down is None
            else float(np.clip(float(risk_smoothing_down), 0.0, 1.0))
        )
        self.max_risk_step_up = (
            None if max_risk_step_up is None else max(0.0, float(max_risk_step_up))
        )
        self.max_risk_step_down = (
            None if max_risk_step_down is None else max(0.0, float(max_risk_step_down))
        )
        self.candidate_buffer = max(1, int(candidate_buffer))
        self.drawdown_window = max(2, int(drawdown_window))
        self.drawdown_limit = float(drawdown_limit)
        self.drawdown_penalty = float(np.clip(float(drawdown_penalty), 0.0, 1.0))
        self.market_index = market_index
        self.market_trend_window = max(2, int(market_trend_window))
        self.market_trend_thresh = float(market_trend_thresh)
        self.market_trend_penalty = float(np.clip(float(market_trend_penalty), 0.0, 1.0))
        self.market_trend_boost_thresh = (
            None if market_trend_boost_thresh is None else float(market_trend_boost_thresh)
        )
        self.market_trend_boost = float(max(0.0, float(market_trend_boost)))
        self.market_bull_floor_thresh = (
            None if market_bull_floor_thresh is None else float(market_bull_floor_thresh)
        )
        self.market_bull_risk_floor = (
            None if market_bull_risk_floor is None else float(market_bull_risk_floor)
        )
        self.include_current_positions = bool(include_current_positions)
        self.market_drawdown_window = max(2, int(market_drawdown_window))
        self.market_drawdown_limit = (
            None if market_drawdown_limit is None else float(market_drawdown_limit)
        )
        self.market_drawdown_penalty = float(np.clip(float(market_drawdown_penalty), 0.0, 1.0))
        self.crash_guard = bool(crash_guard)
        self.crash_drawdown_limit = (
            None if crash_drawdown_limit is None else float(crash_drawdown_limit)
        )
        self.crash_return_lookback = max(1, int(crash_return_lookback))
        self.crash_return_limit = (
            None if crash_return_limit is None else float(crash_return_limit)
        )
        self.crash_penalty = float(np.clip(float(crash_penalty), 0.0, 1.0))
        self.crash_cooldown_steps = max(0, int(crash_cooldown_steps))
        self.min_history = max(5, int(min_history))
        self.base_risk_degree = float(self.risk_degree)
        self._smoothed_risk_degree = float(np.clip(self.base_risk_degree, self.risk_floor, self.risk_ceiling))
        self._cache_span = None
        self._close_cache = pd.DataFrame()
        self._cached_instruments = set()
        self._market_close_cache = None
        self._crash_cooldown_left = 0

    def _clip_risk(self, value: float, floor: Optional[float] = None) -> float:
        if floor is None:
            lower = self.risk_floor
        else:
            lower = float(np.clip(float(floor), self.risk_floor, self.risk_ceiling))
        return float(np.clip(float(value), lower, self.risk_ceiling))

    def _get_signal_series(self, trade_step: int) -> Optional[pd.Series]:
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return None
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        pred_score = pred_score.dropna()
        return pred_score if not pred_score.empty else None

    def _ensure_cache_span(self, reference_date) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if self._cache_span is not None:
            return self._cache_span
        cal = getattr(self.trade_calendar, "_calendar", None)
        if cal is not None and len(cal) > 0:
            first_trade = pd.Timestamp(cal[0])
            last_trade = pd.Timestamp(cal[-1])
        else:
            first_trade = pd.Timestamp(reference_date)
            last_trade = pd.Timestamp(reference_date)
        lookback = max(self.risk_window + 1, self.drawdown_window + 1, self.min_history + 1, self.market_trend_window + 1)
        cache_start = get_date_by_shift(first_trade, -lookback + 1, future=True, clip_shift=True)
        self._cache_span = (pd.Timestamp(cache_start), pd.Timestamp(last_trade))
        return self._cache_span

    def _ensure_close_cache(self, instruments, reference_date):
        if len(instruments) == 0:
            return
        missing = [inst for inst in instruments if inst not in self._cached_instruments]
        if not missing:
            return
        cache_start, cache_end = self._ensure_cache_span(reference_date)
        close = D.features(missing, ["$close"], start_time=cache_start, end_time=cache_end)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            self._cached_instruments.update(missing)
            return
        close_wide = close.unstack("instrument").sort_index()
        if self._close_cache.empty:
            self._close_cache = close_wide
        else:
            self._close_cache = self._close_cache.join(close_wide, how="outer")
        self._cached_instruments.update(missing)

    def _ensure_market_cache(self, reference_date):
        if not self.market_index:
            return
        if self._market_close_cache is not None:
            return
        cache_start, cache_end = self._ensure_cache_span(reference_date)
        close = D.features([self.market_index], ["$close"], start_time=cache_start, end_time=cache_end)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            self._market_close_cache = pd.Series(dtype=float)
            return
        if isinstance(close.index, pd.MultiIndex):
            close = close.droplevel("instrument")
        self._market_close_cache = close.sort_index()

    def _candidate_risk_scale(self, instruments, trade_start_time) -> float:
        if len(instruments) == 0:
            return 1.0

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        lookback = max(self.risk_window + 1, self.drawdown_window + 1, self.min_history + 1)
        start = get_date_by_shift(pre_date, -lookback + 1, future=True, clip_shift=True)
        self._ensure_close_cache(instruments, pre_date)
        if self._close_cache.empty:
            return 1.0
        close = self._close_cache.reindex(columns=instruments)
        close = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(pre_date))]
        close = close.dropna(how="all")
        if close.empty or close.shape[0] < 2:
            return 1.0
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if ret.empty:
            return 1.0
        eq_ret = ret.mean(axis=1, skipna=True).dropna()
        if len(eq_ret) < self.min_history:
            return 1.0

        ann_vol = float(eq_ret.tail(self.risk_window).std(ddof=0) * np.sqrt(252))
        if not np.isfinite(ann_vol) or ann_vol <= 1e-8:
            vol_scale = 1.0
        else:
            vol_scale = float(self.risk_target_ann / ann_vol)

        dd_scale = 1.0
        dd_window = eq_ret.tail(self.drawdown_window)
        drawdown = np.nan
        if len(dd_window) >= 2:
            nav = (1.0 + dd_window).cumprod()
            drawdown = float((nav / nav.cummax() - 1.0).min())
            if np.isfinite(drawdown) and drawdown <= -abs(self.drawdown_limit):
                dd_scale = self.drawdown_penalty

        crash_scale = 1.0
        if self.crash_guard:
            crash_trigger = False
            if (
                self.crash_drawdown_limit is not None
                and np.isfinite(drawdown)
                and drawdown <= -abs(self.crash_drawdown_limit)
            ):
                crash_trigger = True
            if self.crash_return_limit is not None and len(eq_ret) >= self.crash_return_lookback:
                recent_return = float((1.0 + eq_ret.tail(self.crash_return_lookback)).prod() - 1.0)
                if np.isfinite(recent_return) and recent_return <= -abs(self.crash_return_limit):
                    crash_trigger = True
            if crash_trigger:
                self._crash_cooldown_left = max(self._crash_cooldown_left, self.crash_cooldown_steps)
            if self._crash_cooldown_left > 0:
                crash_scale = self.crash_penalty
                self._crash_cooldown_left = max(0, self._crash_cooldown_left - 1)

        return max(0.0, vol_scale) * dd_scale * crash_scale

    def _market_trend_scale(self, trade_start_time) -> Tuple[float, Optional[float]]:
        if not self.market_index:
            return 1.0, None

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        start = get_date_by_shift(pre_date, -self.market_trend_window + 1, future=True, clip_shift=True)
        self._ensure_market_cache(pre_date)
        close = self._market_close_cache
        if close is None or close.empty:
            return 1.0, None
        close = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(pre_date))]
        close = close.dropna()
        if len(close) < 2:
            return 1.0, None

        trend = float(close.iloc[-1] / close.iloc[0] - 1.0)
        if np.isfinite(trend):
            if trend < self.market_trend_thresh:
                return self.market_trend_penalty, trend
            if self.market_trend_boost_thresh is not None and trend > self.market_trend_boost_thresh:
                return self.market_trend_boost, trend
        return 1.0, trend

    def _market_drawdown_scale(self, trade_start_time) -> float:
        if (
            not self.market_index
            or self.market_drawdown_limit is None
            or self.market_drawdown_penalty >= 1.0
        ):
            return 1.0
        pre_date = get_pre_trading_date(trade_start_time, future=True)
        start = get_date_by_shift(
            pre_date, -self.market_drawdown_window + 1, future=True, clip_shift=True
        )
        self._ensure_market_cache(pre_date)
        close = self._market_close_cache
        if close is None or close.empty:
            return 1.0
        close = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(pre_date))]
        close = close.dropna()
        if len(close) < 2:
            return 1.0
        drawdown = float((close / close.cummax() - 1.0).min())
        if np.isfinite(drawdown) and drawdown <= -abs(self.market_drawdown_limit):
            return self.market_drawdown_penalty
        return 1.0

    def _market_bull_floor(self, trend: Optional[float]) -> float:
        if self.market_bull_floor_thresh is None or self.market_bull_risk_floor is None:
            return self.risk_floor
        if trend is None or not np.isfinite(trend):
            return self.risk_floor
        if trend > self.market_bull_floor_thresh:
            return float(np.clip(self.market_bull_risk_floor, self.risk_floor, self.risk_ceiling))
        return self.risk_floor

    def _compute_dynamic_risk_degree(self, trade_step: int, trade_start_time) -> float:
        if not self.dynamic_risk:
            return self._clip_risk(self.base_risk_degree)

        pred_score = self._get_signal_series(trade_step)
        if pred_score is None:
            return self._smoothed_risk_degree

        topn = max(1, int(self.topk) * self.candidate_buffer)
        candidates = pred_score.sort_values(ascending=False).head(topn).index.tolist()
        if self.include_current_positions:
            current_list = list(getattr(self.trade_position, "get_stock_list", lambda: [])())
            if current_list:
                candidates = list(dict.fromkeys(candidates + current_list))
        basket_scale = self._candidate_risk_scale(candidates, trade_start_time)
        trend_scale, trend = self._market_trend_scale(trade_start_time)
        market_dd_scale = self._market_drawdown_scale(trade_start_time)
        bull_floor = self._market_bull_floor(trend)

        prev = self._smoothed_risk_degree
        target = self._clip_risk(
            self.base_risk_degree * basket_scale * trend_scale * market_dd_scale,
            floor=bull_floor,
        )
        if target > prev and self.max_risk_step_up is not None:
            target = min(target, prev + self.max_risk_step_up)
        elif target < prev and self.max_risk_step_down is not None:
            target = max(target, prev - self.max_risk_step_down)

        alpha = self.risk_smoothing_up if target >= prev else self.risk_smoothing_down
        smoothed = (1.0 - alpha) * prev + alpha * target
        self._smoothed_risk_degree = self._clip_risk(smoothed, floor=bull_floor)
        return self._smoothed_risk_degree

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            return TradeDecisionWO([], self)

        dynamic_risk_degree = self._compute_dynamic_risk_degree(trade_step, trade_start_time)
        original_risk_degree = self.risk_degree
        self.risk_degree = dynamic_risk_degree
        try:
            return super().generate_trade_decision(execute_result)
        finally:
            self.risk_degree = original_risk_degree


class WeeklyRiskManagedTopkDropoutStrategy(RiskManagedTopkDropoutStrategy):
    """
    RiskManagedTopkDropoutStrategy that only trades on a specific weekday.

    rebalance_weekday: 0=Monday, 4=Friday
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def _is_rebalance_day(self, trade_start_time) -> bool:
        # Rebalance on the specified weekday; if it's a holiday, use the next trading day.
        cal = getattr(self.trade_calendar, "_calendar", None)
        if cal is None or len(cal) == 0:
            return trade_start_time.weekday() == self.rebalance_weekday
        week_start = trade_start_time.normalize() - pd.Timedelta(days=trade_start_time.weekday())
        target_date = week_start + pd.Timedelta(days=self.rebalance_weekday)
        target = np.datetime64(target_date)
        idx = np.searchsorted(cal, target, side="left")
        if idx >= len(cal):
            return False
        candidate = pd.Timestamp(cal[idx]).normalize()
        return candidate == trade_start_time.normalize()

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            # Avoid out-of-range access on the last calendar step.
            return TradeDecisionWO([], self)
        if not self._is_rebalance_day(trade_start_time):
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
        ret = close.pct_change(fill_method=None).dropna(how="all")
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

    def _is_rebalance_day(self, trade_start_time) -> bool:
        # Rebalance on the specified weekday; if it's a holiday, use the next trading day.
        cal = getattr(self.trade_calendar, "_calendar", None)
        if cal is None or len(cal) == 0:
            return trade_start_time.weekday() == self.rebalance_weekday
        week_start = trade_start_time.normalize() - pd.Timedelta(days=trade_start_time.weekday())
        target_date = week_start + pd.Timedelta(days=self.rebalance_weekday)
        target = np.datetime64(target_date)
        idx = np.searchsorted(cal, target, side="left")
        if idx >= len(cal):
            return False
        candidate = pd.Timestamp(cal[idx]).normalize()
        return candidate == trade_start_time.normalize()

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        try:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        except IndexError:
            # Avoid out-of-range access on the last calendar step.
            return TradeDecisionWO([], self)
        if not self._is_rebalance_day(trade_start_time):
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)
