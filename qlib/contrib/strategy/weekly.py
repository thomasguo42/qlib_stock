# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from qlib.backtest.decision import TradeDecisionWO
from qlib.data import D
from qlib.utils import get_date_by_shift, get_pre_trading_date

from .benchmark_aware import (
    benchmark_weights_from_marketcap,
    build_benchmark_aware_weights,
    limit_turnover_toward_target,
    normalize_long_weights,
)
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
        pred_score = self._apply_feature_score_controls(pred_score, pred_start_time, pred_end_time)
        pred_score = self._apply_sector_cap(pred_score)
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
        feature_score_weights: Optional[Dict[str, float]] = None,
        feature_min_percentiles: Optional[Dict[str, float]] = None,
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
        self.feature_score_weights = TopkDropoutStrategy._normalize_feature_control_map(feature_score_weights)
        self.feature_min_percentiles = TopkDropoutStrategy._normalize_feature_control_map(
            feature_min_percentiles, lower=0.0, upper=1.0
        )
        self._feature_control_cache: Dict[Tuple[str, ...], Dict[str, object]] = {}

    _cs_zscore = staticmethod(TopkDropoutStrategy._cs_zscore)
    _demote_masked_scores = staticmethod(TopkDropoutStrategy._demote_masked_scores)
    _feature_frame_by_instrument = staticmethod(TopkDropoutStrategy._feature_frame_by_instrument)
    _feature_panel = staticmethod(TopkDropoutStrategy._feature_panel)
    _load_feature_controls = TopkDropoutStrategy._load_feature_controls
    _apply_feature_score_controls = TopkDropoutStrategy._apply_feature_score_controls

    def _apply_trade_score_controls(self, score: pd.Series, trade_start_time) -> pd.Series:
        if not self.feature_score_weights and not self.feature_min_percentiles:
            return score
        pred_date = get_pre_trading_date(trade_start_time, future=True)
        return self._apply_feature_score_controls(score, pred_date, pred_date)

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
        score = self._apply_trade_score_controls(score, trade_start_time).dropna()
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


class BenchmarkAwareScoreWeightedStrategy(ScoreWeightedStrategy):
    """
    Score-weighted alpha sleeve blended with a point-in-time benchmark proxy.

    The benchmark proxy can come from an explicit qlib weight field or from a
    PIT market-cap field. It is meant as a practical bridge until official
    historical index/ETF holdings are available.
    """

    def __init__(
        self,
        *,
        benchmark_core_weight: float = 0.40,
        benchmark_topn: int = 500,
        benchmark_weight_field: Optional[str] = None,
        benchmark_marketcap_field: str = "$marketcap_q",
        benchmark_tickers: Optional[Iterable[str]] = None,
        benchmark_tickers_file: Optional[str] = None,
        max_active_weight: Optional[float] = None,
        benchmark_max_weight: Optional[float] = None,
        max_turnover: Optional[float] = None,
        min_trade_weight: float = 0.0,
        min_position_weight: float = 0.0,
        max_holdings: Optional[int] = None,
        sector_map_csv: Optional[str] = None,
        sector_ticker_col: str = "ticker",
        sector_col: str = "sector",
        max_sector_count: Optional[int] = None,
        max_sector_weight: Optional[float] = None,
        dynamic_alpha_weight: bool = False,
        alpha_quality_window: int = 63,
        alpha_quality_min_history: int = 20,
        alpha_quality_lower_excess: float = -0.03,
        alpha_quality_upper_excess: float = 0.03,
        min_alpha_scale: float = 0.0,
        max_alpha_scale: float = 1.0,
        dynamic_risk: bool = False,
        risk_floor: float = 0.0,
        risk_ceiling: Optional[float] = None,
        risk_smoothing: float = 1.0,
        risk_smoothing_up: Optional[float] = None,
        risk_smoothing_down: Optional[float] = None,
        max_risk_step_up: Optional[float] = None,
        max_risk_step_down: Optional[float] = None,
        market_index: Optional[str] = None,
        market_trend_window: int = 63,
        market_trend_thresh: float = -0.04,
        market_trend_penalty: float = 0.50,
        market_trend_boost_thresh: Optional[float] = None,
        market_trend_boost: float = 1.0,
        market_drawdown_window: int = 126,
        market_drawdown_limit: Optional[float] = None,
        market_drawdown_penalty: float = 1.0,
        crash_guard: bool = False,
        crash_return_lookback: int = 5,
        crash_return_limit: Optional[float] = None,
        crash_penalty: float = 1.0,
        crash_cooldown_steps: int = 0,
        risk_min_history: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_risk_degree = float(self.risk_degree)
        self.benchmark_core_weight = float(np.clip(float(benchmark_core_weight), 0.0, 1.0))
        self.benchmark_topn = max(1, int(benchmark_topn))
        self.benchmark_weight_field = benchmark_weight_field
        self.benchmark_marketcap_field = str(benchmark_marketcap_field)
        self.benchmark_tickers = self._coerce_ticker_list(benchmark_tickers)
        self.benchmark_tickers_file = benchmark_tickers_file
        self.max_active_weight = None if max_active_weight is None else max(0.0, float(max_active_weight))
        self.benchmark_max_weight = (
            None if benchmark_max_weight is None else max(0.0, float(benchmark_max_weight))
        )
        self.max_turnover = None if max_turnover is None else max(0.0, float(max_turnover))
        self.min_trade_weight = max(0.0, float(min_trade_weight))
        self.min_position_weight = max(0.0, float(min_position_weight))
        self.max_holdings = None if max_holdings is None else max(1, int(max_holdings))
        self.sector_map_csv = sector_map_csv
        self.sector_ticker_col = sector_ticker_col
        self.sector_col = sector_col
        if max_sector_count is None and max_sector_weight is not None:
            max_sector_count = int(np.floor(float(max_sector_weight) * int(self.topk)))
        self.max_sector_count = None if max_sector_count is None else max(1, int(max_sector_count))
        self.dynamic_alpha_weight = bool(dynamic_alpha_weight)
        self.alpha_quality_window = max(2, int(alpha_quality_window))
        self.alpha_quality_min_history = max(2, int(alpha_quality_min_history))
        self.alpha_quality_lower_excess = float(alpha_quality_lower_excess)
        self.alpha_quality_upper_excess = float(alpha_quality_upper_excess)
        self.min_alpha_scale = float(np.clip(float(min_alpha_scale), 0.0, 1.0))
        self.max_alpha_scale = float(np.clip(float(max_alpha_scale), self.min_alpha_scale, 1.0))
        self.dynamic_risk = bool(dynamic_risk)
        self.risk_floor = max(0.0, float(risk_floor))
        ceiling = self.base_risk_degree if risk_ceiling is None else float(risk_ceiling)
        self.risk_ceiling = float(np.clip(ceiling, self.risk_floor, 1.0))
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
        self.max_risk_step_up = None if max_risk_step_up is None else max(0.0, float(max_risk_step_up))
        self.max_risk_step_down = None if max_risk_step_down is None else max(0.0, float(max_risk_step_down))
        self.market_index = str(market_index).strip().upper() if market_index else None
        self.market_trend_window = max(2, int(market_trend_window))
        self.market_trend_thresh = float(market_trend_thresh)
        self.market_trend_penalty = float(np.clip(float(market_trend_penalty), 0.0, 1.0))
        self.market_trend_boost_thresh = (
            None if market_trend_boost_thresh is None else float(market_trend_boost_thresh)
        )
        self.market_trend_boost = max(0.0, float(market_trend_boost))
        self.market_drawdown_window = max(2, int(market_drawdown_window))
        self.market_drawdown_limit = (
            None if market_drawdown_limit is None else abs(float(market_drawdown_limit))
        )
        self.market_drawdown_penalty = float(np.clip(float(market_drawdown_penalty), 0.0, 1.0))
        self.crash_guard = bool(crash_guard)
        self.crash_return_lookback = max(1, int(crash_return_lookback))
        self.crash_return_limit = None if crash_return_limit is None else abs(float(crash_return_limit))
        self.crash_penalty = float(np.clip(float(crash_penalty), 0.0, 1.0))
        self.crash_cooldown_steps = max(0, int(crash_cooldown_steps))
        self.risk_min_history = max(2, int(risk_min_history))
        self._smoothed_risk_degree = float(np.clip(self.base_risk_degree, self.risk_floor, self.risk_ceiling))
        self._risk_market_cache = pd.Series(dtype=float)
        self._risk_market_cache_span: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._crash_cooldown_left = 0
        self._sector_map = None
        self._benchmark_ticker_cache: Optional[List[str]] = None

    @staticmethod
    def _coerce_ticker_list(tickers: Optional[Iterable[str]]) -> List[str]:
        if tickers is None:
            return []
        if isinstance(tickers, str):
            raw = tickers.replace(",", "\n").splitlines()
        else:
            raw = list(tickers)
        out: List[str] = []
        seen = set()
        for item in raw:
            ticker = str(item).strip().upper()
            if not ticker or ticker.startswith("#") or ticker in seen:
                continue
            out.append(ticker)
            seen.add(ticker)
        return out

    def _configured_benchmark_tickers(self) -> List[str]:
        if self._benchmark_ticker_cache is not None:
            return self._benchmark_ticker_cache
        tickers = list(self.benchmark_tickers)
        if self.benchmark_tickers_file:
            path = Path(self.benchmark_tickers_file).expanduser()
            if path.exists():
                tickers.extend(path.read_text(encoding="utf-8").splitlines())
        self._benchmark_ticker_cache = self._coerce_ticker_list(tickers)
        return self._benchmark_ticker_cache

    @staticmethod
    def _series_from_features(data, field: str) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if field in data.columns:
                data = data[field]
            elif data.shape[1] == 1:
                data = data.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        if not isinstance(data, pd.Series) or data.empty:
            return pd.Series(dtype=float)
        out = data.copy()
        if isinstance(out.index, pd.MultiIndex):
            level = "instrument" if "instrument" in out.index.names else 0
            out = out.groupby(level=level).last()
        out.index = out.index.astype(str)
        return pd.to_numeric(out, errors="coerce")

    def _benchmark_weights(self, instruments, trade_start_time) -> pd.Series:
        configured_tickers = self._configured_benchmark_tickers()
        if configured_tickers:
            pre_date = get_pre_trading_date(trade_start_time, future=True)
            try:
                data = D.features(configured_tickers, ["$close"], start_time=pre_date, end_time=pre_date)
            except Exception:
                data = None
            close = self._series_from_features(data, "$close")
            close = close.replace([np.inf, -np.inf], np.nan).dropna()
            available = close[close > 0].index.astype(str).tolist()
            if not available:
                return pd.Series(dtype=float)
            return normalize_long_weights(pd.Series(1.0, index=available, dtype=float))
        if not instruments:
            return pd.Series(dtype=float)
        pre_date = get_pre_trading_date(trade_start_time, future=True)
        field = self.benchmark_weight_field or self.benchmark_marketcap_field
        data = D.features(list(instruments), [field], start_time=pre_date, end_time=pre_date)
        raw = self._series_from_features(data, field)
        if raw.empty:
            return pd.Series(dtype=float)
        if self.benchmark_weight_field:
            weights = normalize_long_weights(raw)
            if self.benchmark_topn > 0:
                weights = normalize_long_weights(weights.sort_values(ascending=False).head(self.benchmark_topn))
            return weights
        return benchmark_weights_from_marketcap(raw, topn=self.benchmark_topn)

    def _load_sector_map(self) -> Dict[str, str]:
        if self._sector_map is not None:
            return self._sector_map
        if not self.sector_map_csv:
            self._sector_map = {}
            return self._sector_map
        path = Path(self.sector_map_csv).expanduser()
        if not path.exists():
            self._sector_map = {}
            return self._sector_map
        df = pd.read_csv(path, usecols=lambda c: c in {self.sector_ticker_col, self.sector_col}, low_memory=False)
        if self.sector_ticker_col not in df.columns or self.sector_col not in df.columns:
            self._sector_map = {}
            return self._sector_map
        tickers = df[self.sector_ticker_col].astype(str).str.upper().str.strip()
        sectors = df[self.sector_col].astype(str).str.strip()
        sectors = sectors.mask(sectors.eq("") | sectors.str.lower().isin({"nan", "none"}), "__UNKNOWN__")
        sector_map = pd.Series(sectors.values, index=tickers.values)
        sector_map = sector_map[~sector_map.index.duplicated(keep="last")]
        self._sector_map = sector_map.to_dict()
        return self._sector_map

    def _apply_sector_cap(self, score: pd.Series) -> pd.Series:
        if self.max_sector_count is None or self.max_sector_count <= 0 or score is None or score.empty:
            return score
        sector_map = self._load_sector_map()
        if not sector_map:
            return score
        sorted_score = score.dropna().sort_values(ascending=False)
        if sorted_score.empty:
            return score
        spread = float(sorted_score.max() - sorted_score.min())
        if not np.isfinite(spread):
            spread = 1.0
        floor = float(sorted_score.min() - max(1.0, spread))
        step = max(1e-9, max(1.0, spread) * 1e-9)
        counts: Dict[str, int] = {}
        adjusted = sorted_score.copy()
        for inst in sorted_score.index:
            sector = sector_map.get(str(inst).upper().strip(), "__UNKNOWN__")
            count = counts.get(sector, 0)
            if count < self.max_sector_count:
                counts[sector] = count + 1
                continue
            adjusted.loc[inst] = floor
            floor -= step
        return adjusted.reindex(score.index)

    @staticmethod
    def _current_stock_weights(current) -> pd.Series:
        if current is None or not hasattr(current, "get_stock_weight_dict"):
            return pd.Series(dtype=float)
        try:
            weights = pd.Series(current.get_stock_weight_dict(only_stock=True), dtype=float)
        except Exception:
            return pd.Series(dtype=float)
        weights.index = weights.index.astype(str)
        return weights

    def _trailing_weighted_excess_return(
        self,
        alpha_weights: pd.Series,
        benchmark_weights: pd.Series,
        trade_start_time,
    ) -> Optional[float]:
        alpha = normalize_long_weights(alpha_weights)
        bench = normalize_long_weights(benchmark_weights)
        if alpha.empty or bench.empty:
            return None

        pre_date = get_pre_trading_date(trade_start_time, future=True)
        start = get_date_by_shift(
            pre_date,
            -self.alpha_quality_window + 1,
            future=True,
            clip_shift=True,
        )
        instruments = alpha.index.union(bench.index).astype(str).tolist()
        if not instruments:
            return None
        close = D.features(instruments, ["$close"], start_time=start, end_time=pre_date)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if not isinstance(close, pd.Series) or close.empty:
            return None
        if isinstance(close.index, pd.MultiIndex):
            close = close.unstack("instrument")
        else:
            return None
        close = close.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if len(close) < self.alpha_quality_min_history:
            return None

        first = close.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
        last = close.apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
        ret = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        ret = ret[(first.reindex(ret.index) > 0) & (last.reindex(ret.index) > 0)]
        if ret.empty:
            return None

        alpha = alpha.reindex(ret.index).dropna()
        bench = bench.reindex(ret.index).dropna()
        if float(alpha.sum()) < 0.50 or float(bench.sum()) < 0.50:
            return None
        alpha = normalize_long_weights(alpha)
        bench = normalize_long_weights(bench)
        if alpha.empty or bench.empty:
            return None

        alpha_ret = float((alpha * ret.reindex(alpha.index)).sum())
        bench_ret = float((bench * ret.reindex(bench.index)).sum())
        if not np.isfinite(alpha_ret) or not np.isfinite(bench_ret):
            return None
        return alpha_ret - bench_ret

    def _alpha_quality_scale(self, excess_return: Optional[float]) -> float:
        if not self.dynamic_alpha_weight:
            return 1.0
        if excess_return is None or not np.isfinite(excess_return):
            return 1.0
        lower = min(self.alpha_quality_lower_excess, self.alpha_quality_upper_excess)
        upper = max(self.alpha_quality_lower_excess, self.alpha_quality_upper_excess)
        if upper <= lower + 1e-12:
            return self.max_alpha_scale if excess_return >= upper else self.min_alpha_scale
        if excess_return <= lower:
            return self.min_alpha_scale
        if excess_return >= upper:
            return self.max_alpha_scale
        frac = (float(excess_return) - lower) / (upper - lower)
        return float(self.min_alpha_scale + frac * (self.max_alpha_scale - self.min_alpha_scale))

    def _effective_benchmark_core_weight(
        self,
        alpha_weights: pd.Series,
        benchmark_weights: pd.Series,
        trade_start_time,
    ) -> float:
        if not self.dynamic_alpha_weight:
            return self.benchmark_core_weight
        scale = self._alpha_quality_scale(
            self._trailing_weighted_excess_return(alpha_weights, benchmark_weights, trade_start_time)
        )
        alpha_weight = (1.0 - self.benchmark_core_weight) * scale
        return float(np.clip(1.0 - alpha_weight, 0.0, 1.0))

    @staticmethod
    def _market_close_series(data, instrument: str) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if "$close" in data.columns:
                data = data["$close"]
            elif data.shape[1] == 1:
                data = data.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        if not isinstance(data, pd.Series) or data.empty:
            return pd.Series(dtype=float)
        out = data.copy()
        if isinstance(out.index, pd.MultiIndex):
            names = list(out.index.names)
            if "instrument" in names:
                try:
                    out = out.xs(instrument, level="instrument", drop_level=True)
                except KeyError:
                    return pd.Series(dtype=float)
            elif out.index.nlevels >= 2:
                try:
                    out = out.xs(instrument, level=0, drop_level=True)
                except KeyError:
                    return pd.Series(dtype=float)
            if isinstance(out.index, pd.MultiIndex) and "datetime" in out.index.names:
                out = out.droplevel([n for n in out.index.names if n != "datetime"])
        out.index = pd.DatetimeIndex(out.index).normalize()
        out = pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out = out[out > 0].sort_index()
        return out[~out.index.duplicated(keep="last")]

    def _market_risk_close_history(self, trade_start_time) -> pd.Series:
        if not self.market_index:
            return pd.Series(dtype=float)
        pre_date = pd.Timestamp(get_pre_trading_date(trade_start_time, future=True)).normalize()
        lookback = max(
            self.market_trend_window + 1,
            self.market_drawdown_window + 1,
            self.crash_return_lookback + 1,
            self.risk_min_history + 1,
        )
        if (
            self._risk_market_cache_span is None
            or self._risk_market_cache_span[0] > pre_date
            or self._risk_market_cache_span[1] < pre_date
        ):
            try:
                cal = getattr(self.trade_calendar, "_calendar", None)
            except Exception:
                cal = None
            if cal is not None and len(cal) > 0:
                cache_start_ref = pd.Timestamp(cal[0])
                cache_end = pd.Timestamp(cal[-1])
            else:
                cache_start_ref = pre_date
                cache_end = pre_date
            try:
                cache_start = pd.Timestamp(
                    get_date_by_shift(cache_start_ref, -lookback + 1, future=True, clip_shift=True)
                )
            except Exception:
                cache_start = pd.Timestamp(cache_start_ref) - pd.Timedelta(days=lookback * 3)
            raw = D.features([self.market_index], ["$close"], start_time=cache_start, end_time=cache_end)
            self._risk_market_cache = self._market_close_series(raw, self.market_index)
            self._risk_market_cache_span = (cache_start.normalize(), cache_end.normalize())
        close = self._risk_market_cache
        if close.empty:
            return close
        return close[close.index <= pre_date]

    def _clip_dynamic_risk(self, value: float) -> float:
        return float(np.clip(float(value), self.risk_floor, self.risk_ceiling))

    def _compute_dynamic_risk_degree_for_date(self, trade_start_time) -> float:
        if not self.dynamic_risk or not self.market_index:
            return self.base_risk_degree

        close = self._market_risk_close_history(trade_start_time)
        if len(close) < self.risk_min_history:
            return self._smoothed_risk_degree

        target = float(self.base_risk_degree)
        trend_window = close.tail(self.market_trend_window)
        if len(trend_window) >= min(self.risk_min_history, self.market_trend_window):
            trend = float(trend_window.iloc[-1] / trend_window.iloc[0] - 1.0)
            if np.isfinite(trend):
                if trend < self.market_trend_thresh:
                    target *= self.market_trend_penalty
                elif self.market_trend_boost_thresh is not None and trend > self.market_trend_boost_thresh:
                    target *= self.market_trend_boost

        if self.market_drawdown_limit is not None:
            dd_window = close.tail(self.market_drawdown_window)
            if len(dd_window) >= min(self.risk_min_history, self.market_drawdown_window):
                drawdown = float((dd_window / dd_window.cummax() - 1.0).min())
                if np.isfinite(drawdown) and drawdown <= -self.market_drawdown_limit:
                    target *= self.market_drawdown_penalty

        if self.crash_guard and self.crash_return_limit is not None:
            if len(close) >= self.crash_return_lookback + 1:
                recent = close.tail(self.crash_return_lookback + 1)
                recent_return = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
                if np.isfinite(recent_return) and recent_return <= -self.crash_return_limit:
                    self._crash_cooldown_left = max(self._crash_cooldown_left, self.crash_cooldown_steps)
            if self._crash_cooldown_left > 0:
                target *= self.crash_penalty
                self._crash_cooldown_left = max(0, self._crash_cooldown_left - 1)

        prev = self._smoothed_risk_degree
        target = self._clip_dynamic_risk(target)
        if target > prev and self.max_risk_step_up is not None:
            target = min(target, prev + self.max_risk_step_up)
        elif target < prev and self.max_risk_step_down is not None:
            target = max(target, prev - self.max_risk_step_down)
        alpha = self.risk_smoothing_up if target >= prev else self.risk_smoothing_down
        smoothed = (1.0 - alpha) * prev + alpha * target
        self._smoothed_risk_degree = self._clip_dynamic_risk(smoothed)
        return self._smoothed_risk_degree

    def get_risk_degree(self, trade_step=None):
        if not self.dynamic_risk:
            return super().get_risk_degree(trade_step)
        try:
            if trade_step is None:
                trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, _ = self.trade_calendar.get_step_time(trade_step)
        except Exception:
            return self._smoothed_risk_degree
        return self._compute_dynamic_risk_degree_for_date(trade_start_time)

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        if score.empty:
            return {}
        score = self._apply_trade_score_controls(score, trade_start_time).dropna()
        if score.empty:
            return {}
        score = self._apply_sector_cap(score).dropna()
        if score.empty:
            return {}

        sorted_scores = score.sort_values(ascending=False)
        cand_n = min(len(sorted_scores), max(self.topk, 1) * self.liquidity_buffer)
        candidates = sorted_scores.iloc[:cand_n]
        candidates = self._filter_by_liquidity(candidates, trade_start_time)

        bench = self._benchmark_weights(sorted_scores.index.tolist(), trade_start_time)
        if candidates.empty and bench.empty:
            return {}
        alpha_top = candidates.sort_values(ascending=False).head(self.topk)
        alpha_weights = self._score_to_weight(alpha_top) if not alpha_top.empty else pd.Series(dtype=float)
        benchmark_core_weight = self._effective_benchmark_core_weight(
            alpha_weights,
            bench,
            trade_start_time,
        )
        weights = build_benchmark_aware_weights(
            candidates,
            bench,
            topk=self.topk,
            benchmark_core_weight=benchmark_core_weight,
                alpha_weighting=self.weighting,
                alpha_temperature=self.temperature,
                max_weight=self.max_weight,
                max_benchmark_weight=self.benchmark_max_weight,
                max_active_weight=self.max_active_weight,
            )
        weights = self._apply_vol_scale(weights, trade_start_time)
        if (
            self.max_turnover is not None
            or self.min_trade_weight > 0
            or self.min_position_weight > 0
            or self.max_holdings is not None
        ):
            weights = limit_turnover_toward_target(
                weights,
                self._current_stock_weights(current),
                max_turnover=self.max_turnover,
                min_trade_weight=self.min_trade_weight,
                min_position_weight=self.min_position_weight,
                max_positions=self.max_holdings,
            )
        return {inst: float(w) for inst, w in weights.items() if w > 0}


class HedgedBenchmarkAwareScoreWeightedStrategy(BenchmarkAwareScoreWeightedStrategy):
    """
    Benchmark-aware strategy with an optional long-only hedge ETF sleeve.

    This deliberately supports only long hedge instruments, such as inverse ETFs
    present in the qlib provider. The current order generation path is not
    treated as audited for short targets.
    """

    def __init__(
        self,
        *,
        hedge_mode: str = "long_hedge",
        hedge_tickers: Optional[Iterable[str]] = None,
        hedge_tickers_file: Optional[str] = None,
        hedge_weight: float = 0.0,
        hedge_min_weight: float = 0.0,
        hedge_max_weight: float = 0.35,
        hedge_trend_window: int = 63,
        hedge_trend_thresh: Optional[float] = -0.04,
        hedge_drawdown_window: int = 126,
        hedge_drawdown_limit: Optional[float] = 0.10,
        hedge_crash_return_lookback: int = 5,
        hedge_crash_return_limit: Optional[float] = 0.06,
        hedge_min_history: int = 40,
        hedge_smoothing: float = 1.0,
        hedge_smoothing_up: Optional[float] = None,
        hedge_smoothing_down: Optional[float] = None,
        hedge_max_step_up: Optional[float] = None,
        hedge_max_step_down: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hedge_mode = str(hedge_mode or "off").strip().lower()
        self.hedge_tickers = self._coerce_ticker_list(hedge_tickers)
        self.hedge_tickers_file = hedge_tickers_file
        self.hedge_min_weight = float(np.clip(float(hedge_min_weight), 0.0, 1.0))
        self.hedge_max_weight = float(np.clip(float(hedge_max_weight), self.hedge_min_weight, 1.0))
        self.hedge_weight = float(np.clip(float(hedge_weight), self.hedge_min_weight, self.hedge_max_weight))
        self.hedge_trend_window = max(2, int(hedge_trend_window))
        self.hedge_trend_thresh = None if hedge_trend_thresh is None else float(hedge_trend_thresh)
        self.hedge_drawdown_window = max(2, int(hedge_drawdown_window))
        self.hedge_drawdown_limit = None if hedge_drawdown_limit is None else abs(float(hedge_drawdown_limit))
        self.hedge_crash_return_lookback = max(1, int(hedge_crash_return_lookback))
        self.hedge_crash_return_limit = (
            None if hedge_crash_return_limit is None else abs(float(hedge_crash_return_limit))
        )
        self.hedge_min_history = max(2, int(hedge_min_history))
        self.hedge_smoothing = float(np.clip(float(hedge_smoothing), 0.0, 1.0))
        self.hedge_smoothing_up = (
            self.hedge_smoothing
            if hedge_smoothing_up is None
            else float(np.clip(float(hedge_smoothing_up), 0.0, 1.0))
        )
        self.hedge_smoothing_down = (
            self.hedge_smoothing
            if hedge_smoothing_down is None
            else float(np.clip(float(hedge_smoothing_down), 0.0, 1.0))
        )
        self.hedge_max_step_up = None if hedge_max_step_up is None else max(0.0, float(hedge_max_step_up))
        self.hedge_max_step_down = None if hedge_max_step_down is None else max(0.0, float(hedge_max_step_down))
        self._hedge_ticker_cache: Optional[List[str]] = None
        self._smoothed_hedge_weight = self.hedge_weight
        self._last_hedge_weight = 0.0

    def _configured_hedge_tickers(self) -> List[str]:
        if self._hedge_ticker_cache is not None:
            return self._hedge_ticker_cache
        tickers = list(self.hedge_tickers)
        if self.hedge_tickers_file:
            path = Path(self.hedge_tickers_file).expanduser()
            if path.exists():
                tickers.extend(path.read_text(encoding="utf-8").splitlines())
        self._hedge_ticker_cache = self._coerce_ticker_list(tickers)
        return self._hedge_ticker_cache

    def _clip_hedge_weight(self, value: float) -> float:
        return float(np.clip(float(value), self.hedge_min_weight, self.hedge_max_weight))

    def _compute_hedge_weight_for_date(self, trade_start_time) -> float:
        if self.hedge_mode not in {"long_hedge", "long", "inverse_etf"}:
            self._last_hedge_weight = 0.0
            return 0.0

        target = float(self.hedge_weight)
        close = self._market_risk_close_history(trade_start_time)
        if len(close) >= self.hedge_min_history:
            if self.hedge_trend_thresh is not None:
                trend = close.tail(self.hedge_trend_window)
                if len(trend) >= min(self.hedge_min_history, self.hedge_trend_window):
                    trend_ret = float(trend.iloc[-1] / trend.iloc[0] - 1.0)
                    if np.isfinite(trend_ret) and trend_ret < self.hedge_trend_thresh:
                        target = max(target, self.hedge_max_weight)

            if self.hedge_drawdown_limit is not None:
                dd_window = close.tail(self.hedge_drawdown_window)
                if len(dd_window) >= min(self.hedge_min_history, self.hedge_drawdown_window):
                    drawdown = float((dd_window / dd_window.cummax() - 1.0).min())
                    if np.isfinite(drawdown) and drawdown <= -self.hedge_drawdown_limit:
                        target = max(target, self.hedge_max_weight)

            if self.hedge_crash_return_limit is not None and len(close) >= self.hedge_crash_return_lookback + 1:
                recent = close.tail(self.hedge_crash_return_lookback + 1)
                recent_return = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
                if np.isfinite(recent_return) and recent_return <= -self.hedge_crash_return_limit:
                    target = max(target, self.hedge_max_weight)

        prev = self._smoothed_hedge_weight
        target = self._clip_hedge_weight(target)
        if target > prev and self.hedge_max_step_up is not None:
            target = min(target, prev + self.hedge_max_step_up)
        elif target < prev and self.hedge_max_step_down is not None:
            target = max(target, prev - self.hedge_max_step_down)
        alpha = self.hedge_smoothing_up if target >= prev else self.hedge_smoothing_down
        smoothed = (1.0 - alpha) * prev + alpha * target
        self._smoothed_hedge_weight = self._clip_hedge_weight(smoothed)
        self._last_hedge_weight = self._smoothed_hedge_weight
        return self._smoothed_hedge_weight

    def _hedge_weights(self, trade_start_time) -> pd.Series:
        tickers = self._configured_hedge_tickers()
        if not tickers:
            return pd.Series(dtype=float)
        pre_date = get_pre_trading_date(trade_start_time, future=True)
        try:
            data = D.features(tickers, ["$close"], start_time=pre_date, end_time=pre_date)
        except Exception:
            return pd.Series(dtype=float)
        close = self._series_from_features(data, "$close")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        available = close[close > 0].index.astype(str).tolist()
        if not available:
            return pd.Series(dtype=float)
        return normalize_long_weights(pd.Series(1.0, index=available, dtype=float))

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        base_dict = super().generate_target_weight_position(score, current, trade_start_time, trade_end_time)
        base = normalize_long_weights(pd.Series(base_dict, dtype=float))
        if base.empty:
            return {}

        hedge_sleeve = self._compute_hedge_weight_for_date(trade_start_time)
        if hedge_sleeve <= 1e-12:
            return {inst: float(w) for inst, w in base.items() if w > 0}

        hedge = self._hedge_weights(trade_start_time)
        if hedge.empty:
            self._last_hedge_weight = 0.0
            return {inst: float(w) for inst, w in base.items() if w > 0}

        hedge_sleeve = self._clip_hedge_weight(hedge_sleeve)
        target = base.mul(1.0 - hedge_sleeve).add(hedge.mul(hedge_sleeve), fill_value=0.0)
        target = normalize_long_weights(target)
        if (
            self.max_turnover is not None
            or self.min_trade_weight > 0
            or self.min_position_weight > 0
            or self.max_holdings is not None
        ):
            max_positions = None if self.max_holdings is None else int(self.max_holdings) + len(hedge)
            target = limit_turnover_toward_target(
                target,
                self._current_stock_weights(current),
                max_turnover=self.max_turnover,
                min_trade_weight=self.min_trade_weight,
                min_position_weight=self.min_position_weight,
                max_positions=max_positions,
            )
        return {inst: float(w) for inst, w in target.items() if w > 0}


class WeeklyBenchmarkAwareScoreWeightedStrategy(BenchmarkAwareScoreWeightedStrategy):
    """
    BenchmarkAwareScoreWeightedStrategy that only trades on a specific weekday.
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def _is_rebalance_day(self, trade_start_time) -> bool:
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
            return TradeDecisionWO([], self)
        if not self._is_rebalance_day(trade_start_time):
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)


class WeeklyHedgedBenchmarkAwareScoreWeightedStrategy(HedgedBenchmarkAwareScoreWeightedStrategy):
    """
    HedgedBenchmarkAwareScoreWeightedStrategy that only trades on a specific weekday.
    """

    def __init__(self, *args, rebalance_weekday: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebalance_weekday = rebalance_weekday

    def _is_rebalance_day(self, trade_start_time) -> bool:
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
            return TradeDecisionWO([], self)
        if not self._is_rebalance_day(trade_start_time):
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)
