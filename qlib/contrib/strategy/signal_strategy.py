# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd

from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree


class TopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    # 4. Regenerate results with forbid_all_trade_at_limit set to false and flip the default to false, as it is consistent with reality.
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        sector_map_csv: Optional[str] = None,
        sector_ticker_col: str = "ticker",
        sector_col: str = "sector",
        max_sector_count: Optional[int] = None,
        max_sector_weight: Optional[float] = None,
        feature_score_weights: Optional[Dict[str, float]] = None,
        feature_min_percentiles: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        forbid_all_trade_at_limit : bool
            if forbid all trades when limit_up or limit_down reached.

            if forbid_all_trade_at_limit:

                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at
                limit down, though allowed in reality.

            else:

                strategy will sell at limit up and buy ad limit down.
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.sector_map_csv = sector_map_csv
        self.sector_ticker_col = sector_ticker_col
        self.sector_col = sector_col
        if max_sector_count is None and max_sector_weight is not None:
            max_sector_count = int(np.floor(float(max_sector_weight) * int(topk)))
        self.max_sector_count = None if max_sector_count is None else max(1, int(max_sector_count))
        self._sector_map = None
        self.feature_score_weights = self._normalize_feature_control_map(feature_score_weights)
        self.feature_min_percentiles = self._normalize_feature_control_map(
            feature_min_percentiles, lower=0.0, upper=1.0
        )
        self._feature_control_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    @staticmethod
    def _normalize_feature_control_map(
        values: Optional[Dict[str, Any]], *, lower: Optional[float] = None, upper: Optional[float] = None
    ) -> Dict[str, float]:
        if not values:
            return {}
        if not isinstance(values, dict):
            raise TypeError("feature controls must be a mapping from qlib feature expression to numeric value")
        out: Dict[str, float] = {}
        for field, raw in values.items():
            name = str(field).strip()
            if not name:
                continue
            value = float(raw)
            if not np.isfinite(value):
                continue
            if lower is not None:
                value = max(float(lower), value)
            if upper is not None:
                value = min(float(upper), value)
            out[name] = value
        return out

    @staticmethod
    def _cs_zscore(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        std = float(numeric.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=values.index, dtype=float)
        mean = float(numeric.mean())
        return ((numeric - mean) / std).fillna(0.0).astype(float)

    @staticmethod
    def _demote_masked_scores(scores: pd.Series, mask: pd.Series) -> pd.Series:
        if scores.empty:
            return scores
        adjusted = scores.copy()
        mask = mask.reindex(adjusted.index).fillna(False).astype(bool)
        if not bool(mask.any()):
            return adjusted
        finite_scores = adjusted.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_scores.empty:
            return adjusted
        spread = float(finite_scores.max() - finite_scores.min())
        if not np.isfinite(spread):
            spread = 1.0
        floor = float(finite_scores.min() - max(1.0, spread))
        step = max(1e-9, max(1.0, spread) * 1e-9)
        for inst in adjusted.index[mask.to_numpy()]:
            adjusted.loc[inst] = floor
            floor -= step
        return adjusted

    @staticmethod
    def _feature_frame_by_instrument(features, fields: List[str], instruments: List[str]) -> pd.DataFrame:
        if features is None:
            return pd.DataFrame(index=pd.Index(instruments))
        if isinstance(features, pd.Series):
            features = features.to_frame(fields[0] if len(fields) == 1 else "feature")
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.DataFrame(index=pd.Index(instruments))
        frame = features.copy()
        if len(fields) == 1 and fields[0] not in frame.columns and frame.shape[1] == 1:
            frame = frame.rename(columns={frame.columns[0]: fields[0]})
        cols = [field for field in fields if field in frame.columns]
        if not cols:
            return pd.DataFrame(index=pd.Index(instruments))
        frame = frame[cols]
        if isinstance(frame.index, pd.MultiIndex):
            level = "instrument" if "instrument" in frame.index.names else frame.index.nlevels - 1
            frame = frame.groupby(level=level).last()
        frame.index = frame.index.astype(str)
        return frame.reindex([str(inst) for inst in instruments])

    @staticmethod
    def _feature_panel(features, fields: List[str]) -> pd.DataFrame:
        if features is None:
            return pd.DataFrame()
        if isinstance(features, pd.Series):
            features = features.to_frame(fields[0] if len(fields) == 1 else "feature")
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.DataFrame()
        frame = features.copy()
        if len(fields) == 1 and fields[0] not in frame.columns and frame.shape[1] == 1:
            frame = frame.rename(columns={frame.columns[0]: fields[0]})
        cols = [field for field in fields if field in frame.columns]
        if not cols:
            return pd.DataFrame()
        frame = frame[cols]
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == ["instrument", "datetime"]:
            frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()
        return frame.sort_index()

    def _load_feature_controls(
        self, instruments: List[str], start_time, end_time, fields: List[str]
    ) -> pd.DataFrame:
        if not instruments or not fields:
            return pd.DataFrame(index=pd.Index(instruments))
        key = tuple(fields)
        cache = self._feature_control_cache.setdefault(
            key,
            {"frame": pd.DataFrame(), "instruments": set(), "start_time": None, "end_time": None},
        )
        requested = [str(inst) for inst in instruments]
        missing = [inst for inst in requested if inst not in cache["instruments"]]
        if missing:
            cal = getattr(self.trade_calendar, "_calendar", None)
            if cal is not None and len(cal) > 0:
                cache_start, cache_end = pd.Timestamp(cal[0]), pd.Timestamp(cal[-1])
            else:
                cache_start, cache_end = pd.Timestamp(start_time), pd.Timestamp(end_time)
            data = D.features(missing, fields, start_time=cache_start, end_time=cache_end)
            panel = self._feature_panel(data, fields)
            if cache["frame"].empty:
                cache["frame"] = panel
            elif not panel.empty:
                cache["frame"] = pd.concat([cache["frame"], panel]).sort_index()
            cache["instruments"].update(missing)
            cache["start_time"] = cache_start
            cache["end_time"] = cache_end
        frame = cache["frame"]
        if frame.empty:
            return pd.DataFrame(index=pd.Index(requested))
        if isinstance(frame.index, pd.MultiIndex) and "datetime" in frame.index.names:
            dt = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
            frame = frame[(dt >= pd.Timestamp(start_time)) & (dt <= pd.Timestamp(end_time))]
        return self._feature_frame_by_instrument(frame, fields, requested)

    def _apply_feature_score_controls(self, pred_score: pd.Series, pred_start_time, pred_end_time) -> pd.Series:
        if (
            pred_score is None
            or pred_score.empty
            or (not self.feature_score_weights and not self.feature_min_percentiles)
        ):
            return pred_score
        fields = list(dict.fromkeys([*self.feature_score_weights.keys(), *self.feature_min_percentiles.keys()]))
        instruments = [str(inst) for inst in pred_score.index]
        features = self._load_feature_controls(instruments, pred_start_time, pred_end_time, fields)
        if features.empty:
            return pred_score

        base = pd.to_numeric(pred_score, errors="coerce")
        adjusted = self._cs_zscore(base) if self.feature_score_weights else base.astype(float).copy()
        for field, weight in self.feature_score_weights.items():
            if field not in features.columns:
                continue
            feature = pd.to_numeric(features[field], errors="coerce").reindex(instruments)
            adjusted = adjusted + float(weight) * self._cs_zscore(feature).reindex(adjusted.index).fillna(0.0)

        for field, min_pct in self.feature_min_percentiles.items():
            if field not in features.columns:
                continue
            feature = pd.to_numeric(features[field], errors="coerce").reindex(instruments)
            pct = feature.rank(pct=True)
            adjusted = self._demote_masked_scores(adjusted, pct.isna() | (pct < float(min_pct)))
        return adjusted.where(base.notna(), np.nan).reindex(pred_score.index)

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

    def _apply_sector_cap(self, pred_score: pd.Series) -> pd.Series:
        if self.max_sector_count is None or self.max_sector_count <= 0 or pred_score is None or pred_score.empty:
            return pred_score
        sector_map = self._load_sector_map()
        if not sector_map:
            return pred_score

        sorted_score = pred_score.dropna().sort_values(ascending=False)
        if sorted_score.empty:
            return pred_score
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
        return adjusted.reindex(pred_score.index)

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        pred_score = self._apply_feature_score_controls(pred_score, pred_start_time, pred_end_time)
        pred_score = self._apply_sector_cap(pred_score)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        # current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report

            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:

                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)


class EnhancedIndexingStrategy(WeightStrategyBase):
    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping={},
        optimizer_kwargs={},
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        if EnhancedIndexingOptimizer is None:
            raise ImportError("EnhancedIndexingOptimizer is unavailable; install a compatible cvxpy/numpy stack")
        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # NOTE: for stocks missing specific_risk, we always assume it has the highest volatility
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # previous trade date

        # load risk data
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"no risk data for {pre_date:%Y-%m-%d}, skip optimization")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # transform score
        # NOTE: for stocks missing score, we always assume they have the lowest score
        score = score.reindex(universe).fillna(score.min()).values

        # get current weight
        # NOTE: if a stock is not in universe, its current weight will be zero
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "current weight has negative values"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # sum of weight should be risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"previous total holdings excess risk degree (current: {cur_weight.sum()})")

        # load bench weight
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # whether stock tradable
        # NOTE: currently we use last day volume to check whether tradable
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # mask force sell
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # optimize
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("trade date: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("number of holding stocks: {}".format(len(target_weight_position)))
            self.logger.info("total holding weight: {:.6f}".format(weight.sum()))

        return target_weight_position
