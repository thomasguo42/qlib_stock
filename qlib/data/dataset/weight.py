# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import numpy as np
import pandas as pd


class Reweighter:
    def __init__(self, *args, **kwargs):
        """
        To initialize the Reweighter, users should provide specific methods to let reweighter do the reweighting (such as sample-wise, rule-based).
        """
        raise NotImplementedError()

    def reweight(self, data: object) -> object:
        """
        Get weights for data

        Parameters
        ----------
        data : object
            The input data.
            The first dimension is the index of samples

        Returns
        -------
        object:
            the weights info for the data
        """
        raise NotImplementedError(f"This type of input is not supported")


class RecencyReweighter(Reweighter):
    """
    Apply exponentially decaying sample weights by trading date.

    The most recent date in the prepared dataset receives weight 1.0. Older
    dates decay by half every `half_life_days` observed trading dates, floored
    at `min_weight`.
    """

    def __init__(self, half_life_days: int = 252, min_weight: float = 0.25):
        self.half_life_days = max(1, int(half_life_days))
        self.min_weight = min(1.0, max(0.0, float(min_weight)))

    def reweight(self, data: object) -> object:
        if not hasattr(data, "index"):
            raise TypeError("RecencyReweighter expects data with a pandas-like index")
        index = data.index
        if isinstance(index, pd.MultiIndex) and "datetime" in index.names:
            dates = pd.DatetimeIndex(index.get_level_values("datetime")).normalize()
        else:
            dates = pd.DatetimeIndex(index).normalize()
        if len(dates) == 0:
            return pd.Series(dtype=float, index=index)

        unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
        date_pos = {dt: pos for pos, dt in enumerate(unique_dates)}
        latest_pos = len(unique_dates) - 1
        ages = pd.Series([latest_pos - date_pos[pd.Timestamp(dt)] for dt in dates], index=index, dtype=float)
        decay = ages.map(lambda age: math.pow(0.5, float(age) / float(self.half_life_days)))
        weights = self.min_weight + (1.0 - self.min_weight) * decay
        return weights.astype(float)


class RegimeRecencyReweighter(Reweighter):
    """
    Apply sample weights that balance dates and market regimes while optionally
    preserving recency emphasis.

    This is intended for cross-sectional equity models where raw sample counts
    can let a few dense or repeated market states dominate training.  Regime
    balancing uses a lagged market-state feature from the feature group, so it
    does not depend on future labels.
    """

    def __init__(
        self,
        half_life_days: int = 252,
        min_weight: float = 0.25,
        regime_feature: str = "MKT_QQQ_RET_63D_LAG1",
        regime_thresholds=None,
        date_balance: bool = True,
        year_balance: bool = False,
        regime_balance: bool = True,
        label_tail_weight: float = 0.0,
        label_tail_quantile: float = 0.20,
        max_weight: float = 5.0,
    ):
        self.half_life_days = max(0, int(half_life_days))
        self.min_weight = min(1.0, max(0.0, float(min_weight)))
        self.regime_feature = str(regime_feature or "").strip()
        self.regime_thresholds = tuple(float(v) for v in (regime_thresholds or (-0.04, 0.03)))
        self.date_balance = bool(date_balance)
        self.year_balance = bool(year_balance)
        self.regime_balance = bool(regime_balance)
        self.label_tail_weight = max(0.0, float(label_tail_weight))
        self.label_tail_quantile = min(0.49, max(0.0, float(label_tail_quantile)))
        self.max_weight = max(1.0, float(max_weight))

    @staticmethod
    def _dates(index) -> pd.DatetimeIndex:
        if isinstance(index, pd.MultiIndex) and "datetime" in index.names:
            return pd.DatetimeIndex(index.get_level_values("datetime")).normalize()
        return pd.DatetimeIndex(index).normalize()

    @staticmethod
    def _canonical(name: object) -> str:
        text = str(name).strip()
        if text.startswith("$"):
            text = text[1:]
        return text.upper()

    def _resolve_column(self, data: pd.DataFrame, group: str, field: str):
        wanted = self._canonical(field)
        if isinstance(data.columns, pd.MultiIndex):
            direct = (group, field)
            if direct in data.columns:
                return direct
            for col in data.columns:
                if not isinstance(col, tuple) or len(col) < 2:
                    continue
                if str(col[0]) != str(group):
                    continue
                if self._canonical(col[-1]) == wanted:
                    return col
        else:
            if field in data.columns:
                return field
            for col in data.columns:
                if self._canonical(col) == wanted:
                    return col
        return None

    @staticmethod
    def _normalize(weights: pd.Series) -> pd.Series:
        weights = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mean = float(weights.mean(skipna=True)) if len(weights) else float("nan")
        if not math.isfinite(mean) or mean <= 0:
            return pd.Series(1.0, index=weights.index, dtype=float)
        return (weights / mean).astype(float)

    def _recency_weights(self, index, dates: pd.DatetimeIndex) -> pd.Series:
        if self.half_life_days <= 0:
            return pd.Series(1.0, index=index, dtype=float)
        unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
        date_pos = {dt: pos for pos, dt in enumerate(unique_dates)}
        latest_pos = len(unique_dates) - 1
        ages = pd.Series([latest_pos - date_pos[pd.Timestamp(dt)] for dt in dates], index=index, dtype=float)
        decay = ages.map(lambda age: math.pow(0.5, float(age) / float(self.half_life_days)))
        return (self.min_weight + (1.0 - self.min_weight) * decay).astype(float)

    def _date_balance_weights(self, index, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.date_balance:
            return pd.Series(1.0, index=index, dtype=float)
        counts = pd.Series(dates, index=index).map(pd.Series(dates).value_counts())
        return self._normalize(1.0 / counts.astype(float).clip(lower=1.0))

    def _year_balance_weights(self, index, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.year_balance:
            return pd.Series(1.0, index=index, dtype=float)
        years = pd.Series(dates.year, index=index)
        counts = years.map(years.value_counts())
        return self._normalize(1.0 / counts.astype(float).clip(lower=1.0))

    def _regime_balance_weights(self, data: pd.DataFrame, index, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.regime_balance or not self.regime_feature:
            return pd.Series(1.0, index=index, dtype=float)
        col = self._resolve_column(data, "feature", self.regime_feature)
        if col is None:
            return pd.Series(1.0, index=index, dtype=float)
        values = pd.to_numeric(data.loc[:, col], errors="coerce")
        by_date = values.groupby(dates).mean()
        thresholds = sorted(self.regime_thresholds)
        bins = [-np.inf] + thresholds + [np.inf]
        regime_by_date = pd.cut(by_date, bins=bins, labels=False, include_lowest=True)
        regime = pd.Series(dates, index=index).map(regime_by_date).fillna(-1).astype(int)
        counts = regime.map(regime.value_counts())
        return self._normalize(1.0 / counts.astype(float).clip(lower=1.0))

    def _label_tail_weights(self, data: pd.DataFrame, index, dates: pd.DatetimeIndex) -> pd.Series:
        if self.label_tail_weight <= 0 or self.label_tail_quantile <= 0:
            return pd.Series(1.0, index=index, dtype=float)
        label_cols = []
        if isinstance(data.columns, pd.MultiIndex):
            label_cols = [col for col in data.columns if isinstance(col, tuple) and len(col) > 0 and col[0] == "label"]
        elif "label" in data.columns:
            label_cols = ["label"]
        if not label_cols:
            return pd.Series(1.0, index=index, dtype=float)
        label = pd.to_numeric(data.loc[:, label_cols[0]], errors="coerce")
        ranks = label.groupby(dates).rank(method="first", pct=True)
        q = self.label_tail_quantile
        tail = (ranks <= q) | (ranks >= (1.0 - q))
        weights = pd.Series(1.0, index=index, dtype=float)
        weights.loc[tail.fillna(False)] += self.label_tail_weight
        return self._normalize(weights)

    def reweight(self, data: object) -> object:
        if not hasattr(data, "index"):
            raise TypeError("RegimeRecencyReweighter expects data with a pandas-like index")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(index=data.index)
        index = data.index
        if len(index) == 0:
            return pd.Series(dtype=float, index=index)
        dates = self._dates(index)
        weights = pd.Series(1.0, index=index, dtype=float)
        weights *= self._recency_weights(index, dates)
        weights *= self._date_balance_weights(index, dates)
        weights *= self._year_balance_weights(index, dates)
        weights *= self._regime_balance_weights(data, index, dates)
        weights *= self._label_tail_weights(data, index, dates)
        weights = self._normalize(weights)
        return weights.clip(lower=0.0, upper=self.max_weight).astype(float)
