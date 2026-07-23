# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Hashable, Iterable, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model


def _normalize_alias(value: object) -> str:
    text = str(value).strip()
    if text.startswith("$"):
        text = text[1:]
    return text.upper()


def _column_aliases(column: Hashable) -> Iterable[str]:
    yield _normalize_alias(column)
    if isinstance(column, tuple):
        for item in column:
            yield _normalize_alias(item)
        if column:
            yield _normalize_alias(column[-1])


def _date_level(index: pd.Index):
    if isinstance(index, pd.MultiIndex):
        return "datetime" if "datetime" in index.names else 0
    return None


def _daily_zscore(values: pd.Series, index: pd.Index) -> pd.Series:
    level = _date_level(index)
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if level is None:
        std = float(numeric.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=index, dtype=float)
        return ((numeric - float(numeric.mean())) / std).fillna(0.0)

    def score_day(day: pd.Series) -> pd.Series:
        std = float(day.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=day.index, dtype=float)
        return ((day - float(day.mean())) / std).fillna(0.0)

    return numeric.groupby(level=level, group_keys=False).apply(score_day).reindex(index)


class FeatureWeightedScoreModel(Model):
    """Deterministic cross-sectional weighted feature scorer.

    This model is intentionally non-iterative.  It is useful as a release
    baseline when a pre-audited factor composite is stronger than a fitted
    model and we need to avoid fitting noise in walk-forward windows.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        normalize_by_date: bool = True,
        missing: str = "raise",
        regime_feature: Optional[str] = None,
        regime_threshold: float = 0.0,
        risk_on_weights: Optional[Dict[str, float]] = None,
        risk_off_weights: Optional[Dict[str, float]] = None,
    ):
        if not weights:
            raise ValueError("FeatureWeightedScoreModel requires at least one feature weight")
        if missing not in {"raise", "ignore"}:
            raise ValueError("missing must be 'raise' or 'ignore'")
        self.weights = {str(name): float(weight) for name, weight in weights.items()}
        self.normalize_by_date = bool(normalize_by_date)
        self.missing = str(missing)
        self.regime_feature = str(regime_feature).strip() if regime_feature else None
        self.regime_threshold = float(regime_threshold)
        self.risk_on_weights = {str(name): float(weight) for name, weight in (risk_on_weights or {}).items()}
        self.risk_off_weights = {str(name): float(weight) for name, weight in (risk_off_weights or {}).items()}
        if bool(self.regime_feature) != bool(self.risk_on_weights and self.risk_off_weights):
            raise ValueError("regime_feature, risk_on_weights, and risk_off_weights must be provided together")
        self.fitted = False

    @staticmethod
    def _feature_columns(frame: pd.DataFrame) -> Dict[str, Hashable]:
        aliases: Dict[str, Hashable] = {}
        for column in frame.columns:
            for alias in _column_aliases(column):
                aliases.setdefault(alias, column)
        return aliases

    def _resolve_column(self, features: pd.DataFrame, aliases: Dict[str, Hashable], name: str):
        column = aliases.get(_normalize_alias(name))
        if column is None and self.missing == "raise":
            raise KeyError(f"missing feature columns: {[str(name)]}")
        return column

    def _score_with_weights(
        self,
        features: pd.DataFrame,
        aliases: Dict[str, Hashable],
        weights: Dict[str, float],
    ) -> pd.Series:
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.Series(dtype=float)
        score = pd.Series(0.0, index=features.index, dtype=float)
        missing = []
        for name, weight in weights.items():
            column = aliases.get(_normalize_alias(name))
            if column is None:
                missing.append(str(name))
                continue
            values = pd.to_numeric(features[column], errors="coerce").astype(float)
            if self.normalize_by_date:
                values = _daily_zscore(values, features.index)
            else:
                values = values.fillna(0.0)
            score = score + float(weight) * values.reindex(features.index).fillna(0.0)
        if missing and self.missing == "raise":
            raise KeyError(f"missing feature columns: {missing}")
        score.name = "score"
        return score.replace([np.inf, -np.inf], np.nan)

    def _score_frame(self, features: pd.DataFrame) -> pd.Series:
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.Series(dtype=float)
        aliases = self._feature_columns(features)
        if not self.regime_feature:
            return self._score_with_weights(features, aliases, self.weights)

        regime_col = self._resolve_column(features, aliases, self.regime_feature)
        if regime_col is None:
            return self._score_with_weights(features, aliases, self.weights)
        risk_on = self._score_with_weights(features, aliases, self.risk_on_weights)
        risk_off = self._score_with_weights(features, aliases, self.risk_off_weights)
        regime = pd.to_numeric(features[regime_col], errors="coerce")
        use_risk_on = regime >= self.regime_threshold
        score = risk_off.where(~use_risk_on, risk_on)
        score.name = "score"
        return score.replace([np.inf, -np.inf], np.nan)

    def fit(self, dataset: DatasetH, **kwargs):
        self.fitted = True
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self._score_frame(features)


class RegimeSleeveScoreModel(Model):
    """Deterministic factor-sleeve scorer with no-lookahead market-state routing.

    Each sleeve is a weighted feature composite.  The final score is a
    non-negative blend of sleeve scores selected by lagged market-state
    features.  This keeps regime logic explicit and auditable while avoiding a
    fitted model when the research goal is to test robust portfolio behavior.
    """

    def __init__(
        self,
        sleeves: Dict[str, Dict[str, float]],
        state_weights: Dict[str, Dict[str, float]],
        default_state: str = "chop",
        normalize_by_date: bool = True,
        missing: str = "raise",
        trend_feature: str = "MKT_QQQ_RET_63D_LAG1",
        fast_trend_feature: str = "MKT_QQQ_RET_20D_LAG1",
        drawdown_feature: str = "MKT_QQQ_DD_126D_LAG1",
        vol_feature: Optional[str] = "MKT_QQQ_VOL_20D_LAG1",
        breadth_feature: Optional[str] = "MKT_BREADTH_RET63_POS_LAG1",
        risk_on_threshold: float = 0.03,
        risk_off_threshold: float = -0.04,
        recovery_fast_threshold: float = 0.02,
        fading_fast_threshold: float = -0.02,
        drawdown_limit: float = -0.10,
        drawdown_warning: float = -0.06,
        high_vol_threshold: Optional[float] = 0.35,
        breadth_threshold: Optional[float] = 0.45,
    ):
        if not sleeves:
            raise ValueError("RegimeSleeveScoreModel requires at least one sleeve")
        if not state_weights:
            raise ValueError("RegimeSleeveScoreModel requires state_weights")
        if missing not in {"raise", "ignore"}:
            raise ValueError("missing must be 'raise' or 'ignore'")
        self.sleeves = {
            str(name): {str(feature): float(weight) for feature, weight in weights.items()}
            for name, weights in sleeves.items()
        }
        for name, weights in self.sleeves.items():
            if not weights:
                raise ValueError(f"sleeve {name!r} has no feature weights")
        self.state_weights = {
            str(state): {str(sleeve): float(weight) for sleeve, weight in weights.items()}
            for state, weights in state_weights.items()
        }
        sleeve_names = set(self.sleeves)
        for state, weights in self.state_weights.items():
            if not weights:
                raise ValueError(f"state {state!r} has no sleeve weights")
            unknown = sorted(set(weights) - sleeve_names)
            if unknown:
                raise ValueError(f"state {state!r} references unknown sleeves: {unknown}")
            total = 0.0
            for sleeve, weight in weights.items():
                if float(weight) < 0:
                    raise ValueError(f"state {state!r} sleeve {sleeve!r} has negative weight")
                total += float(weight)
            if total <= 0:
                raise ValueError(f"state {state!r} sleeve weights must sum positive")
        self.default_state = str(default_state)
        if self.default_state not in self.state_weights:
            raise ValueError("default_state must be present in state_weights")
        self.normalize_by_date = bool(normalize_by_date)
        self.missing = str(missing)
        self.trend_feature = str(trend_feature)
        self.fast_trend_feature = str(fast_trend_feature)
        self.drawdown_feature = str(drawdown_feature)
        self.vol_feature = None if vol_feature is None else str(vol_feature)
        self.breadth_feature = None if breadth_feature is None else str(breadth_feature)
        self.risk_on_threshold = float(risk_on_threshold)
        self.risk_off_threshold = float(risk_off_threshold)
        self.recovery_fast_threshold = float(recovery_fast_threshold)
        self.fading_fast_threshold = float(fading_fast_threshold)
        self.drawdown_limit = float(drawdown_limit)
        self.drawdown_warning = float(drawdown_warning)
        self.high_vol_threshold = None if high_vol_threshold is None else float(high_vol_threshold)
        self.breadth_threshold = None if breadth_threshold is None else float(breadth_threshold)
        self.fitted = False

    @staticmethod
    def _feature_columns(frame: pd.DataFrame) -> Dict[str, Hashable]:
        return FeatureWeightedScoreModel._feature_columns(frame)

    def _score_with_weights(
        self,
        features: pd.DataFrame,
        aliases: Dict[str, Hashable],
        weights: Dict[str, float],
    ) -> pd.Series:
        scorer = FeatureWeightedScoreModel(
            weights,
            normalize_by_date=self.normalize_by_date,
            missing=self.missing,
        )
        return scorer._score_with_weights(features, aliases, weights)

    def _feature_series(
        self,
        features: pd.DataFrame,
        aliases: Dict[str, Hashable],
        name: Optional[str],
        *,
        required: bool = False,
    ) -> pd.Series:
        if not name:
            return pd.Series(np.nan, index=features.index, dtype=float)
        column = aliases.get(_normalize_alias(name))
        if column is None:
            if required or self.missing == "raise":
                raise KeyError(f"missing feature columns: {[str(name)]}")
            return pd.Series(np.nan, index=features.index, dtype=float)
        return pd.to_numeric(features[column], errors="coerce").astype(float).reindex(features.index)

    def _state_for_frame(self, features: pd.DataFrame, aliases: Dict[str, Hashable]) -> pd.Series:
        trend = self._feature_series(features, aliases, self.trend_feature, required=True)
        fast = self._feature_series(features, aliases, self.fast_trend_feature, required=True)
        drawdown = self._feature_series(features, aliases, self.drawdown_feature, required=True)
        vol = self._feature_series(features, aliases, self.vol_feature)
        breadth = self._feature_series(features, aliases, self.breadth_feature)

        state = pd.Series(self.default_state, index=features.index, dtype=object)
        risk_off = (trend <= self.risk_off_threshold) | (drawdown <= self.drawdown_limit)
        if self.breadth_threshold is not None:
            risk_off = risk_off | ((breadth < self.breadth_threshold) & (trend < 0.0))

        risk_on = (trend >= self.risk_on_threshold) & (drawdown > self.drawdown_warning)
        if self.high_vol_threshold is not None:
            risk_on = risk_on & (vol.isna() | (vol <= self.high_vol_threshold))

        fading = ((trend > 0.0) & (fast <= self.fading_fast_threshold)) | (
            (drawdown <= self.drawdown_warning) & ~risk_off
        )
        recovery = risk_off & (fast >= self.recovery_fast_threshold)

        if "risk_on" in self.state_weights:
            state.loc[risk_on.fillna(False)] = "risk_on"
        if "fading" in self.state_weights:
            state.loc[fading.fillna(False)] = "fading"
        if "risk_off" in self.state_weights:
            state.loc[risk_off.fillna(False)] = "risk_off"
        if "recovery" in self.state_weights:
            state.loc[recovery.fillna(False)] = "recovery"
        return state

    @staticmethod
    def _blend_sleeves(sleeve_scores: Dict[str, pd.Series], weights: Dict[str, float], index: pd.Index) -> pd.Series:
        total = float(sum(float(weight) for weight in weights.values()))
        if total <= 0:
            return pd.Series(0.0, index=index, dtype=float)
        score = pd.Series(0.0, index=index, dtype=float)
        for sleeve, weight in weights.items():
            score = score + (float(weight) / total) * sleeve_scores[sleeve].reindex(index).fillna(0.0)
        return score

    def _score_frame(self, features: pd.DataFrame) -> pd.Series:
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.Series(dtype=float)
        aliases = self._feature_columns(features)
        sleeve_scores = {
            name: self._score_with_weights(features, aliases, weights)
            for name, weights in self.sleeves.items()
        }
        states = self._state_for_frame(features, aliases)
        default_score = self._blend_sleeves(sleeve_scores, self.state_weights[self.default_state], features.index)
        score = default_score.copy()
        for state, weights in self.state_weights.items():
            if state == self.default_state:
                continue
            mask = states == state
            if bool(mask.any()):
                blended = self._blend_sleeves(sleeve_scores, weights, features.index)
                score.loc[mask] = blended.loc[mask]
        score.name = "score"
        return score.replace([np.inf, -np.inf], np.nan)

    def fit(self, dataset: DatasetH, **kwargs):
        self.fitted = True
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self._score_frame(features)


class ICSelectedScoreModel(Model):
    """Walk-forward deterministic scorer that selects factors on the train split only."""

    def __init__(
        self,
        max_features: int = 8,
        min_selected_features: int = 3,
        min_abs_ic: float = 0.0025,
        min_ic_days: int = 120,
        min_daily_count: int = 30,
        min_coverage: float = 0.30,
        min_same_sign_years: int = 2,
        min_worst_year_signed_ic: float = -0.015,
        min_recent_signed_ic: float = 0.0,
        recent_window_days: int = 756,
        selection_step_days: int = 5,
        recent_weight: float = 0.50,
        weight_power: float = 1.0,
        normalize_by_date: bool = True,
        missing: str = "raise",
        fallback_to_best: bool = True,
        regime_feature: Optional[str] = None,
        risk_on_threshold: float = 0.03,
        risk_off_threshold: float = -0.04,
        regime_min_ic_days: Optional[int] = None,
        top_quantile: float = 0.20,
        min_topq_spread: float = -np.inf,
        min_worst_year_topq_spread: float = -np.inf,
        tail_weight: float = 0.0,
    ):
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if min_selected_features <= 0:
            raise ValueError("min_selected_features must be positive")
        if missing not in {"raise", "ignore"}:
            raise ValueError("missing must be 'raise' or 'ignore'")
        self.max_features = int(max_features)
        self.min_selected_features = int(min_selected_features)
        self.min_abs_ic = float(min_abs_ic)
        self.min_ic_days = int(min_ic_days)
        self.min_daily_count = int(min_daily_count)
        self.min_coverage = float(min_coverage)
        self.min_same_sign_years = int(min_same_sign_years)
        self.min_worst_year_signed_ic = float(min_worst_year_signed_ic)
        self.min_recent_signed_ic = float(min_recent_signed_ic)
        self.recent_window_days = int(recent_window_days)
        self.selection_step_days = max(1, int(selection_step_days))
        self.recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
        self.weight_power = float(weight_power)
        self.normalize_by_date = bool(normalize_by_date)
        self.missing = str(missing)
        self.fallback_to_best = bool(fallback_to_best)
        self.regime_feature = str(regime_feature).strip() if regime_feature else None
        self.risk_on_threshold = float(risk_on_threshold)
        self.risk_off_threshold = float(risk_off_threshold)
        self.regime_min_ic_days = None if regime_min_ic_days is None else int(regime_min_ic_days)
        self.top_quantile = float(np.clip(top_quantile, 0.01, 0.49))
        self.min_topq_spread = float(min_topq_spread)
        self.min_worst_year_topq_spread = float(min_worst_year_topq_spread)
        self.tail_weight = float(max(0.0, tail_weight))
        self.fitted = False
        self.selected_weights_: Dict[str, float] = {}
        self.selected_weights_by_state_: Dict[str, Dict[str, float]] = {}
        self.selection_summary_ = pd.DataFrame()
        self.selection_summary_by_state_: Dict[str, pd.DataFrame] = {}
        self.fallback_used_ = False
        self.fallback_used_by_state_: Dict[str, bool] = {}

    @staticmethod
    def _feature_name(column: Hashable) -> str:
        if isinstance(column, tuple) and column:
            return str(column[-1])
        return str(column)

    @staticmethod
    def _split_feature_label(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("empty training data from dataset")
        try:
            features = frame["feature"]
            labels = frame["label"]
        except KeyError as exc:
            raise ValueError("ICSelectedScoreModel requires feature and label columns") from exc
        if isinstance(labels, pd.DataFrame):
            if labels.shape[1] != 1:
                raise ValueError("ICSelectedScoreModel doesn't support multi-label training")
            label = pd.Series(np.squeeze(labels.values), index=labels.index, name=str(labels.columns[0]))
        else:
            label = pd.Series(labels, index=labels.index)
        return features, pd.to_numeric(label, errors="coerce").astype(float)

    @staticmethod
    def _daily_rank_ic(feature: pd.Series, label: pd.Series, min_daily_count: int) -> pd.Series:
        aligned = pd.concat(
            [
                pd.to_numeric(feature, errors="coerce").astype(float).rename("feature"),
                pd.to_numeric(label, errors="coerce").astype(float).rename("label"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)
        level = _date_level(aligned.index)
        if level is None:
            aligned = aligned.dropna()
            if len(aligned) < min_daily_count:
                return pd.Series(dtype=float)
            return pd.Series([aligned["feature"].corr(aligned["label"], method="spearman")])

        rows = []
        for date, day in aligned.groupby(level=level, sort=True):
            day = day.dropna()
            if len(day) < min_daily_count:
                continue
            ic = day["feature"].corr(day["label"], method="spearman")
            if np.isfinite(ic):
                rows.append((pd.Timestamp(date), float(ic)))
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series([value for _, value in rows], index=pd.DatetimeIndex([date for date, _ in rows]))

    def _summarize_feature(self, name: str, values: pd.Series, label: pd.Series) -> Dict[str, float]:
        valid_feature = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        valid_label = label.replace([np.inf, -np.inf], np.nan).notna()
        coverage = float((valid_feature & valid_label).mean()) if len(label) else 0.0
        daily_ic = self._daily_rank_ic(values, label, self.min_daily_count)
        if daily_ic.empty:
            return {
                "feature": name,
                "coverage": coverage,
                "ic_days": 0,
                "mean_ic": np.nan,
                "abs_mean_ic": np.nan,
                "recent_mean_ic": np.nan,
                "same_sign_years": 0,
                "year_count": 0,
                "worst_year_signed_ic": np.nan,
                "selection_score": np.nan,
            }

        mean_ic = float(daily_ic.mean())
        abs_mean_ic = abs(mean_ic)
        direction = 1.0 if mean_ic >= 0 else -1.0
        if self.recent_window_days > 0 and len(daily_ic) > self.recent_window_days:
            recent_ic = daily_ic.iloc[-self.recent_window_days :]
        else:
            recent_ic = daily_ic
        recent_mean_ic = float(recent_ic.mean()) if len(recent_ic) else np.nan
        signed_recent_ic = direction * recent_mean_ic if np.isfinite(recent_mean_ic) else np.nan

        if isinstance(daily_ic.index, pd.DatetimeIndex):
            year_means = daily_ic.groupby(daily_ic.index.year).mean()
        else:
            year_means = pd.Series([daily_ic.mean()])
        signed_years = direction * year_means
        same_sign_years = int((signed_years > 0.0).sum())
        year_count = int(signed_years.notna().sum())
        worst_year_signed_ic = float(signed_years.min()) if year_count else np.nan
        stability = (same_sign_years / year_count) if year_count else 0.0
        recent_component = max(0.0, signed_recent_ic) if np.isfinite(signed_recent_ic) else 0.0
        base_component = abs_mean_ic
        selection_score = ((1.0 - self.recent_weight) * base_component + self.recent_weight * recent_component) * stability
        return {
            "feature": name,
            "coverage": coverage,
            "ic_days": int(len(daily_ic)),
            "mean_ic": mean_ic,
            "abs_mean_ic": abs_mean_ic,
            "recent_mean_ic": recent_mean_ic,
            "signed_recent_ic": signed_recent_ic,
            "same_sign_years": same_sign_years,
            "year_count": year_count,
            "worst_year_signed_ic": worst_year_signed_ic,
            "selection_score": float(selection_score),
        }

    def _coerce_feature_frame(self, features: pd.DataFrame, label: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        feature_names = [self._feature_name(column) for column in features.columns]
        numeric_features = features.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.copy()
        numeric_features.columns = feature_names
        numeric_label = pd.to_numeric(label.reindex(features.index), errors="coerce").astype(np.float32)
        numeric_label = numeric_label.replace([np.inf, -np.inf], np.nan)
        return numeric_features, numeric_label

    def _sample_selection_dates(
        self,
        numeric_features: pd.DataFrame,
        numeric_label: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        level = _date_level(numeric_features.index)
        if level is None or self.selection_step_days <= 1:
            return numeric_features, numeric_label
        dates = pd.Index(numeric_features.index.get_level_values(level).unique()).sort_values()
        if len(dates) <= self.selection_step_days:
            return numeric_features, numeric_label
        keep_dates = set(pd.Timestamp(date) for date in dates[:: self.selection_step_days])
        date_values = numeric_features.index.get_level_values(level)
        mask = [pd.Timestamp(date) in keep_dates for date in date_values]
        return numeric_features.loc[mask], numeric_label.loc[mask]

    def _daily_rank_ic_frame(self, numeric_features: pd.DataFrame, numeric_label: pd.Series) -> pd.DataFrame:
        feature_names = [str(column) for column in numeric_features.columns]
        level = _date_level(numeric_features.index)

        if level is None:
            valid_label = numeric_label.notna()
            if int(valid_label.sum()) < self.min_daily_count:
                return pd.DataFrame(columns=feature_names, dtype=float)
            x = numeric_features.loc[valid_label]
            y_rank = numeric_label.loc[valid_label].rank(method="average")
            valid_counts = x.notna().sum()
            x_rank = x.rank(axis=0, method="average")
            ic = x_rank.corrwith(y_rank, axis=0, method="pearson")
            ic.loc[valid_counts < self.min_daily_count] = np.nan
            return pd.DataFrame([ic], columns=feature_names)

        rows = []
        dates = []
        for date, x_day in numeric_features.groupby(level=level, sort=True):
            y_day = numeric_label.reindex(x_day.index)
            valid_label = y_day.notna()
            if int(valid_label.sum()) < self.min_daily_count:
                continue
            x_day = x_day.loc[valid_label]
            y_day = y_day.loc[valid_label]
            valid_counts = x_day.notna().sum()
            usable = valid_counts[valid_counts >= self.min_daily_count].index
            if len(usable) == 0:
                continue
            x_rank = x_day.loc[:, usable].rank(axis=0, method="average")
            y_rank = y_day.rank(method="average")
            ic = x_rank.corrwith(y_rank, axis=0, method="pearson").reindex(feature_names)
            rows.append(ic)
            dates.append(pd.Timestamp(date))
        if not rows:
            return pd.DataFrame(columns=feature_names, dtype=float)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=feature_names)

    def _daily_tail_spread_frame(
        self,
        numeric_features: pd.DataFrame,
        numeric_label: pd.Series,
        direction: pd.Series,
    ) -> pd.DataFrame:
        feature_names = [str(column) for column in numeric_features.columns]
        level = _date_level(numeric_features.index)
        direction = direction.reindex(feature_names).fillna(1.0)

        def tail_spread(x_day: pd.DataFrame, y_day: pd.Series) -> pd.Series:
            valid_label = y_day.notna()
            if int(valid_label.sum()) < self.min_daily_count:
                return pd.Series(np.nan, index=feature_names, dtype=float)
            x_day = x_day.loc[valid_label]
            y_day = y_day.loc[valid_label]
            signed = x_day.mul(direction, axis=1)
            ranks = signed.rank(axis=0, pct=True, method="average")
            top_mask = ranks >= (1.0 - self.top_quantile)
            bottom_mask = ranks <= self.top_quantile
            top_count = top_mask.sum(axis=0)
            bottom_count = bottom_mask.sum(axis=0)
            top_sum = top_mask.astype(float).mul(y_day, axis=0).sum(axis=0)
            bottom_sum = bottom_mask.astype(float).mul(y_day, axis=0).sum(axis=0)
            spread = (top_sum / top_count.replace(0, np.nan)) - (bottom_sum / bottom_count.replace(0, np.nan))
            spread.loc[(top_count < 2) | (bottom_count < 2)] = np.nan
            return spread.reindex(feature_names)

        if level is None:
            return pd.DataFrame([tail_spread(numeric_features, numeric_label)], columns=feature_names)

        rows = []
        dates = []
        for date, x_day in numeric_features.groupby(level=level, sort=True):
            rows.append(tail_spread(x_day, numeric_label.reindex(x_day.index)))
            dates.append(pd.Timestamp(date))
        if not rows:
            return pd.DataFrame(columns=feature_names, dtype=float)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=feature_names)

    def _summarize_features(self, features: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
        numeric_features, numeric_label = self._coerce_feature_frame(features, label)
        feature_names = [str(column) for column in numeric_features.columns]
        valid_label = numeric_label.notna()
        coverage = numeric_features.notna().mul(valid_label.astype(bool), axis=0).mean()
        ic_features, ic_label = self._sample_selection_dates(numeric_features, numeric_label)
        daily_ic = self._daily_rank_ic_frame(ic_features, ic_label)
        if daily_ic.empty:
            return pd.DataFrame(
                {
                    "feature": feature_names,
                    "coverage": coverage.reindex(feature_names).fillna(0.0).values,
                    "ic_days": 0,
                    "mean_ic": np.nan,
                    "abs_mean_ic": np.nan,
                    "recent_mean_ic": np.nan,
                    "signed_recent_ic": np.nan,
                    "same_sign_years": 0,
                    "year_count": 0,
                    "worst_year_signed_ic": np.nan,
                    "topq_spread": np.nan,
                    "recent_topq_spread": np.nan,
                    "worst_year_topq_spread": np.nan,
                    "selection_score": np.nan,
                }
            )

        mean_ic = daily_ic.mean(axis=0, skipna=True)
        abs_mean_ic = mean_ic.abs()
        direction = pd.Series(np.where(mean_ic >= 0.0, 1.0, -1.0), index=mean_ic.index)
        recent_ic = daily_ic.iloc[-self.recent_window_days :] if self.recent_window_days > 0 else daily_ic
        recent_mean_ic = recent_ic.mean(axis=0, skipna=True)
        signed_recent_ic = direction * recent_mean_ic
        tail_spread = self._daily_tail_spread_frame(ic_features, ic_label, direction)
        mean_topq_spread = tail_spread.mean(axis=0, skipna=True)
        recent_tail = tail_spread.iloc[-self.recent_window_days :] if self.recent_window_days > 0 else tail_spread
        recent_topq_spread = recent_tail.mean(axis=0, skipna=True)
        if isinstance(daily_ic.index, pd.DatetimeIndex):
            year_means = daily_ic.groupby(daily_ic.index.year).mean()
        else:
            year_means = pd.DataFrame([mean_ic], columns=daily_ic.columns)
        signed_years = year_means.mul(direction, axis=1)
        same_sign_years = (signed_years > 0.0).sum(axis=0)
        year_count = signed_years.notna().sum(axis=0)
        worst_year_signed_ic = signed_years.min(axis=0, skipna=True)
        stability = same_sign_years.div(year_count.replace(0, np.nan)).fillna(0.0)
        recent_component = signed_recent_ic.clip(lower=0.0).fillna(0.0)
        tail_component = mean_topq_spread.clip(lower=0.0).fillna(0.0)
        selection_score = (
            (1.0 - self.recent_weight) * abs_mean_ic.fillna(0.0)
            + self.recent_weight * recent_component
            + self.tail_weight * tail_component
        ) * stability
        if isinstance(tail_spread.index, pd.DatetimeIndex):
            tail_year_means = tail_spread.groupby(tail_spread.index.year).mean()
        else:
            tail_year_means = pd.DataFrame([mean_topq_spread], columns=tail_spread.columns)
        worst_year_topq_spread = tail_year_means.min(axis=0, skipna=True)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "coverage": coverage.reindex(feature_names).fillna(0.0).values,
                "ic_days": daily_ic.notna().sum(axis=0).reindex(feature_names).fillna(0).astype(int).values,
                "mean_ic": mean_ic.reindex(feature_names).values,
                "abs_mean_ic": abs_mean_ic.reindex(feature_names).values,
                "recent_mean_ic": recent_mean_ic.reindex(feature_names).values,
                "signed_recent_ic": signed_recent_ic.reindex(feature_names).values,
                "same_sign_years": same_sign_years.reindex(feature_names).fillna(0).astype(int).values,
                "year_count": year_count.reindex(feature_names).fillna(0).astype(int).values,
                "worst_year_signed_ic": worst_year_signed_ic.reindex(feature_names).values,
                "topq_spread": mean_topq_spread.reindex(feature_names).values,
                "recent_topq_spread": recent_topq_spread.reindex(feature_names).values,
                "worst_year_topq_spread": worst_year_topq_spread.reindex(feature_names).values,
                "selection_score": selection_score.reindex(feature_names).values,
            }
        )

    def _select_rows(self, summary: pd.DataFrame, *, min_ic_days: Optional[int] = None) -> Tuple[pd.DataFrame, bool]:
        min_ic_days = self.min_ic_days if min_ic_days is None else int(min_ic_days)
        tail_ok = pd.Series(True, index=summary.index)
        if np.isfinite(self.min_topq_spread) and "topq_spread" in summary.columns:
            tail_ok &= summary["topq_spread"] >= self.min_topq_spread
        if np.isfinite(self.min_worst_year_topq_spread) and "worst_year_topq_spread" in summary.columns:
            tail_ok &= summary["worst_year_topq_spread"] >= self.min_worst_year_topq_spread
        strict = summary[
            (summary["coverage"] >= self.min_coverage)
            & (summary["ic_days"] >= min_ic_days)
            & (summary["abs_mean_ic"] >= self.min_abs_ic)
            & (summary["same_sign_years"] >= self.min_same_sign_years)
            & (summary["worst_year_signed_ic"] >= self.min_worst_year_signed_ic)
            & (summary["signed_recent_ic"] >= self.min_recent_signed_ic)
            & tail_ok
            & summary["selection_score"].replace([np.inf, -np.inf], np.nan).notna()
        ].copy()
        fallback_used = False
        if len(strict) >= min(self.min_selected_features, self.max_features) or not self.fallback_to_best:
            selected = strict
        else:
            selected = summary[
                (summary["coverage"] >= self.min_coverage)
                & (summary["ic_days"] >= max(10, min(min_ic_days, 60)))
                & summary["selection_score"].replace([np.inf, -np.inf], np.nan).notna()
            ].copy()
            fallback_used = True
        selected = selected.sort_values(
            ["selection_score", "abs_mean_ic", "ic_days"],
            ascending=[False, False, False],
        )
        return selected.head(self.max_features), fallback_used

    def _weights_from_selection(self, selected: pd.DataFrame) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for _, row in selected.iterrows():
            mean_ic = float(row["mean_ic"])
            if not np.isfinite(mean_ic) or mean_ic == 0.0:
                continue
            raw_weight = np.sign(mean_ic) * (abs(mean_ic) ** self.weight_power)
            if np.isfinite(raw_weight) and raw_weight != 0.0:
                weights[str(row["feature"])] = float(raw_weight)
        total_abs = float(sum(abs(weight) for weight in weights.values()))
        if total_abs <= 0.0:
            return {}
        return {name: weight / total_abs for name, weight in weights.items()}

    def _weights_from_summary(
        self,
        summary: pd.DataFrame,
        *,
        min_ic_days: Optional[int] = None,
    ) -> Tuple[Dict[str, float], bool]:
        selected, fallback_used = self._select_rows(summary, min_ic_days=min_ic_days)
        return self._weights_from_selection(selected), fallback_used

    def _regime_masks(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        if not self.regime_feature:
            return {}
        aliases = FeatureWeightedScoreModel._feature_columns(features)
        column = aliases.get(_normalize_alias(self.regime_feature))
        if column is None:
            if self.missing == "raise":
                raise KeyError(f"missing feature columns: {[self.regime_feature]}")
            return {}
        regime = pd.to_numeric(features[column], errors="coerce").astype(float)
        return {
            "risk_off": regime <= self.risk_off_threshold,
            "risk_on": regime >= self.risk_on_threshold,
            "neutral": (regime > self.risk_off_threshold) & (regime < self.risk_on_threshold),
        }

    def fit(self, dataset: DatasetH, **kwargs):
        frame = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        features, label = self._split_feature_label(frame)
        summary = self._summarize_features(features, label)
        if summary.empty:
            raise ValueError("no feature columns available for IC selection")
        weights, fallback_used = self._weights_from_summary(summary)
        if not weights:
            raise ValueError("ICSelectedScoreModel could not select any predictive features")
        self.selection_summary_ = summary.sort_values(
            ["selection_score", "abs_mean_ic"],
            ascending=[False, False],
        ).reset_index(drop=True)
        self.selected_weights_ = weights
        self.fallback_used_ = fallback_used
        self.selected_weights_by_state_ = {}
        self.selection_summary_by_state_ = {}
        self.fallback_used_by_state_ = {}
        for state, mask in self._regime_masks(features).items():
            mask = mask.reindex(features.index).fillna(False)
            if int(mask.sum()) < self.min_daily_count:
                continue
            state_summary = self._summarize_features(features.loc[mask], label.loc[mask])
            if state_summary.empty:
                continue
            state_min_ic_days = self.regime_min_ic_days if self.regime_min_ic_days is not None else max(30, self.min_ic_days // 3)
            state_weights, state_fallback = self._weights_from_summary(
                state_summary,
                min_ic_days=state_min_ic_days,
            )
            if not state_weights:
                continue
            self.selected_weights_by_state_[state] = state_weights
            self.selection_summary_by_state_[state] = state_summary.sort_values(
                ["selection_score", "abs_mean_ic"],
                ascending=[False, False],
            ).reset_index(drop=True)
            self.fallback_used_by_state_[state] = state_fallback
        self.fitted = True
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        aliases = FeatureWeightedScoreModel._feature_columns(features)
        base_scorer = FeatureWeightedScoreModel(
            self.selected_weights_,
            normalize_by_date=self.normalize_by_date,
            missing=self.missing,
        )
        score = base_scorer._score_frame(features)
        if not self.selected_weights_by_state_ or not self.regime_feature:
            return score
        regime_col = aliases.get(_normalize_alias(self.regime_feature))
        if regime_col is None:
            if self.missing == "raise":
                raise KeyError(f"missing feature columns: {[self.regime_feature]}")
            return score
        regime = pd.to_numeric(features[regime_col], errors="coerce").astype(float)
        masks = {
            "risk_off": regime <= self.risk_off_threshold,
            "risk_on": regime >= self.risk_on_threshold,
            "neutral": (regime > self.risk_off_threshold) & (regime < self.risk_on_threshold),
        }
        for state, weights in self.selected_weights_by_state_.items():
            mask = masks.get(state)
            if mask is None or not bool(mask.fillna(False).any()):
                continue
            state_scorer = FeatureWeightedScoreModel(
                weights,
                normalize_by_date=self.normalize_by_date,
                missing=self.missing,
            )
            state_score = state_scorer._score_frame(features)
            score.loc[mask.fillna(False)] = state_score.loc[mask.fillna(False)]
        score.name = "score"
        return score.replace([np.inf, -np.inf], np.nan)


class StackedSignalScoreModel(Model):
    """Train-only blend of auditable feature sleeves.

    Each sleeve is a deterministic weighted feature composite.  ``fit`` scores
    each sleeve on the training split, estimates daily rank-IC against the
    training label, then blends only sleeves that pass simple coverage and
    stability gates.  This keeps the model interpretable while allowing FMP,
    Sharadar, and momentum/regime signals to compete for weight walk-forward.
    """

    def __init__(
        self,
        sleeves: Dict[str, Dict[str, float]],
        max_sleeves: Optional[int] = None,
        min_selected_sleeves: int = 1,
        min_abs_ic: float = 0.0,
        min_ic_days: int = 60,
        min_daily_count: int = 30,
        min_coverage: float = 0.30,
        min_recent_signed_ic: float = -np.inf,
        min_worst_year_signed_ic: float = -np.inf,
        recent_window_days: int = 756,
        recent_weight: float = 0.50,
        weight_power: float = 1.0,
        normalize_by_date: bool = True,
        normalize_sleeve_scores_by_date: bool = True,
        missing: str = "raise",
        allow_negative_sleeve_weights: bool = True,
        fallback_to_equal: bool = True,
    ):
        if not sleeves:
            raise ValueError("StackedSignalScoreModel requires at least one sleeve")
        if missing not in {"raise", "ignore"}:
            raise ValueError("missing must be 'raise' or 'ignore'")
        self.sleeves = {
            str(name): {str(feature): float(weight) for feature, weight in weights.items()}
            for name, weights in sleeves.items()
        }
        for name, weights in self.sleeves.items():
            if not weights:
                raise ValueError(f"sleeve {name!r} has no feature weights")
        self.max_sleeves = None if max_sleeves is None else max(1, int(max_sleeves))
        self.min_selected_sleeves = max(1, int(min_selected_sleeves))
        self.min_abs_ic = float(min_abs_ic)
        self.min_ic_days = int(min_ic_days)
        self.min_daily_count = int(min_daily_count)
        self.min_coverage = float(min_coverage)
        self.min_recent_signed_ic = float(min_recent_signed_ic)
        self.min_worst_year_signed_ic = float(min_worst_year_signed_ic)
        self.recent_window_days = int(recent_window_days)
        self.recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
        self.weight_power = float(weight_power)
        self.normalize_by_date = bool(normalize_by_date)
        self.normalize_sleeve_scores_by_date = bool(normalize_sleeve_scores_by_date)
        self.missing = str(missing)
        self.allow_negative_sleeve_weights = bool(allow_negative_sleeve_weights)
        self.fallback_to_equal = bool(fallback_to_equal)
        self.fitted = False
        self.sleeve_summary_ = pd.DataFrame()
        self.selected_sleeve_weights_: Dict[str, float] = {}
        self.fallback_used_ = False

    def _score_sleeves(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame) or features.empty:
            return pd.DataFrame(index=getattr(features, "index", None))
        scores: Dict[str, pd.Series] = {}
        for name, weights in self.sleeves.items():
            scorer = FeatureWeightedScoreModel(
                weights,
                normalize_by_date=self.normalize_by_date,
                missing=self.missing,
            )
            score = scorer._score_frame(features).reindex(features.index)
            if self.normalize_sleeve_scores_by_date:
                score = _daily_zscore(score, features.index)
            scores[name] = score
        return pd.DataFrame(scores, index=features.index).replace([np.inf, -np.inf], np.nan)

    def _summarize_sleeves(self, sleeve_scores: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
        rows = []
        label = pd.to_numeric(label.reindex(sleeve_scores.index), errors="coerce").astype(float)
        valid_label = label.replace([np.inf, -np.inf], np.nan).notna()
        for name in sleeve_scores.columns:
            values = pd.to_numeric(sleeve_scores[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            coverage = float((values.notna() & valid_label).mean()) if len(values) else 0.0
            daily_ic = ICSelectedScoreModel._daily_rank_ic(values, label, self.min_daily_count)
            if daily_ic.empty:
                rows.append(
                    {
                        "sleeve": str(name),
                        "coverage": coverage,
                        "ic_days": 0,
                        "mean_ic": np.nan,
                        "abs_mean_ic": np.nan,
                        "recent_mean_ic": np.nan,
                        "signed_recent_ic": np.nan,
                        "same_sign_years": 0,
                        "year_count": 0,
                        "worst_year_signed_ic": np.nan,
                        "selection_score": np.nan,
                    }
                )
                continue
            mean_ic = float(daily_ic.mean())
            direction = 1.0 if mean_ic >= 0.0 else -1.0
            recent_ic = daily_ic.iloc[-self.recent_window_days :] if self.recent_window_days > 0 else daily_ic
            recent_mean_ic = float(recent_ic.mean()) if len(recent_ic) else np.nan
            signed_recent_ic = direction * recent_mean_ic if np.isfinite(recent_mean_ic) else np.nan
            if isinstance(daily_ic.index, pd.DatetimeIndex):
                year_means = daily_ic.groupby(daily_ic.index.year).mean()
            else:
                year_means = pd.Series([daily_ic.mean()])
            signed_years = direction * year_means
            same_sign_years = int((signed_years > 0.0).sum())
            year_count = int(signed_years.notna().sum())
            worst_year_signed_ic = float(signed_years.min()) if year_count else np.nan
            stability = (same_sign_years / year_count) if year_count else 0.0
            recent_component = max(0.0, signed_recent_ic) if np.isfinite(signed_recent_ic) else 0.0
            selection_score = (
                (1.0 - self.recent_weight) * abs(mean_ic) + self.recent_weight * recent_component
            ) * stability
            rows.append(
                {
                    "sleeve": str(name),
                    "coverage": coverage,
                    "ic_days": int(len(daily_ic)),
                    "mean_ic": mean_ic,
                    "abs_mean_ic": abs(mean_ic),
                    "recent_mean_ic": recent_mean_ic,
                    "signed_recent_ic": signed_recent_ic,
                    "same_sign_years": same_sign_years,
                    "year_count": year_count,
                    "worst_year_signed_ic": worst_year_signed_ic,
                    "selection_score": float(selection_score),
                }
            )
        return pd.DataFrame(rows)

    def _select_sleeve_weights(self, summary: pd.DataFrame) -> Tuple[Dict[str, float], bool]:
        if summary.empty:
            return {}, False
        strict = summary[
            (summary["coverage"] >= self.min_coverage)
            & (summary["ic_days"] >= self.min_ic_days)
            & (summary["abs_mean_ic"] >= self.min_abs_ic)
            & (summary["signed_recent_ic"] >= self.min_recent_signed_ic)
            & (summary["worst_year_signed_ic"] >= self.min_worst_year_signed_ic)
            & summary["selection_score"].replace([np.inf, -np.inf], np.nan).notna()
        ].copy()
        fallback_used = False
        if len(strict) < min(self.min_selected_sleeves, self.max_sleeves or len(self.sleeves)):
            fallback_used = True
            selected = summary[
                (summary["coverage"] >= self.min_coverage)
                & (summary["ic_days"] >= max(5, min(self.min_ic_days, 30)))
                & summary["selection_score"].replace([np.inf, -np.inf], np.nan).notna()
            ].copy()
        else:
            selected = strict
        selected = selected.sort_values(["selection_score", "abs_mean_ic", "ic_days"], ascending=[False, False, False])
        if self.max_sleeves is not None:
            selected = selected.head(self.max_sleeves)
        if selected.empty:
            if not self.fallback_to_equal:
                return {}, fallback_used
            names = list(self.sleeves)
            return {name: 1.0 / len(names) for name in names}, True

        raw: Dict[str, float] = {}
        for _, row in selected.iterrows():
            mean_ic = float(row["mean_ic"])
            if not np.isfinite(mean_ic) or mean_ic == 0.0:
                continue
            if not self.allow_negative_sleeve_weights and mean_ic < 0.0:
                continue
            raw[str(row["sleeve"])] = float(np.sign(mean_ic) * (abs(mean_ic) ** self.weight_power))
        total_abs = float(sum(abs(weight) for weight in raw.values()))
        if total_abs <= 0:
            if not self.fallback_to_equal:
                return {}, fallback_used
            names = selected["sleeve"].astype(str).tolist() or list(self.sleeves)
            return {name: 1.0 / len(names) for name in names}, True
        return {name: weight / total_abs for name, weight in raw.items()}, fallback_used

    def _blend(self, sleeve_scores: pd.DataFrame) -> pd.Series:
        if sleeve_scores.empty:
            return pd.Series(dtype=float)
        score = pd.Series(0.0, index=sleeve_scores.index, dtype=float)
        for name, weight in self.selected_sleeve_weights_.items():
            if name not in sleeve_scores.columns:
                if self.missing == "raise":
                    raise KeyError(f"missing sleeve scores: {[name]}")
                continue
            score = score + float(weight) * pd.to_numeric(sleeve_scores[name], errors="coerce").fillna(0.0)
        score.name = "score"
        return score.replace([np.inf, -np.inf], np.nan)

    def fit(self, dataset: DatasetH, **kwargs):
        frame = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        features, label = ICSelectedScoreModel._split_feature_label(frame)
        sleeve_scores = self._score_sleeves(features)
        summary = self._summarize_sleeves(sleeve_scores, label)
        weights, fallback_used = self._select_sleeve_weights(summary)
        if not weights:
            raise ValueError("StackedSignalScoreModel could not select any sleeves")
        self.sleeve_summary_ = summary.sort_values(
            ["selection_score", "abs_mean_ic"],
            ascending=[False, False],
        ).reset_index(drop=True)
        self.selected_sleeve_weights_ = weights
        self.fallback_used_ = fallback_used
        self.fitted = True
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self._blend(self._score_sleeves(features))
