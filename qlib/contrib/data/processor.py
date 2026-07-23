import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...log import TimeInspector
from ...data.dataset.processor import Processor, get_group_columns
from ...utils.data import robust_zscore, zscore


def _read_pickle_compat(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if not str(getattr(e, "name", "")).startswith("numpy._core"):
            raise
        import importlib

        sys.modules.setdefault("numpy._core", np.core)
        for name in ("numeric", "multiarray", "umath"):
            sys.modules.setdefault(f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}"))
        return pd.read_pickle(path)


def _parse_positive_int_sequence(value, *, name: str) -> Tuple[int, ...]:
    if isinstance(value, str):
        raw = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, Iterable):
        raw = list(value)
    else:
        raw = [value]
    out = []
    for item in raw:
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive")
        out.append(parsed)
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(out)


def _parse_float_sequence(value, *, name: str) -> Tuple[float, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raw = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, Iterable):
        raw = list(value)
    else:
        raw = [value]
    out = tuple(float(item) for item in raw)
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    return out


class ConfigSectionProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    def __init__(self, fields_group=None, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        self.fields_group = None

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        def _label_norm(x):
            x = x - x.mean()  # copy
            x /= x.std()
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        def _feature_norm(x):
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                x.fillna(0, inplace=True)
            return x

        TimeInspector.set_time_mark()

        # Copy the focus part and change it to single level
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        # Label
        cols = df_focus.columns[df_focus.columns.str.contains("^LABEL")]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_label_norm)

        # Features
        cols = df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.25).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")]
        df_focus[cols] = (
            df_focus[cols].apply(lambda x: x**0.5).groupby(level="datetime", group_keys=False).apply(_feature_norm)
        )

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VWAP",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            "VSUMP",
            "VSUMN",
            "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df_focus.columns[df_focus.columns.str.contains(pat) & (~df_focus.columns.isin(["HIGH0", "LOW0"]))]
        df_focus[cols] = df_focus[cols].groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df_focus[cols] = df_focus[cols].apply(np.log).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^RSQR")]
        df_focus[cols] = df_focus[cols].fillna(0).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (x - 1) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")]
        df_focus[cols] = (
            df_focus[cols]
            .apply(lambda x: (1 - x) ** 0.5)
            .groupby(level="datetime", group_keys=False)
            .apply(_feature_norm)
        )

        cols = df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")]
        df_focus[cols] = df_focus[cols].apply(np.exp).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^WVMA")]
        df_focus[cols] = df_focus[cols].apply(np.log1p).groupby(level="datetime", group_keys=False).apply(_feature_norm)

        df[selected_cols] = df_focus.values

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df


class BenchmarkExcessLabel(Processor):
    """
    Convert raw forward return labels into benchmark-relative excess labels.

    This processor is intended for `learn_processors` only.
    It subtracts the benchmark forward return over the same label horizon.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: int = 10,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
    ):
        self.benchmark_pkl = str(benchmark_pkl)
        self.label_horizon_days = max(1, int(label_horizon_days))
        self.label_ref_start_days = max(0, int(label_ref_start_days))
        self.fields_group = fields_group
        self.benchmark_kind = str(benchmark_kind).lower()
        self.fill_method = fill_method
        self.drop_missing = bool(drop_missing)
        self._bench_forward = self._load_benchmark_forward()

    def _load_benchmark_forward(self) -> pd.Series:
        path = Path(self.benchmark_pkl).expanduser().resolve()
        bench = _read_pickle_compat(path)
        if isinstance(bench, pd.DataFrame):
            if bench.shape[1] != 1:
                raise ValueError(f"benchmark_pkl must be Series or single-column DataFrame: {path}")
            bench = bench.iloc[:, 0]
        if not isinstance(bench, pd.Series):
            raise ValueError(f"benchmark_pkl must be pandas Series: {path}")
        bench = bench.copy()
        bench.index = pd.DatetimeIndex(bench.index)
        bench = bench.sort_index()
        bench = bench[~bench.index.duplicated(keep="last")]

        if self.benchmark_kind not in {"auto", "return", "price"}:
            raise ValueError(f"unsupported benchmark_kind: {self.benchmark_kind}")
        if self.benchmark_kind == "return":
            ret = bench.astype(float)
        elif self.benchmark_kind == "price":
            ret = bench.astype(float).pct_change()
        else:
            # Auto detect: if values are mostly in [-1, 1], treat as returns; otherwise prices.
            sample = bench.dropna().head(5000).astype(float)
            as_return = True if sample.empty else bool((sample.abs() <= 1.0).mean() >= 0.98)
            ret = bench.astype(float) if as_return else bench.astype(float).pct_change()

        ret = ret.replace([np.inf, -np.inf], np.nan)
        if self.fill_method == "ffill":
            ret = ret.ffill()
        elif self.fill_method == "bfill":
            ret = ret.bfill()
        elif self.fill_method is not None:
            ret = ret.fillna(method=self.fill_method)

        # Forward cumulative return aligned with qlib label style:
        # Ref($close, -(s + N))/Ref($close, -s) - 1
        # where s is label_ref_start_days and N is label_horizon_days.
        gross = 1.0 + ret
        cp = gross.cumprod()
        start = self.label_ref_start_days
        fwd = cp.shift(-(start + self.label_horizon_days)) / cp.shift(-start) - 1.0
        return fwd

    def is_for_infer(self) -> bool:
        return False

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("BenchmarkExcessLabel expects MultiIndex with `datetime` level")

        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_vals = self._bench_forward.reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)

        if self.drop_missing:
            valid = ~bench_vals.isna().to_numpy()
            if not valid.all():
                df = df.loc[valid].copy()
                bench_vals = bench_vals[valid]

        target = df.loc[:, cols].sub(bench_vals.to_numpy(), axis=0)
        # Preserve original label dtypes (often float32) to avoid pandas
        # incompatible-assignment warnings on future versions.
        try:
            target = target.astype(df.loc[:, cols].dtypes.to_dict())
        except Exception:
            pass
        df.loc[:, cols] = target
        return df


class RiskTierFilter(Processor):
    """Filter learn samples by point-in-time beta/volatility risk tier.

    This processor is intended for ``learn_processors``. It leaves inference
    data untouched by returning ``False`` from :meth:`is_for_infer`.
    """

    def __init__(
        self,
        fields_group: str = "feature",
        beta_feature: str = "RISK_BETA_QQQ_63D",
        vol_feature: str = "RISK_VOL_20D",
        tier: str = "high",
        low_beta_max: float = 1.05,
        low_vol_max: float = 0.35,
        medium_beta_max: float = 1.50,
        medium_vol_max: float = 0.60,
        min_beta: Optional[float] = None,
        min_vol: Optional[float] = None,
        drop_missing: bool = True,
    ):
        self.fields_group = fields_group
        self.beta_feature = str(beta_feature)
        self.vol_feature = str(vol_feature)
        self.tier = str(tier).lower()
        self.low_beta_max = float(low_beta_max)
        self.low_vol_max = float(low_vol_max)
        self.medium_beta_max = float(medium_beta_max)
        self.medium_vol_max = float(medium_vol_max)
        self.min_beta = min_beta
        self.min_vol = min_vol
        self.drop_missing = bool(drop_missing)

    def is_for_infer(self) -> bool:
        return False

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lstrip("$").upper()

    def _resolve_column(self, df: pd.DataFrame, name: str) -> Tuple[str, str] | str:
        wanted = self._normalize_name(name)
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                if len(col) >= 2 and str(col[0]) == self.fields_group and self._normalize_name(col[1]) == wanted:
                    return col
        else:
            for col in df.columns:
                if self._normalize_name(col) == wanted:
                    return col
        raise KeyError(f"RiskTierFilter missing feature column: {name}")

    def __call__(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("RiskTierFilter expects MultiIndex with `datetime` level")
        beta_col = self._resolve_column(df, self.beta_feature)
        vol_col = self._resolve_column(df, self.vol_feature)
        beta = pd.to_numeric(df[beta_col], errors="coerce")
        vol = pd.to_numeric(df[vol_col], errors="coerce")
        valid = beta.notna() & vol.notna()
        low = (beta <= self.low_beta_max) & (vol <= self.low_vol_max)
        medium = (beta <= self.medium_beta_max) & (vol <= self.medium_vol_max) & ~low
        high = valid & ~(low | medium)
        if self.min_beta is not None:
            high = high & (beta >= float(self.min_beta))
        if self.min_vol is not None:
            high = high & (vol >= float(self.min_vol))

        if self.tier == "high":
            keep = high
        elif self.tier == "medium":
            keep = medium
        elif self.tier == "low":
            keep = low
        elif self.tier in {"non_high", "low_medium"}:
            keep = low | medium
        else:
            raise ValueError(f"unsupported risk tier: {self.tier}")
        if self.drop_missing:
            keep = keep & valid
        else:
            keep = keep | ~valid
        return df.loc[keep.to_numpy()].copy()


class ResidualForwardReturnLabel(BenchmarkExcessLabel):
    """
    Convert raw forward return labels into beta-adjusted residual return labels.

    The label becomes:

        raw_forward_return - beta * benchmark_forward_return

    If `beta_feature` is omitted or unavailable, `fallback_beta` is used. This
    keeps the processor usable across research configs while preventing a model
    from being rewarded for simple market beta when the beta feature is present.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: int = 10,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
        beta_feature: Optional[str] = None,
        feature_group: str = "feature",
        fallback_beta: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 2.0,
        fill_missing_beta: bool = True,
    ):
        super().__init__(
            benchmark_pkl=benchmark_pkl,
            label_horizon_days=label_horizon_days,
            label_ref_start_days=label_ref_start_days,
            fields_group=fields_group,
            benchmark_kind=benchmark_kind,
            fill_method=fill_method,
            drop_missing=drop_missing,
        )
        self.beta_feature = None if beta_feature is None else str(beta_feature).strip()
        self.feature_group = str(feature_group)
        self.fallback_beta = float(fallback_beta)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        if self.beta_max < self.beta_min:
            self.beta_min, self.beta_max = self.beta_max, self.beta_min
        self.fill_missing_beta = bool(fill_missing_beta)

    @staticmethod
    def _canonical_field_name(name: object) -> str:
        text = str(name).strip()
        if text.startswith("$"):
            text = text[1:]
        return text.upper()

    def _resolve_beta_column(self, df: pd.DataFrame):
        if not self.beta_feature:
            return None
        wanted = self._canonical_field_name(self.beta_feature)
        if isinstance(df.columns, pd.MultiIndex):
            direct = (self.feature_group, self.beta_feature)
            if direct in df.columns:
                return direct
            for col in df.columns:
                if not isinstance(col, tuple) or len(col) == 0:
                    continue
                if str(col[0]) != self.feature_group:
                    continue
                if self._canonical_field_name(col[-1]) == wanted:
                    return col
        else:
            if self.beta_feature in df.columns:
                return self.beta_feature
            for col in df.columns:
                if self._canonical_field_name(col) == wanted:
                    return col
        return None

    def _beta_values(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_beta_column(df)
        if col is None:
            beta = pd.Series(self.fallback_beta, index=df.index, dtype=float)
        else:
            beta = pd.to_numeric(df.loc[:, col], errors="coerce")
            if self.fill_missing_beta:
                beta = beta.fillna(self.fallback_beta)
        beta = beta.replace([np.inf, -np.inf], np.nan)
        if self.fill_missing_beta:
            beta = beta.fillna(self.fallback_beta)
        return beta.clip(lower=self.beta_min, upper=self.beta_max)

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("ResidualForwardReturnLabel expects MultiIndex with `datetime` level")

        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_vals = self._bench_forward.reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)

        beta_vals = self._beta_values(df)
        valid = ~bench_vals.isna().to_numpy()
        if not self.fill_missing_beta:
            valid &= ~beta_vals.isna().to_numpy()
        if self.drop_missing and not valid.all():
            df = df.loc[valid].copy()
            bench_vals = bench_vals[valid]
            beta_vals = beta_vals.loc[df.index]

        residual_bench = beta_vals.to_numpy(dtype=float) * bench_vals.to_numpy(dtype=float)
        target = df.loc[:, cols].sub(residual_bench, axis=0)
        try:
            target = target.astype(df.loc[:, cols].dtypes.to_dict())
        except Exception:
            pass
        df.loc[:, cols] = target
        return df


class VolScaledExcessLabel(BenchmarkExcessLabel):
    """
    Convert raw forward return labels into volatility-scaled benchmark excess labels.

    The label becomes:

        (raw_forward_return - benchmark_forward_return) / (vol * sqrt(horizon))

    `vol_feature` should be a point-in-time daily volatility feature available in
    the feature group, for example `RISK_VOL_20D`. If it is omitted or missing,
    `fallback_vol` is used so configs remain runnable, but release research
    should prefer configs where the volatility feature is explicitly present.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: int = 10,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
        vol_feature: Optional[str] = "RISK_VOL_20D",
        feature_group: str = "feature",
        fallback_vol: float = 0.02,
        vol_min: float = 0.0025,
        vol_max: float = 0.20,
        scale_by_horizon: bool = True,
        fill_missing_vol: bool = True,
        clip_abs_label: Optional[float] = None,
    ):
        super().__init__(
            benchmark_pkl=benchmark_pkl,
            label_horizon_days=label_horizon_days,
            label_ref_start_days=label_ref_start_days,
            fields_group=fields_group,
            benchmark_kind=benchmark_kind,
            fill_method=fill_method,
            drop_missing=drop_missing,
        )
        self.vol_feature = None if vol_feature is None else str(vol_feature).strip()
        self.feature_group = str(feature_group)
        self.fallback_vol = float(fallback_vol)
        self.vol_min = float(vol_min)
        self.vol_max = float(vol_max)
        if self.vol_max < self.vol_min:
            self.vol_min, self.vol_max = self.vol_max, self.vol_min
        self.scale_by_horizon = bool(scale_by_horizon)
        self.fill_missing_vol = bool(fill_missing_vol)
        self.clip_abs_label = None if clip_abs_label is None else abs(float(clip_abs_label))

    @staticmethod
    def _canonical_field_name(name: object) -> str:
        text = str(name).strip()
        if text.startswith("$"):
            text = text[1:]
        return text.upper()

    def _resolve_vol_column(self, df: pd.DataFrame):
        if not self.vol_feature:
            return None
        wanted = self._canonical_field_name(self.vol_feature)
        if isinstance(df.columns, pd.MultiIndex):
            direct = (self.feature_group, self.vol_feature)
            if direct in df.columns:
                return direct
            for col in df.columns:
                if not isinstance(col, tuple) or len(col) == 0:
                    continue
                if str(col[0]) != self.feature_group:
                    continue
                if self._canonical_field_name(col[-1]) == wanted:
                    return col
        else:
            if self.vol_feature in df.columns:
                return self.vol_feature
            for col in df.columns:
                if self._canonical_field_name(col) == wanted:
                    return col
        return None

    def _vol_values(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_vol_column(df)
        if col is None:
            vol = pd.Series(self.fallback_vol, index=df.index, dtype=float)
        else:
            vol = pd.to_numeric(df.loc[:, col], errors="coerce")
            if self.fill_missing_vol:
                vol = vol.fillna(self.fallback_vol)
        vol = vol.replace([np.inf, -np.inf], np.nan)
        if self.fill_missing_vol:
            vol = vol.fillna(self.fallback_vol)
        return vol.clip(lower=self.vol_min, upper=self.vol_max)

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("VolScaledExcessLabel expects MultiIndex with `datetime` level")

        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_vals = self._bench_forward.reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)

        vol_vals = self._vol_values(df)
        valid = ~bench_vals.isna().to_numpy()
        if not self.fill_missing_vol:
            valid &= ~vol_vals.isna().to_numpy()
        if self.drop_missing and not valid.all():
            df = df.loc[valid].copy()
            bench_vals = bench_vals[valid]
            vol_vals = vol_vals.loc[df.index]

        denom = vol_vals.to_numpy(dtype=float)
        if self.scale_by_horizon:
            denom = denom * np.sqrt(float(self.label_horizon_days))
        target = df.loc[:, cols].sub(bench_vals.to_numpy(dtype=float), axis=0).div(denom, axis=0)
        if self.clip_abs_label is not None and self.clip_abs_label > 0:
            target = target.clip(lower=-self.clip_abs_label, upper=self.clip_abs_label)
        try:
            target = target.astype(df.loc[:, cols].dtypes.to_dict())
        except Exception:
            pass
        df.loc[:, cols] = target
        return df


class DownsideAdjustedExcessLabel(BenchmarkExcessLabel):
    """
    Convert raw forward return labels into benchmark excess labels with an
    asymmetric penalty for negative excess outcomes.

    The label becomes:

        excess - downside_penalty * max(0, -excess)

    This keeps positive benchmark-relative returns on their natural scale while
    making failed selections more expensive to the learner.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: int = 10,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
        downside_penalty: float = 1.0,
        clip_abs_label: Optional[float] = None,
    ):
        super().__init__(
            benchmark_pkl=benchmark_pkl,
            label_horizon_days=label_horizon_days,
            label_ref_start_days=label_ref_start_days,
            fields_group=fields_group,
            benchmark_kind=benchmark_kind,
            fill_method=fill_method,
            drop_missing=drop_missing,
        )
        self.downside_penalty = max(0.0, float(downside_penalty))
        self.clip_abs_label = None if clip_abs_label is None else abs(float(clip_abs_label))

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("DownsideAdjustedExcessLabel expects MultiIndex with `datetime` level")

        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_vals = self._bench_forward.reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)

        if self.drop_missing:
            valid = ~bench_vals.isna().to_numpy()
            if not valid.all():
                df = df.loc[valid].copy()
                bench_vals = bench_vals[valid]

        excess = df.loc[:, cols].sub(bench_vals.to_numpy(dtype=float), axis=0)
        target = excess - self.downside_penalty * (-excess).clip(lower=0.0)
        if self.clip_abs_label is not None and self.clip_abs_label > 0:
            target = target.clip(lower=-self.clip_abs_label, upper=self.clip_abs_label)
        try:
            target = target.astype(df.loc[:, cols].dtypes.to_dict())
        except Exception:
            pass
        df.loc[:, cols] = target
        return df


class PortfolioUtilityExcessLabel(BenchmarkExcessLabel):
    """
    Convert raw forward return labels into a portfolio-utility excess target.

    The base target is benchmark excess return.  The processor subtracts an
    asymmetric downside penalty and an ex-ante volatility charge, both computed
    from point-in-time data available on the signal date.  This target is meant
    for candidates where release gates care about drawdown and rolling excess,
    not just average forward return.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: int = 10,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
        vol_feature: Optional[str] = "RISK_VOL_20D",
        feature_group: str = "feature",
        fallback_vol: float = 0.02,
        vol_min: float = 0.0025,
        vol_max: float = 0.20,
        downside_penalty: float = 0.75,
        volatility_penalty: float = 0.25,
        scale_vol_by_horizon: bool = True,
        divide_by_vol: bool = False,
        fill_missing_vol: bool = True,
        clip_abs_label: Optional[float] = None,
    ):
        super().__init__(
            benchmark_pkl=benchmark_pkl,
            label_horizon_days=label_horizon_days,
            label_ref_start_days=label_ref_start_days,
            fields_group=fields_group,
            benchmark_kind=benchmark_kind,
            fill_method=fill_method,
            drop_missing=drop_missing,
        )
        self.vol_feature = None if vol_feature is None else str(vol_feature).strip()
        self.feature_group = str(feature_group)
        self.fallback_vol = float(fallback_vol)
        self.vol_min = float(vol_min)
        self.vol_max = float(vol_max)
        if self.vol_max < self.vol_min:
            self.vol_min, self.vol_max = self.vol_max, self.vol_min
        self.downside_penalty = max(0.0, float(downside_penalty))
        self.volatility_penalty = max(0.0, float(volatility_penalty))
        self.scale_vol_by_horizon = bool(scale_vol_by_horizon)
        self.divide_by_vol = bool(divide_by_vol)
        self.fill_missing_vol = bool(fill_missing_vol)
        self.clip_abs_label = None if clip_abs_label is None else abs(float(clip_abs_label))

    @staticmethod
    def _canonical_field_name(name: object) -> str:
        text = str(name).strip()
        if text.startswith("$"):
            text = text[1:]
        return text.upper()

    def _resolve_vol_column(self, df: pd.DataFrame):
        if not self.vol_feature:
            return None
        wanted = self._canonical_field_name(self.vol_feature)
        if isinstance(df.columns, pd.MultiIndex):
            direct = (self.feature_group, self.vol_feature)
            if direct in df.columns:
                return direct
            for col in df.columns:
                if not isinstance(col, tuple) or len(col) == 0:
                    continue
                if str(col[0]) != self.feature_group:
                    continue
                if self._canonical_field_name(col[-1]) == wanted:
                    return col
        else:
            if self.vol_feature in df.columns:
                return self.vol_feature
            for col in df.columns:
                if self._canonical_field_name(col) == wanted:
                    return col
        return None

    def _vol_values(self, df: pd.DataFrame) -> pd.Series:
        col = self._resolve_vol_column(df)
        if col is None:
            vol = pd.Series(self.fallback_vol, index=df.index, dtype=float)
        else:
            vol = pd.to_numeric(df.loc[:, col], errors="coerce")
            if self.fill_missing_vol:
                vol = vol.fillna(self.fallback_vol)
        vol = vol.replace([np.inf, -np.inf], np.nan)
        if self.fill_missing_vol:
            vol = vol.fillna(self.fallback_vol)
        return vol.clip(lower=self.vol_min, upper=self.vol_max)

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("PortfolioUtilityExcessLabel expects MultiIndex with `datetime` level")

        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_vals = self._bench_forward.reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)

        vol_vals = self._vol_values(df)
        valid = ~bench_vals.isna().to_numpy()
        if not self.fill_missing_vol:
            valid &= ~vol_vals.isna().to_numpy()
        if self.drop_missing and not valid.all():
            df = df.loc[valid].copy()
            bench_vals = bench_vals[valid]
            vol_vals = vol_vals.loc[df.index]

        excess = df.loc[:, cols].sub(bench_vals.to_numpy(dtype=float), axis=0)
        downside = (-excess).clip(lower=0.0)
        vol_charge = vol_vals.to_numpy(dtype=float)
        if self.scale_vol_by_horizon:
            vol_charge = vol_charge * np.sqrt(float(self.label_horizon_days))
        target = excess - self.downside_penalty * downside
        target = target.sub(self.volatility_penalty * vol_charge, axis=0)
        if self.divide_by_vol:
            denom = vol_vals.to_numpy(dtype=float)
            if self.scale_vol_by_horizon:
                denom = denom * np.sqrt(float(self.label_horizon_days))
            target = target.div(denom, axis=0)
        if self.clip_abs_label is not None and self.clip_abs_label > 0:
            target = target.clip(lower=-self.clip_abs_label, upper=self.clip_abs_label)
        try:
            target = target.astype(df.loc[:, cols].dtypes.to_dict())
        except Exception:
            pass
        df.loc[:, cols] = target
        return df


class DualHorizonPortfolioUtilityLabel(PortfolioUtilityExcessLabel):
    """
    Collapse multiple raw forward-return labels into one portfolio-utility label.

    Qlib's current LightGBM path expects a single target column.  This processor
    allows configs to define two label horizons, for example 10d and 20d, then
    trains on a weighted utility target:

        sum_h weight_h * (excess_h - downside_penalty * downside_h - vol_penalty * vol_h)

    The first label column is overwritten with the collapsed target.  Extra
    label columns are dropped by default so downstream models still receive a
    one-dimensional label.
    """

    def __init__(
        self,
        benchmark_pkl: str,
        label_horizon_days: Sequence[int] = (10, 20),
        label_weights: Optional[Sequence[float]] = None,
        label_ref_start_days: int = 1,
        fields_group: str = "label",
        benchmark_kind: str = "auto",
        fill_method: Optional[str] = "ffill",
        drop_missing: bool = True,
        vol_feature: Optional[str] = "RISK_VOL_20D",
        feature_group: str = "feature",
        fallback_vol: float = 0.02,
        vol_min: float = 0.0025,
        vol_max: float = 0.20,
        downside_penalty: float = 0.75,
        volatility_penalty: float = 0.25,
        scale_vol_by_horizon: bool = True,
        divide_by_vol: bool = False,
        fill_missing_vol: bool = True,
        clip_abs_label: Optional[float] = None,
        drop_extra_labels: bool = True,
    ):
        horizons = _parse_positive_int_sequence(label_horizon_days, name="label_horizon_days")
        if len(horizons) < 2:
            raise ValueError("DualHorizonPortfolioUtilityLabel requires at least two label_horizon_days")
        raw_weights = _parse_float_sequence(label_weights, name="label_weights") if label_weights is not None else tuple()
        if not raw_weights:
            raw_weights = tuple(1.0 for _ in horizons)
        if len(raw_weights) != len(horizons):
            raise ValueError("label_weights length must match label_horizon_days length")
        if any(weight < 0 for weight in raw_weights):
            raise ValueError("label_weights values must be non-negative")
        weight_sum = float(sum(raw_weights))
        if weight_sum <= 0:
            raise ValueError("label_weights must sum to a positive value")
        weights = tuple(float(weight) / weight_sum for weight in raw_weights)

        super().__init__(
            benchmark_pkl=benchmark_pkl,
            label_horizon_days=int(horizons[0]),
            label_ref_start_days=label_ref_start_days,
            fields_group=fields_group,
            benchmark_kind=benchmark_kind,
            fill_method=fill_method,
            drop_missing=drop_missing,
            vol_feature=vol_feature,
            feature_group=feature_group,
            fallback_vol=fallback_vol,
            vol_min=vol_min,
            vol_max=vol_max,
            downside_penalty=downside_penalty,
            volatility_penalty=volatility_penalty,
            scale_vol_by_horizon=scale_vol_by_horizon,
            divide_by_vol=divide_by_vol,
            fill_missing_vol=fill_missing_vol,
            clip_abs_label=clip_abs_label,
        )
        self.label_horizon_days_list = horizons
        self.label_weights = weights
        self.drop_extra_labels = bool(drop_extra_labels)

        self._bench_forward_by_horizon = {int(horizons[0]): self._bench_forward}
        original_horizon = self.label_horizon_days
        for horizon in horizons[1:]:
            self.label_horizon_days = int(horizon)
            self._bench_forward_by_horizon[int(horizon)] = self._load_benchmark_forward()
        self.label_horizon_days = original_horizon
        self._bench_forward = self._bench_forward_by_horizon[int(horizons[0])]

    def _aligned_benchmark(self, dts: pd.DatetimeIndex, horizon: int) -> pd.Series:
        bench_vals = self._bench_forward_by_horizon[int(horizon)].reindex(dts)
        if self.fill_method == "ffill":
            bench_vals = bench_vals.ffill()
        elif self.fill_method == "bfill":
            bench_vals = bench_vals.bfill()
        elif self.fill_method is not None:
            bench_vals = bench_vals.fillna(method=self.fill_method)
        return bench_vals

    def __call__(self, df: pd.DataFrame):
        cols = list(get_group_columns(df, self.fields_group))
        if len(cols) == 0:
            return df
        if len(cols) < len(self.label_horizon_days_list):
            raise ValueError(
                "DualHorizonPortfolioUtilityLabel expected at least "
                f"{len(self.label_horizon_days_list)} label columns, found {len(cols)}"
            )
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise ValueError("DualHorizonPortfolioUtilityLabel expects MultiIndex with `datetime` level")

        label_cols = cols[: len(self.label_horizon_days_list)]
        dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
        bench_by_horizon = {
            int(horizon): self._aligned_benchmark(dts, int(horizon))
            for horizon in self.label_horizon_days_list
        }

        vol_vals = self._vol_values(df)
        valid = np.ones(len(df), dtype=bool)
        for bench_vals in bench_by_horizon.values():
            valid &= ~bench_vals.isna().to_numpy()
        if not self.fill_missing_vol:
            valid &= ~vol_vals.isna().to_numpy()
        if self.drop_missing and not valid.all():
            df = df.loc[valid].copy()
            dts = pd.DatetimeIndex(df.index.get_level_values("datetime"))
            bench_by_horizon = {horizon: bench_vals[valid] for horizon, bench_vals in bench_by_horizon.items()}
            vol_vals = vol_vals.loc[df.index]

        collapsed = pd.Series(0.0, index=df.index, dtype=float)
        for col, horizon, weight in zip(label_cols, self.label_horizon_days_list, self.label_weights):
            bench_vals = bench_by_horizon[int(horizon)]
            raw = pd.to_numeric(df.loc[:, col], errors="coerce")
            excess = raw - bench_vals.to_numpy(dtype=float)
            downside = (-excess).clip(lower=0.0)
            vol_charge = vol_vals.to_numpy(dtype=float)
            if self.scale_vol_by_horizon:
                vol_charge = vol_charge * np.sqrt(float(horizon))
            target = excess - self.downside_penalty * downside
            target = target - self.volatility_penalty * vol_charge
            if self.divide_by_vol:
                denom = vol_vals.to_numpy(dtype=float)
                if self.scale_vol_by_horizon:
                    denom = denom * np.sqrt(float(horizon))
                target = target / denom
            collapsed = collapsed + float(weight) * pd.Series(target, index=df.index, dtype=float)

        if self.clip_abs_label is not None and self.clip_abs_label > 0:
            collapsed = collapsed.clip(lower=-self.clip_abs_label, upper=self.clip_abs_label)
        try:
            collapsed = collapsed.astype(df.loc[:, label_cols[0]].dtype)
        except Exception:
            pass
        df.loc[:, label_cols[0]] = collapsed
        if self.drop_extra_labels and len(label_cols) > 1:
            df = df.drop(columns=label_cols[1:])
        return df


class GroupNeutralize(Processor):
    """
    Cross-sectionally neutralize feature or label columns by a static instrument group.

    The group map is usually built from a security master, e.g. Sharadar
    TICKERS `sector` or `industry`. For each date and group, this subtracts
    the group mean from the selected columns. Small or missing groups fall
    back to the full daily mean so the transform remains usable at inference.
    """

    def __init__(
        self,
        group_map_csv: str,
        fields_group: str = "label",
        ticker_col: str = "ticker",
        group_col: str = "sector",
        min_group_size: int = 5,
        missing_group: str = "__UNKNOWN__",
        scale: bool = False,
    ):
        self.group_map_csv = str(group_map_csv)
        self.fields_group = fields_group
        self.ticker_col = str(ticker_col)
        self.group_col = str(group_col)
        self.min_group_size = max(1, int(min_group_size))
        self.missing_group = str(missing_group)
        self.scale = bool(scale)
        self._group_map = self._load_group_map()

    def _load_group_map(self) -> Dict[str, str]:
        path = Path(self.group_map_csv).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"group_map_csv not found: {path}")
        df = pd.read_csv(path, usecols=lambda c: c in {self.ticker_col, self.group_col})
        missing = {self.ticker_col, self.group_col}.difference(df.columns)
        if missing:
            raise ValueError(f"group_map_csv missing columns {sorted(missing)}: {path}")
        tickers = df[self.ticker_col].astype(str).str.upper().str.strip()
        groups = df[self.group_col].astype(str).str.strip()
        groups = groups.mask(groups.eq("") | groups.str.lower().isin({"nan", "none"}), self.missing_group)
        out = pd.Series(groups.values, index=tickers.values)
        out = out[~out.index.duplicated(keep="last")]
        return out.to_dict()

    def __call__(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names or "instrument" not in df.index.names:
            raise ValueError("GroupNeutralize expects MultiIndex with `datetime` and `instrument` levels")
        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df

        focus = df.loc[:, cols]
        dts = pd.Index(df.index.get_level_values("datetime"), name="datetime")
        instruments = pd.Index(df.index.get_level_values("instrument")).astype(str).str.upper().str.strip()
        groups = pd.Index([self._group_map.get(inst, self.missing_group) for inst in instruments], name="group")

        grouped = focus.groupby([dts, groups], sort=False)
        group_mean = grouped.transform("mean")
        group_count = pd.Series(1, index=df.index).groupby([dts, groups], sort=False).transform("sum")
        daily_mean = focus.groupby(dts, sort=False).transform("mean")
        use_daily = group_count.to_numpy() < self.min_group_size
        center = group_mean.copy()
        if use_daily.any():
            center.iloc[use_daily, :] = daily_mean.iloc[use_daily, :].values
        neutral = focus - center

        if self.scale:
            group_std = grouped.transform("std")
            daily_std = focus.groupby(dts, sort=False).transform("std")
            denom = group_std.copy()
            if use_daily.any():
                denom.iloc[use_daily, :] = daily_std.iloc[use_daily, :].values
            denom = denom.replace(0, np.nan)
            neutral = neutral / denom

        df.loc[:, cols] = neutral.values
        return df


class SelectiveCSZScoreNorm(Processor):
    """
    Cross-sectionally normalize selected feature columns while leaving excluded
    columns unchanged.

    This is useful for mixed feature sets. Stock-varying features should usually
    be normalized within each date, but market-wide regime fields are constant
    across the cross-section and robust z-score normalization would turn them
    into NaN.
    """

    def __init__(
        self,
        fields_group: str = "feature",
        method: str = "robust",
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        exclude_prefixes: Optional[Iterable[str]] = None,
    ):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"unsupported normalization method: {method}")
        self.include = {str(x) for x in include or []}
        self.exclude = {str(x) for x in exclude or []}
        self.exclude_prefixes = tuple(str(x) for x in exclude_prefixes or [])

    @staticmethod
    def _display_name(col) -> str:
        if isinstance(col, tuple):
            return str(col[-1])
        return str(col)

    def _selected_columns(self, cols) -> list:
        selected = []
        for col in cols:
            name = self._display_name(col)
            if self.include and name not in self.include:
                continue
            if name in self.exclude:
                continue
            if self.exclude_prefixes and name.startswith(self.exclude_prefixes):
                continue
            selected.append(col)
        return selected

    def __call__(self, df: pd.DataFrame):
        cols = get_group_columns(df, self.fields_group)
        selected = self._selected_columns(cols)
        if not selected:
            return df
        with pd.option_context("mode.chained_assignment", None):
            df[selected] = df[selected].groupby("datetime", group_keys=False).apply(self.zscore_func)
        return df
