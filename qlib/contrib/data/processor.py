from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...log import TimeInspector
from ...data.dataset.processor import Processor, get_group_columns


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
        bench = pd.read_pickle(path)
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
