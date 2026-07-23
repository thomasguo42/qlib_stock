# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, List, Optional

from .handler import Alpha158, check_transform_proc
from .loader import Alpha158DL
from ...data.dataset.handler import DataHandlerLP


DEFAULT_PIT_FIELDS: List[str] = [
    "assets",
    "liabilities",
    "equity",
    "revenue",
    "netinc",
    "ebit",
    "ebitda",
    "cashneq",
    "debt",
    "fcf",
    "capex",
    "workingcapital",
    "currentratio",
    "grossmargin",
    "netmargin",
    "roe",
    "roa",
    "roic",
    "eps",
    "epsdil",
    "bvps",
    "shareswa",
    "shareswadil",
    "divyield",
    "dps",
]


class Alpha158WithPIT(Alpha158):
    """
    Alpha158 features augmented with PIT fundamentals.

    The PIT fields will be accessed via the P operator, e.g. P($$assets_q).
    """

    def __init__(
        self,
        *args,
        pit_fields: Optional[Iterable[str]] = None,
        pit_interval: str = "q",
        extra_fields: Optional[Iterable[str]] = None,
        extra_names: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        self.pit_fields = list(pit_fields) if pit_fields is not None else DEFAULT_PIT_FIELDS
        self.pit_interval = pit_interval.lower()
        self.extra_fields = [str(f).strip() for f in (extra_fields or []) if str(f).strip()]
        if extra_names is None:
            self.extra_names = [f"EXTRA_{i}" for i in range(len(self.extra_fields))]
        else:
            self.extra_names = [str(n).strip() for n in extra_names]
        if len(self.extra_fields) != len(self.extra_names):
            raise ValueError("extra_fields and extra_names must have the same length")
        super().__init__(*args, **kwargs)

    def get_feature_config(self):
        fields, names = Alpha158DL.get_feature_config()

        pit_fields = []
        pit_names = []
        for raw in self.pit_fields:
            if raw is None:
                continue
            field = str(raw).strip().lower()
            if not field:
                continue
            pit_fields.append(f"P($${field}_{self.pit_interval})")
            pit_names.append(f"{field.upper()}_{self.pit_interval.upper()}")

        extra_fields = list(self.extra_fields)
        extra_names = list(self.extra_names)

        return fields + pit_fields + extra_fields, names + pit_names + extra_names


class SharadarFeatureHandler(DataHandlerLP):
    """
    Sharadar PIT/extra feature handler without the built-in Alpha158 technical features.

    This is useful for conservative factor-only baselines where broad Alpha158 price/volume
    features would otherwise dominate model capacity and make regime diagnostics harder.
    """

    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        pit_fields: Optional[Iterable[str]] = None,
        pit_interval: str = "q",
        extra_fields: Optional[Iterable[str]] = None,
        extra_names: Optional[Iterable[str]] = None,
        label=None,
        **kwargs,
    ):
        self.pit_fields = list(pit_fields or [])
        self.pit_interval = str(pit_interval).lower()
        self.extra_fields = [str(f).strip() for f in (extra_fields or []) if str(f).strip()]
        if extra_names is None:
            self.extra_names = [f"EXTRA_{i}" for i in range(len(self.extra_fields))]
        else:
            self.extra_names = [str(n).strip() for n in extra_names]
        if len(self.extra_fields) != len(self.extra_names):
            raise ValueError("extra_fields and extra_names must have the same length")

        infer_processors = check_transform_proc(infer_processors or [], fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors or [], fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": label or self.get_label_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        fields = []
        names = []
        for raw in self.pit_fields:
            if raw is None:
                continue
            field = str(raw).strip().lower()
            if not field:
                continue
            fields.append(f"P($${field}_{self.pit_interval})")
            names.append(f"{field.upper()}_{self.pit_interval.upper()}")
        fields.extend(self.extra_fields)
        names.extend(self.extra_names)
        if not fields:
            raise ValueError("SharadarFeatureHandler requires at least one pit_field or extra_field")
        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
