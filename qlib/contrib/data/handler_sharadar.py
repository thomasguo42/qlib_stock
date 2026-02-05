# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, List, Optional

from .handler import Alpha158
from .loader import Alpha158DL


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
