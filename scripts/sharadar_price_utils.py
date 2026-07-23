#!/usr/bin/env python
"""Utilities for converting Sharadar SEP prices into Qlib price fields."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


SEP_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj"]
QLIB_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "factor"]


def prepare_sep_qlib_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw Sharadar SEP frame to Qlib daily OHLCV fields.

    Sharadar SEP has three close concepts in the local bundle:
    - close: split-adjusted close
    - closeadj: split- and dividend-adjusted close
    - closeunadj: raw exchange close before split/dividend adjustment

    Qlib's price fields should be on the adjusted close scale. Applying
    closeadj / closeunadj directly to the already split-adjusted close
    double-applies stock splits, so OHLC must be scaled by closeadj / close.
    """
    if raw.empty or "date" not in raw.columns or "closeadj" not in raw.columns or "close" not in raw.columns:
        return pd.DataFrame(columns=QLIB_PRICE_COLUMNS)

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for col in _present(["open", "high", "low", "close", "volume", "closeadj", "closeunadj"], df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    split_adjusted_close = df["close"].replace(0, np.nan)
    price_factor = (df["closeadj"] / split_adjusted_close).replace([np.inf, -np.inf], np.nan)
    if "closeunadj" in df.columns:
        trade_factor = (df["closeadj"] / df["closeunadj"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    else:
        trade_factor = price_factor

    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]
    for col in ["open", "high", "low"]:
        out[col] = df[col] * price_factor if col in df.columns else np.nan
    out["close"] = df["closeadj"]
    out["volume"] = df["volume"] if "volume" in df.columns else np.nan
    out["factor"] = trade_factor
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.loc[:, QLIB_PRICE_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"])


def _present(candidates: Iterable[str], columns: Iterable[str]) -> list[str]:
    cols = set(columns)
    return [col for col in candidates if col in cols]
