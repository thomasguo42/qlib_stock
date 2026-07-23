import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_price_utils():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "sharadar_price_utils.py"
    spec = importlib.util.spec_from_file_location("sharadar_price_utils", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_sep_qlib_frame_does_not_double_adjust_split_prices():
    mod = _load_price_utils()
    raw = pd.DataFrame(
        {
            "date": ["2026-03-30", "2026-04-06"],
            "open": [162.60, 166.25],
            "high": [166.0, 178.0],
            "low": [160.0, 165.0],
            "close": [164.70, 176.19],
            "volume": [10_211_000, 8_191_000],
            "closeadj": [164.70, 176.19],
            "closeunadj": [4117.51, 176.19],
        }
    )

    out = mod.prepare_sep_qlib_frame(raw)

    assert out["close"].tolist() == [164.70, 176.19]
    assert out["open"].tolist() == [162.60, 166.25]
    assert round(float(out.loc[0, "factor"]), 6) == round(164.70 / 4117.51, 6)
    assert float(out.loc[1, "factor"]) == 1.0


def test_prepare_sep_qlib_frame_handles_reverse_split_without_artificial_crash():
    mod = _load_price_utils()
    raw = pd.DataFrame(
        {
            "date": ["2026-03-30", "2026-03-31"],
            "open": [9.847, 7.91],
            "high": [10.225, 8.455],
            "low": [7.602, 7.48],
            "close": [8.035, 7.91],
            "volume": [330_000, 406_000],
            "closeadj": [8.035, 7.91],
            "closeunadj": [0.321, 7.91],
        }
    )

    out = mod.prepare_sep_qlib_frame(raw)
    ret = out["close"].iloc[1] / out["close"].iloc[0] - 1.0

    assert round(float(ret), 6) == round(7.91 / 8.035 - 1.0, 6)
    assert round(float(out.loc[0, "factor"]), 6) == round(8.035 / 0.321, 6)
    assert float(out.loc[1, "factor"]) == 1.0


def test_prepare_sep_qlib_frame_applies_dividend_adjustment_to_ohlc():
    mod = _load_price_utils()
    raw = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "open": [101.0],
            "high": [103.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
            "closeadj": [98.0],
            "closeunadj": [1000.0],
        }
    )

    out = mod.prepare_sep_qlib_frame(raw)

    assert round(float(out.loc[0, "open"]), 6) == 98.98
    assert float(out.loc[0, "close"]) == 98.0
    assert float(out.loc[0, "volume"]) == 1000.0
    assert float(out.loc[0, "factor"]) == 0.098
