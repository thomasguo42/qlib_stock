import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "rebuild_us_sharadar_pit_ratio_features.py"
    spec = importlib.util.spec_from_file_location("rebuild_us_sharadar_pit_ratio_features", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_ratio_frame_uses_datekey_forward_fill():
    mod = _load_module()
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-01-08"))
    sf1 = pd.DataFrame(
        {
            "datekey": ["2024-01-03"],
            "calendardate": ["2023-12-31"],
            "assets": [100.0],
            "equity": [50.0],
            "revenue": [25.0],
            "netinc": [5.0],
            "ebitda": [7.5],
            "cashneq": [10.0],
            "debt": [20.0],
            "fcf": [4.0],
            "capex": [-2.0],
            "dps": [1.0],
            "eps": [2.0],
            "bvps": [10.0],
            "shareswa": [2.0],
            "marketcap": [1000.0],
        }
    )
    close = pd.Series([10.0, 10.0, 20.0, 20.0, 20.0], index=cal)

    out = mod._compute_ratio_frame(sf1, close, cal, date_col="datekey")

    assert pd.isna(out.loc[pd.Timestamp("2024-01-02"), "roe_q"])
    assert out.loc[pd.Timestamp("2024-01-03"), "roe_q"] == pytest.approx(0.1)
    assert out.loc[pd.Timestamp("2024-01-04"), "book_px_q"] == pytest.approx(0.5)
    assert out.loc[pd.Timestamp("2024-01-04"), "fcf_yield_q"] == pytest.approx(0.1)
    assert out.loc[pd.Timestamp("2024-01-04"), "marketcap_q"] == pytest.approx(1000.0)
    assert out.loc[pd.Timestamp("2024-01-04"), "log_marketcap_q"] > 6.9
