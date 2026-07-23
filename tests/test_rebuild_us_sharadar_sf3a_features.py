import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rebuild_us_sharadar_sf3a_features.py"
    spec = importlib.util.spec_from_file_location("rebuild_us_sharadar_sf3a_features", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_overwrite_feature_bins_writes_lagged_fields(tmp_path):
    mod = _load_module()
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2020-01-01\n2020-01-02\n2020-01-03\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "inst13f_totalvalue_daily": [1.0, 2.0],
            "other_field": [9.0, 9.0],
        }
    ).to_csv(prepared / "AAPL.csv", index=False)

    written = mod._overwrite_feature_bins(prepared, provider, prefix="inst13f", max_workers=1)

    assert written == 1
    out = provider / "features" / "aapl" / "inst13f_totalvalue_daily.day.bin"
    arr = np.fromfile(out, dtype="<f")
    assert arr.tolist() == [1.0, 1.0, 2.0]
