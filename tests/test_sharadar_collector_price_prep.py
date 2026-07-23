import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_collector_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "data_collector" / "sharadar" / "collector.py"
    spec = importlib.util.spec_from_file_location("sharadar_collector", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_qlib_csv_uses_correct_adjusted_price_scale(tmp_path):
    mod = _load_collector_module()
    sep_dir = tmp_path / "sep"
    out_dir = tmp_path / "qlib_csv"
    sep_dir.mkdir()
    pd.DataFrame(
        {
            "ticker": ["AGL", "AGL"],
            "date": ["2026-03-30", "2026-03-31"],
            "open": [9.847, 7.91],
            "high": [10.225, 8.455],
            "low": [7.602, 7.48],
            "close": [8.035, 7.91],
            "volume": [330_000, 406_000],
            "closeadj": [8.035, 7.91],
            "closeunadj": [0.321, 7.91],
        }
    ).to_csv(sep_dir / "AGL.csv", index=False)

    collector = mod.SharadarCollector(api_key="dummy", out_dir=str(tmp_path))
    collector.prepare_qlib_csv(str(sep_dir), str(out_dir))
    out = pd.read_csv(out_dir / "AGL.csv")

    assert out["close"].tolist() == [8.035, 7.91]
    assert out["open"].tolist() == [9.847, 7.91]
    assert round(float(out.loc[0, "factor"]), 6) == round(8.035 / 0.321, 6)
