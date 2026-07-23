import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "evaluate_us_sharadar_feature_ic.py"
    spec = importlib.util.spec_from_file_location("evaluate_us_sharadar_feature_ic", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_feature_candidates_extract_extra_names():
    mod = _load_module()
    cfg = {
        "data_handler_config": {
            "extra_fields": ["P($$netinc_q) / (P($$equity_q) + 1e-12)"],
            "extra_names": ["ROE_Q"],
        },
        "task": {"dataset": {"kwargs": {"handler": {"class": "Alpha158WithPIT"}}}},
    }

    assert mod._feature_candidates(cfg, "extra") == [
        ("extra", "P($$netinc_q) / (P($$equity_q) + 1e-12)", "ROE_Q")
    ]


def test_sample_month_dates_uses_tail_month_ends():
    mod = _load_module()
    cal = pd.bdate_range("2024-01-01", "2024-04-30")

    out = mod._sample_month_dates(cal, max_months=2, sample_from="tail")

    assert out == [pd.Timestamp("2024-03-29"), pd.Timestamp("2024-04-30")]


def test_daily_ic_requires_min_daily_count():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    feature = pd.Series([1.0, 2.0, 3.0], index=idx)
    label = pd.Series([1.0, 4.0, 9.0], index=idx)

    ic = mod._daily_ic(feature, label, min_daily_count=2)

    assert list(ic.index) == [pd.Timestamp("2024-01-02")]
    assert abs(ic.iloc[0] - 1.0) < 1e-12
