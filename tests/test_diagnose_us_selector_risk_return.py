import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "diagnose_us_selector_risk_return.py"
    spec = importlib.util.spec_from_file_location("diagnose_us_selector_risk_return", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_risk_tier_thresholds():
    mod = _load_module()

    assert mod.risk_tier(0.8, 0.2) == "low"
    assert mod.risk_tier(1.2, 0.5) == "medium"
    assert mod.risk_tier(1.8, 0.4) == "high"
    assert mod.risk_tier(1.0, 0.8) == "high"
    assert mod.risk_tier(None, 0.2) == "unknown"


def test_score_band_thresholds():
    mod = _load_module()

    assert mod.score_band(0.995) == "top_1pct"
    assert mod.score_band(0.98) == "top_2_5pct"
    assert mod.score_band(0.96) == "top_5pct"
    assert mod.score_band(0.91) == "top_10pct"
    assert mod.score_band(0.8) == "upper_mid"


def test_build_summary_groups_score_and_risk():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "score_band": ["top_1pct", "top_1pct", "top_1pct", "top_5pct"],
            "risk_tier": ["high", "high", "low", "low"],
            "excess_return_20d": [0.03, -0.01, 0.02, -0.02],
            "beta_qqq_63d": [1.8, 1.7, 0.6, 0.5],
            "vol_20d": [0.8, 0.7, 0.2, 0.2],
            "score_percentile": [0.995, 0.992, 0.991, 0.96],
        }
    )

    out = mod.build_summary(frame, horizon=20, group_cols=["score_band", "risk_tier"], min_samples=1)

    high = out[(out["score_band"] == "top_1pct") & (out["risk_tier"] == "high")].iloc[0]
    assert high["sample_count"] == 2
    assert high["median"] == pytest.approx(0.01)
    assert high["hit_rate"] == 0.5
