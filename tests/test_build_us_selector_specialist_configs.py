import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_us_selector_specialist_configs.py"
    spec = importlib.util.spec_from_file_location("build_us_selector_specialist_configs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_config():
    return {
        "qlib_init": {"exp_manager": {"kwargs": {"default_exp_name": "base_exp"}}},
        "data_handler_config": {
            "extra_fields": ["$risk_ret_20d"],
            "extra_names": ["RISK_RET_20D"],
            "learn_processors": [{"class": "DropnaLabel"}],
        },
        "task": {
            "model": {
                "class": "RegimeSleeveScoreModel",
                "kwargs": {
                    "sleeves": {
                        "qqq_leadership": {"RISK_RELRET_QQQ_63D": 1.0},
                        "momentum_regime": {"RISK_RET_63D": 1.0},
                        "quality_value": {"FCF_YIELD_Q": 1.0},
                    },
                    "state_weights": {
                        "risk_on": {"qqq_leadership": 0.2, "momentum_regime": 0.2, "quality_value": 0.6},
                        "chop": {"qqq_leadership": 0.1, "momentum_regime": 0.2, "quality_value": 0.7},
                    },
                    "default_state": "chop",
                },
            }
        },
    }


def test_build_high_risk_upside_config_adds_filter_and_risk_features():
    mod = _load_module()

    cfg = mod.build_high_risk_upside_config(_base_config())
    dh = cfg["data_handler_config"]
    processors = dh["learn_processors"]

    assert "RISK_BETA_QQQ_63D" in dh["extra_names"]
    assert "RISK_VOL_20D" in dh["extra_names"]
    assert processors[-1]["class"] == "RiskTierFilter"
    assert processors[-1]["kwargs"]["tier"] == "high"
    assert cfg["selector_specialist"]["kind"] == "high_risk_upside"
    assert cfg["experiment_name"] == "base_exp_high_risk_upside"


def test_build_high_risk_upside_config_reweights_regime_sleeves():
    mod = _load_module()

    cfg = mod.build_high_risk_upside_config(_base_config())
    weights = cfg["task"]["model"]["kwargs"]["state_weights"]["risk_on"]

    assert weights["qqq_leadership"] >= 0.45
    assert weights["quality_value"] <= 0.20
    assert round(sum(weights.values()), 10) == 1.0
