import importlib.util
import sys
from pathlib import Path

import yaml


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_us_dual_target_selector_config.py"
    spec = importlib.util.spec_from_file_location("build_us_dual_target_selector_config", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_config_rewrites_labels_processor_and_sec_features(tmp_path):
    mod = _load_module()
    base = tmp_path / "base.yaml"
    out = tmp_path / "dual.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data_handler_config": {
                    "label": [["Ref($close, -11)/Ref($close, -1) - 1", "LABEL0"]],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {
                            "class": "BenchmarkExcessLabel",
                            "module_path": "qlib.contrib.data.processor",
                            "kwargs": {"label_horizon_days": 10},
                        },
                    ],
                    "extra_fields": ["$existing"],
                    "extra_names": ["EXISTING"],
                },
                "task": {
                    "model": {
                        "class": "StackedSignalScoreModel",
                        "kwargs": {"sleeves": {"base": {"EXISTING": 1.0}}, "max_sleeves": 1},
                    }
                },
                "port_analysis_config": {"strategy": {"kwargs": {"hold_thresh": 10}}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    mod.build_config(
        base_config=base,
        out=out,
        horizons=[10, 20],
        weights=[0.25, 0.75],
        benchmark_pkl="/bench.pkl",
        benchmark_kind="return",
        label_ref_start_days=1,
        vol_feature="RISK_VOL_20D",
        downside_penalty=0.75,
        volatility_penalty=0.2,
        clip_abs_label=0.0,
        include_sec_features=True,
    )

    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    dh = cfg["data_handler_config"]
    assert dh["label"] == [
        ["Ref($close, -11)/Ref($close, -1) - 1", "Ref($close, -21)/Ref($close, -1) - 1"],
        ["LABEL0", "LABEL1"],
    ]
    label_proc = [p for p in dh["learn_processors"] if p.get("class") == "DualHorizonPortfolioUtilityLabel"][0]
    assert label_proc["kwargs"]["label_horizon_days"] == [10, 20]
    assert label_proc["kwargs"]["label_weights"] == [0.25, 0.75]
    assert cfg["port_analysis_config"]["strategy"]["kwargs"]["hold_thresh"] == 20
    assert "$sec_alpha_event_composite" in dh["extra_fields"]
    assert "SEC_ALPHA_EVENT_COMPOSITE" in dh["extra_names"]
    assert "sec_event" in cfg["task"]["model"]["kwargs"]["sleeves"]
    assert cfg["task"]["model"]["kwargs"]["max_sleeves"] == 2


def test_parse_csv_floats_normalizes_weights():
    mod = _load_module()

    assert mod.parse_csv_floats("1,3", expected=2) == [0.25, 0.75]
