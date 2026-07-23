import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "walkforward_ensemble_us_sharadar.py"
    spec = importlib.util.spec_from_file_location("walkforward_ensemble_us_sharadar", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return {
        "qlib_init": {
            "provider_uri": "/old/provider",
            "exp_manager": {"kwargs": {"uri": "/old/mlruns", "default_exp_name": "exp"}},
        },
        "data_handler_config": {
            "end_time": "2026-04-30",
            "learn_processors": [
                {"class": "BenchmarkExcessLabel", "kwargs": {"benchmark_pkl": "/old/bench.pkl"}},
            ],
            "label": [["Ref($close, -21)/Ref($close, -1) - 1"], ["LABEL0"]],
        },
        "port_analysis_config": {"backtest": {"start_time": "2024-01-01", "end_time": "2026-04-30"}},
        "task": {
            "model": {"class": "LGBRankerModel"},
            "dataset": {
                "kwargs": {
                    "segments": {"test": ["2024-02-01", "2026-04-30"]},
                    "handler": {"kwargs": {"learn_processors": []}},
                }
            },
        },
    }


def test_write_member_runtime_config_overrides_paths_and_dates(tmp_path):
    mod = _load_module()
    source = tmp_path / "member.yaml"
    source.write_text(yaml.safe_dump(_config()), encoding="utf-8")

    out, cfg = mod._write_member_runtime_config(
        source_config=source,
        runtime_config=tmp_path / "runtime" / "member.yaml",
        provider_uri="/new/provider",
        mlruns_uri="sqlite:////tmp/mlflow.db",
        benchmark_pkl="/new/bench.pkl",
        test_start="2024-02-01",
        test_end="2026-05-05",
        override_label_benchmark_pkl=True,
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert cfg["qlib_init"]["provider_uri"] == "/new/provider"
    assert data["qlib_init"]["exp_manager"]["kwargs"]["uri"] == "sqlite:////tmp/mlflow.db"
    assert data["data_handler_config"]["learn_processors"][0]["kwargs"]["benchmark_pkl"] == "/new/bench.pkl"
    assert data["data_handler_config"]["end_time"] == "2026-05-05"
    assert data["port_analysis_config"]["backtest"]["start_time"] == "2024-02-01"
    assert data["task"]["dataset"]["kwargs"]["segments"]["test"] == ["2024-02-01", "2026-05-05"]


def test_build_member_train_command_includes_walkforward_options(tmp_path):
    mod = _load_module()

    cmd = mod._build_member_train_command(
        python_bin="/venv/main/bin/python",
        walkforward_script=tmp_path / "walkforward.py",
        config=tmp_path / "runtime.yaml",
        provider_uri="/provider",
        test_start="2024-02-01",
        test_end="2026-05-05",
        test_block="year",
        valid_days=63,
        exp_name="member_exp",
        out_pred=tmp_path / "pred.pkl",
        manifest=tmp_path / "manifest.json",
        embargo_days=20,
        train_lookback_days=756,
    )

    assert cmd[:4] == ["/venv/main/bin/python", str(tmp_path / "walkforward.py"), "--config", str(tmp_path / "runtime.yaml")]
    assert cmd[cmd.index("--exp_name") + 1] == "member_exp"
    assert cmd[cmd.index("--embargo_days") + 1] == "20"
    assert cmd[cmd.index("--train_lookback_days") + 1] == "756"


def test_flatten_used_runs_adds_member_metadata():
    mod = _load_module()
    member = {
        "name": "v9",
        "role": "defensive",
        "experiment": "exp",
        "mlruns_uri": "/mlruns",
        "model_class": "LGBRankerModel",
        "manifest": {"used_runs": [{"run_id": "abc", "test": ["2025-01-02", "2025-12-31"]}]},
    }

    rows = mod._flatten_used_runs(member)

    assert rows == [
        {
            "member": "v9",
            "role": "defensive",
            "run_id": "abc",
            "test": ["2025-01-02", "2025-12-31"],
            "experiment": "exp",
            "mlruns_uri": "/mlruns",
            "model_class": "LGBRankerModel",
        }
    ]


def test_date_summary_reports_prediction_span():
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2026-01-01", periods=2), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    assert mod._date_summary(pred) == {
        "rows": 4,
        "dates": 2,
        "start": "2026-01-01",
        "end": "2026-01-02",
    }
