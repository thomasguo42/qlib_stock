import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "summarize_us_sharadar_release_failures.py"
    spec = importlib.util.spec_from_file_location("summarize_us_sharadar_release_failures", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_log_extracts_release_failure_map(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "candidate.log"
    log.write_text(
        """
== Config Summary ==
- config: /Stock/qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml
- gate_profile: qqq_release
- benchmark_config: QQQ
- benchmark_source: pkl:/root/.qlib/qlib_data/us_data/bench_qqq.pkl
- label_horizon_days: 5

== Config Consistency Checks ==
check | status | detail
--- | --- | ---
valid_test_embargo | PASS | gap=9 trading_days >= required=5
qqq_release_config_benchmark | FAIL | benchmark=AAPL expected QQQ

== Training Diagnostics ==
check | status | detail
--- | --- | ---
training_best_iteration_min | PASS | min=6, median=35.0, threshold=5, weak=[]
- training_diagnostics_overall: PASS

== Model Quality Checks ==
check | status | detail
--- | --- | ---
model_quality_full_mean_ic | FAIL | 0.0028 >= 0.0100
- model_quality_overall: FAIL

== External Baseline Gates ==
check | status | detail
--- | --- | ---
baseline_QQQ_full_excess_ann | FAIL | 0.0527 >= 0.0800
baseline_QQQ_full_mdd_abs | PASS | |-0.3315| <= 0.3500
baseline_QQQ_rolling_pass_rate | FAIL | 0.6667 >= 0.8500 (passed=6/9, min_excess_ann=0.0000)

== External Baseline Regime Gates ==
check | status | detail
--- | --- | ---
baseline_regime_QQQ_down_days_excess_ann | FAIL | -0.3530 >= 0.2000 (n_days=242 >= 40)

== Rolling Walk-Forward Checks ==
- rolling_pass_rate: 0.6667
- rolling_worst_excess_ann: -0.1249
- rolling_overall: FAIL (pass_rate=0.6667 >= threshold=0.8500)

== Release Decision ==
- release_ready: FAIL
- release_missing_or_failed: model_quality,external_baseline_gates,rolling
""",
        encoding="utf-8",
    )

    row = mod.parse_log(log)

    assert row["candidate"] == "demo_v1"
    assert row["release_ready"] == "FAIL"
    assert row["release_missing_or_failed"] == "model_quality,external_baseline_gates,rolling"
    assert row["valid_test_embargo_status"] == "PASS"
    assert row["training_best_iteration_min"] == 6.0
    assert row["model_quality_full_mean_ic"] == 0.0028
    assert row["baseline_QQQ_full_excess_ann"] == 0.0527
    assert row["baseline_QQQ_full_mdd_abs"] == 0.3315
    assert row["baseline_QQQ_rolling_pass_rate"] == 0.6667
    assert row["baseline_regime_QQQ_down_days_excess_ann"] == -0.3530
    assert row["rolling_pass_rate"] == 0.6667
    assert row["rolling_overall"] == "FAIL"
    assert "model_quality_full_mean_ic" in row["fail_checks"]
    assert "baseline_QQQ_full_excess_ann" in row["fail_checks"]
    assert "release_ready" in row["fail_checks"]
    assert row["fail_count"] >= 5


def test_write_csv_and_markdown(tmp_path: Path):
    mod = _load_module()
    rows = [
        {
            "source_log": "/tmp/a.log",
            "candidate": "a",
            "release_ready": "FAIL",
            "baseline_QQQ_full_excess_ann": 0.01,
            "fail_count": 2,
        }
    ]
    csv_path = tmp_path / "failure_map.csv"
    md_path = tmp_path / "failure_map.md"

    mod._write_csv(csv_path, rows)
    mod._write_markdown(md_path, rows)

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "candidate" in csv_text
    assert "a" in csv_text
    assert "| candidate | release_ready |" in md_text
    assert "| a | FAIL |" in md_text


def test_candidate_name_strips_runtime_suffix():
    mod = _load_module()

    name = mod._candidate_from_config(
        "/tmp/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.runtime.yaml",
        "fallback.log",
    )

    assert name == "demo_v1"
