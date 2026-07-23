import argparse
from pathlib import Path
import importlib.util

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rank_us_sharadar_candidates.py"
    spec = importlib.util.spec_from_file_location("rank_us_sharadar_candidates", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
_config_in_cmd = _mod._config_in_cmd
_infer_run_dir_from_pred = _mod._infer_run_dir_from_pred
_pbo_from_performance_matrix = _mod._pbo_from_performance_matrix
_deflated_sharpe_probability = _mod._deflated_sharpe_probability
_run_is_dirty = _mod._run_is_dirty
_flatten_baseline_results = _mod._flatten_baseline_results
_candidate_release_decision = _mod._candidate_release_decision
_select_release_candidate = _mod._select_release_candidate
_resolve_rolling_settings = _mod._resolve_rolling_settings


def test_infer_run_dir_from_pred(tmp_path: Path):
    run_dir = tmp_path / "mlruns" / "123" / "abc123"
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True)
    pred = art_dir / "pred.pkl"
    pred.write_bytes(b"x")

    inferred = _infer_run_dir_from_pred(pred)
    assert inferred == run_dir


def test_config_in_cmd(tmp_path: Path):
    cfg = tmp_path / "workflow.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    assert _config_in_cmd(cfg, f"qrun {cfg.name}")
    assert _config_in_cmd(cfg, f"qrun {cfg}")
    assert not _config_in_cmd(cfg, "qrun other.yaml")


def test_run_is_dirty(tmp_path: Path):
    run_dir = tmp_path / "mlruns" / "1" / "runx"
    art = run_dir / "artifacts"
    art.mkdir(parents=True)

    fp = art / "code_status.txt"
    fp.write_text("On branch main\nnothing to commit, working tree clean\n", encoding="utf-8")
    assert _run_is_dirty(run_dir) is False

    fp.write_text("Changes not staged for commit:\n  modified: file.py\n", encoding="utf-8")
    assert _run_is_dirty(run_dir) is True


def test_deflated_sharpe_probability_penalizes_more_trials():
    ret = pd.Series([0.01] * 20 + [-0.002] * 10 + [0.004] * 20)

    one_trial = _deflated_sharpe_probability(ret, n_trials=1)
    many_trials = _deflated_sharpe_probability(ret, n_trials=50)

    assert one_trial is not None and many_trials is not None
    assert many_trials < one_trial


def test_pbo_from_performance_matrix_detects_available_estimate():
    matrix = pd.DataFrame(
        {
            "2022": [0.20, 0.05, 0.01],
            "2023": [0.18, 0.04, 0.02],
            "2024": [-0.10, 0.05, 0.03],
            "2025": [-0.12, 0.06, 0.04],
        },
        index=["overfit", "steady", "low"],
    )

    out = _pbo_from_performance_matrix(matrix, max_combinations=32)

    assert out["available"] is True
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["slices"] == 4


def test_flatten_baseline_results_extracts_latest_rolling_metrics():
    comparison = [
        ("QQQ", "full", {"excess_ann_return": 0.05, "excess_ir": 0.6}),
        ("QQQ", "stress x3 open", {"excess_ann_return": 0.03}),
    ]
    rolling = [
        ("QQQ", "w1", {"excess_ann_return": 0.02, "status": 1.0}),
        ("QQQ", "w2", {"excess_ann_return": -0.01, "status": 0.0}),
    ]
    checks = [("baseline_QQQ_latest_rolling_excess_ann", False, "bad")]

    out = _flatten_baseline_results(comparison, rolling, checks)

    assert out["baseline_ok"] is False
    assert out["qqq_full_excess"] == 0.05
    assert out["qqq_latest_rolling"] == -0.01
    assert out["qqq_rolling_pass_rate"] == 0.5


def test_resolve_rolling_settings_uses_profile_defaults_and_yaml_mode():
    args = argparse.Namespace(
        rolling_window_days=None,
        rolling_step_days=None,
        rolling_min_days=None,
        rolling_min_excess_ann=None,
        rolling_min_ir=None,
        rolling_max_mdd_abs=None,
        rolling_max_turnover=None,
        rolling_min_pass_rate=None,
        rolling_mode=None,
        rolling_ir_metric=None,
    )

    out = _resolve_rolling_settings(args, {"mode": "continuous", "ir_metric": "excess"}, "growth")

    assert out["mode"] == "continuous"
    assert out["ir_metric"] == "excess"
    assert out["window_days"] == 126
    assert out["step_days"] == 63
    assert out["min_days"] == 63
    assert out["min_pass_rate"] == 0.65


def test_resolve_rolling_settings_prefers_yaml_over_cli():
    args = argparse.Namespace(
        rolling_window_days=504,
        rolling_step_days=252,
        rolling_min_days=252,
        rolling_min_excess_ann=0.02,
        rolling_min_ir=0.5,
        rolling_max_mdd_abs=0.25,
        rolling_max_turnover=0.05,
        rolling_min_pass_rate=0.9,
        rolling_mode="independent",
        rolling_ir_metric="strategy",
    )

    out = _resolve_rolling_settings(
        args,
        {
            "window_days": 126,
            "step_days": 63,
            "min_days": 63,
            "min_excess_ann": 0.0,
            "min_ir": 0.0,
            "max_mdd_abs": 0.45,
            "max_turnover": 0.2,
            "min_pass_rate": 0.65,
            "mode": "continuous",
            "ir_metric": "excess",
        },
        "release",
    )

    assert out["window_days"] == 126
    assert out["step_days"] == 63
    assert out["min_days"] == 63
    assert out["min_excess_ann"] == 0.0
    assert out["min_ir"] == 0.0
    assert out["max_mdd_abs"] == 0.45
    assert out["max_turnover"] == 0.2
    assert out["min_pass_rate"] == 0.65
    assert out["mode"] == "continuous"
    assert out["ir_metric"] == "excess"


def test_candidate_release_decision_requires_baseline_when_requested():
    row = {
        "strategy_feasibility_ok": True,
        "gates_ok": True,
        "rolling_ok": True,
        "baseline_ok": False,
    }

    decision = _candidate_release_decision(row, require_baseline=True)

    assert decision["release_ready"] is False
    assert decision["missing_or_failed"] == ["external_baseline_gates"]


def test_select_release_candidate_returns_first_ready_candidate():
    rows = [{"candidate": "bad", "release_ready": False}, {"candidate": "good", "release_ready": True}]

    assert _select_release_candidate(rows)["candidate"] == "good"
