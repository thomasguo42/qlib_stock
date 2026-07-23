import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "run_us_sharadar_research_grid.py"
    spec = importlib.util.spec_from_file_location("run_us_sharadar_research_grid", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**kwargs):
    base = {
        "python_bin": "/venv/main/bin/python",
        "release_script": "scripts/run_us_sharadar_release.py",
        "provider_uri": "/root/.qlib/qlib_data/us_data",
        "benchmark_pkl": "/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        "train_mode": "walkforward",
        "walkforward_ensemble_script": "scripts/walkforward_ensemble_us_sharadar.py",
        "ensemble_primary_config": "",
        "ensemble_defensive_config": "",
        "ensemble_primary_name": "primary",
        "ensemble_defensive_name": "defensive",
        "ensemble_primary_weight": 0.70,
        "ensemble_defensive_weight": 0.30,
        "ensemble_normalize": "rank_zscore",
        "override_label_benchmark_pkl": False,
        "mlruns_uri": "",
        "test_start": "2022-01-03",
        "test_end": "2026-05-05",
        "start": "",
        "end": "",
        "test_block": "year",
        "valid_days": 63,
        "embargo_days": None,
        "train_lookback_days": None,
        "model_quality_mode": "strict",
        "training_min_best_iteration": 5,
        "skip_train": False,
        "allow_pred_beyond_test_segment": False,
        "skip_strategy_weighted_quality": False,
        "skip_rebalance_interval_quality": False,
        "skip_active_risk": False,
        "run_preflight_audits": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_config_key_strips_standard_prefix_and_suffix():
    mod = _load_module()

    path = Path(
        "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h60_etfcore_v1_topk40.yaml"
    )

    assert mod._config_key(path) == "qv_risk_sf3a_regime_ranker_h60_etfcore_v1"


def test_collect_configs_sorts_dedupes_and_skips_missing(tmp_path: Path):
    mod = _load_module()
    b = tmp_path / "b.yaml"
    a = tmp_path / "a.yaml"
    a.write_text("x: 1\n", encoding="utf-8")
    b.write_text("x: 2\n", encoding="utf-8")

    out = mod._collect_configs([str(b), str(a), str(a), str(tmp_path / "missing.yaml")], [], max_configs=0)

    assert out == [a.resolve(), b.resolve()]


def test_build_release_command_uses_grid_paths_and_skips_repeated_preflights(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(_args(), cfg.resolve(), tmp_path / "grid" / "tag", "tag")
    cmd = plan["cmd"]

    assert plan["key"] == "demo_v1"
    assert str(plan["pred"]).endswith("/grid/tag/preds/demo_v1_tag_pred.pkl")
    assert "--walkforward_exp_name" in cmd
    assert "demo_v1_grid_tag" in cmd
    assert "--skip_universe_integrity" in cmd
    assert "--skip_price_adjustment_integrity" in cmd
    assert "--skip_hedge_feasibility" in cmd
    assert "--test_start" in cmd
    assert "2022-01-03" in cmd
    assert "--gate_profile" in cmd
    assert "growth" in cmd
    assert "--baseline_tickers" in cmd
    assert "QQQ,SPY,IXIC" in cmd
    assert cmd[cmd.index("--benchmark_pkl") + 1].endswith("/bench_qqq.pkl")
    assert "--preserve_label_benchmark_pkl" in cmd
    assert "--trial_registry" in cmd
    assert "--candidate_name" in cmd
    assert "demo_v1" in cmd


def test_build_release_command_passes_recency_and_trial_count(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(
        _args(
            recency_half_life_days=252,
            recency_min_weight=0.20,
            baseline_pkl_map="IXIC=/tmp/ixic.pkl",
            trial_registry=str(tmp_path / "registry.jsonl"),
        ),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
        trial_count=8,
    )
    cmd = plan["cmd"]

    assert cmd[cmd.index("--recency_half_life_days") + 1] == "252"
    assert cmd[cmd.index("--recency_min_weight") + 1] == "0.2"
    assert cmd[cmd.index("--baseline_pkl_map") + 1] == "IXIC=/tmp/ixic.pkl"
    assert cmd[cmd.index("--trial_count") + 1] == "8"


def test_build_release_command_passes_regime_reweight_args(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(
        _args(
            regime_reweight_feature="MKT_QQQ_RET_63D_LAG1",
            regime_reweight_thresholds="-0.05,0.04",
            label_tail_reweight=0.25,
            label_tail_quantile=0.15,
            sample_max_weight=4.0,
            year_balance_reweight=True,
        ),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
    )
    cmd = plan["cmd"]

    assert cmd[cmd.index("--regime_reweight_feature") + 1] == "MKT_QQQ_RET_63D_LAG1"
    assert "--regime_reweight_thresholds=-0.05,0.04" in cmd
    assert cmd[cmd.index("--label_tail_reweight") + 1] == "0.25"
    assert cmd[cmd.index("--label_tail_quantile") + 1] == "0.15"
    assert cmd[cmd.index("--sample_max_weight") + 1] == "4.0"
    assert "--year_balance_reweight" in cmd


def test_build_release_command_can_opt_into_label_benchmark_override(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(
        _args(override_label_benchmark_pkl=True),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
    )
    cmd = plan["cmd"]

    assert "--override_label_benchmark_pkl" in cmd
    assert "--preserve_label_benchmark_pkl" not in cmd


def test_build_release_command_can_collect_all_diagnostics(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(
        _args(collect_all_diagnostics=True),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
    )
    cmd = plan["cmd"]

    assert "--collect_all_diagnostics" in cmd


def test_build_release_command_preserves_sqlite_mlruns_uri(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    sqlite_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    plan = mod._build_release_command(
        _args(mlruns_uri=sqlite_uri),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
    )
    cmd = plan["cmd"]

    assert cmd[cmd.index("--mlruns_uri") + 1] == sqlite_uri


def test_build_release_command_passes_ensemble_args(tmp_path: Path):
    mod = _load_module()
    cfg = tmp_path / "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_demo_v1_topk40.yaml"
    primary = tmp_path / "primary.yaml"
    defensive = tmp_path / "defensive.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    primary.write_text("x: 1\n", encoding="utf-8")
    defensive.write_text("x: 1\n", encoding="utf-8")

    plan = mod._build_release_command(
        _args(
            train_mode="ensemble",
            ensemble_primary_config=str(primary),
            ensemble_defensive_config=str(defensive),
            ensemble_primary_name="v8",
            ensemble_defensive_name="v9",
            ensemble_primary_weight=0.65,
            ensemble_defensive_weight=0.35,
            ensemble_normalize="rank_zscore",
        ),
        cfg.resolve(),
        tmp_path / "grid" / "tag",
        "tag",
    )
    cmd = plan["cmd"]

    assert cmd[cmd.index("--train_mode") + 1] == "ensemble"
    assert cmd[cmd.index("--ensemble_primary_config") + 1] == str(primary)
    assert cmd[cmd.index("--ensemble_defensive_config") + 1] == str(defensive)
    assert cmd[cmd.index("--ensemble_primary_name") + 1] == "v8"
    assert cmd[cmd.index("--ensemble_defensive_name") + 1] == "v9"
    assert cmd[cmd.index("--ensemble_primary_weight") + 1] == "0.65"
    assert cmd[cmd.index("--ensemble_defensive_weight") + 1] == "0.35"


def test_build_failure_map_command_uses_existing_logs(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "logs" / "a.log"
    log.parent.mkdir()
    log.write_text("- release_ready: FAIL\n", encoding="utf-8")
    run_dir = tmp_path / "grid"

    cmd = mod._build_failure_map_command(
        _args(failure_map_csv="", failure_map_md="", failure_map_script="", skip_failure_map=False),
        run_dir,
        [{"log": str(log)}, {"log": str(tmp_path / "missing.log")}],
    )

    assert cmd[0] == "/venv/main/bin/python"
    assert cmd.count("--log") == 1
    assert str(log.resolve()) in cmd
    assert cmd[cmd.index("--out_csv") + 1] == str((run_dir / "failure_map.csv").resolve())
    assert cmd[cmd.index("--out_md") + 1] == str((run_dir / "failure_map.md").resolve())


def test_build_failure_map_command_can_be_disabled(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "a.log"
    log.write_text("- release_ready: FAIL\n", encoding="utf-8")

    cmd = mod._build_failure_map_command(
        _args(skip_failure_map=True, failure_map_csv="", failure_map_md="", failure_map_script=""),
        tmp_path,
        [{"log": str(log)}],
    )

    assert cmd == []


def test_run_release_subprocess_marks_timeout():
    mod = _load_module()

    returncode, stdout, timed_out = mod._run_release_subprocess(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=1,
    )

    assert returncode == 124
    assert timed_out is True
    assert "TIMEOUT after 1 seconds" in stdout


def test_parse_release_output_extracts_paths():
    mod = _load_module()

    parsed = mod._parse_release_output(
        "train_mode=walkforward\npred=/tmp/pred.pkl\nruntime_config=/tmp/cfg.yaml\n"
        "walkforward_manifest=/tmp/manifest.json\nrelease validation PASS\n"
    )

    assert parsed["pred"] == "/tmp/pred.pkl"
    assert parsed["runtime_config"] == "/tmp/cfg.yaml"
    assert parsed["walkforward_manifest"] == "/tmp/manifest.json"
    assert parsed["release_validation"] == "PASS"


def test_parse_release_output_marks_diagnostics_completion():
    mod = _load_module()

    parsed = mod._parse_release_output(
        "pred=/tmp/pred.pkl\n"
        "- release_ready: FAIL\n"
        "- release_missing_or_failed: model_quality,rolling\n"
        "release validation diagnostics completed\n"
    )

    assert parsed["pred"] == "/tmp/pred.pkl"
    assert parsed["release_validation"] == "DIAGNOSTICS"
    assert parsed["release_ready"] == "FAIL"
    assert parsed["release_missing_or_failed"] == "model_quality,rolling"


def test_target_screen_filter_keeps_only_passing_target_horizon(tmp_path: Path):
    mod = _load_module()
    keep = tmp_path / "keep.yaml"
    skip = tmp_path / "skip.yaml"
    keep.write_text(
        """
data_handler_config:
  label:
    - ["Ref($close, -41)/Ref($close, -1) - 1"]
    - ["LABEL0"]
  learn_processors:
    - class: BenchmarkExcessLabel
      kwargs: {}
    - class: GroupNeutralize
      kwargs:
        fields_group: label
""",
        encoding="utf-8",
    )
    skip.write_text(
        """
data_handler_config:
  label:
    - ["Ref($close, -21)/Ref($close, -1) - 1"]
    - ["LABEL0"]
  learn_processors:
    - class: DownsideAdjustedExcessLabel
      kwargs: {}
""",
        encoding="utf-8",
    )
    screen = tmp_path / "screen.csv"
    screen.write_text(
        "target_kind,horizon_days,screen_pass\nsector_neutral,40,True\ndownside_adjusted_excess,20,False\n",
        encoding="utf-8",
    )

    kept, rows = mod._filter_configs_by_target_screen([keep, skip], str(screen))

    assert kept == [keep]
    assert rows[0]["screen_keys"] == [("sector_neutral", 40), ("benchmark_excess", 40)]
    assert rows[0]["screen_pass"] is True
    assert rows[1]["screen_pass"] is False
