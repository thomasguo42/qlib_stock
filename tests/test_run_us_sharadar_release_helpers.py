import importlib.util
import sys
from pathlib import Path

import yaml


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_us_sharadar_release.py"
    spec = importlib.util.spec_from_file_location("run_us_sharadar_release", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_runtime_config_overrides_paths(tmp_path):
    mod = _load_module()
    cfg = {
        "qlib_init": {
            "provider_uri": "/old/provider",
            "exp_manager": {"kwargs": {"uri": "/old/mlruns", "default_exp_name": "exp"}},
        },
        "data_handler_config": {
            "learn_processors": [
                {"class": "BenchmarkExcessLabel", "kwargs": {"benchmark_pkl": "/old/bench.pkl"}}
            ]
        },
    }

    out = mod._write_runtime_config(
        cfg,
        source_config=tmp_path / "workflow.yaml",
        provider_uri="/new/provider",
        mlruns_uri=tmp_path / "mlruns",
        benchmark_pkl=str(tmp_path / "bench.pkl"),
        tmp_dir=tmp_path,
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["qlib_init"]["provider_uri"] == "/new/provider"
    assert data["qlib_init"]["exp_manager"]["kwargs"]["uri"] == str(tmp_path / "mlruns")
    assert data["data_handler_config"]["learn_processors"][0]["kwargs"]["benchmark_pkl"] == str(tmp_path / "bench.pkl")


def test_write_runtime_config_preserves_label_benchmark_when_requested(tmp_path):
    mod = _load_module()
    cfg = {
        "qlib_init": {
            "provider_uri": "/old/provider",
            "exp_manager": {"kwargs": {"uri": "/old/mlruns", "default_exp_name": "exp"}},
        },
        "data_handler_config": {
            "learn_processors": [
                {"class": "BenchmarkExcessLabel", "kwargs": {"benchmark_pkl": "/old/bench.pkl"}}
            ]
        },
    }

    out = mod._write_runtime_config(
        cfg,
        source_config=tmp_path / "workflow.yaml",
        provider_uri="/new/provider",
        mlruns_uri=tmp_path / "mlruns",
        benchmark_pkl=str(tmp_path / "bench.pkl"),
        tmp_dir=tmp_path,
        override_label_benchmark_pkl=False,
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["qlib_init"]["provider_uri"] == "/new/provider"
    assert data["data_handler_config"]["learn_processors"][0]["kwargs"]["benchmark_pkl"] == "/old/bench.pkl"


def test_write_runtime_config_syncs_runtime_dates(tmp_path):
    mod = _load_module()
    cfg = {
        "qlib_init": {"exp_manager": {"kwargs": {"uri": "/old/mlruns", "default_exp_name": "exp"}}},
        "data_handler_config": {
            "end_time": "2026-01-30",
            "filter_pipe": [{"filter_end_time": "2026-01-30"}],
        },
        "port_analysis_config": {"backtest": {"start_time": "2022-01-01", "end_time": "2026-01-30"}},
        "task": {"dataset": {"kwargs": {"segments": {"test": ["2022-01-03", "2026-01-30"]}}}},
    }

    out = mod._write_runtime_config(
        cfg,
        source_config=tmp_path / "workflow.yaml",
        provider_uri="/provider",
        mlruns_uri=tmp_path / "mlruns",
        benchmark_pkl=str(tmp_path / "bench.pkl"),
        tmp_dir=tmp_path,
        test_start="2022-01-03",
        test_end="2026-04-30",
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["data_handler_config"]["end_time"] == "2026-04-30"
    assert data["data_handler_config"]["filter_pipe"][0]["filter_end_time"] == "2026-04-30"
    assert data["port_analysis_config"]["backtest"]["start_time"] == "2022-01-03"
    assert data["port_analysis_config"]["backtest"]["end_time"] == "2026-04-30"
    assert data["task"]["dataset"]["kwargs"]["segments"]["test"] == ["2022-01-03", "2026-04-30"]


def test_mlruns_uri_helpers_preserve_non_file_schemes(tmp_path):
    mod = _load_module()
    sqlite_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    assert mod._normalize_mlruns_uri(sqlite_uri) == sqlite_uri
    assert mod._mlruns_local_path(sqlite_uri) is None
    assert mod._mlruns_local_path(str(tmp_path / "mlruns")) == (tmp_path / "mlruns").resolve()


def test_write_runtime_config_accepts_sqlite_tracking_uri(tmp_path):
    mod = _load_module()
    sqlite_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = {"qlib_init": {"exp_manager": {"kwargs": {"uri": "/old/mlruns", "default_exp_name": "exp"}}}}

    out = mod._write_runtime_config(
        cfg,
        source_config=tmp_path / "workflow.yaml",
        provider_uri="/provider",
        mlruns_uri=sqlite_uri,
        benchmark_pkl=str(tmp_path / "bench.pkl"),
        tmp_dir=tmp_path,
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["qlib_init"]["exp_manager"]["kwargs"]["uri"] == sqlite_uri


def test_latest_provider_calendar_date(tmp_path):
    mod = _load_module()
    cal = tmp_path / "calendars"
    cal.mkdir()
    (cal / "day.txt").write_text("2026-04-28\n2026-04-29\n2026-04-30\n", encoding="utf-8")

    assert mod._latest_provider_calendar_date(str(tmp_path)) == "2026-04-30"


def test_filter_available_reference_markets(tmp_path):
    mod = _load_module()
    inst = tmp_path / "instruments"
    inst.mkdir()
    (inst / "all.txt").write_text("AAPL\t2000-01-01\t2099-12-31\n", encoding="utf-8")
    (inst / "pit_mrq_large_idx.txt").write_text("AAPL\t2000-01-01\t2099-12-31\n", encoding="utf-8")

    assert mod._filter_available_reference_markets("sp500,nasdaq100,all", str(tmp_path)) == "all"


def test_release_parser_includes_price_adjustment_audit_defaults():
    mod = _load_module()

    args = mod._build_parser().parse_args(["--config", "workflow.yaml"])

    assert args.price_adjustment_validator == "scripts/validate_us_sharadar_price_adjustments.py"
    assert args.hedge_feasibility_auditor == "scripts/audit_us_hedge_feasibility.py"
    assert args.raw_sep_dir == "~/.qlib/sharadar/raw/sep"
    assert args.raw_sfp_dir == "~/.qlib/sharadar/raw/sfp"
    assert args.benchmark_pkl.endswith("/bench_qqq.pkl")
    assert args.preserve_label_benchmark_pkl is True
    assert args.market == "pit_mrq_large_idx"
    assert args.skip_price_adjustment_integrity is False
    assert args.skip_hedge_feasibility is False
    assert args.skip_strategy_weighted_quality is False
    assert args.skip_rebalance_interval_quality is False
    assert args.skip_active_risk is False
    assert args.train_mode == "walkforward"
    assert args.walkforward_ensemble_script == "scripts/walkforward_ensemble_us_sharadar.py"
    assert args.ensemble_primary_weight == 0.70
    assert args.ensemble_defensive_weight == 0.30
    assert args.ensemble_normalize == "rank_zscore"
    assert args.model_quality_mode == "strict"
    assert args.gate_profile == "release"
    assert args.baseline_tickers == "QQQ,SPY,IXIC"
    assert args.baseline_pkl_map == ""
    assert args.check_baseline_gates is False
    assert args.fail_on_baseline_gate_fail is False
    assert args.skip_baseline_gates is False
    assert args.collect_all_diagnostics is False
    assert args.training_min_best_iteration == 5
    assert args.recency_half_life_days == 0
    assert args.recency_min_weight == 0.25
    assert args.strategy_signal_shift == 1
    assert args.year_warmup_days == 5
    assert args.rolling_mode == "continuous"
    assert args.rolling_warmup_days == 5
    assert args.rolling_ir_metric == "excess"
    assert args.skip_cold_start_diagnostics is False
    assert args.hedge_min_history_days == 252
    assert args.hedge_max_missing_ratio == 0.05


def test_release_parser_can_opt_into_label_benchmark_override():
    mod = _load_module()

    args = mod._build_parser().parse_args(["--config", "workflow.yaml", "--override_label_benchmark_pkl"])

    assert args.preserve_label_benchmark_pkl is False


def test_config_requires_hedge_feasibility_for_hedged_strategy_or_tickers():
    mod = _load_module()

    assert mod._config_requires_hedge_feasibility(
        {"port_analysis_config": {"strategy": {"class": "WeeklyHedgedBenchmarkAwareScoreWeightedStrategy"}}}
    )
    assert mod._config_requires_hedge_feasibility(
        {"port_analysis_config": {"strategy": {"kwargs": {"hedge_tickers_file": "hedges.txt"}}}}
    )
    assert not mod._config_requires_hedge_feasibility(
        {"port_analysis_config": {"strategy": {"class": "WeeklyBenchmarkAwareScoreWeightedStrategy"}}}
    )


def test_build_hedge_feasibility_cmd_uses_runtime_config_and_dates(tmp_path):
    mod = _load_module()
    args = mod._build_parser().parse_args(
        [
            "--config",
            "workflow.yaml",
            "--provider_uri",
            str(tmp_path / "provider"),
            "--raw_sfp_dir",
            str(tmp_path / "sfp"),
            "--start",
            "2024-01-01",
            "--end",
            "2026-04-30",
        ]
    )

    cmd = mod._build_hedge_feasibility_cmd(
        args,
        hedge_auditor_path=tmp_path / "audit_us_hedge_feasibility.py",
        train_config_path=tmp_path / "runtime.yaml",
        test_start="2022-01-03",
        test_end="2026-04-30",
    )

    assert cmd[:5] == [
        sys.executable,
        str(tmp_path / "audit_us_hedge_feasibility.py"),
        "--config",
        str(tmp_path / "runtime.yaml"),
        "--provider_uri",
    ]
    assert "--require_inverse" in cmd
    assert "--fail_on_no_hedge" in cmd
    assert cmd[-4:] == ["--start", "2024-01-01", "--end", "2026-04-30"]


def test_release_validation_command_includes_portfolio_quality_gates_by_default():
    mod = _load_module()
    args = mod._build_parser().parse_args(["--config", "workflow.yaml"])
    cmd = []

    mod._append_release_quality_gate_args(cmd, args)

    assert "--check_strategy_weighted_model_quality" in cmd
    assert "--fail_on_strategy_weighted_model_quality_fail" in cmd
    assert "--check_rebalance_interval_quality" in cmd
    assert "--fail_on_rebalance_interval_quality_fail" in cmd
    assert "--check_active_risk" in cmd
    assert "--fail_on_active_risk_fail" in cmd


def test_collect_all_diagnostics_keeps_portfolio_checks_without_fail_fast_flags():
    mod = _load_module()
    args = mod._build_parser().parse_args(["--config", "workflow.yaml", "--collect_all_diagnostics"])
    cmd = []

    mod._append_release_quality_gate_args(cmd, args)

    assert "--check_strategy_weighted_model_quality" in cmd
    assert "--check_rebalance_interval_quality" in cmd
    assert "--check_active_risk" in cmd
    assert "--fail_on_strategy_weighted_model_quality_fail" not in cmd
    assert "--fail_on_rebalance_interval_quality_fail" not in cmd
    assert "--fail_on_active_risk_fail" not in cmd


def test_release_validation_command_can_skip_portfolio_quality_gates():
    mod = _load_module()
    args = mod._build_parser().parse_args(
        [
            "--config",
            "workflow.yaml",
            "--skip_strategy_weighted_quality",
            "--skip_rebalance_interval_quality",
            "--skip_active_risk",
        ]
    )
    cmd = []

    mod._append_release_quality_gate_args(cmd, args)

    assert cmd == []


def test_release_profile_adds_strict_external_baseline_gates():
    mod = _load_module()
    args = mod._build_parser().parse_args(["--config", "workflow.yaml"])
    cmd = []

    mod._append_external_baseline_gate_args(cmd, args)

    assert cmd == [
        "--check_baseline_gates",
        "--baseline_tickers",
        "QQQ,SPY,IXIC",
        "--fail_on_baseline_gate_fail",
    ]


def test_research_profile_can_warn_on_external_baseline_gates():
    mod = _load_module()
    args = mod._build_parser().parse_args(
        [
            "--config",
            "workflow.yaml",
            "--gate_profile",
            "research",
            "--check_baseline_gates",
            "--baseline_tickers",
            "QQQ,SPY",
        ]
    )
    cmd = []

    mod._append_external_baseline_gate_args(cmd, args)

    assert cmd == ["--check_baseline_gates", "--baseline_tickers", "QQQ,SPY"]


def test_collect_all_diagnostics_keeps_external_baseline_check_without_fail_fast():
    mod = _load_module()
    args = mod._build_parser().parse_args(["--config", "workflow.yaml", "--collect_all_diagnostics"])
    cmd = []

    mod._append_external_baseline_gate_args(cmd, args)

    assert cmd == ["--check_baseline_gates", "--baseline_tickers", "QQQ,SPY,IXIC"]


def test_strategy_feasibility_preflight_rejects_impossible_core():
    mod = _load_module()

    rows = mod.strategy_feasibility_check_rows(
        {
            "topk": 40,
            "benchmark_tickers": ["QQQ"],
            "benchmark_topn": 1,
            "benchmark_core_weight": 0.50,
            "max_weight": 0.20,
            "max_holdings": 60,
        },
        "WeeklyBenchmarkAwareScoreWeightedStrategy",
    )
    by_name = {name: ok for name, ok, _ in rows}

    assert by_name["strategy_benchmark_core_capacity"] is False


def test_external_baseline_gate_args_include_pkl_map():
    mod = _load_module()
    args = mod._build_parser().parse_args(
        [
            "--config",
            "workflow.yaml",
            "--check_baseline_gates",
            "--baseline_tickers",
            "QQQ,SPY,ICIC",
            "--baseline_pkl_map",
            "IXIC=/tmp/bench_ixic.pkl",
        ]
    )
    cmd = []

    mod._append_external_baseline_gate_args(cmd, args)

    assert cmd == [
        "--check_baseline_gates",
        "--baseline_tickers",
        "QQQ,SPY,ICIC",
        "--baseline_pkl_map",
        "IXIC=/tmp/bench_ixic.pkl",
        "--fail_on_baseline_gate_fail",
    ]
