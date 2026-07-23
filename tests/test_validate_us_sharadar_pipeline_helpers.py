import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "validate_us_sharadar_pipeline.py"
    spec = importlib.util.spec_from_file_location("validate_us_sharadar_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_count_trade_days_between_excludes_boundaries():
    mod = _load_module()
    calendar = list(pd.bdate_range("2021-01-01", "2021-01-08"))

    assert mod._count_trade_days_between(calendar, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-08")) == 4


def test_warmup_start_for_window_steps_back_on_calendar():
    mod = _load_module()
    calendar = list(pd.bdate_range("2026-01-01", periods=10))

    assert mod._warmup_start_for_window(calendar, pd.Timestamp("2026-01-08"), 3) == pd.Timestamp("2026-01-05")
    assert mod._warmup_start_for_window(calendar, pd.Timestamp("2026-01-08"), 0) == pd.Timestamp("2026-01-08")


def test_run_backtest_eval_window_warms_up_and_slices_metrics(monkeypatch):
    mod = _load_module()
    calendar = list(pd.bdate_range("2026-01-01", periods=10))
    calls = []

    def fake_run_backtest(pred, strategy_cfg, start_time, end_time, benchmark, **kwargs):
        calls.append((pd.Timestamp(start_time), pd.Timestamp(end_time)))
        idx = pd.bdate_range(start_time, end_time)
        return pd.DataFrame(
            {
                "return": [0.0] * len(idx),
                "bench": [0.0] * len(idx),
                "cost": [0.0] * len(idx),
            },
            index=idx,
        )

    monkeypatch.setattr(mod, "_run_backtest", fake_run_backtest)

    report, warmup_start = mod._run_backtest_eval_window(
        pd.DataFrame(),
        {},
        pd.Timestamp("2026-01-08"),
        pd.Timestamp("2026-01-12"),
        pd.Series(dtype=float),
        calendar=calendar,
        warmup_days=3,
    )

    assert warmup_start == pd.Timestamp("2026-01-05")
    assert calls == [(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12"))]
    assert report.index.min() == pd.Timestamp("2026-01-08")


def test_collect_rolling_rows_can_gate_on_excess_ir():
    mod = _load_module()
    windows = [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05"), 3),
        (pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-08"), 3),
    ]
    metrics = [
        {"excess_ann_return": 0.01, "ir": -1.0, "excess_ir": 0.5, "mdd": -0.01, "avg_turnover": 0.01},
        {"excess_ann_return": -0.01, "ir": 1.0, "excess_ir": -0.5, "mdd": -0.01, "avg_turnover": 0.01},
    ]

    rows, passes, worst_excess, worst_ir, worst_mdd = mod._collect_rolling_rows(
        windows,
        metrics_by_window=lambda _s, _e, i: metrics[0 if i == 3 and _s == pd.Timestamp("2026-01-01") else 1],
        ir_metric="excess",
        min_excess_ann=0.0,
        min_ir=0.0,
        max_mdd_abs=0.35,
        max_turnover=0.10,
    )

    assert passes == 1
    assert rows[0][1]["status"] == "PASS"
    assert rows[1][1]["status"] == "FAIL"
    assert worst_excess == -0.01
    assert worst_ir == -0.5
    assert worst_mdd == 0.01


def test_external_baseline_gates_pass_for_qqq_excess_profile():
    mod = _load_module()
    idx = pd.bdate_range("2024-01-02", "2025-12-31")
    report = pd.DataFrame({"return": 0.0008, "bench": 0.0, "cost": 0.0}, index=idx)
    stress_report = pd.DataFrame({"return": 0.00065, "bench": 0.0, "cost": 0.0}, index=idx)
    baseline = pd.Series(0.0004, index=idx, name="QQQ")
    yearly_reports = [
        ("2024", report.loc["2024"]),
        ("2025", report.loc["2025"]),
    ]
    windows = [
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), 252),
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), 252),
    ]
    args = argparse.Namespace(
        stress_cost_mult=3.0,
        stress_deal_price="open",
        baseline_max_missing_ratio=0.01,
        baseline_min_full_excess_ann=0.05,
        baseline_min_stress_excess_ann=0.03,
        baseline_max_full_mdd_abs=0.10,
        baseline_max_mdd_gap=0.05,
        baseline_min_positive_years=2,
        baseline_min_yearly_beat_rate=1.0,
        baseline_min_worst_year_excess_ann=0.05,
        baseline_min_year_days=200,
        baseline_min_rolling_excess_ann=0.05,
        baseline_min_rolling_pass_rate=1.0,
        baseline_min_worst_rolling_excess_ann=0.05,
        baseline_min_latest_rolling_excess_ann=0.05,
    )

    comparison_rows, rolling_rows, checks = mod._evaluate_external_baseline_gates(
        full_report=report,
        stress_report=stress_report,
        yearly_reports=yearly_reports,
        rolling_windows=windows,
        baseline_returns={"QQQ": baseline},
        args=args,
    )

    assert comparison_rows
    assert len(rolling_rows) == 2
    assert all(ok for _, ok, _ in checks)
    assert any(name == "baseline_QQQ_full_excess_ann" for name, _, _ in checks)


def test_external_baseline_gates_reject_underperforming_full_period():
    mod = _load_module()
    idx = pd.bdate_range("2024-01-02", "2025-12-31")
    report = pd.DataFrame({"return": 0.0003, "bench": 0.0, "cost": 0.0}, index=idx)
    stress_report = pd.DataFrame({"return": 0.0003, "bench": 0.0, "cost": 0.0}, index=idx)
    baseline = pd.Series(0.0004, index=idx, name="QQQ")
    args = argparse.Namespace(
        stress_cost_mult=3.0,
        stress_deal_price="open",
        baseline_max_missing_ratio=0.01,
        baseline_min_full_excess_ann=0.00,
        baseline_min_stress_excess_ann=-0.10,
        baseline_max_full_mdd_abs=0.10,
        baseline_max_mdd_gap=0.05,
        baseline_min_positive_years=0,
        baseline_min_yearly_beat_rate=0.0,
        baseline_min_worst_year_excess_ann=-1.0,
        baseline_min_year_days=200,
        baseline_min_rolling_excess_ann=-1.0,
        baseline_min_rolling_pass_rate=0.0,
        baseline_min_worst_rolling_excess_ann=-1.0,
        baseline_min_latest_rolling_excess_ann=-1.0,
    )

    _, _, checks = mod._evaluate_external_baseline_gates(
        full_report=report,
        stress_report=stress_report,
        yearly_reports=[],
        rolling_windows=[],
        baseline_returns={"QQQ": baseline},
        args=args,
    )

    by_name = {name: ok for name, ok, _ in checks}
    assert by_name["baseline_QQQ_full_excess_ann"] is False


def test_training_diagnostics_skip_non_iterative_feature_score_model(tmp_path):
    mod = _load_module()

    for model_class in (
        "FeatureWeightedScoreModel",
        "ICSelectedScoreModel",
        "RegimeSleeveScoreModel",
        "StackedSignalScoreModel",
    ):
        rows = mod._training_diagnostic_rows(
            tmp_path / "missing_manifest.json",
            cfg={"task": {"model": {"class": model_class}}},
            cfg_path=tmp_path / "config.yaml",
            min_best_iteration=5,
        )

        assert rows == [
            (
                "training_non_iterative_model",
                True,
                f"model={model_class}; best-iteration diagnostics not applicable",
            )
        ]


def test_external_baseline_gates_reject_latest_rolling_underperformance():
    mod = _load_module()
    idx = pd.bdate_range("2024-01-02", periods=12)
    good = pd.DataFrame({"return": 0.0010, "bench": 0.0, "cost": 0.0}, index=idx[:6])
    bad = pd.DataFrame({"return": -0.0010, "bench": 0.0, "cost": 0.0}, index=idx[6:])
    report = pd.concat([good, bad])
    stress_report = report.copy()
    baseline = pd.Series(0.0, index=idx, name="QQQ")
    windows = [
        (idx[0], idx[5], 6),
        (idx[6], idx[-1], 6),
    ]
    args = argparse.Namespace(
        stress_cost_mult=3.0,
        stress_deal_price="open",
        baseline_max_missing_ratio=0.01,
        baseline_min_full_excess_ann=-1.0,
        baseline_min_stress_excess_ann=-1.0,
        baseline_max_full_mdd_abs=1.0,
        baseline_max_mdd_gap=1.0,
        baseline_min_positive_years=0,
        baseline_min_yearly_beat_rate=0.0,
        baseline_min_worst_year_excess_ann=-1.0,
        baseline_min_year_days=200,
        baseline_min_rolling_excess_ann=-1.0,
        baseline_min_rolling_pass_rate=0.0,
        baseline_min_worst_rolling_excess_ann=-1.0,
        baseline_min_latest_rolling_excess_ann=0.0,
    )

    _, _, checks = mod._evaluate_external_baseline_gates(
        full_report=report,
        stress_report=stress_report,
        yearly_reports=[],
        rolling_windows=windows,
        baseline_returns={"QQQ": baseline},
        args=args,
    )

    by_name = {name: ok for name, ok, _ in checks}
    assert by_name["baseline_QQQ_latest_rolling_excess_ann"] is False


def test_benchmark_ticker_alias_maps_icic_to_ixic():
    mod = _load_module()

    assert mod._coerce_ticker_list("QQQ,SPY,ICIC,^IXIC") == ["QQQ", "SPY", "IXIC"]


def test_load_external_baseline_returns_uses_pkl_map(tmp_path):
    mod = _load_module()
    idx = pd.bdate_range("2026-01-01", periods=4)
    path = tmp_path / "bench_ixic.pkl"
    pd.Series([0.01, -0.01, 0.02, 0.0], index=idx, name="IXIC").to_pickle(path)

    returns, source = mod._load_external_baseline_returns(
        "ICIC",
        idx[0],
        idx[-1],
        pkl_map={"IXIC": path},
    )

    assert source.startswith("pkl:")
    assert returns.name == "IXIC"
    assert returns.tolist() == [0.01, -0.01, 0.02, 0.0]


def test_ensemble_source_dates_reads_gate_csv(tmp_path):
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2026-01-01", periods=3), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": range(len(idx))}, index=idx)
    gate = pd.DataFrame(
        {
            "datetime": pd.bdate_range("2026-01-01", periods=3),
            "fallback": [False, True, False],
        }
    )
    gate_path = tmp_path / "gate.csv"
    gate.to_csv(gate_path, index=False)

    sources, rows = mod._load_ensemble_source_dates(
        pred,
        gate_csv_path=gate_path,
        primary_name="h20",
        defensive_name="h60",
    )

    assert all(ok for _, ok, _ in rows)
    assert [dt.strftime("%Y-%m-%d") for dt in sources["h60"]] == ["2026-01-02"]
    assert len(sources["h20"]) == 2


def test_active_risk_core_gates_use_etf_core_metrics():
    mod = _load_module()
    full = {
        "days": 20,
        "mean_core_weight_abs_gap": 0.01,
        "mean_core_weight_held": 0.25,
        "mean_stock_overlay_weight": 0.75,
        "mean_max_single_stock_weight": 0.04,
        "mean_stock_names": 30,
        "mean_max_sector_stock_weight": 0.20,
    }
    recent = dict(full)
    args = argparse.Namespace(
        active_risk_min_rebalances=20,
        active_risk_recent_rebalances=13,
        active_risk_max_mean_core_weight_gap=0.08,
        active_risk_max_recent_core_weight_gap=0.10,
        active_risk_min_mean_benchmark_coverage=0.20,
        active_risk_min_recent_benchmark_coverage=0.20,
        active_risk_max_mean_stock_overlay_weight=0.85,
        active_risk_max_mean_single_stock_weight=0.08,
        active_risk_max_mean_portfolio_names=80,
        active_risk_max_mean_sector_active_weight=0.35,
    )

    rows = mod._active_risk_gate_rows(full, recent, args)
    by_name = {name: ok for name, ok, _ in rows}

    assert by_name["active_risk_mean_core_weight_gap"] is True
    assert by_name["active_risk_mean_max_single_stock_weight"] is True
    assert "active_risk_mean_max_abs_active_weight" not in by_name


def test_strategy_feasibility_rejects_impossible_single_etf_core():
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


def test_strategy_feasibility_allows_benchmark_specific_cap():
    mod = _load_module()

    rows = mod.strategy_feasibility_check_rows(
        {
            "topk": 40,
            "benchmark_tickers": ["QQQ"],
            "benchmark_topn": 1,
            "benchmark_core_weight": 0.50,
            "max_weight": 0.20,
            "benchmark_max_weight": 1.0,
            "max_holdings": 60,
        },
        "WeeklyBenchmarkAwareScoreWeightedStrategy",
    )

    assert all(ok for _, ok, _ in rows)


def test_release_decision_requires_all_release_checks():
    mod = _load_module()

    decision = mod.release_decision(
        {"model_quality": True, "rolling": False, "external_baseline_gates": None},
        required_checks=["model_quality", "rolling", "external_baseline_gates"],
    )

    assert decision["release_ready"] is False
    assert decision["missing_or_failed"] == ["rolling", "external_baseline_gates"]


def test_multiple_testing_ir_haircut_penalizes_more_trials():
    mod = _load_module()

    one = mod._multiple_testing_ir_haircut(1.0, 252, 1)
    many = mod._multiple_testing_ir_haircut(1.0, 252, 100)

    assert one["haircut_ir"] == 1.0
    assert many["haircut_ir"] < one["haircut_ir"]


def test_required_embargo_defaults_to_label_horizon():
    mod = _load_module()

    assert mod._required_embargo_days(argparse.Namespace(embargo_days=None), 10) == 10
    assert mod._required_embargo_days(argparse.Namespace(embargo_days=3), 10) == 3


def test_sample_active_instruments_filters_tail_population():
    mod = _load_module()
    spans = {
        "AAPL": [(pd.Timestamp("2020-01-01"), pd.Timestamp("2026-04-30"))],
        "OLD": [(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"))],
    }

    assert mod._sample_active_instruments(spans, pd.Timestamp("2026-04-30"), 10, 0) == ["AAPL"]


def test_provenance_checks_require_lagged_sf3a_and_sf1(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_sf3a_features.json").write_text(
        json.dumps({"full_rebuild": True, "availability_lag_days": 45}),
        encoding="utf-8",
    )
    (meta / "sharadar_sf1_pit.json").write_text(
        json.dumps({"date_col": "datekey", "date_offset_days": 0, "dump_to_qlib": True}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=True,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=True,
    )
    cfg = {"extra_fields": ["$inst13f_totalvalue_63d_chg"], "pit_fields": ["assets"]}

    rows = mod._provenance_check_rows(str(tmp_path), cfg, args)
    assert rows
    assert all(ok for _, ok, _ in rows)


def test_provenance_checks_reject_datekey_offset(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_sf1_pit.json").write_text(
        json.dumps({"date_col": "datekey", "date_offset_days": 45, "dump_to_qlib": True}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=True,
    )

    rows = mod._provenance_check_rows(str(tmp_path), {"pit_fields": ["assets"]}, args)
    assert ("sf1_pit_no_extra_datekey_offset", False, "date_offset_days=45") in rows


def test_provenance_checks_detect_pit_only_in_extra_fields(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_sf1_pit.json").write_text(
        json.dumps({"date_col": "datekey", "date_offset_days": 0, "dump_to_qlib": True}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=True,
    )

    rows = mod._provenance_check_rows(
        str(tmp_path),
        {"extra_fields": ["P($$netinc_q) / (P($$equity_q) + 1e-12)"], "pit_fields": []},
        args,
    )

    assert ("sf1_pit_provenance_present", True, str(meta / "sharadar_sf1_pit.json")) in rows


def test_provenance_checks_detect_prebuilt_sf1_ratio_features(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_sf1_ratio_features.json").write_text(
        json.dumps(
            {
                "date_col": "datekey",
                "date_offset_days": 0,
                "dump_to_qlib": True,
                "ratio_fields": sorted(mod.SF1_RATIO_FIELDS),
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=True,
    )

    rows = mod._provenance_check_rows(str(tmp_path), {"extra_fields": ["$roe_q"], "pit_fields": []}, args)

    assert ("sf1_ratio_provenance_present", True, str(meta / "sharadar_sf1_ratio_features.json")) in rows
    assert ("sf1_ratio_fields_complete", True, f"fields={len(mod.SF1_RATIO_FIELDS)}/{len(mod.SF1_RATIO_FIELDS)}") in rows


def test_provenance_checks_require_model_features_and_etf_sources(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_model_features.json").write_text(
        json.dumps(
            {
                "dump_to_qlib": True,
                "prepared_csv_files": 10,
                "feature_bins_written": 20,
                "feature_fields": [
                    "risk_ret_20d",
                    "mkt_spy_ret_63d",
                    "mkt_breadth_ret63_pos_lag1",
                    "mkt_dispersion_ret63_lag1",
                ],
                "market_etf_close_counts": {"SPY": 100},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=False,
        require_model_feature_provenance=True,
        allow_static_metadata_features=False,
    )

    rows = mod._provenance_check_rows(
        str(tmp_path),
        {
            "extra_fields": [
                "$risk_ret_20d",
                "$mkt_spy_ret_63d",
                "$mkt_breadth_ret63_pos_lag1",
                "$mkt_dispersion_ret63_lag1",
            ]
        },
        args,
    )

    assert ("model_feature_provenance_present", True, str(meta / "sharadar_model_features.json")) in rows
    assert ("model_feature_fields_complete", True, "required=4, missing=[]") in rows
    assert ("model_feature_market_etf_sources", True, "required_etfs=['SPY'], missing_or_empty=[]") in rows


def test_provenance_checks_require_lagged_fmp_event_features(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "fmp_event_features.json").write_text(
        json.dumps(
            {
                "dump_to_qlib": True,
                "availability_lag_days": 1,
                "feature_columns": [
                    "fmp_earn_count_20d_sum",
                    "fmp_eps_surprise_pct_latest",
                    "fmp_alpha_event_composite",
                ],
                "directional_feature_version": 2,
                "directional_feature_columns": ["fmp_alpha_event_composite"],
                "tickers": 100,
                "feature_bins_written": 200,
                "excluded_datasets": [
                    "analyst_estimates_annual",
                    "analyst_estimates_quarter",
                    "price_target_summary",
                    "price_target_consensus",
                ],
                "dataset_classification": {
                    "pit_safe_event_history": ["earnings", "grades"],
                    "excluded_current_only_or_unproven_asof": [
                        "analyst_estimates_annual",
                        "analyst_estimates_quarter",
                        "price_target_summary",
                        "price_target_consensus",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=False,
        require_model_feature_provenance=False,
        allow_static_metadata_features=False,
        require_fmp_feature_provenance=True,
        fmp_feature_min_availability_lag_days=1,
    )

    rows = mod._provenance_check_rows(
        str(tmp_path),
        {"extra_fields": ["$fmp_earn_count_20d_sum", "$fmp_eps_surprise_pct_latest", "$fmp_alpha_event_composite"]},
        args,
    )

    assert ("fmp_feature_provenance_present", True, str(meta / "fmp_event_features.json")) in rows
    assert ("fmp_feature_availability_lag", True, "1 >= 1 calendar_days") in rows
    assert ("fmp_feature_fields_complete", True, "required=3, missing=[]") in rows
    assert ("fmp_feature_current_only_inputs_excluded", True, "excluded=['analyst_estimates_annual', 'analyst_estimates_quarter', 'price_target_consensus', 'price_target_summary']") in rows
    assert ("fmp_directional_feature_version", True, "2 >= 2") in rows
    assert ("fmp_directional_fields_listed", True, "required=1, missing=[]") in rows
    assert (
        "fmp_dataset_classification_present",
        True,
        "pit_safe=['earnings', 'grades'], excluded=['analyst_estimates_annual', 'analyst_estimates_quarter', 'price_target_consensus', 'price_target_summary']",
    ) in rows


def test_provenance_checks_reject_current_static_metadata_by_default(tmp_path):
    mod = _load_module()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sharadar_model_features.json").write_text(
        json.dumps(
            {
                "dump_to_qlib": True,
                "prepared_csv_files": 10,
                "feature_bins_written": 20,
                "feature_fields": ["meta_scalemarketcap"],
                "market_etf_close_counts": {},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        require_sf3a_provenance=False,
        sf3a_min_availability_lag_days=45,
        require_sf1_pit_provenance=False,
        require_model_feature_provenance=True,
        allow_static_metadata_features=False,
    )

    rows = mod._provenance_check_rows(str(tmp_path), {"extra_fields": ["$meta_scalemarketcap"]}, args)

    assert (
        "model_feature_static_metadata_allowed",
        False,
        "uses_meta=True, allow_static_metadata_features=False",
    ) in rows


def test_manifest_checks_validate_all_task_embargoes(monkeypatch, tmp_path):
    mod = _load_module()
    manifest = {
        "embargo_days": 2,
        "tasks": [
            {
                "train": ["2021-01-01", "2021-01-08"],
                "valid": ["2021-01-13", "2021-01-15"],
                "test": ["2021-01-20", "2021-01-29"],
            }
        ],
    }
    fp = tmp_path / "pred.manifest.json"
    fp.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(mod, "_get_calendar_span", lambda start, end: list(pd.bdate_range(start, end)))

    rows = mod._manifest_check_rows(
        fp,
        args=argparse.Namespace(embargo_days=None),
        label_horizon=2,
        pred=None,
    )
    assert rows
    assert all(ok for _, ok, _ in rows)


def test_manifest_checks_require_predictions_for_each_test_window(monkeypatch, tmp_path):
    mod = _load_module()
    manifest = {
        "embargo_days": 2,
        "tasks": [
            {
                "train": ["2021-01-01", "2021-01-08"],
                "valid": ["2021-01-13", "2021-01-15"],
                "test": ["2021-01-20", "2021-01-29"],
            },
            {
                "train": ["2021-01-01", "2021-01-22"],
                "valid": ["2021-01-27", "2021-01-29"],
                "test": ["2021-02-03", "2021-02-12"],
            },
        ],
    }
    fp = tmp_path / "pred.manifest.json"
    fp.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(mod, "_get_calendar_span", lambda start, end: list(pd.bdate_range(start, end)))
    pred = pd.DataFrame(
        {"score": [0.1]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-01-20"), "AAPL")],
            names=["datetime", "instrument"],
        ),
    )

    rows = mod._manifest_check_rows(
        fp,
        args=argparse.Namespace(embargo_days=None),
        label_horizon=2,
        pred=pred,
    )

    assert ("walkforward_manifest_tests_have_predictions", False, "empty_test_windows=1") in rows


def test_training_diagnostics_detect_weak_best_iteration(tmp_path):
    mod = _load_module()
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "1"
    run = exp / "run_a"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    (exp / "meta.yaml").write_text("name: exp_a\n", encoding="utf-8")
    (metrics / "l2.valid").write_text(
        "1 0.100 0\n2 0.110 1\n3 0.120 2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"experiment": "exp_a", "used_runs": [{"run_id": "run_a"}]}),
        encoding="utf-8",
    )
    cfg = {"qlib_init": {"exp_manager": {"kwargs": {"uri": str(mlruns)}}}}

    rows = mod._training_diagnostic_rows(
        manifest,
        cfg=cfg,
        cfg_path=tmp_path / "config.yaml",
        min_best_iteration=5,
    )

    assert ("training_metrics_present", True, "metrics=1/1, missing=[]") in rows
    assert rows[-1][0] == "training_best_iteration_min"
    assert rows[-1][1] is False
    assert "min=1" in rows[-1][2]


def test_training_diagnostics_reads_sqlite_mlflow_metrics(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mod = _load_module()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp_id = client.create_experiment("exp_sqlite")
    with mlflow.start_run(experiment_id=exp_id, run_name="fold", nested=False) as run:
        run_id = run.info.run_id
        mlflow.log_metric("l2.valid", 0.30, step=0)
        mlflow.log_metric("l2.valid", 0.20, step=1)
        mlflow.log_metric("l2.valid", 0.25, step=2)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"experiment": "exp_sqlite", "used_runs": [{"run_id": run_id}]}),
        encoding="utf-8",
    )
    cfg = {"qlib_init": {"exp_manager": {"kwargs": {"uri": uri}}}}

    rows = mod._training_diagnostic_rows(
        manifest,
        cfg=cfg,
        cfg_path=tmp_path / "config.yaml",
        min_best_iteration=2,
    )

    assert ("training_metrics_present", True, "metrics=1/1, missing=[]") in rows
    assert rows[-1][0] == "training_best_iteration_min"
    assert rows[-1][1] is True
    assert "min=2" in rows[-1][2]


def test_training_diagnostics_reads_ensemble_member_runs(tmp_path):
    mod = _load_module()
    mlruns = tmp_path / "mlruns"
    exp = mlruns / "1"
    run = exp / "run_ranker"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    (exp / "meta.yaml").write_text("name: ranker_exp\n", encoding="utf-8")
    (metrics / "ndcg.valid").write_text(
        "1 0.100 0\n2 0.120 1\n3 0.115 2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "ensemble_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ensemble": True,
                "used_runs": [
                    {
                        "member": "v8",
                        "run_id": "run_stack",
                        "experiment": "stack_exp",
                        "mlruns_uri": str(mlruns),
                        "model_class": "StackedSignalScoreModel",
                    },
                    {
                        "member": "v9",
                        "run_id": "run_ranker",
                        "experiment": "ranker_exp",
                        "mlruns_uri": str(mlruns),
                        "model_class": "LGBRankerModel",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = mod._training_diagnostic_rows(
        manifest,
        cfg={"task": {"model": {"class": "StackedSignalScoreModel"}}},
        cfg_path=tmp_path / "config.yaml",
        min_best_iteration=2,
    )

    assert ("training_ensemble_non_iterative_runs", True, "skipped=1") in rows
    assert ("training_metrics_present", True, "metrics=1, skipped_non_iterative=1, total=2, missing=[]") in rows
    assert ("training_best_iteration_min", True, "min=2, median=2.0, threshold=2, weak=[]") in rows


def test_model_quality_metrics_detect_recent_negative_ic():
    mod = _load_module()
    rows = []
    for day in pd.bdate_range("2026-01-01", periods=8):
        for i in range(10):
            score = float(i)
            # First half has correct ordering, second half is inverted.
            label = score if day < pd.Timestamp("2026-01-07") else -score
            rows.append((day, f"S{i}", score, label))
    idx = pd.MultiIndex.from_tuples([(d, s) for d, s, _, _ in rows], names=["datetime", "instrument"])
    frame = pd.DataFrame({"score": [r[2] for r in rows], "label": [r[3] for r in rows]}, index=idx)

    full = mod._model_quality_metrics(frame, topk=3, min_daily_count=5, recent_days=None)
    recent = mod._model_quality_metrics(frame, topk=3, min_daily_count=5, recent_days=4)

    assert full["days"] == 8
    assert abs(full["mean_ic"]) < 1e-12
    assert abs(recent["mean_ic"] + 1.0) < 1e-12
    assert recent["topq_minus_bottomq"] < 0


def test_apply_strategy_score_controls_uses_feature_percentile_filter():
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-05")], ["A", "B", "C", "D"]],
        names=["datetime", "instrument"],
    )
    score = pd.Series([4.0, 3.0, 2.0, 1.0], index=idx)
    feat = pd.DataFrame({"$log_marketcap_q": [10.0, 20.0, 30.0, 40.0]}, index=idx)

    adjusted = mod._apply_strategy_score_controls(
        score,
        feat,
        feature_score_weights={},
        feature_min_percentiles={"$log_marketcap_q": 0.75},
    )

    assert adjusted.sort_values(ascending=False).head(2).index.get_level_values("instrument").tolist() == ["C", "D"]


def test_filter_model_frame_by_weekday_keeps_rebalance_days():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-05"), "AAPL"),
            (pd.Timestamp("2026-01-06"), "AAPL"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"score": [1.0, 2.0], "label": [0.1, 0.2]}, index=idx)

    out = mod._filter_model_frame_by_weekday(frame, 0)

    assert list(out.index.get_level_values("datetime")) == [pd.Timestamp("2026-01-05")]


def test_rebalance_signal_dates_use_previous_trading_day():
    mod = _load_module()
    cal = list(pd.to_datetime(["2025-12-29", "2025-12-30", "2026-01-02", "2026-01-05", "2026-01-06"]))

    assert mod._rebalance_trade_dates(cal, 0) == [pd.Timestamp("2025-12-29"), pd.Timestamp("2026-01-05")]
    assert mod._rebalance_signal_dates(
        cal,
        0,
        start=pd.Timestamp("2026-01-02"),
        end=pd.Timestamp("2026-01-06"),
    ) == [pd.Timestamp("2026-01-02")]


def test_model_quality_gate_rows_fail_recent_inversion():
    mod = _load_module()
    args = argparse.Namespace(
        model_quality_min_days=4,
        model_quality_min_recent_days=4,
        model_quality_min_full_mean_ic=0.0,
        model_quality_min_full_topq_spread=0.0,
        model_quality_min_recent_mean_ic=0.0,
        model_quality_min_recent_pos_ic_rate=0.50,
        model_quality_min_recent_topk_mean_label=0.0,
        model_quality_min_recent_topq_spread=0.0,
    )
    full = {"days": 8, "mean_ic": 0.01, "topq_minus_bottomq": 0.02}
    recent = {
        "days": 4,
        "mean_ic": -0.10,
        "pos_ic_rate": 0.25,
        "topk_mean_label": -0.01,
        "topq_minus_bottomq": -0.02,
    }

    rows = mod._model_quality_gate_rows(full, recent, args)

    assert ("model_quality_recent_mean_ic", False, "-0.1000 >= 0.0000") in rows
    assert ("model_quality_recent_topq_spread", False, "-0.0200 >= 0.0000") in rows


def test_model_quality_gate_rows_support_custom_prefix():
    mod = _load_module()
    args = argparse.Namespace(
        model_quality_min_days=4,
        model_quality_min_recent_days=4,
        model_quality_min_full_mean_ic=0.0,
        model_quality_min_full_topq_spread=0.0,
        model_quality_min_recent_mean_ic=0.0,
        model_quality_min_recent_pos_ic_rate=0.50,
        model_quality_min_recent_topk_mean_label=0.0,
        model_quality_min_recent_topq_spread=0.0,
    )

    rows = mod._model_quality_gate_rows(
        {"days": 1, "mean_ic": -0.1, "topq_minus_bottomq": -0.1},
        {"days": 1, "mean_ic": -0.1, "pos_ic_rate": 0.0, "topk_mean_label": -0.1, "topq_minus_bottomq": -0.1},
        args,
        prefix="rebalance_model_quality",
    )

    assert rows[0][0] == "rebalance_model_quality_full_days"


def test_model_quality_gate_rows_include_yearly_instability():
    mod = _load_module()
    args = argparse.Namespace(
        model_quality_min_days=4,
        model_quality_min_recent_days=4,
        model_quality_min_full_mean_ic=0.0,
        model_quality_min_full_topq_spread=0.0,
        model_quality_min_recent_mean_ic=0.0,
        model_quality_min_recent_pos_ic_rate=0.50,
        model_quality_min_recent_topk_mean_label=0.0,
        model_quality_min_recent_topq_spread=0.0,
        model_quality_min_year_days=4,
        model_quality_min_positive_years=2,
        model_quality_min_worst_year_mean_ic=0.0,
        model_quality_min_worst_year_topk_mean_label=0.0,
        model_quality_min_worst_year_topq_spread=0.0,
    )
    full = {"days": 8, "mean_ic": 0.01, "topq_minus_bottomq": 0.02}
    recent = {"days": 4, "mean_ic": 0.01, "pos_ic_rate": 0.75, "topk_mean_label": 0.01, "topq_minus_bottomq": 0.02}
    yearly = [
        ("2024", {"days": 4, "mean_ic": 0.02, "topk_mean_label": 0.01, "topq_minus_bottomq": 0.01}),
        ("2025", {"days": 4, "mean_ic": -0.03, "topk_mean_label": -0.02, "topq_minus_bottomq": -0.01}),
    ]

    rows = mod._model_quality_gate_rows(full, recent, args, yearly=yearly)

    assert ("model_quality_positive_ic_years", False, "1 >= 2 (eligible_years=2, min_year_days=4, skipped_short_years=0)") in rows
    assert ("model_quality_worst_year_mean_ic", False, "-0.0300 >= 0.0000") in rows
    assert ("model_quality_worst_year_topk_label", False, "-0.0200 >= 0.0000") in rows


def test_model_quality_positive_year_requirement_caps_to_eligible_years():
    mod = _load_module()
    args = argparse.Namespace(
        model_quality_min_days=4,
        model_quality_min_recent_days=4,
        model_quality_min_full_mean_ic=0.0,
        model_quality_min_full_topq_spread=0.0,
        model_quality_min_recent_mean_ic=0.0,
        model_quality_min_recent_pos_ic_rate=0.50,
        model_quality_min_recent_topk_mean_label=0.0,
        model_quality_min_recent_topq_spread=0.0,
        model_quality_min_year_days=4,
        model_quality_min_positive_years=3,
        model_quality_min_worst_year_mean_ic=-1.0,
        model_quality_min_worst_year_topk_mean_label=-1.0,
        model_quality_min_worst_year_topq_spread=-1.0,
    )
    full = {"days": 8, "mean_ic": 0.01, "topq_minus_bottomq": 0.02}
    recent = {"days": 4, "mean_ic": 0.01, "pos_ic_rate": 0.75, "topk_mean_label": 0.01, "topq_minus_bottomq": 0.02}
    yearly = [
        ("2024", {"days": 4, "mean_ic": 0.02, "topk_mean_label": 0.01, "topq_minus_bottomq": 0.01}),
        ("2025", {"days": 4, "mean_ic": 0.03, "topk_mean_label": 0.02, "topq_minus_bottomq": 0.01}),
        ("2026", {"days": 2, "mean_ic": -0.03, "topk_mean_label": -0.02, "topq_minus_bottomq": -0.01}),
    ]

    rows = mod._model_quality_gate_rows(full, recent, args, yearly=yearly)

    assert (
        "model_quality_positive_ic_years",
        True,
        "2 >= 2 (eligible_years=2, min_year_days=4, skipped_short_years=1, configured_min_positive_years=3)",
    ) in rows


def test_robustness_positive_year_requirement_caps_to_eligible_years():
    mod = _load_module()

    rows = mod._evaluate_robustness_gates(
        {"excess_ann_return": 0.06, "ir": 1.0, "mdd": -0.10, "avg_turnover": 0.04},
        {"excess_ann_return": 0.03},
        [
            ("2024", {"n_days": 252, "excess_ann_return": 0.04}),
            ("2025", {"n_days": 252, "excess_ann_return": 0.03}),
            ("2026", {"n_days": 80, "excess_ann_return": -0.01}),
        ],
        min_full_excess_ann=0.05,
        min_full_ir=0.35,
        max_full_mdd_abs=0.40,
        min_stress_excess_ann=0.02,
        max_turnover=0.20,
        min_positive_excess_years=3,
        min_worst_year_excess_ann=-0.25,
        min_year_days=200,
    )

    assert (
        "positive_excess_years",
        True,
        "2 >= 2 (eligible_years=2, min_year_days=200, skipped_short_years=1, configured_min_positive_years=3)",
    ) in rows


def test_active_risk_gate_rows_detect_excessive_benchmark_drift():
    mod = _load_module()
    args = argparse.Namespace(
        active_risk_min_rebalances=4,
        active_risk_recent_rebalances=2,
        active_risk_max_mean_active_share=0.85,
        active_risk_max_recent_mean_active_share=0.85,
        active_risk_max_mean_abs_active_weight=0.08,
        active_risk_min_mean_benchmark_coverage=0.30,
        active_risk_min_recent_benchmark_coverage=0.30,
        active_risk_max_mean_sector_active_weight=0.35,
        active_risk_max_mean_portfolio_names=160,
    )
    full = {
        "days": 4,
        "mean_active_share": 0.92,
        "mean_max_abs_active_weight": 0.10,
        "mean_benchmark_weight_held": 0.20,
        "mean_max_abs_sector_active_weight": 0.40,
        "mean_portfolio_names": 220,
    }
    recent = {"days": 2, "mean_active_share": 0.93, "mean_benchmark_weight_held": 0.15}

    rows = mod._active_risk_gate_rows(full, recent, args)

    assert ("active_risk_mean_active_share", False, "0.9200 <= 0.8500") in rows
    assert ("active_risk_mean_benchmark_coverage", False, "0.2000 >= 0.3000") in rows
    assert ("active_risk_mean_portfolio_names", False, "220.0000 <= 160.0000") in rows


def test_strategy_portfolio_weights_keeps_benchmark_core_independent_of_alpha_pool():
    mod = _load_module()
    scores = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    marketcap = pd.Series({"A": 10.0, "B": 1.0, "C": 100.0, "D": 1000.0})

    weights = mod._strategy_portfolio_weights_for_day(
        scores,
        strategy_cfg={
            "topk": 2,
            "benchmark_topn": 2,
            "liquidity_buffer": 1,
            "benchmark_core_weight": 0.50,
            "weighting": "rank",
        },
        strategy_class="WeeklyBenchmarkAwareScoreWeightedStrategy",
        topk=2,
        marketcap=marketcap,
    )

    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert {"C", "D"}.issubset(set(weights.index))


def test_strategy_portfolio_weights_uses_explicit_benchmark_tickers_without_marketcap():
    mod = _load_module()
    scores = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0})

    weights = mod._strategy_portfolio_weights_for_day(
        scores,
        strategy_cfg={
            "topk": 2,
            "benchmark_tickers": ["SPY", "QQQ"],
            "benchmark_core_weight": 0.80,
            "liquidity_buffer": 1,
            "weighting": "equal",
        },
        strategy_class="WeeklyBenchmarkAwareScoreWeightedStrategy",
        topk=2,
    )

    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert abs(float(weights.loc["SPY"]) - 0.40) < 1e-12
    assert abs(float(weights.loc["QQQ"]) - 0.40) < 1e-12
    assert abs(float(weights.loc["A"]) - 0.10) < 1e-12
    assert abs(float(weights.loc["B"]) - 0.10) < 1e-12


def test_strategy_benchmark_tickers_dedupes_inline_and_file_values(tmp_path):
    mod = _load_module()
    tickers = tmp_path / "bench.txt"
    tickers.write_text("qqq\nIWM\nQQQ\n", encoding="utf-8")

    out = mod._strategy_benchmark_tickers(
        {
            "benchmark_tickers": "SPY, QQQ",
            "benchmark_tickers_file": str(tickers),
        }
    )

    assert out == ["SPY", "QQQ", "IWM"]


def test_strategy_hedge_tickers_dedupes_inline_and_file_values(tmp_path):
    mod = _load_module()
    tickers = tmp_path / "hedges.txt"
    tickers.write_text("psq\nRWM\nPSQ\n", encoding="utf-8")

    out = mod._strategy_hedge_tickers(
        {
            "hedge_tickers": "SH, PSQ",
            "hedge_tickers_file": str(tickers),
        }
    )

    assert out == ["SH", "PSQ", "RWM"]


def test_strategy_portfolio_weights_applies_hedge_overlay():
    mod = _load_module()
    scores = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0})
    benchmark_weights = pd.Series({"A": 0.50, "B": 0.50})

    weights = mod._strategy_portfolio_weights_for_day(
        scores,
        strategy_cfg={
            "topk": 2,
            "benchmark_core_weight": 0.0,
            "benchmark_topn": 2,
            "liquidity_buffer": 1,
            "hedge_tickers": ["SH", "PSQ"],
            "hedge_weight": 0.25,
        },
        strategy_class="WeeklyHedgedBenchmarkAwareScoreWeightedStrategy",
        topk=2,
        benchmark_weights=benchmark_weights,
        available_hedges=["SH"],
    )

    assert round(float(weights.sum()), 8) == 1.0
    assert round(float(weights.loc["SH"]), 8) == 0.25
    assert round(float(weights.drop("SH").sum()), 8) == 0.75


def test_strategy_dynamic_hedge_weight_raises_on_weak_market():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-01"), "SPY"),
            (pd.Timestamp("2026-01-02"), "SPY"),
            (pd.Timestamp("2026-01-05"), "SPY"),
        ],
        names=["datetime", "instrument"],
    )
    close_history = pd.Series([100.0, 98.0, 94.0], index=idx)

    state = {}
    weight = mod._strategy_dynamic_hedge_weight(
        {
            "market_index": "SPY",
            "hedge_weight": 0.0,
            "hedge_max_weight": 0.35,
            "hedge_trend_window": 3,
            "hedge_trend_thresh": -0.02,
            "hedge_min_history": 3,
            "hedge_smoothing_up": 1.0,
        },
        close_history,
        pd.Timestamp("2026-01-06"),
        state,
    )

    assert round(weight, 8) == 0.35
    assert round(state["last_hedge_weight"], 8) == 0.35


def test_strategy_portfolio_weights_applies_turnover_cap_from_current_weights():
    mod = _load_module()
    scores = pd.Series({"C": 4.0, "D": 3.0, "B": 2.0, "A": 1.0})
    marketcap = pd.Series({"A": 10.0, "B": 10.0, "C": 100.0, "D": 100.0})
    current = pd.Series({"A": 0.70, "B": 0.30})

    weights = mod._strategy_portfolio_weights_for_day(
        scores,
        strategy_cfg={
            "topk": 1,
            "benchmark_topn": 2,
            "liquidity_buffer": 1,
            "benchmark_core_weight": 0.0,
            "max_turnover": 0.10,
            "max_holdings": 2,
        },
        strategy_class="WeeklyBenchmarkAwareScoreWeightedStrategy",
        topk=1,
        marketcap=marketcap,
        current_weights=current,
    )

    idx = weights.index.union(current.index)
    turnover = 0.5 * (weights.reindex(idx).fillna(0.0) - current.reindex(idx).fillna(0.0)).abs().sum()
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert len(weights) <= 2
    assert turnover <= 0.10 + 1e-12


def test_strategy_weighted_quality_gate_rows_detect_recent_underperformance():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "weighted_label": [0.010, 0.006, -0.010, -0.008],
            "label_weight_coverage": [1.0, 1.0, 0.96, 0.96],
            "portfolio_names": [40, 40, 40, 40],
        },
        index=pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"]),
    )
    full = mod._strategy_weighted_quality_metrics(frame)
    recent = mod._strategy_weighted_quality_metrics(frame, recent_rebalances=2)
    args = argparse.Namespace(
        strategy_weighted_min_rebalances=4,
        strategy_weighted_min_recent_rebalances=2,
        strategy_weighted_min_full_mean_label=0.0,
        strategy_weighted_min_recent_mean_label=0.0,
        strategy_weighted_min_positive_label_rate=0.50,
        strategy_weighted_min_recent_positive_label_rate=0.50,
        strategy_weighted_min_mean_label_weight_coverage=0.95,
        strategy_weighted_min_recent_label_weight_coverage=0.95,
        strategy_weighted_min_year_rebalances=4,
        strategy_weighted_min_positive_years=1,
        strategy_weighted_min_worst_year_mean_label=0.0,
    )

    rows = mod._strategy_weighted_quality_gate_rows(full, recent, args, yearly=[("2026", full)])

    assert ("strategy_weighted_recent_mean_label", False, "-0.0090 >= 0.0000") in rows
    assert ("strategy_weighted_recent_positive_label_rate", False, "0.0000 >= 0.5000") in rows


def test_strategy_weighted_quality_positive_years_caps_to_eligible_years():
    mod = _load_module()
    full = {
        "rebalances": 50,
        "mean_weighted_label": 0.005,
        "positive_label_rate": 0.60,
        "mean_label_weight_coverage": 1.0,
    }
    recent = {
        "rebalances": 10,
        "mean_weighted_label": 0.004,
        "positive_label_rate": 0.60,
        "mean_label_weight_coverage": 1.0,
    }
    yearly = [
        ("2024", {"rebalances": 25, "mean_weighted_label": 0.006}),
        ("2025", {"rebalances": 25, "mean_weighted_label": 0.001}),
        ("2026", {"rebalances": 5, "mean_weighted_label": 0.007}),
    ]
    args = argparse.Namespace(
        strategy_weighted_min_rebalances=20,
        strategy_weighted_min_recent_rebalances=8,
        strategy_weighted_min_full_mean_label=0.0,
        strategy_weighted_min_recent_mean_label=0.0,
        strategy_weighted_min_positive_label_rate=0.50,
        strategy_weighted_min_recent_positive_label_rate=0.50,
        strategy_weighted_min_mean_label_weight_coverage=0.95,
        strategy_weighted_min_recent_label_weight_coverage=0.95,
        strategy_weighted_min_year_rebalances=20,
        strategy_weighted_min_positive_years=3,
        strategy_weighted_min_worst_year_mean_label=-0.01,
    )

    rows = mod._strategy_weighted_quality_gate_rows(full, recent, args, yearly=yearly)

    assert (
        "strategy_weighted_positive_years",
        True,
        "2 >= 2 (eligible_years=2, min_year_rebalances=20, skipped_short_years=1, configured_min_positive_years=3)",
    ) in rows


def test_benchmark_interval_return_compounds_between_rebalance_dates():
    mod = _load_module()
    bench = pd.Series(
        [0.10, 0.01, -0.02, 0.03],
        index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
    )

    ret = mod._benchmark_interval_return(bench, pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08"))

    assert round(ret, 8) == round((1.01 * 0.98 * 1.03) - 1.0, 8)


def test_rebalance_interval_quality_gate_rows_detect_untradable_actual_horizon():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "holding_days": [5, 5, 5, 5],
            "portfolio_return": [0.010, 0.002, -0.015, -0.010],
            "benchmark_return": [0.000, 0.000, 0.000, 0.000],
            "excess_return": [0.010, 0.002, -0.015, -0.010],
            "active_vs_proxy_return": [0.005, 0.003, -0.008, -0.007],
            "proxy_vs_benchmark_return": [0.005, -0.001, -0.007, -0.003],
            "return_weight_coverage": [1.0, 1.0, 0.99, 0.99],
            "proxy_return_weight_coverage": [1.0, 1.0, 1.0, 1.0],
            "portfolio_names": [120, 120, 120, 120],
        },
        index=pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"]),
    )
    full = mod._rebalance_interval_quality_metrics(frame)
    recent = mod._rebalance_interval_quality_metrics(frame, recent_rebalances=2)
    args = argparse.Namespace(
        rebalance_interval_min_rebalances=4,
        rebalance_interval_min_recent_rebalances=2,
        rebalance_interval_min_full_ann_excess=0.03,
        rebalance_interval_min_recent_ann_excess=0.0,
        rebalance_interval_min_positive_excess_rate=0.50,
        rebalance_interval_min_recent_positive_excess_rate=0.50,
        rebalance_interval_min_mean_return_weight_coverage=0.98,
        rebalance_interval_min_recent_return_weight_coverage=0.98,
        rebalance_interval_max_abs_proxy_tracking_ann=0.05,
        rebalance_interval_min_year_rebalances=4,
        rebalance_interval_min_positive_years=1,
        rebalance_interval_min_worst_year_ann_excess=0.0,
    )

    rows = mod._rebalance_interval_quality_gate_rows(full, recent, args, yearly=[("2026", full)])

    assert ("rebalance_interval_recent_positive_excess_rate", False, "0.0000 >= 0.5000") in rows
    assert any(name == "rebalance_interval_recent_ann_excess" and not ok for name, ok, _ in rows)


def test_rebalance_interval_quality_positive_years_caps_to_eligible_years():
    mod = _load_module()
    full = {
        "rebalances": 50,
        "ann_excess_return": 0.04,
        "positive_excess_rate": 0.60,
        "mean_return_weight_coverage": 1.0,
        "ann_proxy_vs_benchmark_return": 0.0,
    }
    recent = {
        "rebalances": 10,
        "ann_excess_return": 0.03,
        "positive_excess_rate": 0.60,
        "mean_return_weight_coverage": 1.0,
        "ann_proxy_vs_benchmark_return": 0.0,
    }
    yearly = [
        ("2024", {"rebalances": 25, "ann_excess_return": 0.05}),
        ("2025", {"rebalances": 25, "ann_excess_return": 0.01}),
        ("2026", {"rebalances": 5, "ann_excess_return": 0.08}),
    ]
    args = argparse.Namespace(
        rebalance_interval_min_rebalances=20,
        rebalance_interval_min_recent_rebalances=8,
        rebalance_interval_min_full_ann_excess=0.0,
        rebalance_interval_min_recent_ann_excess=0.0,
        rebalance_interval_min_positive_excess_rate=0.50,
        rebalance_interval_min_recent_positive_excess_rate=0.50,
        rebalance_interval_min_mean_return_weight_coverage=0.98,
        rebalance_interval_min_recent_return_weight_coverage=0.98,
        rebalance_interval_max_abs_proxy_tracking_ann=0.05,
        rebalance_interval_min_year_rebalances=20,
        rebalance_interval_min_positive_years=3,
        rebalance_interval_min_worst_year_ann_excess=-0.01,
    )

    rows = mod._rebalance_interval_quality_gate_rows(full, recent, args, yearly=yearly)

    assert (
        "rebalance_interval_positive_years",
        True,
        "2 >= 2 (eligible_years=2, min_year_rebalances=20, skipped_short_years=1, configured_min_positive_years=3)",
    ) in rows


def test_growth_rebalance_interval_profile_tolerates_payoff_skew_without_relaxing_release_profile():
    mod = _load_module()

    assert mod.REBALANCE_INTERVAL_QUALITY_PRESETS["growth"]["rebalance_interval_min_recent_positive_excess_rate"] == 0.35
    assert mod.REBALANCE_INTERVAL_QUALITY_PRESETS["release"]["rebalance_interval_min_recent_positive_excess_rate"] == 0.50


def test_rebalance_interval_quality_frame_uses_trade_dates_with_prior_signal(monkeypatch):
    mod = _load_module()
    from qlib.data import D

    calendar = list(
        pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-12",
            ]
        )
    )
    pred_idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-05"), "A"),
            (pd.Timestamp("2026-01-05"), "B"),
            (pd.Timestamp("2026-01-09"), "A"),
            (pd.Timestamp("2026-01-09"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [2.0, 1.0, 1.0, 5.0, 2.0, 1.0]}, index=pred_idx)

    close = {
        (pd.Timestamp("2026-01-02"), "A"): 100.0,
        (pd.Timestamp("2026-01-02"), "B"): 100.0,
        (pd.Timestamp("2026-01-05"), "A"): 100.0,
        (pd.Timestamp("2026-01-05"), "B"): 100.0,
        (pd.Timestamp("2026-01-09"), "A"): 105.0,
        (pd.Timestamp("2026-01-09"), "B"): 95.0,
        (pd.Timestamp("2026-01-12"), "A"): 110.0,
        (pd.Timestamp("2026-01-12"), "B"): 90.0,
    }

    def fake_features(instruments, fields, start_time=None, end_time=None, **kwargs):
        dates = [dt for dt in calendar if pd.Timestamp(start_time) <= dt <= pd.Timestamp(end_time)]
        idx = pd.MultiIndex.from_product([dates, list(instruments)], names=["datetime", "instrument"])
        data = {}
        for field in fields:
            data[field] = [close.get((dt, inst), 100.0) for dt, inst in idx]
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(D, "features", fake_features, raising=False)
    frame = mod._rebalance_interval_quality_frame(
        pred,
        benchmark=pd.Series(0.0, index=pd.DatetimeIndex(calendar)),
        bt_start=pd.Timestamp("2026-01-05"),
        bt_end=pd.Timestamp("2026-01-12"),
        bt_calendar=calendar,
        strategy_cfg={"topk": 1, "rebalance_weekday": 0, "weighting": "rank", "liquidity_buffer": 1, "risk_degree": 0.5},
        strategy_class="WeeklyScoreWeightedStrategy",
        rebalance_weekday=0,
        args=argparse.Namespace(
            rebalance_interval_marketcap_field=None,
            active_risk_marketcap_field="$marketcap_q",
            rebalance_interval_price_field="",
            rebalance_interval_deal_price="close",
            rebalance_interval_topk=0,
            strategy_signal_shift=1,
        ),
    )

    assert list(frame.index) == [pd.Timestamp("2026-01-05")]
    assert frame.iloc[0]["signal_datetime"] == pd.Timestamp("2026-01-02")
    assert frame.iloc[0]["exit_datetime"] == pd.Timestamp("2026-01-12")
    assert round(float(frame.iloc[0]["portfolio_return"]), 8) == 0.05


def test_rebalance_interval_quality_frame_applies_dynamic_risk(monkeypatch):
    mod = _load_module()
    from qlib.data import D

    calendar = list(
        pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-12",
            ]
        )
    )
    pred_idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-09"), "A"),
            (pd.Timestamp("2026-01-09"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [2.0, 1.0, 2.0, 1.0]}, index=pred_idx)

    close = {
        (pd.Timestamp("2026-01-01"), "SPY"): 100.0,
        (pd.Timestamp("2026-01-02"), "SPY"): 94.0,
        (pd.Timestamp("2026-01-05"), "SPY"): 94.0,
        (pd.Timestamp("2026-01-05"), "A"): 100.0,
        (pd.Timestamp("2026-01-05"), "B"): 100.0,
        (pd.Timestamp("2026-01-12"), "A"): 110.0,
        (pd.Timestamp("2026-01-12"), "B"): 100.0,
        (pd.Timestamp("2026-01-12"), "SPY"): 94.0,
    }

    def fake_features(instruments, fields, start_time=None, end_time=None, **kwargs):
        dates = [dt for dt in calendar if pd.Timestamp(start_time) <= dt <= pd.Timestamp(end_time)]
        idx = pd.MultiIndex.from_product([dates, list(instruments)], names=["datetime", "instrument"])
        data = {}
        for field in fields:
            data[field] = [close.get((dt, inst), 100.0) for dt, inst in idx]
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(D, "features", fake_features, raising=False)
    frame = mod._rebalance_interval_quality_frame(
        pred,
        benchmark=pd.Series(0.0, index=pd.DatetimeIndex(calendar)),
        bt_start=pd.Timestamp("2026-01-05"),
        bt_end=pd.Timestamp("2026-01-12"),
        bt_calendar=calendar,
        strategy_cfg={
            "topk": 1,
            "rebalance_weekday": 0,
            "weighting": "rank",
            "liquidity_buffer": 1,
            "risk_degree": 1.0,
            "dynamic_risk": True,
            "market_index": "SPY",
            "market_trend_window": 2,
            "market_trend_thresh": -0.02,
            "market_trend_penalty": 0.25,
            "risk_min_history": 2,
            "risk_smoothing_down": 1.0,
        },
        strategy_class="WeeklyScoreWeightedStrategy",
        rebalance_weekday=0,
        args=argparse.Namespace(
            rebalance_interval_marketcap_field=None,
            active_risk_marketcap_field="$marketcap_q",
            rebalance_interval_price_field="",
            rebalance_interval_deal_price="close",
            rebalance_interval_topk=0,
            strategy_signal_shift=1,
        ),
    )

    assert round(float(frame.iloc[0]["risk_degree"]), 8) == 0.25
    assert round(float(frame.iloc[0]["portfolio_return"]), 8) == 0.025


def test_apply_sector_cap_to_score_matches_strategy_cap():
    mod = _load_module()
    score = pd.Series([5.0, 4.0, 3.0, 2.0], index=["A", "B", "C", "D"])
    strategy_cfg = {"topk": 4, "max_sector_weight": 0.50}
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health"}

    adjusted = mod._apply_sector_cap_to_score(score, strategy_cfg, sector_map)

    assert adjusted.sort_values(ascending=False).head(3).index.tolist() == ["A", "B", "D"]


def test_load_strategy_sector_map_preserves_duplicate_sector_values(tmp_path):
    mod = _load_module()
    path = tmp_path / "tickers.csv"
    path.write_text("ticker,sector\nA,Tech\nB,Tech\nC,Health\n", encoding="utf-8")

    sector_map = mod._load_strategy_sector_map({"sector_map_csv": str(path)})

    assert sector_map == {"A": "Tech", "B": "Tech", "C": "Health"}
