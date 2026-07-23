import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "diagnose_us_sharadar_candidate.py"
    spec = importlib.util.spec_from_file_location("diagnose_us_sharadar_candidate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_summary_flags_rank_normalized_mse_and_missing_sf3a():
    mod = _load_module()
    cfg = {
        "data_handler_config": {
            "infer_processors": [{"class": "CSZScoreNorm"}],
            "learn_processors": [{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
            "extra_fields": ["$mkt_spy_ret_63d", "$risk_beta_spy_63d"],
        },
        "task": {"model": {"class": "LGBModel", "kwargs": {"loss": "mse"}}},
    }

    summary = mod._model_summary(cfg)

    assert summary["model_class"] == "LGBModel"
    assert summary["objective"] == "mse"
    assert summary["uses_rank_label_processor"] is True
    assert summary["uses_true_ranker"] is False
    assert summary["uses_plain_cszscore"] is True
    assert summary["market_field_count"] == 1
    assert summary["sf3a_field_count"] == 0
    assert summary["direct_risk_alpha_field_count"] == 1


def test_best_iterations_reads_mlflow_metric_files(tmp_path):
    mod = _load_module()
    run_id = "abc123"
    metric_dir = tmp_path / "1" / run_id / "metrics"
    metric_dir.mkdir(parents=True)
    (metric_dir / "l2.valid").write_text(
        "100 0.50 0\n101 0.40 1\n102 0.45 2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"used_runs": [{"run_id": run_id, "test": ["2022-01-01", "2022-12-31"]}]}))

    rows = mod._best_iterations(manifest, tmp_path)

    assert rows == [
        {
            "run_id": run_id,
            "test": ["2022-01-01", "2022-12-31"],
            "metric_name": "l2.valid",
            "metric_points": 3,
            "best_iteration": 2,
            "best_valid_metric": 0.40,
        }
    ]


def test_best_iterations_treats_ndcg_as_higher_is_better(tmp_path):
    mod = _load_module()
    run_id = "ranker123"
    metric_dir = tmp_path / "1" / run_id / "metrics"
    metric_dir.mkdir(parents=True)
    (metric_dir / "ndcg_100.valid").write_text(
        "100 0.30 0\n101 0.45 1\n102 0.35 2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"used_runs": [{"run_id": run_id, "test": ["2022-01-01", "2022-12-31"]}]}))

    rows = mod._best_iterations(manifest, tmp_path)

    assert rows[0]["metric_name"] == "ndcg_100.valid"
    assert rows[0]["best_iteration"] == 2
    assert rows[0]["best_valid_metric"] == 0.45


def test_model_summary_counts_stable_feature_groups():
    mod = _load_module()
    cfg = {
        "data_handler_config": {
            "infer_processors": [{"class": "SelectiveCSZScoreNorm"}],
            "learn_processors": [],
            "extra_fields": [
                "$roe_q",
                "$earn_yield_q",
                "$risk_ret_252d",
                "$inst13f_shrvalue_252d_pct",
                "$log_marketcap_q",
            ],
            "extra_names": [
                "ROE_Q",
                "EARN_YIELD_Q",
                "RISK_RET_252D",
                "INST13F_SHRVALUE_252D_PCT",
                "LOG_MARKETCAP_Q",
            ],
        },
        "task": {"model": {"class": "LGBModel", "kwargs": {"loss": "mse"}}},
    }

    summary = mod._model_summary(cfg)

    assert summary["direct_risk_alpha_field_count"] == 0
    assert summary["quality_value_field_count"] == 2
    assert summary["momentum_field_count"] == 1
    assert summary["sf3a_field_count"] == 1
    assert summary["size_field_count"] == 1


def test_extra_feature_name_mapping_handles_alpha158_offset():
    mod = _load_module()
    cfg = {
        "data_handler_config": {"extra_names": ["ROE_Q", "RISK_VOL_63D"]},
        "task": {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "class": "Alpha158WithPIT",
                        "kwargs": {"extra_names": ["ROE_Q", "RISK_VOL_63D"]},
                    }
                }
            }
        },
    }

    assert mod._extra_name_for_feature("Column_158", cfg) == "ROE_Q"
    assert mod._extra_name_for_feature("Column_159", cfg) == "RISK_VOL_63D"
    assert mod._feature_group("RISK_VOL_63D") == "direct_risk_alpha"


def test_weekly_topk_overlap():
    mod = _load_module()
    rows = [
        {"datetime": "2024-01-05", "instrument": "A", "rank": 1},
        {"datetime": "2024-01-05", "instrument": "B", "rank": 2},
        {"datetime": "2024-01-12", "instrument": "B", "rank": 1},
        {"datetime": "2024-01-12", "instrument": "C", "rank": 2},
    ]

    overlap = mod._weekly_topk_overlap(__import__("pandas").DataFrame(rows))

    assert overlap == 0.5


def test_apply_strategy_feature_controls_demotes_low_percentile_names():
    mod = _load_module()
    import pandas as pd

    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-05")], ["A", "B", "C", "D"]],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([4.0, 3.0, 2.0, 1.0], index=idx)
    feat = pd.DataFrame({"$log_marketcap_q": [10.0, 20.0, 30.0, 40.0]}, index=idx)
    cfg = {
        "port_analysis_config": {
            "strategy": {
                "kwargs": {
                    "feature_min_percentiles": {"$log_marketcap_q": 0.75},
                }
            }
        }
    }

    adjusted = mod._apply_strategy_feature_controls(pred, feat, cfg)

    assert adjusted.sort_values(ascending=False).head(2).index.get_level_values("instrument").tolist() == ["C", "D"]


def test_parse_attribution_windows_and_window_summary():
    mod = _load_module()
    import pandas as pd

    windows = mod._parse_attribution_windows(["recent:2024-01-01:2024-01-31"])
    top = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")],
            "instrument": ["A", "B"],
            "score": [1.0, 2.0],
            "excess_label": [0.05, -0.01],
            "risk_ret_20d_pct": [0.80, 0.40],
            "mkt_qqq_ret_63d_lag1": [0.05, -0.08],
            "mkt_qqq_ret_20d_lag1": [0.03, 0.04],
            "mkt_qqq_dd_126d_lag1": [-0.02, -0.12],
            "mkt_breadth_ret63_pos_lag1": [0.60, 0.30],
            "sector": ["Tech", "Health"],
        }
    )
    bench = pd.Series(0.001, index=pd.bdate_range("2024-01-01", "2024-01-31"))

    out = mod._window_attribution(
        top,
        feature_cols=["risk_ret_20d"],
        exposure_cols=["risk_ret_20d_pct"],
        market_cols=[
            "mkt_qqq_ret_63d_lag1",
            "mkt_qqq_ret_20d_lag1",
            "mkt_qqq_dd_126d_lag1",
            "mkt_breadth_ret63_pos_lag1",
        ],
        benchmark_returns=bench,
        windows=windows,
    )

    assert out["recent"]["topk_rows"] == 2
    assert abs(out["recent"]["mean_excess_label"] - 0.02) < 1e-12
    assert round(out["recent"]["feature_percentile_mean"]["risk_ret_20d_pct"], 8) == 0.6
    assert round(out["recent"]["market_state_mean"]["mkt_qqq_ret_63d_lag1"], 8) == -0.015
    assert out["recent"]["regime_state_share"]["risk_on"] == 0.5
    assert out["recent"]["regime_state_share"]["recovery"] == 0.5


def test_failed_windows_from_validation_log(tmp_path):
    mod = _load_module()
    log = tmp_path / "validate.log"
    log.write_text(
        "window | status | detail\n"
        "2023-01-03->2023-07-05 | FAIL | excess_ann_return=-0.12\n"
        "QQQ | 2025-01-02->2025-07-03 (126d) | FAIL | excess_ann_return=-0.30\n"
        "2023-01-03->2023-07-05 | FAIL | duplicate should be ignored\n",
        encoding="utf-8",
    )

    windows = mod._failed_windows_from_validation_log(log)

    assert windows == [
        ("failed_rolling_1", pd.Timestamp("2023-01-03"), pd.Timestamp("2023-07-05")),
        ("failed_rolling_2", pd.Timestamp("2025-01-02"), pd.Timestamp("2025-07-03")),
    ]
