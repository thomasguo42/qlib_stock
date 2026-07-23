import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_us_stock_selector_report.py"
    spec = importlib.util.spec_from_file_location("generate_us_stock_selector_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_int_list_dedupes_and_sorts():
    mod = _load_module()

    assert mod.parse_int_list("20,5,10,5") == [5, 10, 20]


def test_select_score_date_uses_latest_on_or_before_request():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-01"), "AAPL"),
            (pd.Timestamp("2026-05-05"), "MSFT"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([1.0, 2.0], index=idx)

    assert mod.select_score_date(pred, "latest") == pd.Timestamp("2026-05-05")
    assert mod.select_score_date(pred, "2026-05-03") == pd.Timestamp("2026-05-01")


def test_rank_scores_for_date_adds_percentiles_and_buckets():
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-05-01")], ["A", "B", "C", "D"]],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([0.2, 0.9, -0.1, 0.5], index=idx)

    out = mod.rank_scores_for_date(pred, pd.Timestamp("2026-05-01"), topn=2)

    assert out["symbol"].tolist() == ["B", "D"]
    assert out["rank"].tolist() == [1, 2]
    assert out.loc[0, "score_percentile"] == 1.0
    assert out.loc[0, "score_bucket"] == "top_1pct"


def test_summarize_returns_reports_empirical_bands():
    mod = _load_module()

    out = mod.summarize_returns(pd.Series([-0.10, -0.02, 0.03, 0.05]))

    assert out["sample_count"] == 4
    assert out["hit_rate"] == 0.5
    assert out["worst"] == -0.10
    assert out["best"] == 0.05
    assert abs(out["median"] - 0.005) < 1e-12


def test_make_risk_flags_combines_candidate_warnings():
    mod = _load_module()

    flags = mod.make_risk_flags(
        {
            "beta_qqq_63d": 1.6,
            "vol20_ann": 0.7,
            "drawdown_63d": -0.2,
            "avg_dollar_vol_20d": 5_000_000,
            "fmp_event_freshness": 0.01,
            "hist_20d_p10": -0.12,
        }
    )

    assert flags.split(",") == [
        "high_qqq_beta",
        "high_vol",
        "deep_recent_drawdown",
        "low_liquidity",
        "stale_fmp_event",
        "wide_20d_downside",
    ]


def test_selector_category_helpers_classify_edge_risk_confidence():
    mod = _load_module()
    row = {
        "score_percentile": 0.995,
        "score_z": 2.1,
        "hist_20d_sample_count": 1200,
        "hist_20d_hit_rate": 0.62,
        "hist_20d_median": 0.022,
        "hist_20d_p10": -0.05,
        "beta_qqq_63d": 1.1,
        "vol20_ann": 0.32,
        "vol63_ann": 0.34,
        "drawdown_63d": -0.04,
        "drawdown_126d": -0.06,
        "avg_dollar_vol_20d": 100_000_000,
        "fmp_event_coverage": 0.7,
        "fmp_event_freshness": 0.3,
        "risk_flags": "none",
        "top_driver": "momentum_regime",
        "sleeve_contributions": '{"momentum": 1.0}',
    }

    edge, _ = mod.classify_expected_edge(row, 20)
    risk, _ = mod.classify_risk(row, 20)
    confidence, _ = mod.classify_confidence(row, 20)

    assert edge == "high"
    assert risk == "low"
    assert confidence == "high"
    assert mod.selector_category_for_tiers(edge, risk, confidence) == "Core Candidate"


def test_add_selector_categories_marks_high_risk_edge_as_aggressive():
    mod = _load_module()
    report = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "score_percentile": 0.995,
                "score_z": 2.0,
                "hist_20d_sample_count": 800,
                "hist_20d_hit_rate": 0.60,
                "hist_20d_median": 0.02,
                "hist_20d_p10": -0.16,
                "beta_qqq_63d": 1.7,
                "vol20_ann": 0.75,
                "vol63_ann": 0.65,
                "drawdown_63d": -0.08,
                "drawdown_126d": -0.12,
                "avg_dollar_vol_20d": 200_000_000,
                "fmp_event_coverage": 0.3,
                "fmp_event_freshness": 0.2,
                "risk_flags": "high_qqq_beta,high_vol,wide_20d_downside",
                "top_driver": "fmp_events",
            }
        ]
    )

    out = mod.add_selector_categories(report, 20)

    assert out.loc[0, "expected_edge_tier"] == "high"
    assert out.loc[0, "risk_tier"] == "high"
    assert out.loc[0, "confidence_tier"] in {"medium", "high"}
    assert out.loc[0, "strict_assessment"] == "Aggressive Upside"
    assert out.loc[0, "selector_action"] == "Speculative / High Upside"
    assert out.loc[0, "selector_category"] == "Speculative / High Upside"
    assert out.loc[0, "selector_source"] == "high_risk_upside_proxy"
    assert "edge=high" in out.loc[0, "strict_assessment_reason"]


def test_mark_core_watchlist_prioritizes_actionable_names():
    mod = _load_module()
    report = pd.DataFrame(
        [
            {
                "symbol": "A",
                "rank": 1,
                "selector_rank": 1,
                "selector_action": "Avoid",
                "expected_edge_tier": "low",
                "risk_tier": "high",
                "selector_utility_score": 5.0,
            },
            {
                "symbol": "B",
                "rank": 2,
                "selector_rank": 2,
                "selector_action": "Best Current Candidates",
                "expected_edge_tier": "high",
                "risk_tier": "low",
                "selector_utility_score": 2.0,
            },
            {
                "symbol": "C",
                "rank": 3,
                "selector_rank": 3,
                "selector_action": "Speculative / High Upside",
                "expected_edge_tier": "high",
                "risk_tier": "high",
                "selector_utility_score": 3.0,
            },
            {
                "symbol": "D",
                "rank": 4,
                "selector_rank": 4,
                "selector_action": "Watchlist Only",
                "expected_edge_tier": "medium",
                "risk_tier": "medium",
                "selector_utility_score": 1.0,
            },
        ]
    )

    out = mod.mark_core_watchlist(report, core_topn=2)
    core = out.loc[out["in_core_watchlist"]].sort_values("core_rank")

    assert core["symbol"].tolist() == ["B", "C"]
    assert core["core_role"].tolist() == ["core_5", "core_5"]
    assert bool(out.loc[out["symbol"] == "A", "in_core_watchlist"].iloc[0]) is False


def test_infer_regime_state_matches_config_thresholds():
    mod = _load_module()
    kwargs = {
        "state_weights": {"risk_on": {}, "risk_off": {}, "chop": {}},
        "default_state": "chop",
        "risk_on_threshold": 0.03,
        "risk_off_threshold": -0.04,
        "drawdown_warning": -0.06,
        "drawdown_limit": -0.10,
        "breadth_threshold": 0.45,
    }

    state = mod.infer_regime_state(
        {
            "MKT_QQQ_RET_63D_LAG1": 0.05,
            "MKT_QQQ_RET_20D_LAG1": 0.02,
            "MKT_QQQ_DD_126D_LAG1": -0.02,
            "MKT_QQQ_VOL_20D_LAG1": 0.20,
            "MKT_BREADTH_RET63_POS_LAG1": 0.60,
        },
        kwargs,
    )

    assert state == "risk_on"


def test_build_markdown_report_contains_watchlist_and_calibration():
    mod = _load_module()
    report = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "AAPL",
                "sector": "Technology",
                "score_percentile": 0.99,
                "score_bucket": "top_1pct",
                "hist_20d_hit_rate": 0.6,
                "hist_20d_median": 0.03,
                "hist_20d_p10": -0.04,
                "beta_qqq_63d": 1.2,
                "vol20_ann": 0.35,
                "risk_flags": "none",
                "top_driver": "momentum_regime",
                "expected_edge_tier": "high",
                "risk_tier": "low",
                "confidence_tier": "high",
                "selector_action": "Best Current Candidates",
                "selector_source": "risk_adjusted_utility",
                "selector_source_reason": "selected by utility",
                "selector_category": "Best Current Candidates",
                "strict_assessment": "Core Candidate",
                "selector_utility_score": 1.2,
                "in_core_watchlist": True,
                "core_rank": 1,
                "core_reason": "highest-priority current candidate",
                "model_rank": 1,
                "category_reason": "top diversified utility",
                "strict_assessment_reason": "edge=high; risk=low; confidence=high",
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "score_bucket": "top_1pct",
                "horizon_days": 20,
                "sample_count": 100,
                "hit_rate": 0.6,
                "median": 0.03,
                "p10": -0.04,
                "p25": -0.01,
                "p75": 0.06,
            }
        ]
    )

    text = mod.build_markdown_report(
        report=report,
        calibration=calibration,
        metadata={"generated_utc": "now", "score_date": "2026-05-01", "topn": 1},
        primary_horizon=20,
    )

    assert "# US Stock Selector Report" in text
    assert "## Core 5 Shortlist" in text
    assert "| 1 | 1 | AAPL | Best Current Candidates | high | low | 1.20 | highest-priority current candidate |" in text
    assert "### Best Current Candidates" in text
    assert "| 1 | 1 | AAPL | Technology | 99.0%" in text
    assert "| Best Current Candidates | 1 |" in text
    assert "source `risk_adjusted_utility`" in text
    assert "Calibrated 20d median excess 3.0%" in text


def test_json_safe_replaces_nonfinite_values():
    mod = _load_module()

    text = json.dumps(mod.json_safe({"x": float("nan"), "y": pd.Timestamp("2026-05-01")}))

    assert text == '{"x": null, "y": "2026-05-01"}'


def test_human_decision_template_is_not_overwritten(tmp_path):
    mod = _load_module()
    path = tmp_path / "human_decisions.csv"
    report = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "rank": [1, 2]})

    assert mod.write_human_decision_template(path, report, "2026-05-01") is True
    first = path.read_text(encoding="utf-8")
    path.write_text(first + "manual,note,row,kept,,,,\n", encoding="utf-8")

    assert mod.write_human_decision_template(path, report, "2026-05-01") is False
    assert "manual,note,row,kept" in path.read_text(encoding="utf-8")


def test_run_id_from_metadata_is_stable_for_tracking_fields():
    mod = _load_module()
    metadata = {
        "score_date": "2026-05-01",
        "config": "/a.yaml",
        "pred": "/p.pkl",
        "provider_uri": "/data",
        "benchmark_pkl": "/qqq.pkl",
        "selector_schema_version": 2,
        "selector_classifier_version": 1,
        "topn": 25,
        "rank_export_n": 100,
        "horizons": [5, 10, 20],
        "primary_horizon": 20,
        "generated_utc": "ignored",
    }

    assert mod.run_id_from_metadata(metadata) == mod.run_id_from_metadata({**metadata, "generated_utc": "later"})
