import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "score_us_stock_selector_reports.py"
    spec = importlib.util.spec_from_file_location("score_us_stock_selector_reports", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_horizon_dates_use_ref_start_and_require_available_exit():
    mod = _load_module()
    cal = pd.bdate_range("2026-05-01", periods=8)

    assert mod.horizon_dates(cal, pd.Timestamp("2026-05-01"), 3, label_ref_start_days=1) == (
        pd.Timestamp("2026-05-04"),
        pd.Timestamp("2026-05-07"),
    )
    assert mod.horizon_dates(cal, pd.Timestamp("2026-05-01"), 20, label_ref_start_days=1) is None


def test_completed_horizons_respects_as_of_cutoff():
    mod = _load_module()
    cal = pd.bdate_range("2026-05-01", periods=10)

    out = mod.completed_horizons(
        cal,
        pd.Timestamp("2026-05-01"),
        [1, 3, 5],
        label_ref_start_days=1,
        as_of=pd.Timestamp("2026-05-07"),
    )

    assert sorted(out) == [1, 2 + 1]  # horizons 1 and 3 only
    assert out[1][1] == pd.Timestamp("2026-05-05")
    assert out[3][1] == pd.Timestamp("2026-05-07")


def test_forward_returns_from_close_uses_entry_and_exit_dates():
    mod = _load_module()
    close = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [50.0, 45.0, 54.0],
        },
        index=pd.to_datetime(["2026-05-01", "2026-05-04", "2026-05-05"]),
    )

    out = mod.forward_returns_from_close(
        close,
        {1: (pd.Timestamp("2026-05-04"), pd.Timestamp("2026-05-05"))},
    )

    assert round(out[1]["AAPL"], 8) == 0.10
    assert round(out[1]["MSFT"], 8) == 0.20


def test_matched_control_symbols_excludes_selected_and_prefers_same_sector():
    mod = _load_module()
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "sector": ["Tech", "Tech", "Tech", "Health", "Tech"],
            "log_marketcap": [10.0, 10.2, 13.0, 10.1, 9.8],
            "risk_vol_20d_feature": [0.3, 0.31, 0.9, 0.3, 0.29],
        }
    )

    controls = mod.matched_control_symbols("AAA", universe, selected_symbols=["AAA", "CCC"], n_controls=2)

    assert controls == ["BBB", "EEE"]


def test_summary_row_reports_expected_gap():
    mod = _load_module()

    row = mod.summary_row(
        run_id="r1",
        score_date="2026-05-01",
        horizon=20,
        group="watchlist_top25",
        values=pd.Series([0.02, 0.04, -0.01]),
        expected_median=0.01,
    )

    assert row["sample_count"] == 3
    assert row["hit_rate"] == 2 / 3
    assert row["median"] == 0.02
    assert row["median_minus_expected"] == 0.01


def test_parse_as_of_accepts_latest_aliases():
    mod = _load_module()

    assert mod.parse_as_of("") is None
    assert mod.parse_as_of("latest") is None
    assert mod.parse_as_of("LOCAL_LATEST") is None
    assert mod.parse_as_of("2026-05-29") == pd.Timestamp("2026-05-29")


def test_watchlist_segment_summary_rows_group_by_selector_categories():
    mod = _load_module()
    outcomes = pd.DataFrame(
        {
            "selector_source": ["risk_adjusted_utility", "high_risk_upside_proxy", "risk_adjusted_utility"],
            "selector_action": ["Best Current Candidates", "Speculative / High Upside", "Best Current Candidates"],
            "selector_category": ["Core Candidate", "Aggressive Upside", "Core Candidate"],
            "strict_assessment": ["Core Candidate", "Aggressive Upside", "Low Priority / Defensive"],
            "expected_edge_tier": ["high", "high", "medium"],
            "risk_tier": ["low", "high", "low"],
            "confidence_tier": ["high", "medium", "high"],
            "selector_utility_decile": [10, 9, 8],
            "qqq_excess_return": [0.03, -0.04, 0.01],
            "expected_bucket_median": [0.02, 0.02, 0.01],
        }
    )

    rows = mod.watchlist_segment_summary_rows(
        run_id="r1",
        score_date="2026-05-01",
        horizon=20,
        outcomes=outcomes,
    )
    by_group = {row["group"]: row for row in rows}

    assert by_group["action_best_current_candidates"]["sample_count"] == 2
    assert by_group["source_risk_adjusted_utility"]["sample_count"] == 2
    assert by_group["source_high_risk_upside_proxy"]["median"] == -0.04
    assert by_group["action_speculative_high_upside"]["median"] == -0.04
    assert by_group["strict_core_candidate"]["sample_count"] == 1
    assert by_group["category_core_candidate"]["sample_count"] == 2
    assert by_group["category_core_candidate"]["median"] == 0.02
    assert by_group["category_aggressive_upside"]["median"] == -0.04
    assert by_group["edge_high"]["sample_count"] == 2
    assert by_group["risk_low"]["hit_rate"] == 1.0
    assert by_group["utility_decile_10"]["sample_count"] == 1
