import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "audit_us_sharadar_targets.py"
    spec = importlib.util.spec_from_file_location("audit_us_sharadar_targets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_score_specs_supports_name_expression_and_weight():
    mod = _load_module()

    parsed = mod._parse_score_specs(["MOM=$risk_ret_20d:1.5", "$fcf_yield_q:-0.25"])

    assert parsed == [("MOM", "$risk_ret_20d", 1.5), ("FCF_YIELD_Q", "$fcf_yield_q", -0.25)]


def test_forward_return_from_close_uses_next_bar_entry():
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5, freq="D"), ["A"]],
        names=["datetime", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 12.0, 15.0, 18.0], index=idx)

    label = mod._forward_return_from_close(close, horizon_days=2, label_ref_start_days=1)

    assert abs(label.loc[(pd.Timestamp("2024-01-01"), "A")] - (15.0 / 11.0 - 1.0)) < 1e-12


def test_neutralize_by_sector_demeans_each_date_sector():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
        ],
        names=["datetime", "instrument"],
    )
    label = pd.Series([0.10, 0.20, -0.05], index=idx)

    out = mod._neutralize_by_sector(label, {"A": "Tech", "B": "Tech", "C": "Health"})

    assert abs(out.loc[(pd.Timestamp("2024-01-02"), "A")] + 0.05) < 1e-12
    assert abs(out.loc[(pd.Timestamp("2024-01-02"), "B")] - 0.05) < 1e-12
    assert abs(out.loc[(pd.Timestamp("2024-01-02"), "C")]) < 1e-12


def test_rank_bucket_summary_averages_daily_bucket_labels():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-03"), "C"),
        ],
        names=["datetime", "instrument"],
    )
    score = pd.Series([3.0, 2.0, 1.0, 1.0, 3.0, 2.0], index=idx)
    label = pd.Series([0.30, 0.20, 0.10, -0.10, 0.40, 0.20], index=idx)

    rows = mod._rank_bucket_summary(
        score,
        label,
        score_name="SCORE",
        target_kind="raw_return",
        horizon_days=5,
        recent_days=1,
        buckets=[(1, 1), (2, 3)],
        min_daily_count=3,
    )

    full_top = next(row for row in rows if row["period"] == "full" and row["bucket"] == "001_001")
    recent_top = next(row for row in rows if row["period"] == "recent_1d" and row["bucket"] == "001_001")
    assert abs(full_top["mean_label"] - 0.35) < 1e-12
    assert abs(recent_top["mean_label"] - 0.40) < 1e-12


def test_top_bottom_classification_marks_quintile_tails():
    mod = _load_module()
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-02")], list("ABCDE")],
        names=["datetime", "instrument"],
    )
    label = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    out = mod._top_bottom_classification(label, top_pct=0.20)

    assert out.loc[(pd.Timestamp("2024-01-02"), "A")] == -1.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "E")] == 1.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "C")] == 0.0


def test_screen_target_candidates_requires_recent_top_bucket_quality():
    mod = _load_module()
    ic_df = pd.DataFrame(
        [
            {
                "target_kind": "benchmark_excess",
                "horizon_days": 20,
                "score": "MOM",
                "mean_ic": 0.02,
                "recent_mean_ic": 0.01,
                "recent_pos_ic_rate": 0.60,
                "positive_years": 3,
                "worst_year_mean_ic": -0.01,
            },
            {
                "target_kind": "raw_return",
                "horizon_days": 20,
                "score": "MOM",
                "mean_ic": 0.03,
                "recent_mean_ic": 0.02,
                "recent_pos_ic_rate": 0.70,
                "positive_years": 4,
                "worst_year_mean_ic": 0.00,
            },
        ]
    )
    bucket_df = pd.DataFrame(
        [
            {
                "target_kind": "benchmark_excess",
                "horizon_days": 20,
                "score": "MOM",
                "period": "recent_63d",
                "bucket": "001_020",
                "mean_label": -0.01,
                "positive_day_rate": 0.40,
            },
            {
                "target_kind": "benchmark_excess",
                "horizon_days": 20,
                "score": "MOM",
                "period": "full",
                "bucket": "001_020",
                "mean_label": 0.02,
                "positive_day_rate": 0.60,
            },
            {
                "target_kind": "raw_return",
                "horizon_days": 20,
                "score": "MOM",
                "period": "recent_63d",
                "bucket": "001_020",
                "mean_label": 0.03,
                "positive_day_rate": 0.65,
            },
            {
                "target_kind": "raw_return",
                "horizon_days": 20,
                "score": "MOM",
                "period": "full",
                "bucket": "001_020",
                "mean_label": 0.02,
                "positive_day_rate": 0.60,
            },
        ]
    )
    target_df = pd.DataFrame(
        [
            {"target_kind": "benchmark_excess", "horizon_days": 20, "period": "recent_63d", "positive_rate": 0.50},
            {"target_kind": "raw_return", "horizon_days": 20, "period": "recent_63d", "positive_rate": 0.55},
        ]
    )

    screened = mod._screen_target_candidates(ic_df, bucket_df, target_df, recent_days=63)

    by_target = {row["target_kind"]: row for row in screened.to_dict("records")}
    assert bool(by_target["raw_return"]["screen_pass"]) is True
    assert bool(by_target["benchmark_excess"]["screen_pass"]) is False
    assert "recent_top_bucket_label" in by_target["benchmark_excess"]["screen_failures"]
