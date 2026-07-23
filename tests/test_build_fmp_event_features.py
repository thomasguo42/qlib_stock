import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_fmp_event_features.py"
    spec = importlib.util.spec_from_file_location("build_fmp_event_features", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_grade_score_maps_common_grades():
    mod = _load_module()

    assert mod.grade_score("Strong Buy") == 2.0
    assert mod.grade_score("Outperform") == 1.0
    assert mod.grade_score("Equal Weight") == 0.0
    assert mod.grade_score("Underperform") == -1.0
    assert mod.grade_score("Strong Sell") == -2.0


def test_load_dataset_prefers_filename_symbol_for_class_shares(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(
        raw / "grades" / "BRK.B.json",
        [{"symbol": "BRK-B", "date": "2026-01-03", "newGrade": "Buy"}],
    )

    df = mod.load_dataset(raw, "grades")

    assert df["symbol"].tolist() == ["BRK.B"]


def test_reserved_header_symbol_files_are_ignored(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(raw / "grades" / "TICKER.json", [{"symbol": "TICKER", "date": "2026-01-03"}])

    assert mod.load_dataset(raw, "grades").empty
    assert mod.raw_symbols(raw, ["grades"]) == []


def test_feature_frames_include_zero_rows_for_uncovered_raw_symbols(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(raw / "earnings" / "AAPL.json", [])
    frames = mod.build_feature_frames(
        raw,
        start="2026-01-01",
        end="2026-01-03",
        windows=[3],
        availability_lag_days=1,
        prefix="fmp",
    )

    out = frames["AAPL"]
    expected_cols = mod.expected_feature_columns("fmp", [3])
    assert list(out.columns) == ["date", *expected_cols]
    assert out.drop(columns=["date"]).sum().sum() == 0.0


def test_build_feature_frames_lags_events_and_excludes_future_earnings(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(
        raw / "earnings" / "AAPL.json",
        [
            {
                "symbol": "AAPL",
                "date": "2026-01-02",
                "epsActual": 1.20,
                "epsEstimated": 1.00,
                "revenueActual": 110.0,
                "revenueEstimated": 100.0,
            },
            {
                "symbol": "AAPL",
                "date": "2026-01-05",
                "epsActual": None,
                "epsEstimated": 1.10,
                "revenueActual": None,
                "revenueEstimated": 120.0,
            },
        ],
    )
    _write_json(
        raw / "grades" / "AAPL.json",
        [
            {
                "symbol": "AAPL",
                "date": "2026-01-03",
                "previousGrade": "Hold",
                "newGrade": "Buy",
                "action": "upgrade",
            }
        ],
    )
    _write_json(
        raw / "grades_historical" / "AAPL.json",
        [
            {
                "symbol": "AAPL",
                "date": "2026-01-01",
                "analystRatingsStrongBuy": 1,
                "analystRatingsBuy": 2,
                "analystRatingsHold": 1,
                "analystRatingsSell": 0,
                "analystRatingsStrongSell": 0,
            }
        ],
    )
    _write_json(
        raw / "price_target_news" / "AAPL.json",
        [
            {
                "symbol": "AAPL",
                "publishedDate": "2026-01-04T20:00:00.000Z",
                "adjPriceTarget": 150.0,
                "priceTarget": 150.0,
                "priceWhenPosted": 100.0,
            }
        ],
    )

    frames = mod.build_feature_frames(
        raw,
        start="2026-01-01",
        end="2026-01-08",
        windows=[3],
        availability_lag_days=1,
        prefix="fmp",
    )

    out = frames["AAPL"].set_index("date")
    assert out.loc["2026-01-02", "fmp_rating_bullish_ratio_snapshot"] == 0.75
    assert out.loc["2026-01-03", "fmp_earn_count_daily"] == 1.0
    assert round(float(out.loc["2026-01-03", "fmp_eps_surprise_pct_latest"]), 6) == 0.2
    assert out.loc["2026-01-04", "fmp_grade_up_daily_sum"] == 1.0
    assert out.loc["2026-01-05", "fmp_pt_count_daily"] == 1.0
    assert out["fmp_earn_count_daily"].sum() == 1.0
    assert "fmp_alpha_event_composite" in out.columns
    assert out.loc["2026-01-03", "fmp_earn_count_days_since_latest"] == 0.0
    assert out.loc["2026-01-04", "fmp_earn_count_days_since_latest"] == 1.0
    assert float(out.loc["2026-01-03", "fmp_alpha_earn_surprise_latest"]) > 0.0
    assert float(out.loc["2026-01-03", "fmp_alpha_rating_bullish"]) > 0.0
    assert float(out.loc["2026-01-03", "fmp_alpha_rating_bearish_penalty"]) == 0.0


def test_main_dry_run_builds_without_provider_calendar(tmp_path, monkeypatch):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(
        raw / "grades" / "MSFT.json",
        [{"symbol": "MSFT", "date": "2026-01-03", "previousGrade": "Hold", "newGrade": "Buy", "action": "upgrade"}],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_fmp_event_features.py",
            "--raw_root",
            str(raw),
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-05",
            "--dry_run",
        ],
    )

    assert mod.main() == 0
