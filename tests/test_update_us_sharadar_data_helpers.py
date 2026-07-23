import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "update_us_sharadar_data.py"
    spec = importlib.util.spec_from_file_location("update_us_sharadar_data", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sep_coverage_gaps_flags_missing_and_lagged_files(tmp_path):
    mod = _load_module()
    sep = tmp_path / "sep"
    sep.mkdir()
    pd.DataFrame({"date": ["2026-04-29", "2026-04-30"]}).to_csv(sep / "AAPL.csv", index=False)
    pd.DataFrame({"date": ["2026-04-20"]}).to_csv(sep / "MSFT.csv", index=False)

    gaps = mod._sep_coverage_gaps(
        sep,
        ["AAPL", "MSFT", "NVDA"],
        required_end="2026-04-30",
        max_lag_days=1,
    )

    assert gaps == {"MSFT": "2026-04-20", "NVDA": "<missing>"}


def test_active_market_tickers_filters_by_instrument_end(tmp_path):
    mod = _load_module()
    inst = tmp_path / "market.txt"
    inst.write_text(
        "AAPL\t2016-01-01\t2026-04-30\nMSFT\t2016-01-01\t2026-04-29\n",
        encoding="utf-8",
    )

    assert mod._active_market_tickers(inst, "2026-04-30") == ["AAPL"]


def test_market_frontier_helpers_ignore_later_provider_calendar(tmp_path):
    mod = _load_module()
    inst = tmp_path / "market.txt"
    inst.write_text(
        "AAPL\t2016-01-01\t2026-04-30\n"
        "MSFT\t2016-01-01\t2026-04-30\n"
        "OLD\t2016-01-01\t2021-06-01\n",
        encoding="utf-8",
    )

    assert mod._market_instruments_max_end(inst) == "2026-04-30"
    assert mod._earliest_date("2026-05-04", "2026-04-30") == "2026-04-30"


def test_extend_market_instruments_uses_current_market_frontier(tmp_path):
    mod = _load_module()
    inst = tmp_path / "market.txt"
    inst.write_text(
        "AAPL\t2016-01-01\t2026-04-30\n"
        "MSFT\t2016-01-01\t2026-04-30\n"
        "OLD\t2016-01-01\t2021-06-01\n",
        encoding="utf-8",
    )

    n = mod._extend_market_instruments(inst, mod._market_instruments_max_end(inst), "2026-05-05")

    assert n == 2
    assert inst.read_text(encoding="utf-8").splitlines() == [
        "AAPL\t2016-01-01\t2026-05-05",
        "MSFT\t2016-01-01\t2026-05-05",
        "OLD\t2016-01-01\t2021-06-01",
    ]


def test_feature_missing_ratio_and_failure_count(tmp_path):
    mod = _load_module()
    features = tmp_path / "features"
    features.mkdir()
    (features / "AAPL.csv").write_text("date,x\n2026-04-30,1\n", encoding="utf-8")

    assert mod._feature_missing_ratio(features, ["AAPL", "MSFT"]) == 0.5
    assert mod._failure_count({"failures": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]}) == 2
    assert mod._failure_count({}) == 0


def test_reuse_raw_report_summarizes_existing_files(tmp_path):
    mod = _load_module()
    raw = tmp_path / "sep"
    raw.mkdir()
    pd.DataFrame({"date": ["2026-04-30", "2026-05-01"]}).to_csv(raw / "AAPL.csv", index=False)
    pd.DataFrame({"date": ["2026-04-29"]}).to_csv(raw / "MSFT.csv", index=False)

    report = mod._reuse_raw_report(raw, ["AAPL", "MSFT", "NVDA"], "date")

    assert report["status"] == "reused_raw"
    assert report["tickers"] == 3
    assert report["files_with_dates"] == 2
    assert report["missing_or_empty"] == 1
    assert report["latest_date"] == "2026-05-01"
    assert report["failures"] == []


def test_prepare_price_delta_uses_closeadj_without_double_split_adjustment(tmp_path):
    mod = _load_module()
    raw = tmp_path / "BKNG.csv"
    pd.DataFrame(
        {
            "date": ["2026-03-30", "2026-04-06"],
            "open": [162.60, 166.25],
            "high": [166.0, 178.0],
            "low": [160.0, 165.0],
            "close": [164.70, 176.19],
            "volume": [10_211_000, 8_191_000],
            "closeadj": [164.70, 176.19],
            "closeunadj": [4117.51, 176.19],
        }
    ).to_csv(raw, index=False)

    out = mod._prepare_price_delta(raw, "2026-03-29")

    assert out["date"].tolist() == ["2026-03-30", "2026-04-06"]
    assert out["close"].tolist() == [164.70, 176.19]
    assert out["open"].tolist() == [162.60, 166.25]
    assert round(float(out.loc[0, "factor"]), 6) == round(164.70 / 4117.51, 6)


def test_prepare_sfp_qlib_frame_adds_symbol_and_adjusted_prices(tmp_path):
    mod = _load_module()
    raw = tmp_path / "SPY.csv"
    pd.DataFrame(
        {
            "ticker": ["SPY", "SPY"],
            "date": ["2026-01-05", "2026-01-02"],
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.0, 100.0],
            "volume": [2000, 1000],
            "closeadj": [50.5, 50.0],
            "closeunadj": [101.0, 100.0],
        }
    ).to_csv(raw, index=False)

    out = mod._prepare_sfp_qlib_frame(raw)

    assert out["symbol"].tolist() == ["SPY", "SPY"]
    assert out["date"].tolist() == ["2026-01-02", "2026-01-05"]
    assert out["close"].tolist() == [50.0, 50.5]
    assert out["open"].tolist() == [50.0, 50.5]


def test_write_sfp_qlib_files_dedupes_tickers_and_skips_missing(tmp_path):
    mod = _load_module()
    raw_dir = tmp_path / "raw_sfp"
    out_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": ["2026-01-02"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
            "closeadj": [100.0],
            "closeunadj": [100.0],
        }
    ).to_csv(raw_dir / "SPY.csv", index=False)

    written = mod._write_sfp_qlib_files(raw_dir, ["spy", "SPY", "QQQ"], out_dir)

    assert written == 1
    assert sorted(p.name for p in out_dir.glob("*.csv")) == ["SPY.csv"]
