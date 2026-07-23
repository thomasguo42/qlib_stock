import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rebuild_us_sharadar_model_features.py"
    spec = importlib.util.spec_from_file_location("rebuild_us_sharadar_model_features", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stock_feature_frame_uses_available_full_history():
    mod = _load_module()
    calendar = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100.0, 160.0, len(calendar)), index=calendar, dtype="float32")
    volume = pd.Series(1_000_000.0, index=calendar, dtype="float32")
    spy = pd.Series(np.linspace(100.0, 140.0, len(calendar)), index=calendar, dtype="float32")
    qqq = pd.Series(np.linspace(100.0, 170.0, len(calendar)), index=calendar, dtype="float32")
    spy_ret = spy.pct_change().astype("float32")
    qqq_ret = qqq.pct_change().astype("float32")
    market_features = pd.DataFrame(index=calendar)
    for col in mod.REGIME_FEATURES + mod.LAGGED_REGIME_FEATURES + mod.MARKET_BREADTH_FEATURES + mod.SECTOR_REGIME_FEATURES:
        market_features[col] = 0.01
    market_features["mkt_spy_ret_63d"] = mod._ret(spy, 63)
    market_features["mkt_qqq_ret_63d"] = mod._ret(qqq, 63)
    market_features["mkt_spy_dd_63d"] = -0.02
    market_features["mkt_spy_vol_20d"] = 0.15
    market_features["mkt_xlk_ret_63d"] = market_features["mkt_spy_ret_63d"] + 0.03

    features = mod._stock_feature_frame(close, volume, spy_ret, qqq_ret, market_features, {"meta_sector_technology": 1.0})

    assert np.isfinite(features.loc[calendar[260], "risk_ret_252d"])
    assert np.isfinite(features.loc[calendar[260], "risk_relret_spy_252d"])
    assert np.isfinite(features.loc[calendar[260], "risk_relret_qqq_252d"])
    assert np.isfinite(features.loc[calendar[260], "risk_beta_qqq_63d"])
    assert "regime_beta_spy63_ret63" in features.columns
    assert "regime_sector_relret63" in features.columns
    assert np.isfinite(features.loc[calendar[260], "regime_beta_spy63_ret63"])
    assert round(float(features.loc[calendar[260], "regime_sector_relret63"]), 6) == 0.03


def test_metadata_rows_parse_scales_and_sector_flags(tmp_path):
    mod = _load_module()
    fp = tmp_path / "tickers.csv"
    pd.DataFrame(
        {
            "table": ["SEP", "SFP"],
            "ticker": ["AAPL", "SPY"],
            "sector": ["Technology", ""],
            "scalemarketcap": ["6 - Mega", ""],
            "scalerevenue": ["5 - Large", ""],
        }
    ).to_csv(fp, index=False)

    rows = mod._metadata_rows(fp)

    assert set(rows) == {"AAPL"}
    assert rows["AAPL"]["meta_scalemarketcap"] == 6.0
    assert rows["AAPL"]["meta_scalerevenue"] == 5.0
    assert rows["AAPL"]["meta_sector_technology"] == 1.0
    assert rows["AAPL"]["meta_sector_healthcare"] == 0.0


def test_market_feature_frame_records_missing_etf_counts(tmp_path):
    mod = _load_module()
    calendar = pd.bdate_range("2020-01-01", periods=80)
    pd.DataFrame(
        {
            "date": calendar.strftime("%Y-%m-%d"),
            "closeadj": np.linspace(100.0, 120.0, len(calendar)),
        }
    ).to_csv(tmp_path / "SPY.csv", index=False)

    frame, daily_returns, counts = mod._market_feature_frame(tmp_path, calendar)

    assert counts["SPY"] == len(calendar)
    assert counts["QQQ"] == 0
    assert counts["IXIC"] == 0
    assert "$SPY" not in frame.columns
    assert "mkt_spy_ret_63d" in frame.columns
    assert "mkt_spy_ret_63d_lag1" in frame.columns
    assert daily_returns["SPY"].notna().sum() > 0
    assert "QQQ" in daily_returns


def test_stock_breadth_feature_frame_uses_prior_day_values(tmp_path):
    mod = _load_module()
    calendar = pd.bdate_range("2020-01-01", periods=90)
    pd.DataFrame({"date": calendar.strftime("%Y-%m-%d"), "closeadj": np.linspace(100.0, 130.0, len(calendar))}).to_csv(
        tmp_path / "AAA.csv",
        index=False,
    )
    pd.DataFrame({"date": calendar.strftime("%Y-%m-%d"), "closeadj": np.linspace(100.0, 80.0, len(calendar))}).to_csv(
        tmp_path / "BBB.csv",
        index=False,
    )
    spy_ret63 = pd.Series(0.0, index=calendar)

    frame, counts = mod._stock_breadth_feature_frame(tmp_path, ["AAA", "BBB"], calendar, spy_ret_63d=spy_ret63)

    assert counts == {"tickers_requested": 2, "tickers_used": 2}
    assert frame.loc[calendar[63], "mkt_breadth_ret63_pos_lag1"] != frame.loc[calendar[64], "mkt_breadth_ret63_pos_lag1"]
    assert frame.loc[calendar[64], "mkt_breadth_ret63_pos_lag1"] == 0.5


def test_overwrite_feature_bins_writes_model_fields(tmp_path):
    mod = _load_module()
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2020-01-01\n2020-01-02\n2020-01-03\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    row = {"date": ["2020-01-02", "2020-01-03"]}
    for i, field in enumerate(mod.FEATURE_FIELDS, start=1):
        row[field] = [float(i), float(i + 1)]
    pd.DataFrame(row).to_csv(prepared / "AAPL.csv", index=False)

    written = mod._overwrite_feature_bins(prepared, provider)

    assert written == len(mod.FEATURE_FIELDS)
    out = provider / "features" / "aapl" / "risk_ret_20d.day.bin"
    arr = np.fromfile(out, dtype="<f")
    assert arr.tolist() == [1.0, 1.0, 2.0]
