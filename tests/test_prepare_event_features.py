import subprocess
import sys
from pathlib import Path

import pandas as pd


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "scripts" / "data_collector" / "sharadar" / "prepare_event_features.py"


def test_prepare_event_features_directory_event_mode(tmp_path):
    in_dir = tmp_path / "sf2"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out_event"

    pd.DataFrame(
        {
            "filingdate": ["2020-01-01", "2020-01-02"],
            "transactionshares": [10, -5],
            "transactionvalue": [1000.0, -500.0],
        }
    ).to_csv(in_dir / "AAPL.csv", index=False)

    cmd = [
        sys.executable,
        str(_script_path()),
        "--input",
        str(in_dir),
        "--out_dir",
        str(out_dir),
        "--ticker_col",
        "ticker",
        "--date_col",
        "filingdate",
        "--value_cols",
        "transactionshares,transactionvalue",
        "--windows",
        "2",
        "--prefix",
        "insider",
        "--aggregation_mode",
        "event",
        "--start",
        "2020-01-01",
        "--resample_end",
        "2020-01-03",
    ]
    subprocess.check_call(cmd)

    out = pd.read_csv(out_dir / "AAPL.csv")
    assert len(out) == 3
    assert "insider_transactionshares_2d_sum" in out.columns
    # day3 has no event, so 2-day rolling sum = day2 + day3 = -5 + 0
    day3 = out.loc[out["date"] == "2020-01-03"].iloc[0]
    assert float(day3["insider_transactionshares_2d_sum"]) == -5.0


def test_prepare_event_features_snapshot_mode(tmp_path):
    in_file = tmp_path / "sf3a.csv"
    out_dir = tmp_path / "out_snapshot"

    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "calendardate": ["2020-01-01", "2020-01-03"],
            "totalvalue": [100.0, 130.0],
        }
    ).to_csv(in_file, index=False)

    cmd = [
        sys.executable,
        str(_script_path()),
        "--input",
        str(in_file),
        "--out_dir",
        str(out_dir),
        "--ticker_col",
        "ticker",
        "--date_col",
        "calendardate",
        "--value_cols",
        "totalvalue",
        "--windows",
        "2",
        "--prefix",
        "inst",
        "--aggregation_mode",
        "snapshot",
        "--start",
        "2020-01-01",
        "--resample_end",
        "2020-01-04",
    ]
    subprocess.check_call(cmd)

    out = pd.read_csv(out_dir / "AAPL.csv")
    day3 = out.loc[out["date"] == "2020-01-03"].iloc[0]
    assert abs(float(day3["inst_totalvalue_2d_chg"]) - 30.0) < 1e-6
    assert abs(float(day3["inst_totalvalue_2d_pct"]) - 0.3) < 1e-6
