import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "validate_us_sharadar_price_adjustments.py"
    spec = importlib.util.spec_from_file_location("validate_us_sharadar_price_adjustments", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_symbol_flags_double_adjusted_split_jump():
    mod = _load_module()
    raw = pd.Series(
        [164.70, 176.19],
        index=pd.DatetimeIndex(["2026-03-30", "2026-04-06"]),
        dtype=float,
    )
    qlib = pd.Series(
        [6.587984, 176.19],
        index=pd.DatetimeIndex(["2026-03-30", "2026-04-06"]),
        dtype=float,
    )

    row = mod._compare_symbol(
        "BKNG",
        raw,
        qlib,
        max_relative_error=1e-4,
        max_return_error=1e-3,
        max_scale_change=1e-3,
        jump_threshold=0.50,
        raw_jump_threshold=0.35,
    )

    assert row["bad_price_days"] == 1
    assert row["suspicious_jump_days"] == 1
    assert row["examples"][0]["date"] == "2026-04-06"


def test_compare_symbol_accepts_matching_closeadj_series():
    mod = _load_module()
    raw = pd.Series(
        [8.035, 7.91],
        index=pd.DatetimeIndex(["2026-03-30", "2026-03-31"]),
        dtype=float,
    )
    qlib = raw.copy()

    row = mod._compare_symbol(
        "AGL",
        raw,
        qlib,
        max_relative_error=1e-4,
        max_return_error=1e-3,
        max_scale_change=1e-3,
        jump_threshold=0.50,
        raw_jump_threshold=0.35,
    )

    assert row["bad_price_days"] == 0
    assert row["suspicious_jump_days"] == 0
    assert row["common_days"] == 2


def test_compare_symbol_accepts_constant_adjustment_scale_when_returns_match():
    mod = _load_module()
    raw = pd.Series(
        [100.0, 101.0, 99.0],
        index=pd.DatetimeIndex(["2026-03-30", "2026-03-31", "2026-04-01"]),
        dtype=float,
    )
    qlib = raw / 10.0

    row = mod._compare_symbol(
        "SPLIT",
        raw,
        qlib,
        max_relative_error=1e-4,
        max_return_error=1e-3,
        max_scale_change=1e-3,
        jump_threshold=0.50,
        raw_jump_threshold=0.35,
    )

    assert row["bad_price_days"] == 0
    assert row["suspicious_jump_days"] == 0
    assert row["max_relative_error"] > 0.80


def test_compare_symbol_flags_scale_change_even_without_large_jump():
    mod = _load_module()
    raw = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.DatetimeIndex(["2026-03-30", "2026-03-31", "2026-04-01"]),
        dtype=float,
    )
    qlib = pd.Series(
        [10.0, 10.1, 20.4],
        index=raw.index,
        dtype=float,
    )

    row = mod._compare_symbol(
        "BROKEN",
        raw,
        qlib,
        max_relative_error=1e-4,
        max_return_error=1e-3,
        max_scale_change=1e-3,
        jump_threshold=0.50,
        raw_jump_threshold=0.35,
    )

    assert row["bad_price_days"] == 1
    assert row["examples"][0]["date"] == "2026-04-01"


def test_read_market_tickers_filters_instruments_outside_validation_window(tmp_path):
    mod = _load_module()
    inst_dir = tmp_path / "instruments"
    inst_dir.mkdir()
    (inst_dir / "market.txt").write_text(
        "OLD\t2016-01-04\t2020-12-31\n"
        "LIVE\t2022-01-03\t2026-04-30\n"
        "FUTURE\t2027-01-04\t2028-12-31\n",
        encoding="utf-8",
    )

    tickers = mod._read_market_tickers(
        tmp_path,
        "market",
        max_tickers=None,
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2026-04-30"),
    )

    assert tickers == ["LIVE"]
