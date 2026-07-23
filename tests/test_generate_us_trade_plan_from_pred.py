import importlib.util
import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "generate_us_trade_plan_from_pred.py"
    spec = importlib.util.spec_from_file_location("generate_us_trade_plan_from_pred", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_positions_csv_preserves_holding_count(tmp_path):
    mod = _load_module()
    fp = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "symbol": ["aapl", "msft"],
            "shares": [10, 5],
            "price": [100.0, 200.0],
            "days_held": [12, 3],
        }
    ).to_csv(fp, index=False)

    positions, missing = mod._load_positions_csv(fp)

    assert missing is False
    assert positions["AAPL"]["amount"] == 10.0
    assert positions["AAPL"]["price"] == 100.0
    assert positions["AAPL"]["count_day"] == 12
    assert positions["MSFT"]["count_day"] == 3


def test_load_positions_csv_reports_missing_holding_count(tmp_path):
    mod = _load_module()
    fp = tmp_path / "positions.csv"
    pd.DataFrame({"ticker": ["AAPL"], "amount": [7]}).to_csv(fp, index=False)

    positions, missing = mod._load_positions_csv(fp)
    assert missing is True
    assert "count_day" not in positions["AAPL"]

    positions, missing = mod._load_positions_csv(fp, default_holding_days=10)
    assert missing is False
    assert positions["AAPL"]["count_day"] == 10


def test_score_date_helpers_use_score_date_not_trade_date():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-04-24"), "AAPL")], names=["datetime", "instrument"]
    )
    pred = pd.DataFrame({"score": [1.25]}, index=idx)

    assert mod._has_score_for_date(pred, pd.Timestamp("2026-04-24"))
    assert not mod._has_score_for_date(pred, pd.Timestamp("2026-04-27"))
    assert mod._score_for_order(pred, pd.Timestamp("2026-04-24"), "aapl") == 1.25


def test_positions_require_explicit_cash():
    mod = _load_module()

    cash, err = mod._resolve_cash(argparse.Namespace(cash=None, capital=100000.0), has_positions=True)
    assert cash is None
    assert "--cash must be explicit" in err

    cash, err = mod._resolve_cash(argparse.Namespace(cash=0.0, capital=100000.0), has_positions=True)
    assert cash == 0.0
    assert err is None

    cash, err = mod._resolve_cash(argparse.Namespace(cash=None, capital=100000.0), has_positions=False)
    assert cash == 100000.0
    assert err is None
