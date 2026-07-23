import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "audit_us_hedge_feasibility.py"
    spec = importlib.util.spec_from_file_location("audit_us_hedge_feasibility", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_strategy_hedge_tickers_dedupes_inline_and_file_values(tmp_path):
    mod = _load_module()
    tickers = tmp_path / "hedges.txt"
    tickers.write_text("psq\nRWM\nPSQ\n#comment\n", encoding="utf-8")
    cfg = {
        "port_analysis_config": {
            "strategy": {
                "kwargs": {
                    "hedge_tickers": "SH, PSQ",
                    "hedge_tickers_file": str(tickers),
                }
            }
        }
    }

    out, path = mod._strategy_hedge_tickers(cfg)

    assert out == ["SH", "PSQ", "RWM"]
    assert path == str(tickers)


def test_audit_rows_require_qlib_history_and_inverse_membership(tmp_path):
    mod = _load_module()
    raw_dir = tmp_path / "sfp"
    raw_dir.mkdir()
    (raw_dir / "SH.csv").write_text("date\n2024-01-02\n2024-01-03\n", encoding="utf-8")
    close = pd.DataFrame(
        {"SH": [10.0, 10.1, 10.2], "SPY": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    rows = mod._audit_rows(
        ["SH", "SPY", "MISSING"],
        raw_sfp_dir=raw_dir,
        qlib_close=close,
        min_history_days=2,
        max_missing_ratio=0.0,
        require_inverse=True,
    )

    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["SH"]["usable"] is True
    assert by_ticker["SPY"]["qlib_ok"] is True
    assert by_ticker["SPY"]["inverse_ok"] is False
    assert by_ticker["SPY"]["usable"] is False
    assert by_ticker["MISSING"]["usable"] is False
