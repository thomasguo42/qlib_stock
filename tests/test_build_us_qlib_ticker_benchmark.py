import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "build_us_qlib_ticker_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_us_qlib_ticker_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_returns_from_close_sorts_deduplicates_and_names_series():
    mod = _load_module()
    close = pd.Series(
        [100.0, 102.0, 101.0, 103.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-03", "2024-01-04"]),
    )

    returns = mod._returns_from_close(close, name="bench_qqq")

    assert returns.name == "bench_qqq"
    assert list(returns.index) == list(pd.to_datetime(["2024-01-03", "2024-01-04"]))
    assert returns.round(6).tolist() == [0.01, 0.019802]


def test_extract_close_series_handles_qlib_multiindex_frame():
    mod = _load_module()
    idx = pd.MultiIndex.from_tuples(
        [
            ("QQQ", pd.Timestamp("2024-01-02")),
            ("QQQ", pd.Timestamp("2024-01-03")),
        ],
        names=["instrument", "datetime"],
    )
    features = pd.DataFrame({"$close": [100.0, 101.0]}, index=idx)

    close = mod._extract_close_series(features)

    assert list(close.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert close.tolist() == [100.0, 101.0]


def test_parse_tickers_aliases_icic_to_ixic():
    mod = _load_module()

    assert mod._parse_tickers("QQQ,SPY,ICIC,^IXIC") == ["QQQ", "SPY", "IXIC"]
