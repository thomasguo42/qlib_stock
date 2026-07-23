import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "build_external_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_external_index_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_ticker_maps_icic_to_ixic():
    mod = _load_module()

    assert mod.canonical_ticker("ICIC") == "IXIC"
    assert mod.NASDAQ_SYMBOLS["IXIC"] == "COMP"


def test_returns_from_close_cleans_sorts_and_names_series():
    mod = _load_module()
    close = pd.Series(
        [100.0, 101.0, 103.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-05"]),
    )

    returns = mod.returns_from_close(close, name="bench_ixic")

    assert returns.name == "bench_ixic"
    assert list(returns.index) == list(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    assert returns.round(6).tolist() == [-0.009901, 0.03]
