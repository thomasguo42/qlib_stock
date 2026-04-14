import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "data_collector" / "sharadar" / "collector.py"
    spec = importlib.util.spec_from_file_location("sharadar_collector", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
_extract_table_names = _mod._extract_table_names
_parse_key_value_csv = _mod._parse_key_value_csv


def test_parse_key_value_csv():
    out = _parse_key_value_csv("ticker=AAPL,date.gte=2020-01-01,dimension=MRQ")
    assert out["ticker"] == "AAPL"
    assert out["date.gte"] == "2020-01-01"
    assert out["dimension"] == "MRQ"


def test_extract_table_names():
    payload = {
        "datatables": [
            {"datatable_code": "SEP"},
            {"name": "SF1"},
            {"name": "Sharadar Fundamentals"},
        ],
        "nested": {"table": "TICKERS"},
    }
    names = _extract_table_names(payload)
    assert "SEP" in names
    assert "SF1" in names
    assert "TICKERS" in names
