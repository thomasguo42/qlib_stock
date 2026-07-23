import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "download_fmp_data.py"
    spec = importlib.util.spec_from_file_location("download_fmp_data", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_url_redaction_hides_api_key():
    mod = _load_module()
    url = mod.build_url("earnings", {"symbol": "AAPL"}, "secret-key")

    assert "secret-key" in url
    redacted = mod.redact_url(url)
    assert "secret-key" not in redacted
    assert "apikey=%3Credacted%3E" in redacted


def test_records_from_payload_handles_common_shapes():
    mod = _load_module()

    assert mod.records_from_payload([{"a": 1}, "bad"]) == [{"a": 1}]
    assert mod.records_from_payload({"data": [{"a": 2}]}) == [{"a": 2}]
    assert mod.records_from_payload({"symbol": "AAPL", "value": 1}) == [{"symbol": "AAPL", "value": 1}]
    assert mod.records_from_payload(None) == []


def test_load_symbols_accepts_qlib_instruments_file(tmp_path):
    mod = _load_module()
    symbols_file = tmp_path / "market.txt"
    symbols_file.write_text(
        "ticker\n"
        "AAPL\t2016-01-04\t2026-05-05\n"
        "msft\t2016-01-04\t2026-05-05\n"
        "# comment\n"
        "NVDA,TSLA\n",
        encoding="utf-8",
    )

    assert mod.load_symbols("AAPL,JPM", str(symbols_file)) == ["AAPL", "JPM", "MSFT", "NVDA", "TSLA"]


def test_dot_class_symbols_use_hyphen_for_fmp_request_but_keep_original_file_symbol():
    mod = _load_module()

    assert mod.fmp_request_symbol("BRK.B") == "BRK-B"
    params = mod.dataset_params("earnings", mod.ENDPOINTS["earnings"], symbol="BF.B")
    assert params["symbol"] == "BF-B"


def test_analyst_audit_flags_no_asof_columns():
    mod = _load_module()
    records = [
        {
            "symbol": "AAPL",
            "date": "2026-09-30",
            "fiscalYear": 2026,
            "epsAvg": 8.1,
            "revenueAvg": 400_000_000_000,
        }
    ]

    audit = mod.audit_records("analyst_estimates_annual", records)

    assert audit["rows"] == 1
    assert audit["schema_hash"]
    assert audit["asof_candidate_columns"] == []
    assert audit["raw_date_candidate_columns"] == ["date"]
    assert any("as-of" in warning for warning in audit["warnings"])
    assert not any("estimate-value" in warning for warning in audit["warnings"])


def test_date_span_and_pit_columns_detect_dates():
    mod = _load_module()
    records = [
        {"symbol": "AAPL", "date": "2026-01-03", "epsActual": 1.0},
        {"symbol": "AAPL", "date": "2026-04-03", "epsActual": 1.1},
    ]

    audit = mod.audit_records("earnings", records)

    assert audit["asof_candidate_columns"] == ["date"]
    assert audit["date_spans"]["date"] == {"min": "2026-01-03", "max": "2026-04-03"}


def test_download_dataset_writes_redacted_error_payload(monkeypatch, tmp_path):
    mod = _load_module()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "fetch_json", _raise)
    entries = mod.download_dataset(
        "earnings",
        mod.ENDPOINTS["earnings"],
        api_key="secret-key",
        out_root=tmp_path,
        symbols=["AAPL"],
        start="2022-01-01",
        end="2026-05-12",
        limit=1000,
        timeout=1,
        retries=0,
        sleep=0,
    )

    assert entries[0]["status"] == "ERROR"
    raw_path = Path(entries[0]["raw_path"])
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert "secret-key" not in json.dumps(payload)
    assert payload["url"].endswith("apikey=%3Credacted%3E")


def test_dry_run_does_not_require_api_key(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_fmp_data.py",
            "--symbols",
            "AAPL",
            "--datasets",
            "earnings",
            "--out_root",
            str(tmp_path),
            "--dry_run",
        ],
    )

    assert mod.main() == 0
