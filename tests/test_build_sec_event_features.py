import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_sec_event_features.py"
    spec = importlib.util.spec_from_file_location("build_sec_event_features", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _submission_payload():
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["0001", "0002", "0003"],
                "form": ["8-K", "10-Q/A", "4"],
                "filingDate": ["2026-01-02", "2026-01-03", "2026-01-04"],
                "acceptanceDateTime": [
                    "2026-01-02T21:00:00.000Z",
                    "2026-01-03T10:00:00.000Z",
                    "2026-01-04T18:30:00.000Z",
                ],
                "isXBRL": [0, 1, 0],
                "items": ["2.02,9.01", "", ""],
            }
        }
    }


def test_recent_records_from_submission_payload_expands_parallel_arrays():
    mod = _load_module()

    rows = mod._records_from_submission_payload(_submission_payload(), "AAPL")

    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAPL"
    assert rows[1]["form"] == "10-Q/A"


def test_load_submission_events_includes_archive_payloads(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(raw / "submissions" / "AAPL.json", _submission_payload())
    _write_json(
        raw / "submissions_archives" / "AAPL" / "CIK0000320193-submissions-001.json",
        {
            "form": ["10-K"],
            "filingDate": ["2020-02-01"],
            "acceptanceDateTime": ["2020-02-01T12:00:00.000Z"],
            "isXBRL": [1],
            "items": [""],
        },
    )

    df = mod.load_submission_events(raw, ["AAPL"])

    assert sorted(df["form"].astype(str).tolist()) == ["10-K", "10-Q/A", "4", "8-K"]


def test_build_feature_frames_lags_sec_filings_and_classifies_forms(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(raw / "submissions" / "AAPL.json", _submission_payload())

    frames = mod.build_feature_frames(
        raw,
        start="2026-01-01",
        end="2026-01-06",
        windows=[3],
        availability_lag_days=1,
        prefix="sec",
    )

    out = frames["AAPL"].set_index("date")
    assert out.loc["2026-01-02", "sec_filing_count_daily"] == 0.0
    assert out.loc["2026-01-03", "sec_8k_count_daily"] == 1.0
    assert out.loc["2026-01-03", "sec_material_8k_count_daily"] == 1.0
    assert out.loc["2026-01-03", "sec_8k_item_count_daily"] == 2.0
    assert out.loc["2026-01-04", "sec_10q_count_daily"] == 1.0
    assert out.loc["2026-01-04", "sec_amend_count_daily"] == 1.0
    assert out.loc["2026-01-04", "sec_xbrl_count_daily"] == 1.0
    assert out.loc["2026-01-03", "sec_filing_days_since_latest"] == 0.0
    assert out.loc["2026-01-06", "sec_filing_count_3d_sum"] == 2.0
    assert "sec_alpha_event_composite" in out.columns


def test_extract_cik_from_sharadar_secfilings_url():
    mod = _load_module()

    assert mod.extract_cik_from_secfilings("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193") == "320193"


def test_main_dry_run_builds_without_provider_calendar(tmp_path, monkeypatch):
    mod = _load_module()
    raw = tmp_path / "raw"
    _write_json(raw / "submissions" / "MSFT.json", _submission_payload())
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("MSFT\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sec_event_features.py",
            "--raw_root",
            str(raw),
            "--symbols_file",
            str(symbols_file),
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-05",
            "--dry_run",
        ],
    )

    assert mod.main() == 0
