import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "refresh_us_sharadar_pit.py"
    spec = importlib.util.spec_from_file_location("refresh_us_sharadar_pit", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_does_not_require_api_key(monkeypatch, tmp_path):
    mod = _load_module()
    provider = tmp_path / "provider"
    inst = provider / "instruments"
    inst.mkdir(parents=True)
    (inst / "pit_mrq_large_idx.txt").write_text(
        "AAPL\t2020-01-01\t2026-04-30\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("NDL_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_us_sharadar_pit.py",
            "--provider_uri",
            str(provider),
            "--out_dir",
            str(tmp_path / "out"),
            "--dump_to_qlib",
            "--dry_run",
        ],
    )

    assert mod.main() == 0


def test_sf1_ticker_coverage(tmp_path):
    mod = _load_module()
    raw = tmp_path / "sf1.csv"
    raw.write_text("Ticker,datekey\nAAPL,2026-01-01\nMSFT,2026-01-01\n", encoding="utf-8")

    covered, missing_ratio = mod._sf1_ticker_coverage(raw, 4)

    assert covered == 2
    assert missing_ratio == 0.5
