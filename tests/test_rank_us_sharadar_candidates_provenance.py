from pathlib import Path
import importlib.util


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rank_us_sharadar_candidates.py"
    spec = importlib.util.spec_from_file_location("rank_us_sharadar_candidates", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
_config_in_cmd = _mod._config_in_cmd
_infer_run_dir_from_pred = _mod._infer_run_dir_from_pred
_run_is_dirty = _mod._run_is_dirty


def test_infer_run_dir_from_pred(tmp_path: Path):
    run_dir = tmp_path / "mlruns" / "123" / "abc123"
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True)
    pred = art_dir / "pred.pkl"
    pred.write_bytes(b"x")

    inferred = _infer_run_dir_from_pred(pred)
    assert inferred == run_dir


def test_config_in_cmd(tmp_path: Path):
    cfg = tmp_path / "workflow.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    assert _config_in_cmd(cfg, f"qrun {cfg.name}")
    assert _config_in_cmd(cfg, f"qrun {cfg}")
    assert not _config_in_cmd(cfg, "qrun other.yaml")


def test_run_is_dirty(tmp_path: Path):
    run_dir = tmp_path / "mlruns" / "1" / "runx"
    art = run_dir / "artifacts"
    art.mkdir(parents=True)

    fp = art / "code_status.txt"
    fp.write_text("On branch main\nnothing to commit, working tree clean\n", encoding="utf-8")
    assert _run_is_dirty(run_dir) is False

    fp.write_text("Changes not staged for commit:\n  modified: file.py\n", encoding="utf-8")
    assert _run_is_dirty(run_dir) is True
