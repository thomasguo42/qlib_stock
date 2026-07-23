import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "sweep_us_sharadar_strategy_params.py"
    spec = importlib.util.spec_from_file_location("sweep_us_sharadar_strategy_params", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_list_helpers():
    mod = _load_module()

    assert mod._parse_float_list("0.60, 0.67") == [0.60, 0.67]
    assert mod._parse_int_list("20, 40") == [20, 40]


def test_strategy_variant_sets_release_sweep_controls():
    mod = _load_module()
    base = {
        "topk": 40,
        "benchmark_topn": 7,
        "benchmark_core_weight": 0.80,
        "max_holdings": 30,
        "max_sector_weight": 0.40,
    }

    out = mod._strategy_variant(
        base,
        benchmark_core_weight=0.67,
        topk=20,
        max_sector_weight=0.35,
        max_weight=0.30,
        benchmark_max_weight=1.0,
        max_turnover=0.06,
        max_active_weight=0.04,
    )

    assert out["topk"] == 20
    assert out["benchmark_core_weight"] == 0.67
    assert out["max_sector_weight"] == 0.35
    assert out["max_weight"] == 0.30
    assert out["benchmark_max_weight"] == 1.0
    assert out["max_turnover"] == 0.06
    assert out["max_active_weight"] == 0.04
    assert out["max_holdings"] >= 27
    assert base["topk"] == 40
