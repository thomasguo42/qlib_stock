import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "ensemble_us_sharadar_predictions.py"
    spec = importlib.util.spec_from_file_location("ensemble_us_sharadar_predictions", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pred(rows):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(dt), inst) for dt, inst, _ in rows],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"score": [score for _, _, score in rows]}, index=idx)


def test_confidence_gate_uses_shifted_history():
    mod = _load_module()
    primary = _pred(
        [
            ("2026-01-01", "A", 3.0),
            ("2026-01-01", "B", 2.0),
            ("2026-01-01", "C", 1.0),
            ("2026-01-02", "A", 3.1),
            ("2026-01-02", "B", 2.0),
            ("2026-01-02", "C", 1.0),
            ("2026-01-05", "A", 3.2),
            ("2026-01-05", "B", 2.0),
            ("2026-01-05", "C", 1.0),
            ("2026-01-06", "A", 0.011),
            ("2026-01-06", "B", 0.010),
            ("2026-01-06", "C", 0.009),
        ]
    )

    gate = mod.confidence_gate_frame(primary, min_history_days=2, min_std_ratio=0.25)

    assert bool(gate.loc[pd.Timestamp("2026-01-05"), "fallback"]) is False
    assert bool(gate.loc[pd.Timestamp("2026-01-06"), "fallback"]) is True
    assert pd.isna(gate.loc[pd.Timestamp("2026-01-01"), "reference_std"])


def test_combine_predictions_falls_back_to_defensive_scores():
    mod = _load_module()
    primary = _pred(
        [
            ("2026-01-01", "A", 3.0),
            ("2026-01-01", "B", 1.0),
            ("2026-01-02", "A", 0.011),
            ("2026-01-02", "B", 0.010),
        ]
    )
    defensive = _pred(
        [
            ("2026-01-01", "A", 0.0),
            ("2026-01-01", "B", 5.0),
            ("2026-01-02", "A", 0.0),
            ("2026-01-02", "B", 5.0),
        ]
    )
    gate = pd.DataFrame(
        {"fallback": [False, True]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )

    out = mod.combine_predictions(primary, defensive, gate=gate)

    assert out.loc[(pd.Timestamp("2026-01-01"), "A"), "score"] == 3.0
    assert out.loc[(pd.Timestamp("2026-01-02"), "B"), "score"] == 5.0


def test_load_prediction_normalizes_index_and_score_column(tmp_path):
    mod = _load_module()
    raw = _pred([("2026-01-01", "aapl", 1.0), ("2026-01-01", "msft", 2.0)]).rename(columns={"score": "pred"})
    path = tmp_path / "pred.pkl"
    raw.to_pickle(path)

    loaded = mod.load_prediction(path)

    assert loaded.index.names == ["datetime", "instrument"]
    assert "AAPL" in loaded.index.get_level_values("instrument")
    assert loaded.columns.tolist() == ["score"]


def test_regime_confidence_gate_uses_shifted_benchmark_state():
    mod = _load_module()
    dates = pd.bdate_range("2026-01-01", periods=8)
    rows = []
    for dt in dates:
        rows.extend([(dt, "A", 3.0), (dt, "B", 2.0), (dt, "C", 1.0)])
    primary = _pred(rows)
    benchmark = pd.Series([0.02, 0.02, -0.20, 0.00, 0.00, 0.00, 0.00, 0.00], index=dates)

    gate = mod.regime_confidence_gate_frame(
        primary,
        benchmark_returns=benchmark,
        min_history_days=2,
        min_std_ratio=0.0,
        trend_window=2,
        min_trend_return=-0.10,
        drawdown_window=3,
        max_drawdown=-0.10,
        vol_window=2,
        max_ann_vol=None,
    )

    assert bool(gate.loc[dates[2], "regime_fallback"]) is False
    assert bool(gate.loc[dates[3], "regime_fallback"]) is True
    assert bool(gate.loc[dates[3], "fallback"]) is True
