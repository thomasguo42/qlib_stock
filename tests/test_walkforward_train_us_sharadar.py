import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qlib.data.dataset.weight import RecencyReweighter, RegimeRecencyReweighter


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "walkforward_train_us_sharadar.py"
    spec = importlib.util.spec_from_file_location("walkforward_train_us_sharadar", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gap(calendar, left_end, right_start):
    return sum(1 for dt in calendar if pd.Timestamp(left_end) < dt < pd.Timestamp(right_start))


def test_build_walkforward_specs_applies_embargo(monkeypatch):
    mod = _load_module()
    cal = list(pd.bdate_range("2021-01-01", "2021-03-31"))
    monkeypatch.setattr(mod, "_align_to_trade_day_start", lambda ts: pd.Timestamp(ts))
    monkeypatch.setattr(mod, "_align_to_trade_day_end", lambda ts: pd.Timestamp(ts))

    specs = mod._build_walkforward_specs(
        cal=cal,
        train_start=pd.Timestamp("2021-01-01"),
        test_segments=[mod.Segment(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-31"))],
        valid_days=5,
        train_lookback_days=None,
        embargo_days=2,
    )

    spec = specs[0]
    assert _gap(cal, spec.valid.end, spec.test.start) == 2
    assert _gap(cal, spec.train.end, spec.valid.start) == 2


def test_infer_label_horizon_from_config():
    mod = _load_module()
    cfg = {"data_handler_config": {"label": [["Ref($close, -11)/Ref($close, -1) - 1"], ["LABEL0"]]}}
    assert mod._infer_label_horizon(cfg) == 10


def test_override_benchmark_pkl_updates_top_level_and_task_handler():
    mod = _load_module()
    top = {
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "BenchmarkExcessLabel", "kwargs": {"benchmark_pkl": "/old.pkl"}},
            {"class": "VolScaledExcessLabel", "kwargs": {"benchmark_pkl": "/old3.pkl"}},
            {"class": "DownsideAdjustedExcessLabel", "kwargs": {"benchmark_pkl": "/old4.pkl"}},
            {"class": "PortfolioUtilityExcessLabel", "kwargs": {"benchmark_pkl": "/old5.pkl"}},
        ]
    }
    task = {
        "learn_processors": [
            {"class": "qlib.contrib.data.processor.ResidualForwardReturnLabel", "kwargs": {"benchmark_pkl": "/old2.pkl"}}
        ]
    }
    cfg = {
        "data_handler_config": top,
        "task": {"dataset": {"kwargs": {"handler": {"kwargs": task}}}},
    }

    assert mod._override_benchmark_pkl(cfg, "/new.pkl") == 5
    assert top["learn_processors"][1]["kwargs"]["benchmark_pkl"] == "/new.pkl"
    assert top["learn_processors"][2]["kwargs"]["benchmark_pkl"] == "/new.pkl"
    assert top["learn_processors"][3]["kwargs"]["benchmark_pkl"] == "/new.pkl"
    assert top["learn_processors"][4]["kwargs"]["benchmark_pkl"] == "/new.pkl"
    assert task["learn_processors"][0]["kwargs"]["benchmark_pkl"] == "/new.pkl"


def test_attach_recency_reweighter_skips_ranker_and_attaches_lgb():
    mod = _load_module()
    lgb_task = {"model": {"class": "LGBModel"}}
    ranker_task = {"model": {"class": "LGBRankerModel"}}

    assert mod._attach_recency_reweighter(lgb_task, half_life_days=63, min_weight=0.25) is True
    assert mod._attach_recency_reweighter(ranker_task, half_life_days=63, min_weight=0.25) is False
    assert lgb_task["reweighter"].half_life_days == 63
    assert "reweighter" not in ranker_task


def test_attach_regime_recency_reweighter_skips_ranker_and_attaches_lgb():
    mod = _load_module()
    lgb_task = {"model": {"class": "LGBModel"}}
    ranker_task = {"model": {"class": "LGBRankerModel"}}

    assert mod._attach_regime_recency_reweighter(
        lgb_task,
        half_life_days=63,
        min_weight=0.25,
        regime_feature="MKT_QQQ_RET_63D_LAG1",
        regime_thresholds=[-0.04, 0.03],
        date_balance=True,
        year_balance=True,
        label_tail_weight=0.5,
        label_tail_quantile=0.2,
        max_weight=4.0,
    ) is True
    assert mod._attach_regime_recency_reweighter(
        ranker_task,
        half_life_days=63,
        min_weight=0.25,
        regime_feature="MKT_QQQ_RET_63D_LAG1",
        regime_thresholds=[-0.04, 0.03],
        date_balance=True,
        year_balance=False,
        label_tail_weight=0.0,
        label_tail_quantile=0.2,
        max_weight=4.0,
    ) is False
    assert isinstance(lgb_task["reweighter"], RegimeRecencyReweighter)
    assert "reweighter" not in ranker_task


def test_model_selection_diagnostics_extracts_ic_selected_model_summary():
    mod = _load_module()

    class Model:
        selected_weights_ = {"B": -0.25, "A": 0.75}
        selected_weights_by_state_ = {"risk_off": {"C": -1.0}}
        fallback_used_ = True
        fallback_used_by_state_ = {"risk_off": False}
        selection_summary_ = pd.DataFrame(
            [
                {
                    "feature": "A",
                    "coverage": 0.95,
                    "ic_days": 200,
                    "mean_ic": 0.02,
                    "recent_mean_ic": 0.03,
                    "same_sign_years": 3,
                    "year_count": 4,
                    "worst_year_signed_ic": -0.01,
                    "selection_score": 0.015,
                }
            ]
        )
        selection_summary_by_state_ = {"risk_off": selection_summary_}

    class Recorder:
        def load_object(self, name):
            assert name == "params.pkl"
            return Model()

    diagnostics = mod._model_selection_diagnostics(Recorder())

    assert diagnostics["selected_feature_count"] == 2
    assert list(diagnostics["selected_weights"]) == ["A", "B"]
    assert diagnostics["fallback_used"] is True
    assert diagnostics["selected_weights_by_state"]["risk_off"] == {"C": -1.0}
    assert diagnostics["fallback_used_by_state"]["risk_off"] is False
    assert diagnostics["selection_summary_top_by_state"]["risk_off"][0]["feature"] == "A"
    assert diagnostics["selection_summary_top"][0]["feature"] == "A"
    assert diagnostics["selection_summary_top"][0]["ic_days"] == 200.0


def test_model_selection_diagnostics_extracts_stacked_sleeve_summary():
    mod = _load_module()

    class Model:
        selected_sleeve_weights_ = {"momentum": -0.40, "quality": 0.60}
        fallback_used_ = False
        sleeve_summary_ = pd.DataFrame(
            [
                {
                    "sleeve": "quality",
                    "coverage": 0.98,
                    "ic_days": 180,
                    "mean_ic": 0.03,
                    "recent_mean_ic": 0.04,
                    "same_sign_years": 2,
                    "year_count": 2,
                    "worst_year_signed_ic": 0.01,
                    "selection_score": 0.025,
                }
            ]
        )

    class Recorder:
        def load_object(self, name):
            assert name == "params.pkl"
            return Model()

    diagnostics = mod._model_selection_diagnostics(Recorder())

    assert diagnostics["selected_sleeve_count"] == 2
    assert list(diagnostics["selected_sleeve_weights"]) == ["quality", "momentum"]
    assert diagnostics["selected_sleeve_weights"]["momentum"] == -0.40
    assert diagnostics["fallback_used"] is False
    assert diagnostics["sleeve_summary_top"][0]["sleeve"] == "quality"
    assert diagnostics["sleeve_summary_top"][0]["ic_days"] == 180.0


def test_recency_reweighter_decays_by_observed_dates():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)

    weights = RecencyReweighter(half_life_days=1, min_weight=0.0).reweight(df)

    assert weights.iloc[-1] == 1.0
    assert round(weights.iloc[-2], 8) == 0.5
    assert round(weights.iloc[-3], 8) == 0.25


def test_regime_recency_reweighter_balances_dates_and_regimes():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            ("feature", "MKT_QQQ_RET_63D_LAG1"): [-0.08, -0.08, -0.08, 0.01, 0.08],
            ("label", "LABEL0"): [-2.0, 0.0, 2.0, 0.5, -0.5],
        },
        index=idx,
    )

    weights = RegimeRecencyReweighter(
        half_life_days=0,
        regime_feature="MKT_QQQ_RET_63D_LAG1",
        regime_thresholds=[-0.04, 0.03],
        date_balance=True,
        label_tail_weight=0.5,
        label_tail_quantile=0.34,
    ).reweight(df)

    assert len(weights) == len(df)
    assert weights.groupby(level="datetime").sum().loc[pd.Timestamp("2024-01-02")] < 2.0
    assert weights.max() <= 5.0
    assert round(float(weights.mean()), 6) == 1.0
