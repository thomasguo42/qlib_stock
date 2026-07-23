import numpy as np
import pandas as pd
import pytest

from qlib.contrib.model.score import (
    FeatureWeightedScoreModel,
    ICSelectedScoreModel,
    RegimeSleeveScoreModel,
    StackedSignalScoreModel,
)


class DummyDataset:
    def __init__(self, features, learn_frame=None):
        self.features = features
        self.learn_frame = learn_frame

    def prepare(self, segment, col_set="feature", data_key=None):
        if col_set == ["feature", "label"]:
            return self.learn_frame if self.learn_frame is not None else self.features
        return self.features


def test_feature_weighted_score_model_scores_by_date_with_aliases():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-05"), "A"),
            (pd.Timestamp("2026-01-05"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame(
        {
            "RISK_RET_63D": [1.0, 3.0, 4.0, 2.0],
            ("feature", "RISK_VOL_20D"): [3.0, 1.0, 1.0, 3.0],
        },
        index=idx,
    )
    model = FeatureWeightedScoreModel({"$risk_ret_63d": 1.0, "RISK_VOL_20D": -1.0})

    model.fit(DummyDataset(features))
    pred = model.predict(DummyDataset(features))

    assert pred.loc[(pd.Timestamp("2026-01-02"), "B")] > pred.loc[(pd.Timestamp("2026-01-02"), "A")]
    assert pred.loc[(pd.Timestamp("2026-01-05"), "A")] > pred.loc[(pd.Timestamp("2026-01-05"), "B")]


def test_feature_weighted_score_model_rejects_missing_features():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-01-02"), "A")], names=["datetime", "instrument"])
    features = pd.DataFrame({"RISK_RET_63D": [1.0]}, index=idx)
    model = FeatureWeightedScoreModel({"RISK_RET_63D": 1.0, "RISK_VOL_20D": -1.0})

    model.fit(DummyDataset(features))

    with pytest.raises(KeyError, match="RISK_VOL_20D"):
        model.predict(DummyDataset(features))


def test_feature_weighted_score_model_switches_weights_by_regime_feature():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-05"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame(
        {
            "MKT_QQQ_RET_63D": [0.05, -0.02],
            "RISK_RET_63D": [2.0, 2.0],
            "RISK_VOL_20D": [3.0, 3.0],
        },
        index=idx,
    )
    model = FeatureWeightedScoreModel(
        {"RISK_VOL_20D": -1.0},
        normalize_by_date=False,
        regime_feature="MKT_QQQ_RET_63D",
        risk_on_weights={"RISK_RET_63D": 1.0},
        risk_off_weights={"RISK_VOL_20D": -1.0},
    )

    model.fit(DummyDataset(features))
    pred = model.predict(DummyDataset(features))

    assert pred.loc[(pd.Timestamp("2026-01-02"), "A")] == pytest.approx(2.0)
    assert pred.loc[(pd.Timestamp("2026-01-05"), "A")] == pytest.approx(-3.0)


def test_feature_weighted_score_model_requires_complete_regime_configuration():
    with pytest.raises(ValueError, match="regime_feature"):
        FeatureWeightedScoreModel(
            {"RISK_VOL_20D": -1.0},
            regime_feature="MKT_QQQ_RET_63D",
            risk_on_weights={"RISK_RET_63D": 1.0},
        )


def test_regime_sleeve_score_model_routes_states_and_blends_sleeves():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-05"), "A"),
            (pd.Timestamp("2026-01-05"), "B"),
            (pd.Timestamp("2026-01-06"), "A"),
            (pd.Timestamp("2026-01-06"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame(
        {
            "MKT_QQQ_RET_63D_LAG1": [0.08, 0.08, -0.08, -0.08, -0.08, -0.08],
            "MKT_QQQ_RET_20D_LAG1": [0.04, 0.04, -0.03, -0.03, 0.04, 0.04],
            "MKT_QQQ_DD_126D_LAG1": [-0.01, -0.01, -0.12, -0.12, -0.12, -0.12],
            "MKT_QQQ_VOL_20D_LAG1": [0.15, 0.15, 0.20, 0.20, 0.20, 0.20],
            "MKT_BREADTH_RET63_POS_LAG1": [0.70, 0.70, 0.30, 0.30, 0.30, 0.30],
            "RISK_RET_63D": [4.0, 1.0, 4.0, 1.0, 4.0, 1.0],
            "RISK_VOL_20D": [4.0, 1.0, 4.0, 1.0, 4.0, 1.0],
        },
        index=idx,
    )
    model = RegimeSleeveScoreModel(
        sleeves={
            "growth": {"RISK_RET_63D": 1.0},
            "defensive": {"RISK_VOL_20D": -1.0},
        },
        state_weights={
            "risk_on": {"growth": 1.0},
            "risk_off": {"defensive": 1.0},
            "recovery": {"growth": 0.5, "defensive": 0.5},
            "chop": {"defensive": 1.0},
        },
    )

    model.fit(DummyDataset(features))
    pred = model.predict(DummyDataset(features))

    assert pred.loc[(pd.Timestamp("2026-01-02"), "A")] > pred.loc[(pd.Timestamp("2026-01-02"), "B")]
    assert pred.loc[(pd.Timestamp("2026-01-05"), "B")] > pred.loc[(pd.Timestamp("2026-01-05"), "A")]
    assert pred.loc[(pd.Timestamp("2026-01-06"), "A")] == pytest.approx(0.0)
    assert pred.loc[(pd.Timestamp("2026-01-06"), "B")] == pytest.approx(0.0)


def test_regime_sleeve_score_model_rejects_unknown_sleeve_reference():
    with pytest.raises(ValueError, match="unknown sleeves"):
        RegimeSleeveScoreModel(
            sleeves={"growth": {"RISK_RET_63D": 1.0}},
            state_weights={"chop": {"missing": 1.0}},
        )


def test_ic_selected_score_model_selects_train_only_positive_feature():
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    instruments = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rank = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    features = pd.DataFrame(
        {
            "GOOD": rank,
            "NOISE": np.resize([0.0, 1.0, 0.0, 1.0], len(idx)),
        },
        index=idx,
    )
    label = pd.Series(rank, index=idx, name="LABEL0")
    learn_frame = pd.concat({"feature": features, "label": pd.DataFrame({"LABEL0": label})}, axis=1)
    model = ICSelectedScoreModel(
        max_features=1,
        min_selected_features=1,
        min_abs_ic=0.50,
        min_ic_days=5,
        min_daily_count=3,
        min_coverage=0.90,
        min_same_sign_years=1,
        min_recent_signed_ic=-1.0,
        selection_step_days=1,
    )

    model.fit(DummyDataset(features, learn_frame=learn_frame))
    pred = model.predict(DummyDataset(features, learn_frame=learn_frame))

    assert model.selected_weights_ == {"GOOD": pytest.approx(1.0)}
    assert pred.loc[(dates[0], "D")] > pred.loc[(dates[0], "A")]


def test_ic_selected_score_model_uses_negative_weight_for_inverse_signal():
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    instruments = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rank = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    features = pd.DataFrame({"INVERSE": -rank}, index=idx)
    label = pd.Series(rank, index=idx, name="LABEL0")
    learn_frame = pd.concat({"feature": features, "label": pd.DataFrame({"LABEL0": label})}, axis=1)
    model = ICSelectedScoreModel(
        max_features=1,
        min_selected_features=1,
        min_abs_ic=0.50,
        min_ic_days=5,
        min_daily_count=3,
        min_coverage=0.90,
        min_same_sign_years=1,
        min_recent_signed_ic=-1.0,
        selection_step_days=1,
    )

    model.fit(DummyDataset(features, learn_frame=learn_frame))
    pred = model.predict(DummyDataset(features, learn_frame=learn_frame))

    assert model.selected_weights_ == {"INVERSE": pytest.approx(-1.0)}
    assert pred.loc[(dates[0], "D")] > pred.loc[(dates[0], "A")]


def test_ic_selected_score_model_routes_regime_specific_weights():
    dates = pd.date_range("2025-01-02", periods=10, freq="B")
    instruments = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rank = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    regime_by_date = np.array([-0.10] * 5 + [0.10] * 5, dtype=float)
    regime = np.repeat(regime_by_date, len(instruments))
    label_values = np.where(regime >= 0.03, rank, -rank)
    features = pd.DataFrame(
        {
            "MKT_QQQ_RET_63D_LAG1": regime,
            "ON": rank,
            "OFF": -rank,
            "BASE": label_values,
        },
        index=idx,
    )
    label = pd.Series(label_values, index=idx, name="LABEL0")
    learn_frame = pd.concat({"feature": features, "label": pd.DataFrame({"LABEL0": label})}, axis=1)
    model = ICSelectedScoreModel(
        max_features=1,
        min_selected_features=1,
        min_abs_ic=0.50,
        min_ic_days=5,
        min_daily_count=3,
        min_coverage=0.90,
        min_same_sign_years=1,
        min_recent_signed_ic=-1.0,
        selection_step_days=1,
        regime_feature="MKT_QQQ_RET_63D_LAG1",
        regime_min_ic_days=3,
    )

    model.fit(DummyDataset(features, learn_frame=learn_frame))
    pred = model.predict(DummyDataset(features, learn_frame=learn_frame))

    assert set(model.selected_weights_by_state_) >= {"risk_on", "risk_off"}
    assert pred.loc[(dates[-1], "D")] > pred.loc[(dates[-1], "A")]
    assert pred.loc[(dates[0], "A")] > pred.loc[(dates[0], "D")]


def test_stacked_signal_score_model_learns_train_only_sleeve_weights():
    dates = pd.date_range("2025-01-02", periods=10, freq="B")
    instruments = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rank = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    features = pd.DataFrame(
        {
            "FMP_GOOD": rank,
            "NOISE": np.resize([0.0, 1.0, 0.0, 1.0], len(idx)),
        },
        index=idx,
    )
    label = pd.Series(rank, index=idx, name="LABEL0")
    learn_frame = pd.concat({"feature": features, "label": pd.DataFrame({"LABEL0": label})}, axis=1)
    model = StackedSignalScoreModel(
        sleeves={
            "fmp": {"FMP_GOOD": 1.0},
            "noise": {"NOISE": 1.0},
        },
        max_sleeves=1,
        min_ic_days=5,
        min_daily_count=3,
        min_coverage=0.90,
        min_recent_signed_ic=-1.0,
    )

    model.fit(DummyDataset(features, learn_frame=learn_frame))
    pred = model.predict(DummyDataset(features, learn_frame=learn_frame))

    assert set(model.selected_sleeve_weights_) == {"fmp"}
    assert model.selected_sleeve_weights_["fmp"] == pytest.approx(1.0)
    assert pred.loc[(dates[0], "D")] > pred.loc[(dates[0], "A")]


def test_stacked_signal_score_model_can_invert_negative_sleeve():
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    instruments = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rank = np.tile(np.arange(len(instruments), dtype=float), len(dates))
    features = pd.DataFrame({"INVERSE": -rank}, index=idx)
    label = pd.Series(rank, index=idx, name="LABEL0")
    learn_frame = pd.concat({"feature": features, "label": pd.DataFrame({"LABEL0": label})}, axis=1)
    model = StackedSignalScoreModel(
        sleeves={"inverse": {"INVERSE": 1.0}},
        min_ic_days=5,
        min_daily_count=3,
        min_coverage=0.90,
        min_recent_signed_ic=-1.0,
    )

    model.fit(DummyDataset(features, learn_frame=learn_frame))
    pred = model.predict(DummyDataset(features, learn_frame=learn_frame))

    assert model.selected_sleeve_weights_["inverse"] == pytest.approx(-1.0)
    assert pred.loc[(dates[0], "D")] > pred.loc[(dates[0], "A")]
