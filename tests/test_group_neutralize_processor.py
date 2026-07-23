from pathlib import Path

import pandas as pd

from qlib.contrib.data.processor import (
    BenchmarkExcessLabel,
    DownsideAdjustedExcessLabel,
    DualHorizonPortfolioUtilityLabel,
    GroupNeutralize,
    PortfolioUtilityExcessLabel,
    ResidualForwardReturnLabel,
    RiskTierFilter,
    VolScaledExcessLabel,
)


def test_group_neutralize_demeans_by_date_and_group(tmp_path: Path):
    group_map = tmp_path / "groups.csv"
    group_map.write_text(
        "ticker,sector\nA,Tech\nB,Tech\nC,Energy\nD,Energy\n",
        encoding="utf-8",
    )
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-02"), "D"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    df = pd.DataFrame([1.0, 3.0, 10.0, 14.0], index=idx, columns=cols)

    out = GroupNeutralize(str(group_map), fields_group="label", min_group_size=2)(df.copy())

    assert out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")] == -1.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")] == 1.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "C"), ("label", "LABEL0")] == -2.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "D"), ("label", "LABEL0")] == 2.0


def test_group_neutralize_small_group_falls_back_to_daily_mean(tmp_path: Path):
    group_map = tmp_path / "groups.csv"
    group_map.write_text("ticker,sector\nA,Tech\nB,Energy\n", encoding="utf-8")
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples([("feature", "X")])
    df = pd.DataFrame([1.0, 5.0], index=idx, columns=cols)

    out = GroupNeutralize(str(group_map), fields_group="feature", min_group_size=2)(df.copy())

    assert out.loc[(pd.Timestamp("2024-01-02"), "A"), ("feature", "X")] == -2.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "B"), ("feature", "X")] == 2.0


def test_risk_tier_filter_keeps_only_high_risk_training_samples():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "LOW"),
            (pd.Timestamp("2024-01-02"), "MED"),
            (pd.Timestamp("2024-01-02"), "BETA"),
            (pd.Timestamp("2024-01-02"), "VOL"),
            (pd.Timestamp("2024-01-02"), "MISS"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("feature", "RISK_BETA_QQQ_63D"),
            ("feature", "RISK_VOL_20D"),
            ("label", "LABEL0"),
        ]
    )
    df = pd.DataFrame(
        [
            [0.8, 0.20, 0.01],
            [1.2, 0.45, 0.02],
            [1.7, 0.40, 0.03],
            [0.9, 0.75, 0.04],
            [float("nan"), 0.75, 0.05],
        ],
        index=idx,
        columns=cols,
    )

    out = RiskTierFilter(tier="high")(df)

    assert out.index.get_level_values("instrument").tolist() == ["BETA", "VOL"]
    assert RiskTierFilter(tier="high").is_for_infer() is False


def test_risk_tier_filter_can_select_non_high_samples():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "LOW"),
            (pd.Timestamp("2024-01-02"), "MED"),
            (pd.Timestamp("2024-01-02"), "HIGH"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            ("feature", "RISK_BETA_QQQ_63D"): [0.6, 1.2, 2.0],
            ("feature", "RISK_VOL_20D"): [0.2, 0.5, 0.8],
        },
        index=idx,
    )

    out = RiskTierFilter(tier="non_high")(df)

    assert out.index.get_level_values("instrument").tolist() == ["LOW", "MED"]


def test_residual_forward_return_label_subtracts_beta_adjusted_benchmark(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("label", "LABEL0"),
            ("feature", "RISK_BETA_SPY_63D"),
        ]
    )
    df = pd.DataFrame([[0.10, 0.5], [0.10, 1.5]], index=idx, columns=cols)

    out = ResidualForwardReturnLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        beta_feature="$risk_beta_spy_63d",
    )(df.copy())

    # Forward benchmark from 2024-01-03 to 2024-01-04 is -1%.
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 0.105
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")], 8) == 0.115


def test_residual_forward_return_label_uses_fallback_beta_when_feature_missing(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    df = pd.DataFrame([0.10], index=idx, columns=cols)

    out = ResidualForwardReturnLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        beta_feature="RISK_BETA_SPY_63D",
        fallback_beta=0.5,
    )(df.copy())

    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 0.105


def test_vol_scaled_excess_label_uses_point_in_time_vol_feature(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("label", "LABEL0"),
            ("feature", "RISK_VOL_20D"),
        ]
    )
    df = pd.DataFrame([[0.10, 0.02], [0.10, 0.04]], index=idx, columns=cols)

    out = VolScaledExcessLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        vol_feature="$risk_vol_20d",
    )(df.copy())

    # Forward benchmark from 2024-01-03 to 2024-01-04 is -1%, so excess is 11%.
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 5.5
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")], 8) == 2.75


def test_vol_scaled_excess_label_falls_back_when_feature_missing(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    df = pd.DataFrame([0.10], index=idx, columns=cols)

    out = VolScaledExcessLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        vol_feature="RISK_VOL_20D",
        fallback_vol=0.05,
    )(df.copy())

    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 2.2


def test_downside_adjusted_excess_label_penalizes_negative_excess(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    df = pd.DataFrame([0.10, -0.02], index=idx, columns=cols)

    out = DownsideAdjustedExcessLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        downside_penalty=1.0,
    )(df.copy())

    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 0.11
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")], 8) == -0.02


def test_portfolio_utility_excess_label_penalizes_downside_and_volatility(tmp_path: Path):
    benchmark = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("label", "LABEL0"),
            ("feature", "RISK_VOL_20D"),
        ]
    )
    df = pd.DataFrame([[0.10, 0.02], [-0.02, 0.04]], index=idx, columns=cols)

    out = PortfolioUtilityExcessLabel(
        str(bench_path),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
        vol_feature="$risk_vol_20d",
        downside_penalty=1.0,
        volatility_penalty=0.5,
    )(df.copy())

    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 0.10
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")], 8) == -0.04


def test_dual_horizon_portfolio_utility_label_collapses_to_single_target(tmp_path: Path):
    benchmark = pd.Series(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
    )
    bench_path = tmp_path / "bench.pkl"
    benchmark.to_pickle(bench_path)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["datetime", "instrument"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("label", "LABEL0"),
            ("label", "LABEL1"),
            ("feature", "RISK_VOL_20D"),
        ]
    )
    df = pd.DataFrame([[0.10, 0.20, 0.02], [-0.02, 0.04, 0.04]], index=idx, columns=cols)

    out = DualHorizonPortfolioUtilityLabel(
        str(bench_path),
        label_horizon_days=[1, 2],
        label_weights=[0.25, 0.75],
        label_ref_start_days=1,
        benchmark_kind="return",
        vol_feature="RISK_VOL_20D",
        downside_penalty=1.0,
        volatility_penalty=0.0,
    )(df.copy())

    assert ("label", "LABEL1") not in out.columns
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "A"), ("label", "LABEL0")], 8) == 0.175
    assert round(out.loc[(pd.Timestamp("2024-01-02"), "B"), ("label", "LABEL0")], 8) == 0.02


def test_benchmark_label_loads_numpy_core_compat_pickle(monkeypatch, tmp_path: Path):
    import qlib.contrib.data.processor as processor_mod

    calls = []
    benchmark = pd.Series(
        [0.00, 0.01, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    def fake_read_pickle(path):
        calls.append(path)
        if len(calls) == 1:
            raise ModuleNotFoundError("No module named 'numpy._core.numeric'", name="numpy._core.numeric")
        return benchmark

    monkeypatch.setattr(processor_mod.pd, "read_pickle", fake_read_pickle)

    proc = BenchmarkExcessLabel(
        str(tmp_path / "old_numpy.pkl"),
        label_horizon_days=1,
        label_ref_start_days=1,
        benchmark_kind="return",
    )

    assert len(calls) == 2
    assert round(proc._bench_forward.loc[pd.Timestamp("2024-01-02")], 8) == 0.02
