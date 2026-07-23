import pandas as pd

from qlib.contrib.strategy.benchmark_aware import (
    active_weight_metrics,
    benchmark_weights_from_marketcap,
    build_benchmark_aware_weights,
    limit_turnover_toward_target,
)
from qlib.contrib.strategy.weekly import BenchmarkAwareScoreWeightedStrategy, HedgedBenchmarkAwareScoreWeightedStrategy


def test_benchmark_aware_weights_keep_benchmark_core():
    scores = pd.Series([10.0, 9.0, 1.0, 0.0], index=["A", "B", "C", "D"])
    marketcap = pd.Series([100.0, 10.0, 500.0, 400.0], index=scores.index)
    bench = benchmark_weights_from_marketcap(marketcap, topn=4)

    alpha_only = build_benchmark_aware_weights(scores, bench, topk=2, benchmark_core_weight=0.0)
    aware = build_benchmark_aware_weights(scores, bench, topk=2, benchmark_core_weight=0.5)

    assert aware.loc["C"] > alpha_only.get("C", 0.0)
    assert aware.loc["D"] > alpha_only.get("D", 0.0)
    assert aware.sum() == 1.0


def test_benchmark_aware_max_active_weight_caps_single_name():
    scores = pd.Series([10.0, 9.0, 1.0], index=["A", "B", "C"])
    bench = pd.Series([0.02, 0.49, 0.49], index=scores.index)

    weights = build_benchmark_aware_weights(
        scores,
        bench,
        topk=1,
        benchmark_core_weight=0.0,
        max_active_weight=0.05,
    )

    assert weights.loc["A"] <= 0.07 + 1e-12
    assert abs(weights.sum() - 1.0) < 1e-12


def test_benchmark_aware_max_benchmark_weight_overrides_stock_cap_for_etf_core():
    scores = pd.Series([10.0, 9.0, 8.0], index=["A", "B", "C"])
    bench = pd.Series([1.0], index=["QQQ"])

    weights = build_benchmark_aware_weights(
        scores,
        bench,
        topk=3,
        benchmark_core_weight=0.50,
        max_weight=0.20,
        max_benchmark_weight=1.0,
    )

    assert abs(float(weights.loc["QQQ"]) - 0.50) < 1e-12
    assert weights.drop("QQQ").max() <= 0.20 + 1e-12


def test_limit_turnover_toward_target_caps_one_way_turnover():
    current = pd.Series({"A": 0.70, "B": 0.30})
    target = pd.Series({"A": 0.10, "B": 0.20, "C": 0.70})

    limited = limit_turnover_toward_target(target, current, max_turnover=0.20)
    idx = target.index.union(current.index)
    turnover = 0.5 * (limited.reindex(idx).fillna(0.0) - current.reindex(idx).fillna(0.0)).abs().sum()

    assert abs(float(limited.sum()) - 1.0) < 1e-12
    assert turnover <= 0.20 + 1e-12
    assert limited.loc["C"] < target.loc["C"]


def test_limit_turnover_toward_target_prunes_residual_positions():
    current = pd.Series({f"OLD{i}": 0.01 for i in range(100)})
    target = pd.Series({"A": 0.60, "B": 0.40})

    limited = limit_turnover_toward_target(target, current, max_turnover=0.10, max_positions=5)

    assert abs(float(limited.sum()) - 1.0) < 1e-12
    assert len(limited) <= 5
    assert {"A", "B"}.issubset(set(limited.index))


def test_active_weight_metrics_reports_benchmark_coverage_and_sector_drift():
    portfolio = pd.Series([0.60, 0.40], index=["A", "B"])
    benchmark = pd.Series([0.50, 0.25, 0.25], index=["A", "B", "C"])
    metrics = active_weight_metrics(
        portfolio,
        benchmark,
        sector_map={"A": "Tech", "B": "Tech", "C": "Health"},
    )

    assert round(metrics["active_share"], 4) == 0.25
    assert round(metrics["benchmark_weight_held"], 4) == 0.75
    assert round(metrics["max_abs_sector_active_weight"], 4) == 0.25


def test_strategy_feature_series_flattens_instrument_datetime_index():
    idx = pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2024-01-01")), ("B", pd.Timestamp("2024-01-01"))],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({"$marketcap_q": [100.0, 200.0]}, index=idx)

    out = BenchmarkAwareScoreWeightedStrategy._series_from_features(frame, "$marketcap_q")

    assert out.to_dict() == {"A": 100.0, "B": 200.0}


def test_benchmark_aware_strategy_applies_sector_cap(tmp_path):
    sector_path = tmp_path / "tickers.csv"
    sector_path.write_text("ticker,sector\nA,Tech\nB,Tech\nC,Tech\nD,Health\n", encoding="utf-8")
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=4,
        signal=pd.Series(dtype=float),
        sector_map_csv=str(sector_path),
        max_sector_weight=0.50,
    )
    score = pd.Series([5.0, 4.0, 3.0, 2.0], index=["A", "B", "C", "D"])

    adjusted = strategy._apply_sector_cap(score)

    assert adjusted.sort_values(ascending=False).head(3).index.tolist() == ["A", "B", "D"]


def test_benchmark_aware_strategy_builds_core_from_full_scored_universe():
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=1,
        signal=pd.Series(dtype=float),
        benchmark_topn=2,
        benchmark_core_weight=0.50,
        liquidity_buffer=1,
    )
    score = pd.Series([4.0, 3.0, 2.0], index=["A", "B", "C"])
    seen = []

    def fake_benchmark_weights(instruments, trade_start_time):
        seen.extend(instruments)
        return pd.Series([0.8, 0.2], index=["B", "C"])

    strategy._benchmark_weights = fake_benchmark_weights

    weights = strategy.generate_target_weight_position(
        score,
        current=None,
        trade_start_time=pd.Timestamp("2026-01-05"),
        trade_end_time=pd.Timestamp("2026-01-05"),
    )

    assert seen == ["A", "B", "C"]
    assert "C" in weights


def test_benchmark_aware_strategy_applies_feature_controls_before_selection(monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.0,
        liquidity_buffer=2,
        feature_min_percentiles={"$log_marketcap_q": 0.75},
    )
    strategy._benchmark_weights = lambda instruments, trade_start_time: pd.Series(dtype=float)

    def fake_feature_controls(instruments, start_time, end_time, fields):
        assert pd.Timestamp(start_time) == pd.Timestamp("2026-01-02")
        assert pd.Timestamp(end_time) == pd.Timestamp("2026-01-02")
        return pd.DataFrame({"$log_marketcap_q": [10.0, 20.0, 30.0, 40.0]}, index=instruments)

    strategy._load_feature_controls = fake_feature_controls
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-02"))

    weights = strategy.generate_target_weight_position(
        pd.Series([4.0, 3.0, 2.0, 1.0], index=["A", "B", "C", "D"]),
        current=None,
        trade_start_time=pd.Timestamp("2026-01-05"),
        trade_end_time=pd.Timestamp("2026-01-05"),
    )

    assert set(weights) == {"C", "D"}


def test_benchmark_aware_strategy_uses_explicit_benchmark_tickers(monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=1,
        signal=pd.Series(dtype=float),
        benchmark_tickers=["SPY", "QQQ", "MISSING"],
    )

    def fake_features(instruments, fields, start_time=None, end_time=None):
        assert instruments == ["SPY", "QQQ", "MISSING"]
        idx = pd.MultiIndex.from_tuples(
            [
                ("SPY", pd.Timestamp("2026-01-02")),
                ("QQQ", pd.Timestamp("2026-01-02")),
                ("MISSING", pd.Timestamp("2026-01-02")),
            ],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame({"$close": [500.0, 400.0, float("nan")]}, index=idx)

    monkeypatch.setattr(weekly_mod.D, "features", fake_features, raising=False)
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-02"))

    weights = strategy._benchmark_weights(["A", "B"], pd.Timestamp("2026-01-05"))

    assert weights.to_dict() == {"SPY": 0.5, "QQQ": 0.5}


def test_benchmark_aware_strategy_loads_benchmark_tickers_file(tmp_path, monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    tickers = tmp_path / "bench.txt"
    tickers.write_text("spy\n\nQQQ\nSPY\n#comment\n", encoding="utf-8")
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=1,
        signal=pd.Series(dtype=float),
        benchmark_tickers_file=str(tickers),
    )

    def fake_features(instruments, fields, start_time=None, end_time=None):
        idx = pd.MultiIndex.from_product(
            [instruments, [pd.Timestamp("2026-01-02")]],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame({"$close": [1.0] * len(idx)}, index=idx)

    monkeypatch.setattr(weekly_mod.D, "features", fake_features, raising=False)
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-02"))

    weights = strategy._benchmark_weights(["A"], pd.Timestamp("2026-01-05"))

    assert weights.to_dict() == {"QQQ": 0.5, "SPY": 0.5}


def test_benchmark_aware_strategy_respects_turnover_cap_with_current_position():
    class FakePosition:
        def get_stock_weight_dict(self, only_stock=True):
            assert only_stock is True
            return {"A": 0.70, "B": 0.30}

    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=1,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.0,
        max_turnover=0.20,
        liquidity_buffer=1,
    )
    strategy._benchmark_weights = lambda instruments, trade_start_time: pd.Series([1.0], index=["C"])
    score = pd.Series([3.0, 2.0, 1.0], index=["C", "B", "A"])

    weights = pd.Series(
        strategy.generate_target_weight_position(
            score,
            current=FakePosition(),
            trade_start_time=pd.Timestamp("2026-01-05"),
            trade_end_time=pd.Timestamp("2026-01-05"),
        )
    )
    current = pd.Series({"A": 0.70, "B": 0.30})
    idx = weights.index.union(current.index)
    turnover = 0.5 * (weights.reindex(idx).fillna(0.0) - current.reindex(idx).fillna(0.0)).abs().sum()

    assert turnover <= 0.20 + 1e-12
    assert weights.loc["C"] <= 0.20 + 1e-12


def test_benchmark_aware_strategy_dynamic_alpha_raises_core_when_alpha_is_weak():
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.70,
        dynamic_alpha_weight=True,
        alpha_quality_lower_excess=-0.02,
        alpha_quality_upper_excess=0.02,
        min_alpha_scale=0.0,
    )
    strategy._trailing_weighted_excess_return = lambda alpha, bench, trade_start_time: -0.03

    core = strategy._effective_benchmark_core_weight(
        pd.Series({"A": 0.6, "B": 0.4}),
        pd.Series({"C": 1.0}),
        pd.Timestamp("2026-01-05"),
    )

    assert core == 1.0


def test_benchmark_aware_strategy_dynamic_alpha_keeps_core_when_alpha_is_strong():
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.70,
        dynamic_alpha_weight=True,
        alpha_quality_lower_excess=-0.02,
        alpha_quality_upper_excess=0.02,
        min_alpha_scale=0.0,
    )
    strategy._trailing_weighted_excess_return = lambda alpha, bench, trade_start_time: 0.03

    core = strategy._effective_benchmark_core_weight(
        pd.Series({"A": 0.6, "B": 0.4}),
        pd.Series({"C": 1.0}),
        pd.Timestamp("2026-01-05"),
    )

    assert core == 0.70


def test_benchmark_aware_strategy_dynamic_alpha_is_neutral_without_history():
    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.70,
        dynamic_alpha_weight=True,
        min_alpha_scale=0.0,
    )
    strategy._trailing_weighted_excess_return = lambda alpha, bench, trade_start_time: None

    core = strategy._effective_benchmark_core_weight(
        pd.Series({"A": 0.6, "B": 0.4}),
        pd.Series({"C": 1.0}),
        pd.Timestamp("2026-01-05"),
    )

    assert core == 0.70


def test_benchmark_aware_strategy_dynamic_risk_reduces_exposure_on_weak_market(monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    strategy = BenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        risk_degree=0.95,
        dynamic_risk=True,
        market_index="SPY",
        market_trend_window=3,
        market_trend_thresh=-0.02,
        market_trend_penalty=0.25,
        risk_min_history=3,
        risk_smoothing_down=1.0,
    )

    idx = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2026-01-01")),
            ("SPY", pd.Timestamp("2026-01-02")),
            ("SPY", pd.Timestamp("2026-01-05")),
        ],
        names=["instrument", "datetime"],
    )
    closes = pd.DataFrame({"$close": [100.0, 98.0, 94.0]}, index=idx)

    monkeypatch.setattr(weekly_mod.D, "features", lambda *args, **kwargs: closes, raising=False)
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-05"))

    risk = strategy._compute_dynamic_risk_degree_for_date(pd.Timestamp("2026-01-06"))

    assert round(risk, 4) == 0.2375


def test_hedged_benchmark_aware_strategy_adds_long_hedge_sleeve(monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    strategy = HedgedBenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        benchmark_core_weight=0.0,
        hedge_tickers=["SH", "PSQ", "MISSING"],
        hedge_weight=0.30,
        liquidity_buffer=1,
    )

    def fake_features(instruments, fields, start_time=None, end_time=None):
        idx = pd.MultiIndex.from_product(
            [instruments, [pd.Timestamp("2026-01-02")]],
            names=["instrument", "datetime"],
        )
        close = {"SH": 10.0, "PSQ": 20.0, "MISSING": float("nan")}
        return pd.DataFrame({"$close": [close.get(inst, 1.0) for inst, _ in idx]}, index=idx)

    monkeypatch.setattr(weekly_mod.D, "features", fake_features, raising=False)
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-02"))

    weights = pd.Series(
        strategy.generate_target_weight_position(
            pd.Series([4.0, 3.0, 2.0], index=["A", "B", "C"]),
            current=None,
            trade_start_time=pd.Timestamp("2026-01-05"),
            trade_end_time=pd.Timestamp("2026-01-05"),
        )
    )

    assert round(float(weights.sum()), 8) == 1.0
    assert round(float(weights[["SH", "PSQ"]].sum()), 8) == 0.30
    assert round(float(weights["SH"]), 8) == 0.15
    assert round(float(weights[["A", "B"]].sum()), 8) == 0.70


def test_hedged_benchmark_aware_strategy_raises_hedge_on_weak_market(monkeypatch):
    import qlib.contrib.strategy.weekly as weekly_mod

    strategy = HedgedBenchmarkAwareScoreWeightedStrategy(
        topk=2,
        signal=pd.Series(dtype=float),
        market_index="SPY",
        hedge_weight=0.0,
        hedge_max_weight=0.35,
        hedge_trend_window=3,
        hedge_trend_thresh=-0.02,
        hedge_min_history=3,
        hedge_smoothing_up=1.0,
    )
    idx = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2026-01-01")),
            ("SPY", pd.Timestamp("2026-01-02")),
            ("SPY", pd.Timestamp("2026-01-05")),
        ],
        names=["instrument", "datetime"],
    )
    closes = pd.DataFrame({"$close": [100.0, 98.0, 94.0]}, index=idx)

    monkeypatch.setattr(weekly_mod.D, "features", lambda *args, **kwargs: closes, raising=False)
    monkeypatch.setattr(weekly_mod, "get_pre_trading_date", lambda trading_date, future=True: pd.Timestamp("2026-01-05"))

    hedge_weight = strategy._compute_hedge_weight_for_date(pd.Timestamp("2026-01-06"))

    assert round(hedge_weight, 8) == 0.35
