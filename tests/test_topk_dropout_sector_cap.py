import pandas as pd

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


def test_apply_sector_cap_demotes_names_after_sector_limit():
    strategy = object.__new__(TopkDropoutStrategy)
    strategy.max_sector_count = 2
    strategy._sector_map = {
        "A": "Technology",
        "B": "Technology",
        "C": "Technology",
        "D": "Financials",
        "E": "Financials",
    }

    scores = pd.Series(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        index=["A", "B", "C", "D", "E"],
    )

    capped = strategy._apply_sector_cap(scores)
    top4 = capped.sort_values(ascending=False).head(4).index.tolist()

    assert top4 == ["A", "B", "D", "E"]
    assert capped.loc["C"] < scores.min()


def test_apply_sector_cap_is_noop_without_map_or_limit():
    strategy = object.__new__(TopkDropoutStrategy)
    strategy.max_sector_count = None
    strategy._sector_map = {"A": "Technology"}
    scores = pd.Series([1.0], index=["A"])

    pd.testing.assert_series_equal(strategy._apply_sector_cap(scores), scores)


def test_load_sector_map_preserves_tickers_with_same_sector(tmp_path):
    path = tmp_path / "tickers.csv"
    path.write_text("ticker,sector\nA,Technology\nB,Technology\nC,Healthcare\n", encoding="utf-8")
    strategy = object.__new__(TopkDropoutStrategy)
    strategy.sector_map_csv = str(path)
    strategy.sector_ticker_col = "ticker"
    strategy.sector_col = "sector"
    strategy._sector_map = None

    sector_map = strategy._load_sector_map()

    assert sector_map == {"A": "Technology", "B": "Technology", "C": "Healthcare"}


def test_apply_feature_min_percentile_demotes_small_names(monkeypatch):
    strategy = object.__new__(TopkDropoutStrategy)
    strategy.feature_score_weights = {}
    strategy.feature_min_percentiles = {"$log_marketcap_q": 0.75}

    def fake_features(instruments, start_time, end_time, fields):
        return pd.DataFrame({"$log_marketcap_q": [10.0, 20.0, 30.0, 40.0]}, index=instruments)

    strategy._load_feature_controls = fake_features
    scores = pd.Series([4.0, 3.0, 2.0, 1.0], index=["A", "B", "C", "D"])

    adjusted = strategy._apply_feature_score_controls(
        scores,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
    )

    assert adjusted.sort_values(ascending=False).head(2).index.tolist() == ["C", "D"]
    assert adjusted.loc["A"] < scores.min()
    assert adjusted.loc["B"] < scores.min()


def test_apply_feature_score_weight_blends_cross_sectional_zscores(monkeypatch):
    strategy = object.__new__(TopkDropoutStrategy)
    strategy.feature_score_weights = {"$log_marketcap_q": 2.0}
    strategy.feature_min_percentiles = {}

    def fake_features(instruments, start_time, end_time, fields):
        return pd.DataFrame({"$log_marketcap_q": [10.0, 20.0, 30.0]}, index=instruments)

    strategy._load_feature_controls = fake_features
    scores = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])

    adjusted = strategy._apply_feature_score_controls(
        scores,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
    )

    assert adjusted.idxmax() == "C"
