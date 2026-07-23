import pandas as pd

from qlib.contrib.model.gbdt import LGBRankerModel


def test_lgb_ranker_relevance_bins_are_per_date():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-02"), "C"),
            (pd.Timestamp("2026-01-05"), "A"),
            (pd.Timestamp("2026-01-05"), "B"),
            (pd.Timestamp("2026-01-05"), "C"),
        ],
        names=["datetime", "instrument"],
    )
    y = pd.Series([0.0, 0.1, 0.2, 1.0, -1.0, 0.0], index=idx)
    model = LGBRankerModel(label_bins=3)

    rel = model._labels_to_relevance(y)

    assert rel.loc[(pd.Timestamp("2026-01-02"), "A")] == 0
    assert rel.loc[(pd.Timestamp("2026-01-02"), "B")] == 1
    assert rel.loc[(pd.Timestamp("2026-01-02"), "C")] == 2
    assert rel.loc[(pd.Timestamp("2026-01-05"), "B")] == 0
    assert rel.loc[(pd.Timestamp("2026-01-05"), "C")] == 1
    assert rel.loc[(pd.Timestamp("2026-01-05"), "A")] == 2


def test_lgb_ranker_group_sizes_follow_sorted_dates():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-02"), "A"),
            (pd.Timestamp("2026-01-02"), "B"),
            (pd.Timestamp("2026-01-05"), "A"),
        ],
        names=["datetime", "instrument"],
    )

    assert LGBRankerModel._group_sizes(idx) == [2, 1]
