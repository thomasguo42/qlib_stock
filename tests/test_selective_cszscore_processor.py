import numpy as np
import pandas as pd

from qlib.contrib.data.processor import SelectiveCSZScoreNorm


def test_selective_cszscore_norm_preserves_excluded_constant_regime_fields():
    idx = pd.MultiIndex.from_product(
        [["2024-01-02"], ["AAPL", "MSFT", "JPM"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            ("feature", "RISK_RET_20D"): [0.01, 0.03, 0.08],
            ("feature", "MKT_SPY_RET_63D"): [0.05, 0.05, 0.05],
        },
        index=idx,
    )

    out = SelectiveCSZScoreNorm(
        fields_group="feature",
        method="robust",
        exclude_prefixes=["MKT_"],
    )(df.copy())

    assert out[("feature", "MKT_SPY_RET_63D")].tolist() == [0.05, 0.05, 0.05]
    assert np.isfinite(out[("feature", "RISK_RET_20D")]).all()
    assert out[("feature", "RISK_RET_20D")].nunique() > 1


def test_selective_cszscore_norm_returns_unchanged_when_all_columns_excluded():
    idx = pd.MultiIndex.from_product(
        [["2024-01-02"], ["AAPL", "MSFT"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {("feature", "MKT_SPY_RET_63D"): [0.05, 0.05]},
        index=idx,
    )

    out = SelectiveCSZScoreNorm(fields_group="feature", exclude_prefixes=["MKT_"])(df.copy())

    pd.testing.assert_frame_equal(out, df)
