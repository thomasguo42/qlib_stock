import numpy as np

from qlib.data.data import LocalPITProvider


def test_latest_period_value_uses_last_available_revision():
    records = np.array(
        [
            (20200115, 201904, 1.0, -1),
            (20200215, 201904, 2.0, -1),
            (20200220, 202001, 3.0, -1),
        ],
        dtype=[("date", "<i4"), ("period", "<i4"), ("value", "<f4"), ("_next", "<i4")],
    )
    prefix = records[: np.searchsorted(records["date"], 20200216, side="right")]

    assert LocalPITProvider._latest_period_value(prefix, 201904, np.nan) == 2.0


def test_latest_period_value_returns_default_when_missing():
    records = np.array(
        [(20200115, 201904, 1.0, -1)],
        dtype=[("date", "<i4"), ("period", "<i4"), ("value", "<f4"), ("_next", "<i4")],
    )

    assert np.isnan(LocalPITProvider._latest_period_value(records, 202001, np.nan))
