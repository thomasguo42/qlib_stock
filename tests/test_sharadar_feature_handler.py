from qlib.contrib.data.handler_sharadar import SharadarFeatureHandler


def test_sharadar_feature_handler_builds_factor_only_feature_config():
    handler = object.__new__(SharadarFeatureHandler)
    handler.pit_fields = ["assets", "roe"]
    handler.pit_interval = "q"
    handler.extra_fields = ["$earn_yield_q", "$fcf_yield_q"]
    handler.extra_names = ["EARN_YIELD_Q", "FCF_YIELD_Q"]

    fields, names = handler.get_feature_config()

    assert fields == ["P($$assets_q)", "P($$roe_q)", "$earn_yield_q", "$fcf_yield_q"]
    assert names == ["ASSETS_Q", "ROE_Q", "EARN_YIELD_Q", "FCF_YIELD_Q"]


def test_sharadar_feature_handler_requires_at_least_one_feature():
    handler = object.__new__(SharadarFeatureHandler)
    handler.pit_fields = []
    handler.pit_interval = "q"
    handler.extra_fields = []
    handler.extra_names = []

    try:
        handler.get_feature_config()
    except ValueError as exc:
        assert "requires at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError")
