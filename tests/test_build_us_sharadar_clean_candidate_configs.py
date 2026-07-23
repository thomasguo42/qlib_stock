import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "build_us_sharadar_clean_candidate_configs.py"
    spec = importlib.util.spec_from_file_location("build_us_sharadar_clean_candidate_configs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_cfg():
    neutral_proc_feature = {
        "class": "GroupNeutralize",
        "kwargs": {"fields_group": "feature"},
    }
    neutral_proc_label = {
        "class": "GroupNeutralize",
        "kwargs": {"fields_group": "label"},
    }
    return {
        "qlib_init": {"exp_manager": {"kwargs": {"default_exp_name": "base"}}},
        "data_handler_config": {
            "infer_processors": [neutral_proc_feature, {"class": "CSZScoreNorm"}],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "BenchmarkExcessLabel", "kwargs": {"label_horizon_days": 10}},
                neutral_proc_label,
            ],
            "extra_fields": ["$roe_q"],
            "extra_names": ["ROE_Q"],
            "label": [["Ref($close, -11)/Ref($close, -1) - 1"], ["LABEL0"]],
        },
        "port_analysis_config": {"strategy": {"kwargs": {"hold_thresh": 10}}},
        "task": {
            "model": {},
            "dataset": {
                "kwargs": {
                    "handler": {"kwargs": {}},
                    "segments": {
                        "train": ["2020-01-01", "2020-12-31"],
                        "valid": ["2021-01-01", "2021-12-31"],
                        "test": ["2022-01-01", "2022-12-31"],
                    },
                }
            },
        },
    }


def test_build_variants_removes_neutralization_for_plain_value_lgb():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    cfg = variants["value6_lgb_v1"]

    assert cfg["task"]["model"]["class"] == "LGBModel"
    assert len(cfg["data_handler_config"]["extra_fields"]) == 6
    assert all(p["class"] != "GroupNeutralize" for p in cfg["data_handler_config"]["infer_processors"])
    assert all(p["class"] != "GroupNeutralize" for p in cfg["data_handler_config"]["learn_processors"])


def test_build_variants_sets_label5_horizon():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["value6_lgb_label5_v1"]

    assert cfg["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert cfg["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 5
    assert cfg["port_analysis_config"]["strategy"]["kwargs"]["hold_thresh"] == 5


def test_build_variants_adds_factor_only_handlers():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    cfg = variants["value3_factors_lgb_v1"]

    handler = cfg["task"]["dataset"]["kwargs"]["handler"]
    assert handler["class"] == "SharadarFeatureHandler"
    assert handler["module_path"] == "qlib.contrib.data.handler_sharadar"
    assert cfg["data_handler_config"]["extra_fields"] == [
        "$earn_yield_q",
        "$fcf_yield_q",
        "$ebitda_margin_q",
    ]
    assert all(p["class"] != "GroupNeutralize" for p in cfg["data_handler_config"]["infer_processors"])
    assert all(p["class"] != "GroupNeutralize" for p in cfg["data_handler_config"]["learn_processors"])


def test_build_variants_adds_supported_risk_regime_features_without_static_meta():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    cfg = variants["value6_risk_lgb_v1"]
    fields = cfg["data_handler_config"]["extra_fields"]
    benchmarkaware = variants["value6_risk_lgb_benchmarkaware_v1"]
    bench_strategy = benchmarkaware["port_analysis_config"]["strategy"]

    assert len(fields) == len(mod.VALUE6) + len(mod.RISK_REGIME)
    assert "$risk_beta_spy_63d" in fields
    assert "$mkt_qqq_ret_63d" in fields
    assert not any(str(field).startswith("$meta_") for field in fields)
    assert not any(str(field).startswith("$mkt_xl") for field in fields)
    assert bench_strategy["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert bench_strategy["kwargs"]["benchmark_core_weight"] == 0.75
    assert bench_strategy["kwargs"]["benchmark_topn"] == 100
    h5_benchmarkaware = variants["value6_risk_lgb_h5_benchmarkaware_v1"]
    assert h5_benchmarkaware["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_benchmarkaware["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 5
    assert h5_benchmarkaware["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h5_benchmarkaware["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.75
    assert "hold_thresh" not in h5_benchmarkaware["port_analysis_config"]["strategy"]["kwargs"]
    h5_lowturn = variants["value6_risk_lgb_h5_benchmarkaware_lowturn_v1"]
    h5_lowturn_kwargs = h5_lowturn["port_analysis_config"]["strategy"]["kwargs"]
    assert h5_lowturn["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_lowturn_kwargs["benchmark_core_weight"] == 0.75
    assert h5_lowturn_kwargs["max_turnover"] == 0.10
    assert h5_lowturn_kwargs["min_trade_weight"] == 0.001
    assert h5_lowturn_kwargs["max_holdings"] == 140
    assert "hold_thresh" not in h5_lowturn_kwargs
    h5_lowturn_core80 = variants["value6_risk_lgb_h5_benchmarkaware_lowturn_core80_v1"]
    h5_lowturn_core80_kwargs = h5_lowturn_core80["port_analysis_config"]["strategy"]["kwargs"]
    assert h5_lowturn_core80["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_lowturn_core80_kwargs["benchmark_core_weight"] == 0.80
    assert h5_lowturn_core80_kwargs["benchmark_topn"] == 100
    assert h5_lowturn_core80_kwargs["max_turnover"] == 0.10
    assert h5_lowturn_core80_kwargs["max_holdings"] == 140
    assert "hold_thresh" not in h5_lowturn_core80_kwargs
    h5_ranker_lowturn = variants["value6_risk_ranker_h5_benchmarkaware_lowturn_v1"]
    h5_ranker_kwargs = h5_ranker_lowturn["port_analysis_config"]["strategy"]["kwargs"]
    assert h5_ranker_lowturn["task"]["model"]["class"] == "LGBRankerModel"
    assert h5_ranker_lowturn["task"]["model"]["kwargs"]["eval_at"] == [40]
    assert h5_ranker_lowturn["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_ranker_lowturn["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 5
    assert h5_ranker_lowturn["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h5_ranker_kwargs["benchmark_core_weight"] == 0.75
    assert h5_ranker_kwargs["max_turnover"] == 0.10
    assert h5_ranker_kwargs["max_holdings"] == 140
    assert "hold_thresh" not in h5_ranker_kwargs
    h5_ranker_etfcore = variants["value6_risk_ranker_h5_etfcore_v1"]
    h5_ranker_etf_kwargs = h5_ranker_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    assert h5_ranker_etfcore["task"]["model"]["class"] == "LGBRankerModel"
    assert h5_ranker_etfcore["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_ranker_etf_kwargs["benchmark_core_weight"] == 0.75
    assert h5_ranker_etf_kwargs["benchmark_topn"] == 7
    assert h5_ranker_etf_kwargs["benchmark_tickers_file"] == "/Stock/qlib/_sfp_benchmark_tickers.txt"
    assert h5_ranker_etf_kwargs["max_weight"] == 0.20
    assert h5_ranker_etf_kwargs["max_holdings"] == 70
    assert "hold_thresh" not in h5_ranker_etf_kwargs


def test_build_variants_adds_rank_normalized_risk_label():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["value6_risk_rank_lgb_v1"]
    processors = cfg["data_handler_config"]["learn_processors"]

    assert processors[-1] == {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    assert cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"] is cfg["data_handler_config"]


def test_build_variants_adds_low_turnover_risk_strategy():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["value6_risk_rank_lowturn_lgb_v1"]
    strategy = cfg["port_analysis_config"]["strategy"]["kwargs"]

    assert strategy["n_drop"] == 4
    assert strategy["candidate_buffer"] == 4
    assert strategy["risk_target_ann"] == 0.15
    assert strategy["risk_ceiling"] == 0.75


def test_build_variants_adds_true_ranker_with_sf3a_and_selective_regime_norm():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["qv_risk_sf3a_regime_ranker_v1"]
    fields = cfg["data_handler_config"]["extra_fields"]
    processors = cfg["data_handler_config"]["infer_processors"]
    strategy = cfg["port_analysis_config"]["strategy"]["kwargs"]

    assert cfg["task"]["model"]["class"] == "LGBRankerModel"
    assert cfg["task"]["model"]["kwargs"]["objective"] == "lambdarank"
    assert not any(p.get("class") == "CSRankNorm" for p in cfg["data_handler_config"]["learn_processors"])
    assert "$inst13f_totalvalue_daily" in fields
    assert "$regime_beta_spy63_ret63" in fields
    assert "$mkt_xlk_ret_63d" in fields
    assert processors == [
        {
            "class": "SelectiveCSZScoreNorm",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "fields_group": "feature",
                "method": "robust",
                "exclude_prefixes": ["MKT_"],
            },
        }
    ]
    assert strategy["n_drop"] == 4
    assert strategy["risk_target_ann"] == 0.14
    assert strategy["risk_ceiling"] == 0.72
    assert strategy["max_sector_weight"] == 0.30
    assert strategy["sector_map_csv"] == "/root/.qlib/sharadar/raw/tickers.csv"
    benchmarkaware = mod.build_variants(_base_cfg())["qv_risk_sf3a_regime_ranker_benchmarkaware_v1"]
    bench_strategy = benchmarkaware["port_analysis_config"]["strategy"]
    bench_kwargs = bench_strategy["kwargs"]
    assert bench_strategy["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert bench_kwargs["benchmark_core_weight"] == 0.80
    assert bench_kwargs["benchmark_topn"] == 120
    assert bench_kwargs["max_sector_weight"] == 0.30


def test_build_variants_adds_true_ranker_horizon_variants():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h5 = variants["qv_risk_sf3a_regime_ranker_h5_v1"]
    h5_etfcore = variants["qv_risk_sf3a_regime_ranker_h5_etfcore_v1"]
    h5_etfactive35 = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_v1"]
    h5_etfactive35_riskguard = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_riskguard_v1"]
    h5_etfactive35_dyn = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha_v1"]
    h5_etfactive35_dyn60 = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha60_v1"]
    h5_etfactive35_dyn70 = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha70_v1"]
    h5_etfactive35_dyn85_riskguard = variants[
        "qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_v1"
    ]
    h5_etfactive35_dyn85_riskguard_dynrisk = variants[
        "qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_dynrisk_v1"
    ]
    h5_etfactive35_residual = variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_v1"]
    h5_etfactive35_residual_hedged = variants[
        "qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_hedged_v1"
    ]
    h5_etfactive31 = variants["qv_risk_sf3a_regime_ranker_h5_etfactive31_v1"]
    h5_lgb_etfcore = variants["qv_risk_sf3a_regime_lgb_h5_etfcore_v1"]
    h5_lgb_rank_etfcore = variants["qv_risk_sf3a_regime_lgb_rank_h5_etfcore_v1"]
    h5_raw_etfcore = variants["qv_risk_sf3a_regime_ranker_h5_raw_etfcore_v1"]
    h20 = variants["qv_risk_sf3a_regime_ranker_h20_v1"]
    h20_etfcore = variants["qv_risk_sf3a_regime_ranker_h20_etfcore_v1"]
    h20_residual_etfcore = variants["qv_risk_sf3a_regime_ranker_h20_residual_etfcore_v1"]
    h20_volscaled_etfcore = variants["qv_risk_sf3a_regime_ranker_h20_volscaled_etfcore_v1"]
    h60 = variants["qv_risk_sf3a_regime_ranker_h60_v1"]
    h60_etfcore = variants["qv_risk_sf3a_regime_ranker_h60_etfcore_v1"]
    h60_residual_etfcore = variants["qv_risk_sf3a_regime_ranker_h60_residual_etfcore_v1"]
    h60_lgb_rank_etfcore = variants["qv_risk_sf3a_regime_lgb_rank_h60_etfcore_v1"]

    assert h5["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 5
    assert h5["port_analysis_config"]["strategy"]["kwargs"]["hold_thresh"] == 5
    assert h5_etfcore["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h5_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_tickers_file"] == "/Stock/qlib/_sfp_benchmark_tickers.txt"
    assert h5_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_topn"] == 7
    assert h5_etfcore["port_analysis_config"]["strategy"]["kwargs"]["max_holdings"] == 70
    active35_kwargs = h5_etfactive35["port_analysis_config"]["strategy"]["kwargs"]
    assert active35_kwargs["benchmark_core_weight"] == 0.65
    assert active35_kwargs["benchmark_topn"] == 7
    assert active35_kwargs["max_active_weight"] == 0.055
    assert active35_kwargs["max_sector_weight"] == 0.35
    riskguard_kwargs = h5_etfactive35_riskguard["port_analysis_config"]["strategy"]["kwargs"]
    assert riskguard_kwargs["benchmark_core_weight"] == 0.65
    assert riskguard_kwargs["feature_score_weights"] == {
        "$log_marketcap_q": 0.15,
        "$risk_vol_20d": -0.20,
        "$risk_beta_spy_63d": -0.10,
        "$earn_yield_q": 0.07,
        "$fcf_yield_q": 0.07,
    }
    assert riskguard_kwargs["feature_min_percentiles"] == {"$log_marketcap_q": 0.20}
    active35_dyn_kwargs = h5_etfactive35_dyn["port_analysis_config"]["strategy"]["kwargs"]
    assert active35_dyn_kwargs["benchmark_core_weight"] == 0.65
    assert active35_dyn_kwargs["dynamic_alpha_weight"] is True
    assert active35_dyn_kwargs["alpha_quality_window"] == 63
    assert active35_dyn_kwargs["alpha_quality_lower_excess"] == -0.03
    assert active35_dyn_kwargs["alpha_quality_upper_excess"] == 0.04
    assert active35_dyn_kwargs["min_alpha_scale"] == 0.30
    active35_dyn60_kwargs = h5_etfactive35_dyn60["port_analysis_config"]["strategy"]["kwargs"]
    assert active35_dyn60_kwargs["benchmark_core_weight"] == 0.65
    assert active35_dyn60_kwargs["dynamic_alpha_weight"] is True
    assert active35_dyn60_kwargs["min_alpha_scale"] == 0.60
    active35_dyn70_kwargs = h5_etfactive35_dyn70["port_analysis_config"]["strategy"]["kwargs"]
    assert active35_dyn70_kwargs["benchmark_core_weight"] == 0.65
    assert active35_dyn70_kwargs["dynamic_alpha_weight"] is True
    assert active35_dyn70_kwargs["min_alpha_scale"] == 0.70
    active35_dyn85_riskguard_kwargs = h5_etfactive35_dyn85_riskguard["port_analysis_config"]["strategy"]["kwargs"]
    assert active35_dyn85_riskguard_kwargs["benchmark_core_weight"] == 0.65
    assert active35_dyn85_riskguard_kwargs["dynamic_alpha_weight"] is True
    assert active35_dyn85_riskguard_kwargs["min_alpha_scale"] == 0.85
    assert active35_dyn85_riskguard_kwargs["feature_score_weights"] == riskguard_kwargs["feature_score_weights"]
    dynrisk_kwargs = h5_etfactive35_dyn85_riskguard_dynrisk["port_analysis_config"]["strategy"]["kwargs"]
    assert dynrisk_kwargs["dynamic_alpha_weight"] is True
    assert dynrisk_kwargs["dynamic_risk"] is True
    assert dynrisk_kwargs["market_index"] == "SPY"
    assert dynrisk_kwargs["market_trend_penalty"] == 0.0
    assert dynrisk_kwargs["market_drawdown_penalty"] == 0.0
    assert dynrisk_kwargs["crash_guard"] is True
    assert dynrisk_kwargs["feature_score_weights"] == riskguard_kwargs["feature_score_weights"]
    residual_proc = h5_etfactive35_residual["data_handler_config"]["learn_processors"][1]
    assert residual_proc["class"] == "ResidualForwardReturnLabel"
    assert residual_proc["module_path"] == "qlib.contrib.data.processor"
    assert residual_proc["kwargs"]["label_horizon_days"] == 5
    assert residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    hedged_strategy = h5_etfactive35_residual_hedged["port_analysis_config"]["strategy"]
    hedged_kwargs = hedged_strategy["kwargs"]
    assert hedged_strategy["class"] == "WeeklyHedgedBenchmarkAwareScoreWeightedStrategy"
    assert hedged_kwargs["hedge_tickers_file"] == "/Stock/qlib/_sfp_hedge_tickers.txt"
    assert hedged_kwargs["hedge_max_weight"] == 0.35
    assert hedged_kwargs["market_index"] == "SPY"
    active31_kwargs = h5_etfactive31["port_analysis_config"]["strategy"]["kwargs"]
    assert active31_kwargs["benchmark_core_weight"] == 0.69
    assert active31_kwargs["benchmark_topn"] == 7
    assert active31_kwargs["max_active_weight"] == 0.050
    assert active31_kwargs["max_sector_weight"] == 0.32
    assert h5_lgb_etfcore["task"]["model"]["class"] == "LGBModel"
    assert h5_lgb_etfcore["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_lgb_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h5_lgb_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_topn"] == 7
    assert h5_lgb_rank_etfcore["task"]["model"]["class"] == "LGBModel"
    assert h5_lgb_rank_etfcore["data_handler_config"]["learn_processors"][-1] == {
        "class": "CSRankNorm",
        "kwargs": {"fields_group": "label"},
    }
    assert h5_lgb_rank_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_topn"] == 7
    assert all(
        p.get("class") != "GroupNeutralize" for p in h5_raw_etfcore["data_handler_config"]["learn_processors"]
    )
    assert h5_raw_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h20["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert h20["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 20
    assert h20["port_analysis_config"]["strategy"]["kwargs"]["hold_thresh"] == 20
    assert h20["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-12-02"
    assert h20["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-12-02"
    assert h20_etfcore["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert h20_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h20_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.80
    assert h20_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_topn"] == 7
    assert "hold_thresh" not in h20_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    residual_proc = h20_residual_etfcore["data_handler_config"]["learn_processors"][1]
    assert residual_proc["class"] == "ResidualForwardReturnLabel"
    assert residual_proc["kwargs"]["label_horizon_days"] == 20
    assert residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    vol_proc = h20_volscaled_etfcore["data_handler_config"]["learn_processors"][1]
    assert vol_proc["class"] == "VolScaledExcessLabel"
    assert vol_proc["kwargs"]["label_horizon_days"] == 20
    assert vol_proc["kwargs"]["vol_feature"] == "RISK_VOL_20D"
    assert "$risk_vol_20d" in h20_volscaled_etfcore["data_handler_config"]["extra_fields"]
    assert h60["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert h60["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 60
    assert h60["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-10-06"
    assert h60["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-10-06"
    assert h60["data_handler_config"]["fit_end_time"] == "2020-10-06"
    assert h60_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert h60_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.80
    assert h60_etfcore["port_analysis_config"]["strategy"]["kwargs"]["max_turnover"] == 0.06
    assert "hold_thresh" not in h60_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    residual_h60_proc = h60_residual_etfcore["data_handler_config"]["learn_processors"][1]
    assert residual_h60_proc["class"] == "ResidualForwardReturnLabel"
    assert residual_h60_proc["kwargs"]["label_horizon_days"] == 60
    assert h60_lgb_rank_etfcore["task"]["model"]["class"] == "LGBModel"
    assert h60_lgb_rank_etfcore["data_handler_config"]["learn_processors"][-1] == {
        "class": "CSRankNorm",
        "kwargs": {"fields_group": "label"},
    }


def test_build_variants_adds_raw_excess_true_ranker_variant():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["qv_risk_sf3a_regime_ranker_raw_v1"]
    processors = cfg["data_handler_config"]["learn_processors"]

    assert cfg["task"]["model"]["class"] == "LGBRankerModel"
    assert all(p.get("class") != "GroupNeutralize" for p in processors)
    assert cfg["qlib_init"]["exp_manager"]["kwargs"]["default_exp_name"].endswith("_raw_v1_topk40")


def test_build_variants_adds_stable20_factor_lgb_rank_candidate():
    mod = _load_module()

    cfg = mod.build_variants(_base_cfg())["stable20_factors_lgb_rank_v1"]
    fields = cfg["data_handler_config"]["extra_fields"]
    processors = cfg["data_handler_config"]["learn_processors"]
    strategy = cfg["port_analysis_config"]["strategy"]["kwargs"]
    handler = cfg["task"]["dataset"]["kwargs"]["handler"]

    assert handler["class"] == "SharadarFeatureHandler"
    assert cfg["task"]["model"]["class"] == "LGBModel"
    assert cfg["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert cfg["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 20
    assert cfg["data_handler_config"]["fit_end_time"] == "2020-12-02"
    assert cfg["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-12-02"
    assert cfg["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-12-02"
    assert processors[-1] == {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    assert "$risk_ret_252d" in fields
    assert "$inst13f_shrvalue_252d_pct" in fields
    assert "$risk_vol_63d" not in fields
    assert "$risk_beta_spy_63d" not in fields
    assert strategy["n_drop"] == 8
    assert strategy["hold_thresh"] == 20
    assert strategy["max_sector_weight"] == 0.40


def test_build_variants_adds_stable20_ranker_and_ablation_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    ranker = variants["stable20_factors_ranker_v1"]
    h10 = variants["stable10_factors_lgb_rank_v1"]
    no13f = variants["stable20_factors_lgb_rank_no13f_v1"]
    no_mom = variants["stable20_factors_lgb_rank_no_mom_v1"]
    capmom = variants["stable20_capmom_lgb_rank_v1"]
    sizeguard = variants["stable20_capmom_sizeguard_lgb_rank_v1"]
    benchmarkaware = variants["stable20_capmom_benchmarkaware_lgb_rank_v1"]
    etfcore = variants["stable20_capmom_etfcore_lgb_rank_v1"]
    sectorneutral_etfcore = variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_v1"]
    sectorneutral_h60_etfcore = variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_v1"]
    release_core67 = variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_topk20"]
    release_core70 = variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core70_topk20"]
    release_v3 = variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v3_core67_lr005_topk40"]
    growth_h60_qqq = variants[
        "stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_qqqexcess_core25_topk30"
    ]
    growth_h20_qqq = variants[
        "stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_qqqexcess_core25_topk30"
    ]
    growth_h20_abs = variants[
        "stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_abs_core25_topk30"
    ]
    growth_h20_residual = variants[
        "stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_betaresqqq_core25_topk30"
    ]
    growth_h40_abs = variants[
        "stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_abs_core25_topk30"
    ]
    growth_h40_residual = variants[
        "stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_betaresqqq_core25_topk30"
    ]
    growth_h5_qqq = variants[
        "stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_qqqexcess_core25_topk30"
    ]
    growth_h5_abs = variants[
        "stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_abs_core25_topk30"
    ]
    growth_h20_vol = variants[
        "stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_volscaledqqq_core25_topk30"
    ]
    growth_h60_vol = variants[
        "stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_volscaledqqq_core25_topk30"
    ]
    growth_h60_residual = variants[
        "stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_betaresqqq_core25_topk30"
    ]
    release_dyn = variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_dynalpha_topk20"]
    dyn_etfcore = variants["stable20_capmom_dynalpha_etfcore_lgb_rank_v1"]
    dynalpha = variants["stable20_capmom_dynalpha_benchmarkaware_lgb_rank_v1"]

    assert ranker["task"]["model"]["class"] == "LGBRankerModel"
    assert all(p.get("class") != "CSRankNorm" for p in ranker["data_handler_config"]["learn_processors"])
    assert h10["data_handler_config"]["label"][0][0] == "Ref($close, -11)/Ref($close, -1) - 1"
    assert h10["port_analysis_config"]["strategy"]["kwargs"]["hold_thresh"] == 10
    assert "$inst13f_shrvalue_252d_pct" not in no13f["data_handler_config"]["extra_fields"]
    assert "$inst13f_percentoftotal_63d_mean" not in no13f["data_handler_config"]["extra_fields"]
    assert "$risk_ret_252d" not in no_mom["data_handler_config"]["extra_fields"]
    assert "$log_marketcap_q" in capmom["data_handler_config"]["extra_fields"]
    assert "$risk_ret_20d" in capmom["data_handler_config"]["extra_fields"]
    assert "$risk_relret_spy_63d" in capmom["data_handler_config"]["extra_fields"]
    assert "$risk_vol_20d" not in capmom["data_handler_config"]["extra_fields"]
    assert sizeguard["data_handler_config"]["extra_fields"] == capmom["data_handler_config"]["extra_fields"]
    assert sizeguard["port_analysis_config"]["strategy"]["kwargs"]["feature_score_weights"] == {
        "$log_marketcap_q": 0.35
    }
    assert sizeguard["port_analysis_config"]["strategy"]["kwargs"]["feature_min_percentiles"] == {
        "$log_marketcap_q": 0.30
    }
    assert benchmarkaware["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    bench_kwargs = benchmarkaware["port_analysis_config"]["strategy"]["kwargs"]
    assert bench_kwargs["benchmark_core_weight"] == 0.80
    assert bench_kwargs["benchmark_topn"] == 120
    assert bench_kwargs["benchmark_marketcap_field"] == "$marketcap_q"
    assert bench_kwargs["sector_map_csv"] == "/root/.qlib/sharadar/raw/tickers.csv"
    assert bench_kwargs["max_sector_weight"] == 0.40
    assert "n_drop" not in bench_kwargs
    etf_kwargs = etfcore["port_analysis_config"]["strategy"]["kwargs"]
    assert etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert etf_kwargs["benchmark_core_weight"] == 0.80
    assert etf_kwargs["benchmark_topn"] == 7
    assert etf_kwargs["benchmark_tickers_file"] == "/Stock/qlib/_sfp_benchmark_tickers.txt"
    assert etf_kwargs["max_weight"] == 0.20
    assert etf_kwargs["max_holdings"] == 60
    dyn_etf_kwargs = dyn_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    sectorneutral_processors = sectorneutral_etfcore["data_handler_config"]["learn_processors"]
    assert any(p.get("class") == "GroupNeutralize" for p in sectorneutral_processors)
    assert sectorneutral_processors[-1] == {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    assert sectorneutral_etfcore["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert sectorneutral_etfcore["port_analysis_config"]["strategy"]["kwargs"]["benchmark_topn"] == 7
    assert "hold_thresh" not in sectorneutral_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    assert sectorneutral_h60_etfcore["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert sectorneutral_h60_etfcore["data_handler_config"]["learn_processors"][1]["kwargs"]["label_horizon_days"] == 60
    assert sectorneutral_h60_etfcore["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-10-06"
    assert sectorneutral_h60_etfcore["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-10-06"
    assert sectorneutral_h60_etfcore["port_analysis_config"]["strategy"]["kwargs"]["max_turnover"] == 0.06
    assert "hold_thresh" not in sectorneutral_h60_etfcore["port_analysis_config"]["strategy"]["kwargs"]
    release_kwargs = release_core67["port_analysis_config"]["strategy"]["kwargs"]
    assert release_kwargs["topk"] == 20
    assert release_kwargs["benchmark_core_weight"] == 0.67
    assert release_kwargs["max_sector_weight"] == 0.35
    assert release_kwargs["max_holdings"] == 80
    assert "hold_thresh" not in release_kwargs
    assert release_core70["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.70
    release_v3_kwargs = release_v3["port_analysis_config"]["strategy"]["kwargs"]
    release_v3_model_kwargs = release_v3["task"]["model"]["kwargs"]
    assert release_v3["qlib_init"]["exp_manager"]["kwargs"]["default_exp_name"].endswith(
        "_release_v3_core67_lr005_topk40"
    )
    assert release_v3_kwargs["topk"] == 40
    assert release_v3_kwargs["benchmark_core_weight"] == 0.67
    assert release_v3_kwargs["max_turnover"] == 0.06
    assert release_v3_kwargs["max_sector_weight"] == 0.35
    assert release_v3_kwargs["max_holdings"] == 80
    assert release_v3_model_kwargs["learning_rate"] == 0.005
    assert release_v3_model_kwargs["num_boost_round"] == 4000
    assert release_v3_model_kwargs["early_stopping_rounds"] == 200
    growth_h60_kwargs = growth_h60_qqq["port_analysis_config"]["strategy"]["kwargs"]
    growth_h60_model_kwargs = growth_h60_qqq["task"]["model"]["kwargs"]
    growth_h60_label = growth_h60_qqq["data_handler_config"]["learn_processors"][1]
    assert growth_h60_qqq["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert growth_h60_kwargs["topk"] == 30
    assert growth_h60_kwargs["benchmark_core_weight"] == 0.25
    assert growth_h60_kwargs["benchmark_topn"] == 1
    assert growth_h60_kwargs["benchmark_tickers"] == ["QQQ"]
    assert growth_h60_kwargs["benchmark_max_weight"] == 1.0
    assert "benchmark_tickers_file" not in growth_h60_kwargs
    assert growth_h60_kwargs["max_turnover"] == 0.15
    assert growth_h60_kwargs["max_sector_weight"] == 0.45
    assert growth_h60_kwargs["max_active_weight"] == 0.12
    assert growth_h60_kwargs["risk_degree"] == 1.0
    assert growth_h60_label["class"] == "BenchmarkExcessLabel"
    assert growth_h60_label["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert growth_h60_label["kwargs"]["benchmark_kind"] == "return"
    assert growth_h60_label["kwargs"]["label_horizon_days"] == 60
    assert growth_h60_model_kwargs["learning_rate"] == 0.005
    assert growth_h60_model_kwargs["num_boost_round"] == 4000
    assert growth_h60_model_kwargs["early_stopping_rounds"] == 200
    growth_h20_kwargs = growth_h20_qqq["port_analysis_config"]["strategy"]["kwargs"]
    growth_h20_label = growth_h20_qqq["data_handler_config"]["learn_processors"][1]
    assert growth_h20_qqq["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert growth_h20_kwargs["topk"] == 30
    assert growth_h20_kwargs["benchmark_core_weight"] == 0.25
    assert growth_h20_kwargs["benchmark_tickers"] == ["QQQ"]
    assert growth_h20_kwargs["benchmark_max_weight"] == 1.0
    assert growth_h20_label["kwargs"]["label_horizon_days"] == 20
    assert growth_h20_label["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    abs_processors = growth_h20_abs["data_handler_config"]["learn_processors"]
    assert growth_h20_abs["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert not any(
        p.get("class")
        in {
            "BenchmarkExcessLabel",
            "ResidualForwardReturnLabel",
            "VolScaledExcessLabel",
            "DownsideAdjustedExcessLabel",
            "PortfolioUtilityExcessLabel",
        }
        for p in abs_processors
    )
    assert abs_processors[-1] == {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    h20_residual_proc = growth_h20_residual["data_handler_config"]["learn_processors"][1]
    assert h20_residual_proc["class"] == "ResidualForwardReturnLabel"
    assert h20_residual_proc["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h20_residual_proc["kwargs"]["label_horizon_days"] == 20
    assert h20_residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    assert "$risk_beta_spy_63d" in growth_h20_residual["data_handler_config"]["extra_fields"]
    assert growth_h20_residual["data_handler_config"]["learn_processors"][-1] == {
        "class": "CSRankNorm",
        "kwargs": {"fields_group": "label"},
    }
    assert growth_h40_abs["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert growth_h40_abs["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-11-03"
    assert growth_h40_abs["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-11-03"
    assert growth_h40_abs["data_handler_config"]["fit_end_time"] == "2020-11-03"
    assert not any(
        p.get("class")
        in {
            "BenchmarkExcessLabel",
            "ResidualForwardReturnLabel",
            "VolScaledExcessLabel",
            "DownsideAdjustedExcessLabel",
            "PortfolioUtilityExcessLabel",
        }
        for p in growth_h40_abs["data_handler_config"]["learn_processors"]
    )
    h40_residual_proc = growth_h40_residual["data_handler_config"]["learn_processors"][1]
    assert growth_h40_residual["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert growth_h40_residual["task"]["dataset"]["kwargs"]["segments"]["train"][1] == "2020-11-03"
    assert growth_h40_residual["task"]["dataset"]["kwargs"]["segments"]["valid"][1] == "2021-11-03"
    assert h40_residual_proc["class"] == "ResidualForwardReturnLabel"
    assert h40_residual_proc["kwargs"]["label_horizon_days"] == 40
    assert h40_residual_proc["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h40_residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    assert "$risk_beta_spy_63d" in growth_h40_residual["data_handler_config"]["extra_fields"]
    h5_kwargs = growth_h5_qqq["port_analysis_config"]["strategy"]["kwargs"]
    h5_label = growth_h5_qqq["data_handler_config"]["learn_processors"][1]
    assert growth_h5_qqq["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert h5_label["class"] == "BenchmarkExcessLabel"
    assert h5_label["kwargs"]["label_horizon_days"] == 5
    assert h5_label["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h5_kwargs["benchmark_tickers"] == ["QQQ"]
    assert h5_kwargs["benchmark_core_weight"] == 0.25
    assert h5_kwargs["benchmark_max_weight"] == 1.0
    assert growth_h5_abs["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert not any(
        p.get("class")
        in {
            "BenchmarkExcessLabel",
            "ResidualForwardReturnLabel",
            "VolScaledExcessLabel",
            "DownsideAdjustedExcessLabel",
            "PortfolioUtilityExcessLabel",
        }
        for p in growth_h5_abs["data_handler_config"]["learn_processors"]
    )
    h20_vol_proc = growth_h20_vol["data_handler_config"]["learn_processors"][1]
    h60_vol_proc = growth_h60_vol["data_handler_config"]["learn_processors"][1]
    assert h20_vol_proc["class"] == "VolScaledExcessLabel"
    assert h20_vol_proc["kwargs"]["label_horizon_days"] == 20
    assert h20_vol_proc["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h20_vol_proc["kwargs"]["clip_abs_label"] == 8.0
    assert "$risk_vol_20d" in growth_h20_vol["data_handler_config"]["extra_fields"]
    assert h60_vol_proc["class"] == "VolScaledExcessLabel"
    assert h60_vol_proc["kwargs"]["label_horizon_days"] == 60
    h60_residual_proc = growth_h60_residual["data_handler_config"]["learn_processors"][1]
    assert h60_residual_proc["class"] == "ResidualForwardReturnLabel"
    assert h60_residual_proc["kwargs"]["label_horizon_days"] == 60
    assert h60_residual_proc["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h60_residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    assert "$risk_beta_spy_63d" in growth_h60_residual["data_handler_config"]["extra_fields"]
    dyn_release_kwargs = release_dyn["port_analysis_config"]["strategy"]["kwargs"]
    assert dyn_release_kwargs["topk"] == 20
    assert dyn_release_kwargs["benchmark_core_weight"] == 0.67
    assert dyn_release_kwargs["dynamic_alpha_weight"] is True
    assert dyn_release_kwargs["alpha_quality_lower_excess"] == -0.01
    assert dyn_release_kwargs["alpha_quality_upper_excess"] == 0.03
    assert dyn_release_kwargs["min_alpha_scale"] == 0.25
    assert dyn_etf_kwargs["benchmark_core_weight"] == 0.70
    assert dyn_etf_kwargs["benchmark_tickers_file"] == "/Stock/qlib/_sfp_benchmark_tickers.txt"
    assert dyn_etf_kwargs["dynamic_alpha_weight"] is True
    assert dyn_etf_kwargs["alpha_quality_lower_excess"] == -0.02
    assert dyn_etf_kwargs["alpha_quality_upper_excess"] == 0.04
    assert dyn_etf_kwargs["max_holdings"] == 70
    dyn_kwargs = dynalpha["port_analysis_config"]["strategy"]["kwargs"]
    assert dynalpha["port_analysis_config"]["strategy"]["class"] == "WeeklyBenchmarkAwareScoreWeightedStrategy"
    assert dyn_kwargs["benchmark_core_weight"] == 0.70
    assert dyn_kwargs["benchmark_topn"] == 160
    assert dyn_kwargs["max_turnover"] == 0.10
    assert dyn_kwargs["max_holdings"] == 180
    assert dyn_kwargs["dynamic_alpha_weight"] is True
    assert dyn_kwargs["alpha_quality_window"] == 63
    assert dyn_kwargs["alpha_quality_lower_excess"] == -0.02
    assert dyn_kwargs["alpha_quality_upper_excess"] == 0.03
    assert dyn_kwargs["min_alpha_scale"] == 0.0


def test_build_variants_adds_nextgen_growth_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h40_qqq = variants["nextgen_growth_v1_h40_qqqexcess_lgb_rank_core35_topk25"]
    h40_ranker = variants["nextgen_growth_v1_h40_qqqexcess_ranker_core35_topk25"]
    h40_spy = variants["nextgen_growth_v1_h40_spyexcess_lgb_rank_core35_topk25"]
    h40_ixic = variants["nextgen_growth_v1_h40_ixicexcess_lgb_rank_core35_topk25"]
    h40_residual = variants["nextgen_growth_v1_h40_qqqresidual_lgb_rank_core35_topk25"]
    h40_downside = variants["nextgen_growth_v1_h40_qqqdownside_lgb_rank_core35_topk25"]

    kwargs = h40_qqq["port_analysis_config"]["strategy"]["kwargs"]
    label = h40_qqq["data_handler_config"]["learn_processors"][1]
    assert h40_qqq["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert h40_qqq["task"]["model"]["class"] == "LGBModel"
    assert h40_qqq["task"]["model"]["kwargs"]["learning_rate"] == 0.005
    assert label["class"] == "BenchmarkExcessLabel"
    assert label["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert h40_qqq["data_handler_config"]["learn_processors"][-1] == {
        "class": "CSRankNorm",
        "kwargs": {"fields_group": "label"},
    }
    assert kwargs["topk"] == 25
    assert kwargs["benchmark_core_weight"] == 0.35
    assert kwargs["benchmark_tickers"] == ["QQQ"]
    assert kwargs["benchmark_max_weight"] == 1.0
    assert kwargs["dynamic_alpha_weight"] is True
    assert kwargs["alpha_quality_lower_excess"] == -0.01
    assert kwargs["alpha_quality_upper_excess"] == 0.03
    assert kwargs["max_active_weight"] == 0.10
    assert kwargs["max_sector_weight"] == 0.40

    assert h40_ranker["task"]["model"]["class"] == "LGBRankerModel"
    assert h40_ranker["task"]["model"]["kwargs"]["eval_at"] == [25, 50]
    assert all(p.get("class") != "CSRankNorm" for p in h40_ranker["data_handler_config"]["learn_processors"])
    assert h40_spy["data_handler_config"]["learn_processors"][1]["kwargs"]["benchmark_pkl"].endswith("bench_spy.pkl")
    assert h40_ixic["data_handler_config"]["learn_processors"][1]["kwargs"]["benchmark_pkl"].endswith("bench_ixic.pkl")
    assert h40_residual["data_handler_config"]["learn_processors"][1]["class"] == "ResidualForwardReturnLabel"
    assert h40_residual["data_handler_config"]["learn_processors"][1]["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    assert h40_downside["data_handler_config"]["learn_processors"][1]["class"] == "DownsideAdjustedExcessLabel"
    assert h40_downside["data_handler_config"]["learn_processors"][1]["kwargs"]["downside_penalty"] == 1.0
    assert h40_downside["data_handler_config"]["learn_processors"][1]["kwargs"]["clip_abs_label"] == 0.50


def test_build_variants_adds_nextgen_growth_v2_regime_residual_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h40 = variants["nextgen_growth_v2_h40_qqqresidual_regime_lgb_core60_topk25"]
    utility = variants["nextgen_growth_v2_h40_qqqutility_regime_lgb_core60_topk25"]
    ranker = variants["nextgen_growth_v2_h40_qqqresidual_regime_ranker_core60_topk25"]
    ranker_excess = variants["nextgen_growth_v2_h40_qqqexcess_regime_ranker_core60_topk25"]

    fields = h40["data_handler_config"]["extra_fields"]
    processors = h40["data_handler_config"]["learn_processors"]
    strategy = h40["port_analysis_config"]["strategy"]["kwargs"]
    model_kwargs = h40["task"]["model"]["kwargs"]

    assert h40["task"]["model"]["class"] == "LGBModel"
    assert model_kwargs["learning_rate"] == 0.005
    assert "$risk_beta_qqq_63d" in fields
    assert "$risk_relret_qqq_63d" in fields
    assert "$mkt_qqq_ret_63d_lag1" in fields
    assert "$mkt_breadth_ret63_pos_lag1" in fields
    assert h40["data_handler_config"]["infer_processors"] == [
        {
            "class": "SelectiveCSZScoreNorm",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "fields_group": "feature",
                "method": "robust",
                "exclude_prefixes": ["MKT_"],
            },
        }
    ]
    assert processors[1]["class"] == "ResidualForwardReturnLabel"
    assert processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert processors[1]["kwargs"]["beta_feature"] == "RISK_BETA_QQQ_63D"
    assert processors[-1] == {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    assert strategy["benchmark_core_weight"] == 0.60
    assert strategy["benchmark_tickers"] == ["QQQ"]
    assert strategy["topk"] == 25
    assert strategy["max_active_weight"] == 0.06
    assert strategy["max_turnover"] == 0.10
    assert strategy["max_sector_weight"] == 0.35
    assert strategy["min_alpha_scale"] == 0.25

    utility_proc = utility["data_handler_config"]["learn_processors"][1]
    assert utility_proc["class"] == "PortfolioUtilityExcessLabel"
    assert utility_proc["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert utility_proc["kwargs"]["downside_penalty"] == 0.50
    assert utility_proc["kwargs"]["volatility_penalty"] == 0.05

    assert ranker["task"]["model"]["class"] == "LGBRankerModel"
    assert ranker["task"]["model"]["kwargs"]["eval_at"] == [25, 50]
    assert ranker["task"]["model"]["kwargs"]["learning_rate"] == 0.015
    assert "$risk_beta_qqq_63d" in ranker["data_handler_config"]["extra_fields"]
    assert ranker["data_handler_config"]["learn_processors"][1]["class"] == "ResidualForwardReturnLabel"
    assert ranker["data_handler_config"]["learn_processors"][1]["kwargs"]["beta_feature"] == "RISK_BETA_QQQ_63D"
    assert all(
        proc.get("class") != "CSRankNorm"
        for proc in ranker["data_handler_config"]["learn_processors"]
        if isinstance(proc, dict)
    )
    assert ranker["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.60
    assert ranker["port_analysis_config"]["strategy"]["kwargs"]["min_alpha_scale"] == 0.25
    assert ranker_excess["data_handler_config"]["learn_processors"][1]["class"] == "BenchmarkExcessLabel"


def test_build_variants_adds_score_growth_candidates_from_target_audit_composite():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    sector = variants["score_growth_v1_h60_sectorneutral_composite_core20_topk30"]
    residual = variants["score_growth_v1_h60_betaresqqq_composite_core20_topk30"]

    sector_model = sector["task"]["model"]
    sector_kwargs = sector["port_analysis_config"]["strategy"]["kwargs"]
    sector_processors = sector["data_handler_config"]["learn_processors"]
    residual_processors = residual["data_handler_config"]["learn_processors"]

    assert sector_model["class"] == "FeatureWeightedScoreModel"
    assert sector_model["module_path"] == "qlib.contrib.model.score"
    assert sector_model["kwargs"]["weights"] == mod.TARGET_AUDIT_COMPOSITE_WEIGHTS
    assert sector_model["kwargs"]["normalize_by_date"] is True
    assert "$risk_vol_20d" in sector["data_handler_config"]["extra_fields"]
    assert "$risk_beta_spy_63d" in sector["data_handler_config"]["extra_fields"]
    assert sector["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert sector_processors[1]["class"] == "BenchmarkExcessLabel"
    assert sector_processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert any(p.get("class") == "GroupNeutralize" for p in sector_processors)
    assert sector_kwargs["topk"] == 30
    assert sector_kwargs["benchmark_core_weight"] == 0.20
    assert sector_kwargs["benchmark_tickers"] == ["QQQ"]
    assert sector_kwargs["max_active_weight"] == 0.12
    assert sector_kwargs["max_sector_weight"] == 0.45

    assert residual["task"]["model"]["class"] == "FeatureWeightedScoreModel"
    assert residual_processors[1]["class"] == "ResidualForwardReturnLabel"
    assert residual_processors[1]["kwargs"]["label_horizon_days"] == 60
    assert residual_processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert residual_processors[1]["kwargs"]["beta_feature"] == "RISK_BETA_SPY_63D"
    assert not any(p.get("class") == "GroupNeutralize" for p in residual_processors)


def test_build_variants_adds_regime_score_growth_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h60 = variants["score_growth_v2_h60_regimeqqq_core20_topk30"]
    h40 = variants["score_growth_v2_h40_regimeqqq_core20_topk30"]

    model = h60["task"]["model"]
    kwargs = model["kwargs"]
    fields = h60["data_handler_config"]["extra_fields"]
    processors = h60["data_handler_config"]["learn_processors"]
    infer_processors = h60["data_handler_config"]["infer_processors"]
    strategy_kwargs = h60["port_analysis_config"]["strategy"]["kwargs"]

    assert model["class"] == "FeatureWeightedScoreModel"
    assert kwargs["weights"] == mod.REGIME_RISK_OFF_WEIGHTS
    assert kwargs["regime_feature"] == "MKT_QQQ_RET_63D"
    assert kwargs["regime_threshold"] == 0.0
    assert kwargs["risk_on_weights"] == mod.REGIME_RISK_ON_WEIGHTS
    assert kwargs["risk_off_weights"] == mod.REGIME_RISK_OFF_WEIGHTS
    assert "$mkt_qqq_ret_63d" in fields
    assert "$risk_vol_20d" in fields
    assert infer_processors == [
        {
            "class": "SelectiveCSZScoreNorm",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "fields_group": "feature",
                "method": "robust",
                "exclude_prefixes": ["MKT_"],
            },
        }
    ]
    assert processors[1]["class"] == "BenchmarkExcessLabel"
    assert processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert not any(p.get("class") == "GroupNeutralize" for p in processors)
    assert not any(p.get("class") == "CSRankNorm" for p in processors)
    assert h60["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert h40["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["benchmark_core_weight"] == 0.20
    assert strategy_kwargs["benchmark_tickers"] == ["QQQ"]
    assert strategy_kwargs["max_active_weight"] == 0.12
    assert strategy_kwargs["max_sector_weight"] == 0.45


def test_build_variants_adds_regime_sleeve_score_growth_v3_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h60 = variants["score_growth_v3_h60_regimesleeve_core45_topk30"]
    h40_defensive = variants["score_growth_v3_h40_regimesleeve_core60_topk25"]

    model = h60["task"]["model"]
    kwargs = model["kwargs"]
    fields = h60["data_handler_config"]["extra_fields"]
    processors = h60["data_handler_config"]["learn_processors"]
    strategy_kwargs = h60["port_analysis_config"]["strategy"]["kwargs"]

    assert model["class"] == "RegimeSleeveScoreModel"
    assert model["module_path"] == "qlib.contrib.model.score"
    assert kwargs["sleeves"] == mod.REGIME_SLEEVE_WEIGHTS
    assert kwargs["state_weights"] == mod.REGIME_STATE_SLEEVE_WEIGHTS
    assert kwargs["trend_feature"] == "MKT_QQQ_RET_63D_LAG1"
    assert kwargs["fast_trend_feature"] == "MKT_QQQ_RET_20D_LAG1"
    assert kwargs["drawdown_feature"] == "MKT_QQQ_DD_126D_LAG1"
    assert "$mkt_qqq_ret_63d_lag1" in fields
    assert "$mkt_qqq_dd_126d_lag1" in fields
    assert "$mkt_breadth_ret63_pos_lag1" in fields
    assert "$risk_vol_63d" in fields
    assert processors[1]["class"] == "BenchmarkExcessLabel"
    assert not any(p.get("class") == "GroupNeutralize" for p in processors)
    assert not any(p.get("class") == "CSRankNorm" for p in processors)
    assert h60["data_handler_config"]["label"][0][0] == "Ref($close, -61)/Ref($close, -1) - 1"
    assert h40_defensive["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["benchmark_core_weight"] == 0.45
    assert strategy_kwargs["benchmark_tickers"] == ["QQQ"]
    assert strategy_kwargs["max_active_weight"] == 0.07
    assert strategy_kwargs["max_turnover"] == 0.10
    assert strategy_kwargs["max_sector_weight"] == 0.35
    assert strategy_kwargs["dynamic_alpha_weight"] is True
    assert strategy_kwargs["min_alpha_scale"] == 0.25
    assert strategy_kwargs["dynamic_risk"] is True
    assert strategy_kwargs["market_index"] == "QQQ"
    assert strategy_kwargs["risk_floor"] == 0.55
    assert strategy_kwargs["market_drawdown_penalty"] == 0.70
    assert strategy_kwargs["crash_guard"] is True
    assert strategy_kwargs["vol_scale"] is True
    defensive_kwargs = h40_defensive["port_analysis_config"]["strategy"]["kwargs"]
    assert defensive_kwargs["benchmark_core_weight"] == 0.60
    assert defensive_kwargs["topk"] == 25
    assert defensive_kwargs["max_active_weight"] == 0.05
    assert defensive_kwargs["min_alpha_scale"] == 0.40


def test_build_variants_adds_ic_selected_score_growth_v4_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h40 = variants["score_growth_v4_h40_qqqexcess_icselect_core45_topk30"]
    residual = variants["score_growth_v4_h40_qqqresidual_icselect_core45_topk30"]
    defensive = variants["score_growth_v4_h40_qqqexcess_icselect_core60_topk25"]

    model = h40["task"]["model"]
    kwargs = model["kwargs"]
    fields = h40["data_handler_config"]["extra_fields"]
    processors = h40["data_handler_config"]["learn_processors"]
    strategy_kwargs = h40["port_analysis_config"]["strategy"]["kwargs"]

    assert model["class"] == "ICSelectedScoreModel"
    assert model["module_path"] == "qlib.contrib.model.score"
    assert kwargs["max_features"] == 8
    assert kwargs["min_recent_signed_ic"] == 0.0
    assert kwargs["selection_step_days"] == 5
    assert kwargs["normalize_by_date"] is True
    assert "$risk_beta_qqq_63d" in fields
    assert "$mkt_qqq_ret_63d_lag1" in fields
    assert processors[1]["class"] == "BenchmarkExcessLabel"
    assert processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert not any(p.get("class") == "GroupNeutralize" for p in processors)
    assert not any(p.get("class") == "CSRankNorm" for p in processors)
    assert h40["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert strategy_kwargs["benchmark_core_weight"] == 0.45
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["max_active_weight"] == 0.07
    assert strategy_kwargs["dynamic_risk"] is True
    assert strategy_kwargs["market_index"] == "QQQ"

    residual_proc = residual["data_handler_config"]["learn_processors"][1]
    assert residual_proc["class"] == "ResidualForwardReturnLabel"
    assert residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_QQQ_63D"
    defensive_kwargs = defensive["port_analysis_config"]["strategy"]["kwargs"]
    assert defensive_kwargs["benchmark_core_weight"] == 0.60
    assert defensive_kwargs["topk"] == 25
    assert defensive_kwargs["max_active_weight"] == 0.05


def test_build_variants_adds_regime_ic_selected_score_growth_v5_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h40 = variants["score_growth_v5_h40_qqqexcess_regimeic_core45_topk30"]
    residual = variants["score_growth_v5_h40_qqqresidual_regimeic_core45_topk30"]

    model = h40["task"]["model"]
    kwargs = model["kwargs"]
    processors = h40["data_handler_config"]["learn_processors"]
    strategy_kwargs = h40["port_analysis_config"]["strategy"]["kwargs"]

    assert model["class"] == "ICSelectedScoreModel"
    assert kwargs["regime_feature"] == "MKT_QQQ_RET_63D_LAG1"
    assert kwargs["risk_on_threshold"] == 0.03
    assert kwargs["risk_off_threshold"] == -0.04
    assert kwargs["regime_min_ic_days"] == 35
    assert kwargs["max_features"] == 6
    assert kwargs["min_selected_features"] == 2
    assert kwargs["min_worst_year_signed_ic"] == -0.03
    assert processors[1]["class"] == "BenchmarkExcessLabel"
    assert not any(p.get("class") == "CSRankNorm" for p in processors)
    assert strategy_kwargs["benchmark_core_weight"] == 0.45
    assert strategy_kwargs["topk"] == 30

    residual_proc = residual["data_handler_config"]["learn_processors"][1]
    assert residual_proc["class"] == "ResidualForwardReturnLabel"
    assert residual_proc["kwargs"]["beta_feature"] == "RISK_BETA_QQQ_63D"


def test_build_variants_adds_tail_regime_ic_selected_score_growth_v6_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h40 = variants["score_growth_v6_h40_qqqexcess_tailregimeic_core45_topk30"]

    kwargs = h40["task"]["model"]["kwargs"]
    assert kwargs["regime_feature"] == "MKT_QQQ_RET_63D_LAG1"
    assert kwargs["min_topq_spread"] == 0.0
    assert kwargs["min_worst_year_topq_spread"] == -0.015
    assert kwargs["tail_weight"] == 0.75
    assert kwargs["top_quantile"] == 0.20


def test_build_variants_adds_fmp_event_growth_v7_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    fmp_only = variants["score_growth_v7_h40_qqqexcess_fmponly_regimeic_core45_topk30"]
    fmp_core = variants["score_growth_v7_h40_qqqexcess_fmpcore_regimeic_core45_topk30"]

    fmp_only_fields = fmp_only["data_handler_config"]["extra_fields"]
    fmp_core_fields = fmp_core["data_handler_config"]["extra_fields"]
    kwargs = fmp_core["task"]["model"]["kwargs"]

    assert "$mkt_qqq_ret_63d_lag1" in fmp_only_fields
    assert "$fmp_earn_count_20d_sum" in fmp_only_fields
    assert "$fmp_eps_surprise_pct_latest" in fmp_only_fields
    assert "$fmp_rating_net_bullish_63d_chg" in fmp_only_fields
    assert "$risk_ret_20d" not in fmp_only_fields
    assert "$risk_ret_20d" in fmp_core_fields
    assert "$fmp_pt_upside_63d_event_mean" in fmp_core_fields
    assert not any("estimate" in str(field).lower() for field in fmp_core_fields)
    assert not any("consensus" in str(field).lower() for field in fmp_core_fields)
    assert fmp_core["data_handler_config"]["label"][0][0] == "Ref($close, -41)/Ref($close, -1) - 1"
    assert fmp_core["data_handler_config"]["learn_processors"][1]["class"] == "BenchmarkExcessLabel"
    assert fmp_core["port_analysis_config"]["strategy"]["kwargs"]["benchmark_tickers"] == ["QQQ"]
    assert fmp_core["task"]["model"]["class"] == "ICSelectedScoreModel"
    assert kwargs["regime_feature"] == "MKT_QQQ_RET_63D_LAG1"
    assert kwargs["max_features"] == 8
    assert kwargs["min_selected_features"] == 2
    assert kwargs["min_topq_spread"] == 0.0
    assert kwargs["tail_weight"] == 0.50


def test_build_variants_adds_fmp_v2_stacked_high_qqq_overlay_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    h10 = variants["score_growth_v8_h10_qqqexcess_fmpv2_stack_core85_topk30"]
    h20 = variants["score_growth_v8_h20_qqqexcess_fmpv2_stack_core85_topk30"]
    h20_core35 = variants["score_growth_v8_h20_qqqexcess_fmpv2_stack_core35_topk30"]
    fmp_only = variants["score_growth_v8_h20_qqqexcess_fmpv2_fmponly_core90_topk25"]

    fields = h20["data_handler_config"]["extra_fields"]
    model = h20["task"]["model"]
    model_kwargs = model["kwargs"]
    strategy_kwargs = h20["port_analysis_config"]["strategy"]["kwargs"]

    assert h10["data_handler_config"]["label"][0][0] == "Ref($close, -11)/Ref($close, -1) - 1"
    assert h10["task"]["model"]["kwargs"]["min_ic_days"] == 60
    assert h10["task"]["dataset"]["kwargs"]["segments"]["train"] == ["2022-01-03", "2023-08-31"]
    assert h10["task"]["dataset"]["kwargs"]["segments"]["valid"] == ["2023-09-18", "2023-12-29"]
    assert h10["task"]["dataset"]["kwargs"]["segments"]["test"][0] == "2024-01-17"
    assert h20["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert h20["benchmark"] == "QQQ"
    assert h20["port_analysis_config"]["backtest"]["benchmark"] == "QQQ"
    assert h20["data_handler_config"]["start_time"] == "2022-01-01"
    assert h20["data_handler_config"]["fit_start_time"] == "2022-01-03"
    assert h20["data_handler_config"]["fit_end_time"] == "2023-07-31"
    assert h20["task"]["dataset"]["kwargs"]["segments"]["train"] == ["2022-01-03", "2023-07-31"]
    assert h20["task"]["dataset"]["kwargs"]["segments"]["valid"] == ["2023-09-01", "2023-12-29"]
    assert h20["task"]["dataset"]["kwargs"]["segments"]["test"][0] == "2024-02-01"
    assert h20["data_handler_config"]["learn_processors"][1]["class"] == "BenchmarkExcessLabel"
    assert h20["data_handler_config"]["learn_processors"][1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert "$fmp_alpha_event_composite" in fields
    assert "$fmp_alpha_rating_bearish_penalty" in fields
    assert "$fmp_earn_count_20d_freshness" in fields
    assert "$risk_relret_qqq_63d" in fields
    assert "$fmp_earn_count_20d_sum" not in fields
    assert not any("estimate" in str(field).lower() for field in fields)
    assert not any("consensus" in str(field).lower() for field in fields)
    assert model["class"] == "StackedSignalScoreModel"
    assert model["module_path"] == "qlib.contrib.model.score"
    assert set(model_kwargs["sleeves"]) == {"fmp_event", "quality_value", "momentum_regime"}
    assert model_kwargs["max_sleeves"] == 3
    assert model_kwargs["min_abs_ic"] == 0.001
    assert strategy_kwargs["benchmark_core_weight"] == 0.85
    assert strategy_kwargs["benchmark_tickers"] == ["QQQ"]
    assert strategy_kwargs["benchmark_max_weight"] == 1.0
    assert strategy_kwargs["dynamic_alpha_weight"] is False
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["max_active_weight"] == 0.08
    assert strategy_kwargs["max_turnover"] == 0.10
    assert "feature_score_weights" not in strategy_kwargs
    assert "feature_min_percentiles" not in strategy_kwargs
    h20_core35_kwargs = h20_core35["port_analysis_config"]["strategy"]["kwargs"]
    assert h20_core35_kwargs["benchmark_core_weight"] == 0.35
    assert h20_core35_kwargs["max_active_weight"] == 0.40
    assert h20_core35_kwargs["max_turnover"] == 0.40
    assert h20_core35_kwargs["max_holdings"] == 110
    assert "feature_score_weights" not in h20_core35_kwargs
    assert "feature_min_percentiles" not in h20_core35_kwargs

    fmp_only_model = fmp_only["task"]["model"]
    fmp_only_kwargs = fmp_only["port_analysis_config"]["strategy"]["kwargs"]
    assert set(fmp_only_model["kwargs"]["sleeves"]) == {"fmp_event"}
    assert fmp_only_kwargs["benchmark_core_weight"] == 0.90
    assert fmp_only_kwargs["topk"] == 25
    assert "$risk_ret_20d" not in fmp_only["data_handler_config"]["extra_fields"]


def test_build_variants_adds_controlled_fmp_whitelist_ablation_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    sharadar_stack = variants["score_growth_v13_h20_qqqexcess_sharadar_stack_core35_topk30"]
    white_stack = variants["score_growth_v13_h20_qqqexcess_fmpwhite_stack_core35_topk30"]
    sharadar_ranker_h5 = variants["score_growth_v14_h5_qqqexcess_sharadar_ranker_core55_topk30"]
    white_ranker_h5 = variants["score_growth_v14_h5_qqqexcess_fmpwhite_ranker_core55_topk30"]
    white_ranker_h20 = variants["score_growth_v14_h20_qqqexcess_fmpwhite_ranker_core60_topk30"]

    white_fields = white_stack["data_handler_config"]["extra_fields"]
    sharadar_fields = sharadar_stack["data_handler_config"]["extra_fields"]
    white_sleeves = white_stack["task"]["model"]["kwargs"]["sleeves"]

    assert not any(str(field).startswith("$fmp_") for field in sharadar_fields)
    assert "$fmp_alpha_earn_surprise_latest" in white_fields
    assert "$fmp_alpha_rating_bearish_penalty" in white_fields
    assert "$fmp_alpha_grade_score_latest" in white_fields
    assert "$fmp_alpha_event_composite" in white_fields
    assert "$fmp_alpha_event_coverage" not in white_fields
    assert "$fmp_alpha_grade_revision_20d" not in white_fields
    assert "$fmp_alpha_pt_upside_latest" not in white_fields
    assert "$fmp_pt_upside_latest" not in white_fields
    assert set(sharadar_stack["task"]["model"]["kwargs"]["sleeves"]) == {"quality_value", "momentum_regime"}
    assert set(white_sleeves) == {"fmp_event", "quality_value", "momentum_regime"}
    assert "FMP_ALPHA_EVENT_COVERAGE" not in white_sleeves["fmp_event"]
    assert white_stack["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.35
    assert white_stack["port_analysis_config"]["strategy"]["kwargs"]["max_active_weight"] == 0.40
    assert white_stack["task"]["dataset"]["kwargs"]["segments"]["train"] == ["2022-01-03", "2023-07-31"]

    assert sharadar_ranker_h5["task"]["model"]["class"] == "LGBRankerModel"
    assert sharadar_ranker_h5["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert not any(str(field).startswith("$fmp_") for field in sharadar_ranker_h5["data_handler_config"]["extra_fields"])
    assert white_ranker_h5["port_analysis_config"]["strategy"]["kwargs"]["dynamic_alpha_weight"] is True
    assert white_ranker_h5["port_analysis_config"]["strategy"]["kwargs"]["min_alpha_scale"] == 0.45
    assert white_ranker_h5["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.55
    assert white_ranker_h20["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert white_ranker_h20["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.60
    assert white_ranker_h20["port_analysis_config"]["strategy"]["kwargs"]["min_alpha_scale"] == 0.55


def test_build_variants_adds_v15_controlled_qqq_leadership_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    stack = variants["score_growth_v15_h20_qqqexcess_fmpleader_stack_core40_topk25"]
    core45 = variants["score_growth_v15_h20_qqqexcess_fmpleader_stack_core45_topk25"]
    regime = variants["score_growth_v15_h20_qqqexcess_fmpleader_regime_core40_topk25"]

    stack_fields = stack["data_handler_config"]["extra_fields"]
    stack_names = stack["data_handler_config"]["extra_names"]
    stack_model = stack["task"]["model"]
    stack_model_kwargs = stack_model["kwargs"]
    stack_strategy_kwargs = stack["port_analysis_config"]["strategy"]["kwargs"]
    regime_model_kwargs = regime["task"]["model"]["kwargs"]

    assert "$regime_sector_relret63" in stack_fields
    assert "$regime_sector_relret63_beta63" in stack_fields
    assert "REGIME_SECTOR_RELRET63" in stack_names
    assert "$fmp_alpha_event_composite" in stack_fields
    assert "$risk_relret_qqq_63d" in stack_fields
    assert stack["benchmark"] == "QQQ"
    assert stack["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"
    assert stack["task"]["dataset"]["kwargs"]["segments"]["train"] == ["2022-01-03", "2023-07-31"]
    assert stack["task"]["dataset"]["kwargs"]["segments"]["test"][0] == "2024-02-01"
    assert stack_model["class"] == "StackedSignalScoreModel"
    assert set(stack_model_kwargs["sleeves"]) == {"fmp_event", "quality_value", "momentum_regime", "qqq_leadership"}
    assert stack_model_kwargs["sleeves"]["qqq_leadership"]["RISK_RELRET_QQQ_63D"] == 1.0
    assert stack_model_kwargs["sleeves"]["qqq_leadership"]["REGIME_SECTOR_RELRET63"] == 0.45
    assert stack_model_kwargs["max_sleeves"] == 4
    assert stack_model_kwargs["min_selected_sleeves"] == 2
    assert stack_model_kwargs["min_abs_ic"] == 0.001
    assert stack_strategy_kwargs["benchmark_core_weight"] == 0.40
    assert stack_strategy_kwargs["benchmark_tickers"] == ["QQQ"]
    assert stack_strategy_kwargs["benchmark_max_weight"] == 1.0
    assert stack_strategy_kwargs["topk"] == 25
    assert stack_strategy_kwargs["max_active_weight"] == 0.40
    assert stack_strategy_kwargs["max_turnover"] == 0.40
    assert stack_strategy_kwargs["max_holdings"] == 110
    assert "hold_thresh" not in stack_strategy_kwargs
    assert all(
        proc.get("class") != "GroupNeutralize" or proc.get("kwargs", {}).get("fields_group") != "label"
        for proc in stack["data_handler_config"]["learn_processors"]
        if isinstance(proc, dict)
    )
    assert all(
        proc.get("class") != "CSRankNorm" or proc.get("kwargs", {}).get("fields_group") != "label"
        for proc in stack["data_handler_config"]["learn_processors"]
        if isinstance(proc, dict)
    )
    assert core45["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.45
    assert regime["task"]["model"]["class"] == "RegimeSleeveScoreModel"
    assert set(regime_model_kwargs["sleeves"]) == {"fmp_event", "quality_value", "momentum_regime", "qqq_leadership"}
    assert regime_model_kwargs["state_weights"]["risk_on"]["qqq_leadership"] > 0.30
    assert regime_model_kwargs["state_weights"]["risk_off"]["qqq_leadership"] == 0.0
    assert regime_model_kwargs["trend_feature"] == "MKT_QQQ_RET_63D_LAG1"
    assert regime["port_analysis_config"]["strategy"]["kwargs"]["benchmark_core_weight"] == 0.40


def test_build_variants_adds_fmp_v10_dynamic_risk_qqq_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    excess = variants["score_growth_v10_h5_qqqexcess_fmpv2_ranker_dynrisk_core45_topk25"]
    utility = variants["score_growth_v10_h5_qqqutility_fmpv2_ranker_dynrisk_core45_topk25"]
    lgb = variants["score_growth_v10_h5_qqqutility_fmpv2_lgb_dynrisk_core45_topk25"]
    stack = variants["score_growth_v10_h20_qqqutility_fmpv2_stack_dynrisk_core45_topk25"]

    strategy_kwargs = utility["port_analysis_config"]["strategy"]["kwargs"]
    utility_processors = utility["data_handler_config"]["learn_processors"]

    assert excess["task"]["model"]["class"] == "LGBRankerModel"
    assert excess["data_handler_config"]["learn_processors"][1]["class"] == "BenchmarkExcessLabel"
    assert utility["task"]["model"]["class"] == "LGBRankerModel"
    assert utility_processors[1]["class"] == "PortfolioUtilityExcessLabel"
    assert utility_processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert utility_processors[1]["kwargs"]["downside_penalty"] == 0.75
    assert utility_processors[1]["kwargs"]["volatility_penalty"] == 0.10
    assert not any(
        proc.get("class") == "GroupNeutralize" and proc.get("kwargs", {}).get("fields_group") == "label"
        for proc in utility_processors
        if isinstance(proc, dict)
    )
    assert strategy_kwargs["benchmark_core_weight"] == 0.45
    assert strategy_kwargs["benchmark_tickers"] == ["QQQ"]
    assert strategy_kwargs["dynamic_risk"] is True
    assert strategy_kwargs["risk_floor"] == 0.50
    assert strategy_kwargs["crash_guard"] is True
    assert strategy_kwargs["crash_return_limit"] == 0.045
    assert strategy_kwargs["feature_score_weights"]["$risk_beta_qqq_63d"] == -0.12
    assert strategy_kwargs["feature_score_weights"]["$fmp_alpha_event_composite"] == 0.04
    assert strategy_kwargs["topk"] == 25
    assert strategy_kwargs["max_active_weight"] == 0.10
    assert lgb["task"]["model"]["class"] == "LGBModel"
    assert any(
        proc.get("class") == "CSRankNorm" and proc.get("kwargs", {}).get("fields_group") == "label"
        for proc in lgb["data_handler_config"]["learn_processors"]
        if isinstance(proc, dict)
    )
    assert stack["task"]["model"]["class"] == "StackedSignalScoreModel"
    assert stack["data_handler_config"]["label"][0][0] == "Ref($close, -21)/Ref($close, -1) - 1"


def test_build_variants_adds_fmp_v11_mildrisk_candidates():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    excess = variants["score_growth_v11_h5_qqqexcess_fmpv2_ranker_mildrisk_core55_topk30"]
    utility = variants["score_growth_v11_h5_qqqutility_fmpv2_ranker_mildrisk_core55_topk30"]
    lgb = variants["score_growth_v11_h5_qqqutility_fmpv2_lgb_mildrisk_core55_topk30"]

    strategy_kwargs = utility["port_analysis_config"]["strategy"]["kwargs"]
    utility_processors = utility["data_handler_config"]["learn_processors"]

    assert excess["task"]["model"]["class"] == "LGBRankerModel"
    assert excess["data_handler_config"]["learn_processors"][1]["class"] == "BenchmarkExcessLabel"
    assert utility_processors[1]["class"] == "PortfolioUtilityExcessLabel"
    assert utility_processors[1]["kwargs"]["downside_penalty"] == 0.60
    assert utility_processors[1]["kwargs"]["volatility_penalty"] == 0.06
    assert strategy_kwargs["benchmark_core_weight"] == 0.55
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["dynamic_risk"] is True
    assert strategy_kwargs["risk_floor"] == 0.85
    assert strategy_kwargs["market_trend_penalty"] == 0.96
    assert strategy_kwargs["market_drawdown_penalty"] == 0.92
    assert strategy_kwargs["crash_penalty"] == 0.90
    assert strategy_kwargs["feature_score_weights"]["$risk_beta_qqq_63d"] == -0.04
    assert strategy_kwargs["feature_score_weights"]["$fmp_alpha_event_composite"] == 0.03
    assert not any(
        proc.get("class") == "GroupNeutralize" and proc.get("kwargs", {}).get("fields_group") == "label"
        for proc in utility_processors
        if isinstance(proc, dict)
    )
    assert lgb["task"]["model"]["class"] == "LGBModel"
    assert any(
        proc.get("class") == "CSRankNorm" and proc.get("kwargs", {}).get("fields_group") == "label"
        for proc in lgb["data_handler_config"]["learn_processors"]
        if isinstance(proc, dict)
    )


def test_build_variants_adds_fmp_v12_sectorneutral_mildrisk_candidate():
    mod = _load_module()

    variants = mod.build_variants(_base_cfg())
    cfg = variants["score_growth_v12_h5_qqqsector_fmpv2_ranker_mildrisk_core55_topk30"]
    processors = cfg["data_handler_config"]["learn_processors"]
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]

    assert cfg["task"]["model"]["class"] == "LGBRankerModel"
    assert cfg["data_handler_config"]["label"][0][0] == "Ref($close, -6)/Ref($close, -1) - 1"
    assert processors[1]["class"] == "BenchmarkExcessLabel"
    assert processors[1]["kwargs"]["benchmark_pkl"] == "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
    assert any(
        proc.get("class") == "GroupNeutralize" and proc.get("kwargs", {}).get("fields_group") == "label"
        for proc in processors
        if isinstance(proc, dict)
    )
    assert strategy_kwargs["benchmark_core_weight"] == 0.55
    assert strategy_kwargs["topk"] == 30
    assert strategy_kwargs["max_holdings"] == 75
    assert strategy_kwargs["dynamic_risk"] is True
    assert strategy_kwargs["risk_floor"] == 0.85
    assert strategy_kwargs["feature_score_weights"]["$risk_beta_qqq_63d"] == -0.04
