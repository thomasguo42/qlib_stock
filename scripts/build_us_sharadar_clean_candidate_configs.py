#!/usr/bin/env python
import argparse
import copy
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


VALUE6: List[Tuple[str, str]] = [
    ("$div_yield_px_q", "DIV_YIELD_PX_Q"),
    ("$book_px_q", "BOOK_PX_Q"),
    ("$earn_yield_q", "EARN_YIELD_Q"),
    ("$leverage_q", "LEVERAGE_Q"),
    ("$fcf_yield_q", "FCF_YIELD_Q"),
    ("$ebitda_margin_q", "EBITDA_MARGIN_Q"),
]

VALUE3: List[Tuple[str, str]] = [
    ("$earn_yield_q", "EARN_YIELD_Q"),
    ("$fcf_yield_q", "FCF_YIELD_Q"),
    ("$ebitda_margin_q", "EBITDA_MARGIN_Q"),
]

RISK_REGIME: List[Tuple[str, str]] = [
    ("$risk_ret_20d", "RISK_RET_20D"),
    ("$risk_ret_63d", "RISK_RET_63D"),
    ("$risk_ret_252d", "RISK_RET_252D"),
    ("$risk_vol_20d", "RISK_VOL_20D"),
    ("$risk_vol_63d", "RISK_VOL_63D"),
    ("$risk_dvol_20d", "RISK_DVOL_20D"),
    ("$risk_beta_spy_63d", "RISK_BETA_SPY_63D"),
    ("$risk_beta_spy_252d", "RISK_BETA_SPY_252D"),
    ("$risk_relret_spy_63d", "RISK_RELRET_SPY_63D"),
    ("$risk_relret_spy_252d", "RISK_RELRET_SPY_252D"),
    ("$mkt_spy_ret_20d", "MKT_SPY_RET_20D"),
    ("$mkt_spy_ret_63d", "MKT_SPY_RET_63D"),
    ("$mkt_spy_vol_20d", "MKT_SPY_VOL_20D"),
    ("$mkt_spy_dd_63d", "MKT_SPY_DD_63D"),
    ("$mkt_qqq_ret_63d", "MKT_QQQ_RET_63D"),
    ("$mkt_qqq_vol_20d", "MKT_QQQ_VOL_20D"),
    ("$mkt_iwm_ret_63d", "MKT_IWM_RET_63D"),
    ("$mkt_iwm_vol_20d", "MKT_IWM_VOL_20D"),
]

QQQ_RISK: List[Tuple[str, str]] = [
    ("$risk_beta_qqq_63d", "RISK_BETA_QQQ_63D"),
    ("$risk_beta_qqq_252d", "RISK_BETA_QQQ_252D"),
    ("$risk_relret_qqq_63d", "RISK_RELRET_QQQ_63D"),
    ("$risk_relret_qqq_252d", "RISK_RELRET_QQQ_252D"),
]

SECTOR_REGIME: List[Tuple[str, str]] = [
    ("$mkt_xlk_ret_63d", "MKT_XLK_RET_63D"),
    ("$mkt_xlf_ret_63d", "MKT_XLF_RET_63D"),
    ("$mkt_xlv_ret_63d", "MKT_XLV_RET_63D"),
    ("$mkt_xly_ret_63d", "MKT_XLY_RET_63D"),
    ("$mkt_xlp_ret_63d", "MKT_XLP_RET_63D"),
    ("$mkt_xle_ret_63d", "MKT_XLE_RET_63D"),
    ("$mkt_xli_ret_63d", "MKT_XLI_RET_63D"),
    ("$mkt_xlb_ret_63d", "MKT_XLB_RET_63D"),
    ("$mkt_xlu_ret_63d", "MKT_XLU_RET_63D"),
    ("$mkt_xlre_ret_63d", "MKT_XLRE_RET_63D"),
    ("$mkt_xlc_ret_63d", "MKT_XLC_RET_63D"),
]

REGIME_INTERACTIONS: List[Tuple[str, str]] = [
    ("$regime_beta_spy63_ret63", "REGIME_BETA_SPY63_RET63"),
    ("$regime_beta_spy63_dd63", "REGIME_BETA_SPY63_DD63"),
    ("$regime_vol20_spyvol20", "REGIME_VOL20_SPYVOL20"),
    ("$regime_relret63_spyret63", "REGIME_RELRET63_SPYRET63"),
    ("$regime_sector_relret63", "REGIME_SECTOR_RELRET63"),
    ("$regime_sector_relret63_beta63", "REGIME_SECTOR_RELRET63_BETA63"),
]

SF3A: List[Tuple[str, str]] = [
    ("$inst13f_totalvalue_daily", "INST13F_TOTALVALUE_DAILY"),
    ("$inst13f_totalvalue_20d_mean", "INST13F_TOTALVALUE_20D_MEAN"),
    ("$inst13f_percentoftotal_63d_mean", "INST13F_PERCENTOFTOTAL_63D_MEAN"),
    ("$inst13f_shrvalue_252d_pct", "INST13F_SHRVALUE_252D_PCT"),
]

SELECTED_SF3A: List[Tuple[str, str]] = [
    ("$inst13f_percentoftotal_63d_mean", "INST13F_PERCENTOFTOTAL_63D_MEAN"),
    ("$inst13f_shrvalue_252d_pct", "INST13F_SHRVALUE_252D_PCT"),
]

QUALITY_VALUE: List[Tuple[str, str]] = [
    ("$roe_q", "ROE_Q"),
    ("$roa_q", "ROA_Q"),
    ("$ebitda_margin_q", "EBITDA_MARGIN_Q"),
    ("$fcf_margin_q", "FCF_MARGIN_Q"),
    ("$leverage_q", "LEVERAGE_Q"),
    ("$cash_assets_q", "CASH_ASSETS_Q"),
    ("$capex_assets_q", "CAPEX_ASSETS_Q"),
    ("$asset_turn_q", "ASSET_TURN_Q"),
    ("$div_yield_px_q", "DIV_YIELD_PX_Q"),
    ("$earn_yield_q", "EARN_YIELD_Q"),
    ("$book_px_q", "BOOK_PX_Q"),
    ("$fcf_yield_q", "FCF_YIELD_Q"),
]

STABLE_MOMENTUM: List[Tuple[str, str]] = [
    ("$risk_ret_252d", "RISK_RET_252D"),
]

LEADERSHIP_MOMENTUM: List[Tuple[str, str]] = [
    ("$risk_ret_20d", "RISK_RET_20D"),
    ("$risk_ret_63d", "RISK_RET_63D"),
    ("$risk_ret_252d", "RISK_RET_252D"),
    ("$risk_relret_spy_63d", "RISK_RELRET_SPY_63D"),
]

SIZE: List[Tuple[str, str]] = [
    ("$log_marketcap_q", "LOG_MARKETCAP_Q"),
]

STABLE20_FEATURES: List[Tuple[str, str]] = QUALITY_VALUE + STABLE_MOMENTUM + SELECTED_SF3A
STABLE20_CAPMOM_FEATURES: List[Tuple[str, str]] = QUALITY_VALUE + SIZE + LEADERSHIP_MOMENTUM + SELECTED_SF3A
COMPOSITE_SCORE_FEATURES: List[Tuple[str, str]] = STABLE20_CAPMOM_FEATURES + [
    ("$risk_vol_20d", "RISK_VOL_20D"),
    ("$risk_beta_spy_63d", "RISK_BETA_SPY_63D"),
]
REGIME_GROWTH_FEATURES: List[Tuple[str, str]] = COMPOSITE_SCORE_FEATURES + [
    ("$mkt_qqq_ret_63d", "MKT_QQQ_RET_63D"),
]
LAGGED_MARKET_STATE: List[Tuple[str, str]] = [
    ("$mkt_spy_ret_63d_lag1", "MKT_SPY_RET_63D_LAG1"),
    ("$mkt_spy_dd_126d_lag1", "MKT_SPY_DD_126D_LAG1"),
    ("$mkt_qqq_ret_20d_lag1", "MKT_QQQ_RET_20D_LAG1"),
    ("$mkt_qqq_ret_63d_lag1", "MKT_QQQ_RET_63D_LAG1"),
    ("$mkt_qqq_ret_252d_lag1", "MKT_QQQ_RET_252D_LAG1"),
    ("$mkt_qqq_vol_20d_lag1", "MKT_QQQ_VOL_20D_LAG1"),
    ("$mkt_qqq_dd_63d_lag1", "MKT_QQQ_DD_63D_LAG1"),
    ("$mkt_qqq_dd_126d_lag1", "MKT_QQQ_DD_126D_LAG1"),
    ("$mkt_iwm_ret_63d_lag1", "MKT_IWM_RET_63D_LAG1"),
]
BREADTH_REGIME: List[Tuple[str, str]] = [
    ("$mkt_breadth_ret20_pos_lag1", "MKT_BREADTH_RET20_POS_LAG1"),
    ("$mkt_breadth_ret63_pos_lag1", "MKT_BREADTH_RET63_POS_LAG1"),
    ("$mkt_breadth_ret63_above_spy_lag1", "MKT_BREADTH_RET63_ABOVE_SPY_LAG1"),
    ("$mkt_dispersion_ret63_lag1", "MKT_DISPERSION_RET63_LAG1"),
]
REGIME_GROWTH_V3_FEATURES: List[Tuple[str, str]] = COMPOSITE_SCORE_FEATURES + [
    ("$risk_vol_63d", "RISK_VOL_63D"),
] + QQQ_RISK + LAGGED_MARKET_STATE + BREADTH_REGIME
FMP_EVENT_CORE: List[Tuple[str, str]] = [
    ("$fmp_earn_count_20d_sum", "FMP_EARN_COUNT_20D_SUM"),
    ("$fmp_earn_count_63d_sum", "FMP_EARN_COUNT_63D_SUM"),
    ("$fmp_eps_surprise_pct_latest", "FMP_EPS_SURPRISE_PCT_LATEST"),
    ("$fmp_eps_surprise_pct_63d_event_mean", "FMP_EPS_SURPRISE_PCT_63D_EVENT_MEAN"),
    ("$fmp_rev_surprise_pct_latest", "FMP_REV_SURPRISE_PCT_LATEST"),
    ("$fmp_rev_surprise_pct_63d_event_mean", "FMP_REV_SURPRISE_PCT_63D_EVENT_MEAN"),
    ("$fmp_grade_count_20d_sum", "FMP_GRADE_COUNT_20D_SUM"),
    ("$fmp_grade_count_63d_sum", "FMP_GRADE_COUNT_63D_SUM"),
    ("$fmp_grade_up_63d_sum", "FMP_GRADE_UP_63D_SUM"),
    ("$fmp_grade_down_63d_sum", "FMP_GRADE_DOWN_63D_SUM"),
    ("$fmp_grade_delta_63d_event_mean", "FMP_GRADE_DELTA_63D_EVENT_MEAN"),
    ("$fmp_grade_new_score_latest", "FMP_GRADE_NEW_SCORE_LATEST"),
    ("$fmp_pt_count_63d_sum", "FMP_PT_COUNT_63D_SUM"),
    ("$fmp_pt_upside_latest", "FMP_PT_UPSIDE_LATEST"),
    ("$fmp_pt_upside_63d_event_mean", "FMP_PT_UPSIDE_63D_EVENT_MEAN"),
    ("$fmp_rating_bullish_ratio_snapshot", "FMP_RATING_BULLISH_RATIO_SNAPSHOT"),
    ("$fmp_rating_bearish_ratio_snapshot", "FMP_RATING_BEARISH_RATIO_SNAPSHOT"),
    ("$fmp_rating_net_bullish_snapshot", "FMP_RATING_NET_BULLISH_SNAPSHOT"),
    ("$fmp_rating_net_bullish_63d_chg", "FMP_RATING_NET_BULLISH_63D_CHG"),
]
FMP_EVENT_DIRECTIONAL_V2: List[Tuple[str, str]] = [
    ("$fmp_alpha_earn_surprise_latest", "FMP_ALPHA_EARN_SURPRISE_LATEST"),
    ("$fmp_alpha_earn_surprise_10d", "FMP_ALPHA_EARN_SURPRISE_10D"),
    ("$fmp_alpha_earn_surprise_20d", "FMP_ALPHA_EARN_SURPRISE_20D"),
    ("$fmp_alpha_earn_surprise_63d", "FMP_ALPHA_EARN_SURPRISE_63D"),
    ("$fmp_alpha_rating_bullish", "FMP_ALPHA_RATING_BULLISH"),
    ("$fmp_alpha_rating_bearish_penalty", "FMP_ALPHA_RATING_BEARISH_PENALTY"),
    ("$fmp_alpha_rating_net_change", "FMP_ALPHA_RATING_NET_CHANGE"),
    ("$fmp_alpha_grade_score_latest", "FMP_ALPHA_GRADE_SCORE_LATEST"),
    ("$fmp_alpha_grade_revision_10d", "FMP_ALPHA_GRADE_REVISION_10D"),
    ("$fmp_alpha_grade_revision_20d", "FMP_ALPHA_GRADE_REVISION_20D"),
    ("$fmp_alpha_grade_revision_63d", "FMP_ALPHA_GRADE_REVISION_63D"),
    ("$fmp_alpha_pt_upside_latest", "FMP_ALPHA_PT_UPSIDE_LATEST"),
    ("$fmp_alpha_pt_upside_20d", "FMP_ALPHA_PT_UPSIDE_20D"),
    ("$fmp_alpha_event_freshness", "FMP_ALPHA_EVENT_FRESHNESS"),
    ("$fmp_alpha_event_coverage", "FMP_ALPHA_EVENT_COVERAGE"),
    ("$fmp_alpha_event_composite", "FMP_ALPHA_EVENT_COMPOSITE"),
]
FMP_EVENT_CORE_V2: List[Tuple[str, str]] = [
    ("$fmp_eps_surprise_pct_latest", "FMP_EPS_SURPRISE_PCT_LATEST"),
    ("$fmp_eps_surprise_pct_63d_event_mean", "FMP_EPS_SURPRISE_PCT_63D_EVENT_MEAN"),
    ("$fmp_rev_surprise_pct_latest", "FMP_REV_SURPRISE_PCT_LATEST"),
    ("$fmp_grade_new_score_latest", "FMP_GRADE_NEW_SCORE_LATEST"),
    ("$fmp_rating_bullish_ratio_snapshot", "FMP_RATING_BULLISH_RATIO_SNAPSHOT"),
    ("$fmp_rating_bearish_ratio_snapshot", "FMP_RATING_BEARISH_RATIO_SNAPSHOT"),
    ("$fmp_rating_net_bullish_snapshot", "FMP_RATING_NET_BULLISH_SNAPSHOT"),
    ("$fmp_rating_net_bullish_63d_chg", "FMP_RATING_NET_BULLISH_63D_CHG"),
    ("$fmp_earn_count_20d_freshness", "FMP_EARN_COUNT_20D_FRESHNESS"),
    ("$fmp_grade_count_20d_freshness", "FMP_GRADE_COUNT_20D_FRESHNESS"),
] + FMP_EVENT_DIRECTIONAL_V2
FMP_EVENT_DIAGNOSTIC_FEATURES: List[Tuple[str, str]] = [
    ("$mkt_qqq_ret_63d_lag1", "MKT_QQQ_RET_63D_LAG1"),
] + FMP_EVENT_CORE
FMP_EVENT_DIAGNOSTIC_FEATURES_V2: List[Tuple[str, str]] = [
    ("$mkt_qqq_ret_63d_lag1", "MKT_QQQ_RET_63D_LAG1"),
] + FMP_EVENT_CORE_V2
FMP_STACK_V2_FEATURES: List[Tuple[str, str]] = REGIME_GROWTH_V3_FEATURES + FMP_EVENT_CORE_V2
FMP_EVENT_WHITELIST_V1: List[Tuple[str, str]] = [
    ("$fmp_alpha_earn_surprise_latest", "FMP_ALPHA_EARN_SURPRISE_LATEST"),
    ("$fmp_alpha_earn_surprise_20d", "FMP_ALPHA_EARN_SURPRISE_20D"),
    ("$fmp_alpha_rating_bullish", "FMP_ALPHA_RATING_BULLISH"),
    ("$fmp_alpha_rating_bearish_penalty", "FMP_ALPHA_RATING_BEARISH_PENALTY"),
    ("$fmp_alpha_grade_score_latest", "FMP_ALPHA_GRADE_SCORE_LATEST"),
    ("$fmp_alpha_event_freshness", "FMP_ALPHA_EVENT_FRESHNESS"),
    ("$fmp_alpha_event_composite", "FMP_ALPHA_EVENT_COMPOSITE"),
]
SHARADAR_STACK_V1_FEATURES: List[Tuple[str, str]] = REGIME_GROWTH_V3_FEATURES
FMP_STACK_WHITELIST_V1_FEATURES: List[Tuple[str, str]] = REGIME_GROWTH_V3_FEATURES + FMP_EVENT_WHITELIST_V1
TARGET_AUDIT_COMPOSITE_WEIGHTS = {
    "RISK_RET_20D": 1.0,
    "RISK_RET_63D": 1.0,
    "RISK_RELRET_SPY_63D": 1.0,
    "BOOK_PX_Q": 1.0,
    "FCF_YIELD_Q": 1.0,
    "EARN_YIELD_Q": 1.0,
    "RISK_VOL_20D": -1.0,
    "RISK_BETA_SPY_63D": -1.0,
}
REGIME_RISK_ON_WEIGHTS = {
    "RISK_RET_20D": 0.50,
    "RISK_RET_63D": 1.00,
    "RISK_RELRET_SPY_63D": 0.75,
    "RISK_RET_252D": 0.25,
    "RISK_VOL_20D": 0.45,
    "RISK_BETA_SPY_63D": 0.45,
    "LOG_MARKETCAP_Q": 0.25,
}
REGIME_RISK_OFF_WEIGHTS = dict(TARGET_AUDIT_COMPOSITE_WEIGHTS)
REGIME_SLEEVE_WEIGHTS = {
    "growth": {
        "RISK_RET_20D": 0.45,
        "RISK_RET_63D": 1.00,
        "RISK_RELRET_SPY_63D": 0.80,
        "RISK_RET_252D": 0.35,
        "LOG_MARKETCAP_Q": 0.20,
        "RISK_VOL_20D": 0.15,
        "RISK_BETA_SPY_63D": 0.15,
    },
    "quality_value": {
        "BOOK_PX_Q": 0.70,
        "FCF_YIELD_Q": 1.00,
        "EARN_YIELD_Q": 0.90,
        "ROE_Q": 0.45,
        "FCF_MARGIN_Q": 0.35,
        "RISK_VOL_20D": -0.35,
        "RISK_BETA_SPY_63D": -0.35,
    },
    "low_vol": {
        "RISK_VOL_20D": -1.00,
        "RISK_VOL_63D": -0.60,
        "RISK_BETA_SPY_63D": -0.70,
        "LOG_MARKETCAP_Q": 0.30,
        "FCF_YIELD_Q": 0.35,
        "EARN_YIELD_Q": 0.25,
    },
    "recovery": {
        "RISK_RET_20D": -0.40,
        "RISK_RET_63D": 0.45,
        "RISK_RELRET_SPY_63D": 0.35,
        "FCF_YIELD_Q": 0.55,
        "EARN_YIELD_Q": 0.45,
        "RISK_VOL_20D": -0.20,
    },
}
REGIME_STATE_SLEEVE_WEIGHTS = {
    "risk_on": {"growth": 0.65, "quality_value": 0.20, "low_vol": 0.05, "recovery": 0.10},
    "fading": {"growth": 0.25, "quality_value": 0.45, "low_vol": 0.20, "recovery": 0.10},
    "chop": {"growth": 0.20, "quality_value": 0.45, "low_vol": 0.25, "recovery": 0.10},
    "risk_off": {"growth": 0.05, "quality_value": 0.50, "low_vol": 0.35, "recovery": 0.10},
    "recovery": {"growth": 0.45, "quality_value": 0.25, "low_vol": 0.10, "recovery": 0.20},
}
FMP_STACK_V2_SLEEVES = {
    "fmp_event": {
        "FMP_ALPHA_EVENT_COMPOSITE": 1.00,
        "FMP_ALPHA_EARN_SURPRISE_LATEST": 0.80,
        "FMP_ALPHA_EARN_SURPRISE_20D": 0.65,
        "FMP_ALPHA_EARN_SURPRISE_63D": 0.35,
        "FMP_ALPHA_RATING_BULLISH": 0.70,
        "FMP_ALPHA_RATING_BEARISH_PENALTY": 0.70,
        "FMP_ALPHA_RATING_NET_CHANGE": 0.45,
        "FMP_ALPHA_GRADE_SCORE_LATEST": 0.45,
        "FMP_ALPHA_PT_UPSIDE_LATEST": 0.15,
        "FMP_ALPHA_EVENT_FRESHNESS": 0.10,
    },
    "quality_value": {
        "FCF_YIELD_Q": 1.00,
        "EARN_YIELD_Q": 0.80,
        "BOOK_PX_Q": 0.55,
        "ROE_Q": 0.50,
        "FCF_MARGIN_Q": 0.35,
        "LEVERAGE_Q": -0.25,
        "RISK_VOL_20D": -0.25,
    },
    "momentum_regime": {
        "RISK_RET_20D": 0.40,
        "RISK_RET_63D": 0.90,
        "RISK_RET_252D": 0.35,
        "RISK_RELRET_QQQ_63D": 0.65,
        "RISK_RELRET_QQQ_252D": 0.30,
        "LOG_MARKETCAP_Q": 0.20,
        "MKT_QQQ_RET_63D_LAG1": 0.10,
    },
}
FMP_LEADERSHIP_STACK_V1_FEATURES: List[Tuple[str, str]] = FMP_STACK_V2_FEATURES + REGIME_INTERACTIONS
FMP_LEADERSHIP_STACK_V1_SLEEVES = {
    "fmp_event": copy.deepcopy(FMP_STACK_V2_SLEEVES["fmp_event"]),
    "quality_value": copy.deepcopy(FMP_STACK_V2_SLEEVES["quality_value"]),
    "momentum_regime": copy.deepcopy(FMP_STACK_V2_SLEEVES["momentum_regime"]),
    "qqq_leadership": {
        "RISK_RELRET_QQQ_63D": 1.00,
        "RISK_RELRET_QQQ_252D": 0.45,
        "RISK_RET_63D": 0.65,
        "RISK_RET_20D": 0.35,
        "REGIME_SECTOR_RELRET63": 0.45,
        "REGIME_SECTOR_RELRET63_BETA63": 0.15,
        "LOG_MARKETCAP_Q": 0.25,
        "RISK_BETA_QQQ_63D": 0.15,
        "RISK_VOL_20D": -0.10,
    },
}
FMP_LEADERSHIP_REGIME_STATE_WEIGHTS = {
    "risk_on": {"fmp_event": 0.30, "quality_value": 0.15, "momentum_regime": 0.20, "qqq_leadership": 0.35},
    "fading": {"fmp_event": 0.42, "quality_value": 0.40, "momentum_regime": 0.12, "qqq_leadership": 0.06},
    "chop": {"fmp_event": 0.40, "quality_value": 0.35, "momentum_regime": 0.15, "qqq_leadership": 0.10},
    "risk_off": {"fmp_event": 0.48, "quality_value": 0.45, "momentum_regime": 0.07, "qqq_leadership": 0.00},
    "recovery": {"fmp_event": 0.35, "quality_value": 0.25, "momentum_regime": 0.20, "qqq_leadership": 0.20},
}
SHARADAR_STACK_V1_SLEEVES = {
    "quality_value": copy.deepcopy(FMP_STACK_V2_SLEEVES["quality_value"]),
    "momentum_regime": copy.deepcopy(FMP_STACK_V2_SLEEVES["momentum_regime"]),
}
FMP_WHITELIST_V1_SLEEVES = {
    "fmp_event": {
        "FMP_ALPHA_EVENT_COMPOSITE": 1.00,
        "FMP_ALPHA_EARN_SURPRISE_LATEST": 0.90,
        "FMP_ALPHA_EARN_SURPRISE_20D": 0.55,
        "FMP_ALPHA_RATING_BULLISH": 0.75,
        "FMP_ALPHA_RATING_BEARISH_PENALTY": 0.75,
        "FMP_ALPHA_GRADE_SCORE_LATEST": 0.50,
        "FMP_ALPHA_EVENT_FRESHNESS": 0.10,
    },
    "quality_value": copy.deepcopy(FMP_STACK_V2_SLEEVES["quality_value"]),
    "momentum_regime": copy.deepcopy(FMP_STACK_V2_SLEEVES["momentum_regime"]),
}
LABEL_PROCESSORS = {
    "BenchmarkExcessLabel",
    "ResidualForwardReturnLabel",
    "VolScaledExcessLabel",
    "DownsideAdjustedExcessLabel",
    "PortfolioUtilityExcessLabel",
    "DualHorizonPortfolioUtilityLabel",
}


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML structure: {path}")
    return data


def _processor_class(proc: Dict) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def _remove_group_neutralize(processors: Iterable[Dict], *, fields_group: str = None) -> List[Dict]:
    out = []
    for proc in processors or []:
        if not isinstance(proc, dict):
            out.append(proc)
            continue
        if _processor_class(proc) != "GroupNeutralize":
            out.append(proc)
            continue
        kwargs = proc.get("kwargs", {}) or {}
        if fields_group is None or str(kwargs.get("fields_group", "")) == str(fields_group):
            continue
        out.append(proc)
    return out


def _remove_label_rank_norm(processors: Iterable[Dict]) -> List[Dict]:
    out = []
    for proc in processors or []:
        if not isinstance(proc, dict):
            out.append(proc)
            continue
        if _processor_class(proc) != "CSRankNorm":
            out.append(proc)
            continue
        kwargs = proc.get("kwargs", {}) or {}
        if str(kwargs.get("fields_group", "")) != "label":
            out.append(proc)
    return out


def _set_extra_features(cfg: Dict, features: List[Tuple[str, str]]) -> None:
    dh = cfg["data_handler_config"]
    dh["extra_fields"] = [field for field, _ in features]
    dh["extra_names"] = [name for _, name in features]


def _append_extra_feature(cfg: Dict, field: str, name: str) -> None:
    dh = cfg["data_handler_config"]
    fields = dh.setdefault("extra_fields", [])
    names = dh.setdefault("extra_names", [])
    if str(field) not in [str(v) for v in fields]:
        fields.append(str(field))
        names.append(str(name))


def _set_factor_only_handler(cfg: Dict) -> None:
    handler = cfg["task"]["dataset"]["kwargs"]["handler"]
    handler["class"] = "SharadarFeatureHandler"
    handler["module_path"] = "qlib.contrib.data.handler_sharadar"


def _set_selective_feature_norm(cfg: Dict, *, exclude_prefixes: Iterable[str] = ("MKT_",)) -> None:
    processors = []
    for proc in cfg["data_handler_config"].get("infer_processors", []) or []:
        if not isinstance(proc, dict):
            processors.append(proc)
            continue
        cls = _processor_class(proc)
        if cls in {"CSZScoreNorm", "SelectiveCSZScoreNorm", "GroupNeutralize"}:
            continue
        processors.append(proc)
    processors.append(
        {
            "class": "SelectiveCSZScoreNorm",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "fields_group": "feature",
                "method": "robust",
                "exclude_prefixes": list(exclude_prefixes),
            },
        }
    )
    cfg["data_handler_config"]["infer_processors"] = processors


def _set_exp_name(cfg: Dict, exp_name: str) -> None:
    qlib_init = cfg.setdefault("qlib_init", {})
    expm = qlib_init.setdefault("exp_manager", {})
    kwargs = expm.setdefault("kwargs", {})
    kwargs["default_exp_name"] = exp_name


def _set_model_lgb(cfg: Dict) -> None:
    cfg["task"]["model"] = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.025,
            "colsample_bytree": 0.70,
            "subsample": 0.80,
            "lambda_l1": 40.0,
            "lambda_l2": 200.0,
            "max_depth": 4,
            "num_leaves": 24,
            "min_data_in_leaf": 400,
            "num_threads": 20,
            "num_boost_round": 1000,
            "early_stopping_rounds": 80,
        },
    }


def _set_slow_lgb_training(cfg: Dict, *, learning_rate: float = 0.005, rounds: int = 4000, early_stopping: int = 200) -> None:
    kwargs = cfg["task"]["model"].setdefault("kwargs", {})
    kwargs["learning_rate"] = float(learning_rate)
    kwargs["num_boost_round"] = int(rounds)
    kwargs["early_stopping_rounds"] = int(early_stopping)


def _set_ranker_training(
    cfg: Dict,
    *,
    learning_rate: float = 0.015,
    rounds: int = 1200,
    early_stopping: int = 120,
) -> None:
    kwargs = cfg["task"]["model"].setdefault("kwargs", {})
    kwargs["learning_rate"] = float(learning_rate)
    kwargs["num_boost_round"] = int(rounds)
    kwargs["early_stopping_rounds"] = int(early_stopping)


def _set_model_ranker(cfg: Dict, *, eval_at: List[int] = None) -> None:
    eval_at = [40, 100] if eval_at is None else [int(v) for v in eval_at]
    cfg["task"]["model"] = {
        "class": "LGBRankerModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": eval_at,
            "label_bins": 8,
            "learning_rate": 0.025,
            "colsample_bytree": 0.70,
            "subsample": 0.80,
            "subsample_freq": 1,
            "lambda_l1": 10.0,
            "lambda_l2": 120.0,
            "max_depth": 4,
            "num_leaves": 24,
            "min_data_in_leaf": 300,
            "num_threads": 20,
            "num_boost_round": 700,
            "early_stopping_rounds": 80,
        },
    }


def _set_feature_weighted_score_model(
    cfg: Dict,
    *,
    weights: Optional[Dict[str, float]] = None,
    regime_feature: Optional[str] = None,
    regime_threshold: float = 0.0,
    risk_on_weights: Optional[Dict[str, float]] = None,
    risk_off_weights: Optional[Dict[str, float]] = None,
) -> None:
    kwargs = {
        "weights": dict(weights or TARGET_AUDIT_COMPOSITE_WEIGHTS),
        "normalize_by_date": True,
        "missing": "raise",
    }
    if regime_feature is not None or risk_on_weights is not None or risk_off_weights is not None:
        if not regime_feature or not risk_on_weights or not risk_off_weights:
            raise ValueError("regime_feature, risk_on_weights, and risk_off_weights must be provided together")
        kwargs.update(
            {
                "regime_feature": str(regime_feature),
                "regime_threshold": float(regime_threshold),
                "risk_on_weights": dict(risk_on_weights),
                "risk_off_weights": dict(risk_off_weights),
            }
        )
    cfg["task"]["model"] = {
        "class": "FeatureWeightedScoreModel",
        "module_path": "qlib.contrib.model.score",
        "kwargs": kwargs,
    }


def _set_ic_selected_score_model(
    cfg: Dict,
    *,
    max_features: int = 8,
    min_selected_features: int = 3,
    min_abs_ic: float = 0.0025,
    min_ic_days: int = 120,
    min_daily_count: int = 30,
    min_coverage: float = 0.30,
    min_same_sign_years: int = 2,
    min_worst_year_signed_ic: float = -0.015,
    min_recent_signed_ic: float = 0.0,
    recent_window_days: int = 756,
    selection_step_days: int = 5,
    recent_weight: float = 0.50,
    regime_feature: Optional[str] = None,
    risk_on_threshold: float = 0.03,
    risk_off_threshold: float = -0.04,
    regime_min_ic_days: Optional[int] = None,
    top_quantile: float = 0.20,
    min_topq_spread: Optional[float] = None,
    min_worst_year_topq_spread: Optional[float] = None,
    tail_weight: float = 0.0,
) -> None:
    kwargs = {
        "max_features": int(max_features),
        "min_selected_features": int(min_selected_features),
        "min_abs_ic": float(min_abs_ic),
        "min_ic_days": int(min_ic_days),
        "min_daily_count": int(min_daily_count),
        "min_coverage": float(min_coverage),
        "min_same_sign_years": int(min_same_sign_years),
        "min_worst_year_signed_ic": float(min_worst_year_signed_ic),
        "min_recent_signed_ic": float(min_recent_signed_ic),
        "recent_window_days": int(recent_window_days),
        "selection_step_days": int(selection_step_days),
        "recent_weight": float(recent_weight),
        "weight_power": 1.0,
        "normalize_by_date": True,
        "missing": "raise",
        "fallback_to_best": True,
        "top_quantile": float(top_quantile),
        "tail_weight": float(tail_weight),
    }
    if min_topq_spread is not None:
        kwargs["min_topq_spread"] = float(min_topq_spread)
    if min_worst_year_topq_spread is not None:
        kwargs["min_worst_year_topq_spread"] = float(min_worst_year_topq_spread)
    if regime_feature:
        kwargs.update(
            {
                "regime_feature": str(regime_feature),
                "risk_on_threshold": float(risk_on_threshold),
                "risk_off_threshold": float(risk_off_threshold),
                "regime_min_ic_days": int(regime_min_ic_days if regime_min_ic_days is not None else max(30, min_ic_days // 3)),
            }
        )
    cfg["task"]["model"] = {
        "class": "ICSelectedScoreModel",
        "module_path": "qlib.contrib.model.score",
        "kwargs": kwargs,
    }


def _set_regime_sleeve_score_model(
    cfg: Dict,
    *,
    sleeves: Optional[Dict[str, Dict[str, float]]] = None,
    state_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    cfg["task"]["model"] = {
        "class": "RegimeSleeveScoreModel",
        "module_path": "qlib.contrib.model.score",
        "kwargs": {
            "sleeves": copy.deepcopy(sleeves or REGIME_SLEEVE_WEIGHTS),
            "state_weights": copy.deepcopy(state_weights or REGIME_STATE_SLEEVE_WEIGHTS),
            "default_state": "chop",
            "normalize_by_date": True,
            "missing": "raise",
            "trend_feature": "MKT_QQQ_RET_63D_LAG1",
            "fast_trend_feature": "MKT_QQQ_RET_20D_LAG1",
            "drawdown_feature": "MKT_QQQ_DD_126D_LAG1",
            "vol_feature": "MKT_QQQ_VOL_20D_LAG1",
            "breadth_feature": "MKT_BREADTH_RET63_POS_LAG1",
            "risk_on_threshold": 0.03,
            "risk_off_threshold": -0.04,
            "recovery_fast_threshold": 0.02,
            "fading_fast_threshold": -0.02,
            "drawdown_limit": -0.10,
            "drawdown_warning": -0.06,
            "high_vol_threshold": 0.35,
            "breadth_threshold": 0.45,
        },
    }


def _set_stacked_signal_score_model(
    cfg: Dict,
    *,
    sleeves: Optional[Dict[str, Dict[str, float]]] = None,
    max_sleeves: int = 3,
    min_selected_sleeves: int = 1,
    min_abs_ic: float = 0.001,
    min_ic_days: int = 80,
    min_recent_signed_ic: float = -0.005,
    min_worst_year_signed_ic: float = -0.04,
) -> None:
    cfg["task"]["model"] = {
        "class": "StackedSignalScoreModel",
        "module_path": "qlib.contrib.model.score",
        "kwargs": {
            "sleeves": copy.deepcopy(sleeves or FMP_STACK_V2_SLEEVES),
            "max_sleeves": int(max_sleeves),
            "min_selected_sleeves": int(min_selected_sleeves),
            "min_abs_ic": float(min_abs_ic),
            "min_ic_days": int(min_ic_days),
            "min_daily_count": 30,
            "min_coverage": 0.30,
            "min_recent_signed_ic": float(min_recent_signed_ic),
            "min_worst_year_signed_ic": float(min_worst_year_signed_ic),
            "recent_window_days": 756,
            "recent_weight": 0.65,
            "weight_power": 1.0,
            "normalize_by_date": True,
            "normalize_sleeve_scores_by_date": True,
            "missing": "raise",
            "allow_negative_sleeve_weights": True,
            "fallback_to_equal": True,
        },
    }


def _set_horizon(cfg: Dict, horizon: int) -> None:
    horizon = int(horizon)
    dh = cfg["data_handler_config"]
    dh["label"] = [[f"Ref($close, -{horizon + 1})/Ref($close, -1) - 1"], ["LABEL0"]]
    for proc in dh.get("learn_processors", []) or []:
        if isinstance(proc, dict) and _processor_class(proc) in LABEL_PROCESSORS:
            proc.setdefault("kwargs", {})["label_horizon_days"] = horizon
    strategy = cfg["port_analysis_config"]["strategy"]
    strategy_kwargs = strategy["kwargs"]
    strategy_class = str(strategy.get("class", "") or "")
    if not strategy_class or "TopkDropout" in strategy_class:
        strategy_kwargs["hold_thresh"] = horizon
    else:
        strategy_kwargs.pop("hold_thresh", None)
    if horizon >= 60:
        segments = cfg["task"]["dataset"]["kwargs"]["segments"]
        segments["train"][1] = "2020-10-06"
        segments["valid"][1] = "2021-10-06"
        dh["fit_end_time"] = "2020-10-06"
    elif horizon >= 40:
        segments = cfg["task"]["dataset"]["kwargs"]["segments"]
        segments["train"][1] = "2020-11-03"
        segments["valid"][1] = "2021-11-03"
        dh["fit_end_time"] = "2020-11-03"
    elif horizon >= 20:
        segments = cfg["task"]["dataset"]["kwargs"]["segments"]
        segments["train"][1] = "2020-12-02"
        segments["valid"][1] = "2021-12-02"
        dh["fit_end_time"] = "2020-12-02"


def _set_fmp_event_segments(cfg: Dict, horizon: int) -> None:
    """Restrict FMP candidates to periods where lagged FMP event features exist."""
    horizon = int(horizon)
    segments = cfg["task"]["dataset"]["kwargs"]["segments"]
    if horizon >= 40:
        train_end = "2023-06-30"
        valid_start = "2023-09-01"
        test_start = "2024-03-01"
    elif horizon >= 20:
        train_end = "2023-07-31"
        valid_start = "2023-09-01"
        test_start = "2024-02-01"
    else:
        train_end = "2023-08-31"
        valid_start = "2023-09-18"
        test_start = "2024-01-17" if horizon >= 10 else "2024-01-16"
    segments["train"] = ["2022-01-03", train_end]
    segments["valid"] = [valid_start, "2023-12-29"]
    segments["test"] = [test_start, "2026-04-30"]
    dh = cfg["data_handler_config"]
    dh["start_time"] = "2022-01-01"
    dh["end_time"] = "2026-04-30"
    dh["fit_start_time"] = "2022-01-03"
    dh["fit_end_time"] = train_end


def _set_benchmark_excess_label(
    cfg: Dict,
    *,
    benchmark_pkl: str,
    benchmark_kind: str = "return",
    label_ref_start_days: int = 1,
) -> None:
    benchmark_symbol = _infer_benchmark_symbol_from_pkl(benchmark_pkl)
    if benchmark_symbol:
        cfg["benchmark"] = benchmark_symbol
        backtest_cfg = cfg.setdefault("port_analysis_config", {}).setdefault("backtest", {})
        backtest_cfg["benchmark"] = benchmark_symbol
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    horizon = _label_processor_horizon(cfg, default=20)
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "BenchmarkExcessLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        proc["kwargs"] = {
            "benchmark_pkl": str(benchmark_pkl),
            "benchmark_kind": str(benchmark_kind),
            "label_horizon_days": int(horizon),
            "label_ref_start_days": int(label_ref_start_days),
        }
        return
    processors.append(
        {
            "class": "BenchmarkExcessLabel",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "benchmark_pkl": str(benchmark_pkl),
                "benchmark_kind": str(benchmark_kind),
                "label_horizon_days": int(horizon),
                "label_ref_start_days": int(label_ref_start_days),
            },
        }
    )


def _infer_benchmark_symbol_from_pkl(benchmark_pkl: str) -> str:
    name = Path(str(benchmark_pkl)).name.lower()
    if "bench_qqq" in name:
        return "QQQ"
    if "bench_spy" in name:
        return "SPY"
    return ""


def _set_absolute_return_label(cfg: Dict) -> None:
    processors = []
    for proc in cfg["data_handler_config"].get("learn_processors", []) or []:
        if isinstance(proc, dict) and _processor_class(proc) in LABEL_PROCESSORS:
            continue
        processors.append(proc)
    cfg["data_handler_config"]["learn_processors"] = processors


def _add_label_rank_norm(cfg: Dict) -> None:
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) != "CSRankNorm":
            continue
        kwargs = proc.get("kwargs", {}) or {}
        if str(kwargs.get("fields_group", "")) == "label":
            return
    processors.append(
        {
            "class": "CSRankNorm",
            "kwargs": {"fields_group": "label"},
        }
    )


def _set_residual_label(
    cfg: Dict,
    *,
    beta_feature: str = "RISK_BETA_SPY_63D",
    fallback_beta: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2.0,
) -> None:
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "ResidualForwardReturnLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        kwargs = proc.setdefault("kwargs", {})
        for key in (
            "vol_feature",
            "fallback_vol",
            "vol_min",
            "vol_max",
            "scale_by_horizon",
            "fill_missing_vol",
            "clip_abs_label",
            "downside_penalty",
        ):
            kwargs.pop(key, None)
        kwargs["beta_feature"] = str(beta_feature)
        kwargs["fallback_beta"] = float(fallback_beta)
        kwargs["beta_min"] = float(beta_min)
        kwargs["beta_max"] = float(beta_max)
        kwargs["fill_missing_beta"] = True
        return
    processors.append(
        {
            "class": "ResidualForwardReturnLabel",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "benchmark_pkl": "/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl",
                "label_horizon_days": 5,
                "label_ref_start_days": 1,
                "beta_feature": str(beta_feature),
                "fallback_beta": float(fallback_beta),
                "beta_min": float(beta_min),
                "beta_max": float(beta_max),
                "fill_missing_beta": True,
            },
        }
    )


def _label_processor_horizon(cfg: Dict, default: int = 10) -> int:
    for proc in cfg["data_handler_config"].get("learn_processors", []) or []:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        kwargs = proc.get("kwargs", {}) or {}
        try:
            return int(kwargs.get("label_horizon_days", default))
        except (TypeError, ValueError):
            return int(default)
    return int(default)


def _set_vol_scaled_label(
    cfg: Dict,
    *,
    vol_feature: str = "RISK_VOL_20D",
    fallback_vol: float = 0.02,
    vol_min: float = 0.0025,
    vol_max: float = 0.20,
    scale_by_horizon: bool = True,
    clip_abs_label: Optional[float] = None,
) -> None:
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "VolScaledExcessLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        kwargs = proc.setdefault("kwargs", {})
        for key in ("beta_feature", "fallback_beta", "beta_min", "beta_max", "fill_missing_beta", "downside_penalty"):
            kwargs.pop(key, None)
        kwargs["vol_feature"] = str(vol_feature)
        kwargs["fallback_vol"] = float(fallback_vol)
        kwargs["vol_min"] = float(vol_min)
        kwargs["vol_max"] = float(vol_max)
        kwargs["scale_by_horizon"] = bool(scale_by_horizon)
        kwargs["fill_missing_vol"] = True
        if clip_abs_label is None:
            kwargs.pop("clip_abs_label", None)
        else:
            kwargs["clip_abs_label"] = float(clip_abs_label)
        return
    processors.append(
        {
            "class": "VolScaledExcessLabel",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "benchmark_pkl": "/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl",
                "label_horizon_days": _label_processor_horizon(cfg, default=10),
                "label_ref_start_days": 1,
                "vol_feature": str(vol_feature),
                "fallback_vol": float(fallback_vol),
                "vol_min": float(vol_min),
                "vol_max": float(vol_max),
                "scale_by_horizon": bool(scale_by_horizon),
                "fill_missing_vol": True,
            },
        }
    )


def _set_downside_adjusted_label(
    cfg: Dict,
    *,
    downside_penalty: float = 1.0,
    clip_abs_label: Optional[float] = None,
) -> None:
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "DownsideAdjustedExcessLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        kwargs = proc.setdefault("kwargs", {})
        for key in (
            "beta_feature",
            "fallback_beta",
            "beta_min",
            "beta_max",
            "fill_missing_beta",
            "vol_feature",
            "fallback_vol",
            "vol_min",
            "vol_max",
            "scale_by_horizon",
            "fill_missing_vol",
        ):
            kwargs.pop(key, None)
        kwargs["downside_penalty"] = float(downside_penalty)
        if clip_abs_label is None:
            kwargs.pop("clip_abs_label", None)
        else:
            kwargs["clip_abs_label"] = float(clip_abs_label)
        return
    processors.append(
        {
            "class": "DownsideAdjustedExcessLabel",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "benchmark_pkl": "/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
                "label_horizon_days": _label_processor_horizon(cfg, default=10),
                "label_ref_start_days": 1,
                "downside_penalty": float(downside_penalty),
            },
        }
    )


def _set_portfolio_utility_label(
    cfg: Dict,
    *,
    benchmark_pkl: str = "/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
    vol_feature: str = "RISK_VOL_20D",
    downside_penalty: float = 0.75,
    volatility_penalty: float = 0.15,
    clip_abs_label: Optional[float] = None,
) -> None:
    processors = cfg["data_handler_config"].setdefault("learn_processors", [])
    horizon = _label_processor_horizon(cfg, default=20)
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if _processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "PortfolioUtilityExcessLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        proc["kwargs"] = {
            "benchmark_pkl": str(benchmark_pkl),
            "label_horizon_days": int(horizon),
            "label_ref_start_days": 1,
            "vol_feature": str(vol_feature),
            "downside_penalty": float(downside_penalty),
            "volatility_penalty": float(volatility_penalty),
            "scale_vol_by_horizon": True,
            "divide_by_vol": False,
            "fill_missing_vol": True,
        }
        if clip_abs_label is not None:
            proc["kwargs"]["clip_abs_label"] = float(clip_abs_label)
        return
    kwargs = {
        "benchmark_pkl": str(benchmark_pkl),
        "label_horizon_days": int(horizon),
        "label_ref_start_days": 1,
        "vol_feature": str(vol_feature),
        "downside_penalty": float(downside_penalty),
        "volatility_penalty": float(volatility_penalty),
        "scale_vol_by_horizon": True,
        "divide_by_vol": False,
        "fill_missing_vol": True,
    }
    if clip_abs_label is not None:
        kwargs["clip_abs_label"] = float(clip_abs_label)
    processors.append(
        {
            "class": "PortfolioUtilityExcessLabel",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": kwargs,
        }
    )


def _set_hedged_benchmark_aware_strategy(
    cfg: Dict,
    *,
    hedge_tickers_file: str = "/Stock/qlib/_sfp_hedge_tickers.txt",
    hedge_max_weight: float = 0.35,
) -> None:
    strategy = cfg["port_analysis_config"]["strategy"]
    strategy["class"] = "WeeklyHedgedBenchmarkAwareScoreWeightedStrategy"
    strategy["module_path"] = "qlib.contrib.strategy"
    kwargs = strategy.setdefault("kwargs", {})
    kwargs.update(
        {
            "hedge_mode": "long_hedge",
            "hedge_tickers_file": str(hedge_tickers_file),
            "hedge_weight": 0.0,
            "hedge_min_weight": 0.0,
            "hedge_max_weight": float(hedge_max_weight),
            "hedge_trend_window": 63,
            "hedge_trend_thresh": -0.04,
            "hedge_drawdown_window": 126,
            "hedge_drawdown_limit": 0.10,
            "hedge_crash_return_lookback": 5,
            "hedge_crash_return_limit": 0.06,
            "hedge_min_history": 40,
            "hedge_smoothing_up": 1.0,
            "hedge_smoothing_down": 0.50,
            "hedge_max_step_up": 0.35,
            "hedge_max_step_down": 0.20,
            "market_index": "SPY",
        }
    )


def _set_low_turnover_strategy(cfg: Dict) -> None:
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy_kwargs["n_drop"] = 4
    strategy_kwargs["candidate_buffer"] = 4
    strategy_kwargs["risk_target_ann"] = 0.15
    strategy_kwargs["risk_ceiling"] = 0.75
    strategy_kwargs["max_risk_step_down"] = 0.16


def _set_conservative_risk_strategy(cfg: Dict) -> None:
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy_kwargs["n_drop"] = 4
    strategy_kwargs["candidate_buffer"] = 4
    strategy_kwargs["risk_target_ann"] = 0.14
    strategy_kwargs["risk_ceiling"] = 0.72
    strategy_kwargs["risk_floor"] = 0.08
    strategy_kwargs["max_risk_step_up"] = 0.05
    strategy_kwargs["max_risk_step_down"] = 0.15
    strategy_kwargs["drawdown_limit"] = 0.05
    strategy_kwargs["drawdown_penalty"] = 0.35
    strategy_kwargs["market_drawdown_limit"] = 0.07
    strategy_kwargs["market_drawdown_penalty"] = 0.50
    strategy_kwargs["sector_map_csv"] = "/root/.qlib/sharadar/raw/tickers.csv"
    strategy_kwargs["sector_ticker_col"] = "ticker"
    strategy_kwargs["sector_col"] = "sector"
    strategy_kwargs["max_sector_weight"] = 0.30


def _set_stable20_strategy(cfg: Dict) -> None:
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy_kwargs["n_drop"] = 8
    strategy_kwargs["candidate_buffer"] = 3
    strategy_kwargs["risk_target_ann"] = 0.15
    strategy_kwargs["risk_ceiling"] = 0.78
    strategy_kwargs["risk_floor"] = 0.10
    strategy_kwargs["max_risk_step_up"] = 0.06
    strategy_kwargs["max_risk_step_down"] = 0.16
    strategy_kwargs["drawdown_limit"] = 0.08
    strategy_kwargs["drawdown_penalty"] = 0.45
    strategy_kwargs["market_drawdown_limit"] = 0.10
    strategy_kwargs["market_drawdown_penalty"] = 0.60
    strategy_kwargs["sector_map_csv"] = "/root/.qlib/sharadar/raw/tickers.csv"
    strategy_kwargs["sector_ticker_col"] = "ticker"
    strategy_kwargs["sector_col"] = "sector"
    strategy_kwargs["max_sector_weight"] = 0.40


def _set_sizeguard_strategy(cfg: Dict) -> None:
    strategy_kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    strategy_kwargs["feature_score_weights"] = {"$log_marketcap_q": 0.35}
    strategy_kwargs["feature_min_percentiles"] = {"$log_marketcap_q": 0.30}


def _set_benchmark_aware_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.80,
    benchmark_topn: int = 120,
    benchmark_tickers: Optional[Iterable[str]] = None,
    benchmark_tickers_file: str = None,
    max_turnover: float = None,
    min_trade_weight: float = 0.0,
    min_position_weight: float = 0.0,
    max_holdings: int = None,
    max_weight: float = 0.05,
    benchmark_max_weight: float = None,
    max_active_weight: float = 0.04,
    dynamic_alpha_weight: bool = False,
    alpha_quality_window: int = 63,
    alpha_quality_lower_excess: float = -0.03,
    alpha_quality_upper_excess: float = 0.03,
    min_alpha_scale: float = 0.0,
) -> None:
    strategy = cfg["port_analysis_config"]["strategy"]
    old_kwargs = strategy.get("kwargs", {}) or {}
    strategy["class"] = "WeeklyBenchmarkAwareScoreWeightedStrategy"
    strategy["module_path"] = "qlib.contrib.strategy"
    new_kwargs = {
        "topk": int(old_kwargs.get("topk", 40)),
        "rebalance_weekday": int(old_kwargs.get("rebalance_weekday", 0)),
        "risk_degree": float(old_kwargs.get("risk_degree", 0.95)),
        "weighting": "rank",
        "benchmark_core_weight": float(benchmark_core_weight),
        "benchmark_topn": int(benchmark_topn),
        "benchmark_marketcap_field": "$marketcap_q",
        "max_weight": float(max_weight),
        "max_active_weight": float(max_active_weight),
        "sector_map_csv": old_kwargs.get("sector_map_csv") or "/root/.qlib/sharadar/raw/tickers.csv",
        "sector_ticker_col": old_kwargs.get("sector_ticker_col") or "ticker",
        "sector_col": old_kwargs.get("sector_col") or "sector",
        "liquidity_window": 20,
        "liquidity_buffer": 3,
        "vol_window": 63,
        "vol_scale": False,
    }
    if benchmark_tickers_file:
        new_kwargs["benchmark_tickers_file"] = str(benchmark_tickers_file)
    if benchmark_tickers:
        new_kwargs["benchmark_tickers"] = [str(t).strip().upper() for t in benchmark_tickers if str(t).strip()]
    if benchmark_max_weight is not None:
        new_kwargs["benchmark_max_weight"] = float(benchmark_max_weight)
    elif benchmark_tickers and len(new_kwargs["benchmark_tickers"]) == 1:
        new_kwargs["benchmark_max_weight"] = 1.0
    if max_turnover is not None:
        new_kwargs["max_turnover"] = float(max_turnover)
    if min_trade_weight is not None and float(min_trade_weight) > 0:
        new_kwargs["min_trade_weight"] = float(min_trade_weight)
    if min_position_weight is not None and float(min_position_weight) > 0:
        new_kwargs["min_position_weight"] = float(min_position_weight)
    if max_holdings is not None:
        new_kwargs["max_holdings"] = int(max_holdings)
    if dynamic_alpha_weight:
        new_kwargs["dynamic_alpha_weight"] = True
        new_kwargs["alpha_quality_window"] = int(alpha_quality_window)
        new_kwargs["alpha_quality_lower_excess"] = float(alpha_quality_lower_excess)
        new_kwargs["alpha_quality_upper_excess"] = float(alpha_quality_upper_excess)
        new_kwargs["min_alpha_scale"] = float(min_alpha_scale)
    for key in ("max_sector_count", "max_sector_weight", "feature_score_weights", "feature_min_percentiles"):
        if key in old_kwargs:
            new_kwargs[key] = old_kwargs[key]
    strategy["kwargs"] = new_kwargs


def _set_nextgen_growth_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.35,
    topk: int = 25,
    max_active_weight: float = 0.10,
    max_turnover: float = 0.15,
) -> None:
    _set_benchmark_aware_strategy(
        cfg,
        benchmark_core_weight=float(benchmark_core_weight),
        benchmark_topn=1,
        benchmark_tickers=["QQQ"],
        max_turnover=float(max_turnover),
        min_trade_weight=0.001,
        max_holdings=55,
        max_weight=0.20,
        benchmark_max_weight=1.0,
        max_active_weight=float(max_active_weight),
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.01,
        alpha_quality_upper_excess=0.03,
        min_alpha_scale=0.0,
    )
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs["topk"] = int(topk)
    kwargs["max_sector_weight"] = 0.40
    kwargs["risk_degree"] = 1.0


def _set_release_grade_growth_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.45,
    topk: int = 30,
    max_active_weight: float = 0.07,
    max_turnover: float = 0.10,
    min_alpha_scale: float = 0.25,
) -> None:
    _set_benchmark_aware_strategy(
        cfg,
        benchmark_core_weight=float(benchmark_core_weight),
        benchmark_topn=1,
        benchmark_tickers=["QQQ"],
        max_turnover=float(max_turnover),
        min_trade_weight=0.001,
        min_position_weight=0.001,
        max_holdings=60,
        max_weight=0.16,
        benchmark_max_weight=1.0,
        max_active_weight=float(max_active_weight),
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.015,
        alpha_quality_upper_excess=0.035,
        min_alpha_scale=float(min_alpha_scale),
    )
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs.update(
        {
            "topk": int(topk),
            "risk_degree": 1.0,
            "max_sector_weight": 0.35,
            "vol_window": 63,
            "vol_scale": True,
            "feature_score_weights": {
                "$log_marketcap_q": 0.15,
                "$risk_vol_20d": -0.10,
            },
            "feature_min_percentiles": {
                "$log_marketcap_q": 0.25,
            },
            "dynamic_risk": True,
            "market_index": "QQQ",
            "risk_floor": 0.55,
            "risk_ceiling": 1.0,
            "risk_smoothing_up": 0.35,
            "risk_smoothing_down": 0.85,
            "max_risk_step_up": 0.10,
            "max_risk_step_down": 0.35,
            "market_trend_window": 63,
            "market_trend_thresh": -0.04,
            "market_trend_penalty": 0.75,
            "market_trend_boost_thresh": 0.10,
            "market_trend_boost": 1.0,
            "market_drawdown_window": 126,
            "market_drawdown_limit": 0.10,
            "market_drawdown_penalty": 0.70,
            "crash_guard": True,
            "crash_return_lookback": 5,
            "crash_return_limit": 0.06,
            "crash_penalty": 0.65,
            "crash_cooldown_steps": 2,
            "risk_min_history": 40,
        }
    )


def _set_high_qqq_overlay_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.85,
    topk: int = 30,
    max_active_weight: float = 0.08,
    max_turnover: float = 0.10,
    max_holdings: int = 55,
) -> None:
    _set_benchmark_aware_strategy(
        cfg,
        benchmark_core_weight=float(benchmark_core_weight),
        benchmark_topn=1,
        benchmark_tickers=["QQQ"],
        max_turnover=float(max_turnover),
        min_trade_weight=0.001,
        min_position_weight=0.001,
        max_holdings=int(max_holdings),
        max_weight=0.12,
        benchmark_max_weight=1.0,
        max_active_weight=float(max_active_weight),
        dynamic_alpha_weight=False,
    )
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs.update(
        {
            "topk": int(topk),
            "risk_degree": 1.0,
            "max_sector_weight": 0.35,
            "vol_window": 63,
            "vol_scale": False,
            "dynamic_alpha_weight": False,
        }
    )
    kwargs.pop("feature_score_weights", None)
    kwargs.pop("feature_min_percentiles", None)


def _set_qqq_overlay_dynamic_alpha(
    cfg: Dict,
    *,
    min_alpha_scale: float,
    lower_excess: float = -0.02,
    upper_excess: float = 0.04,
) -> None:
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs["dynamic_alpha_weight"] = True
    kwargs["alpha_quality_window"] = 63
    kwargs["alpha_quality_lower_excess"] = float(lower_excess)
    kwargs["alpha_quality_upper_excess"] = float(upper_excess)
    kwargs["min_alpha_scale"] = float(min_alpha_scale)
    kwargs["max_alpha_scale"] = 1.0


def _set_high_return_dynamic_qqq_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.45,
    topk: int = 25,
    max_active_weight: float = 0.10,
    max_turnover: float = 0.12,
    min_alpha_scale: float = 0.25,
) -> None:
    _set_release_grade_growth_strategy(
        cfg,
        benchmark_core_weight=float(benchmark_core_weight),
        topk=int(topk),
        max_active_weight=float(max_active_weight),
        max_turnover=float(max_turnover),
        min_alpha_scale=float(min_alpha_scale),
    )
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs.update(
        {
            "max_sector_weight": 0.30,
            "feature_score_weights": {
                "$log_marketcap_q": 0.12,
                "$risk_vol_20d": -0.18,
                "$risk_beta_qqq_63d": -0.12,
                "$risk_relret_qqq_63d": 0.05,
                "$fmp_alpha_event_composite": 0.04,
            },
            "feature_min_percentiles": {
                "$log_marketcap_q": 0.25,
            },
            "alpha_quality_lower_excess": -0.015,
            "alpha_quality_upper_excess": 0.045,
            "min_alpha_scale": float(min_alpha_scale),
            "risk_floor": 0.50,
            "market_trend_thresh": -0.03,
            "market_trend_penalty": 0.70,
            "market_drawdown_limit": 0.08,
            "market_drawdown_penalty": 0.65,
            "crash_return_limit": 0.045,
            "crash_penalty": 0.55,
            "crash_cooldown_steps": 3,
        }
    )


def _set_mild_dynamic_qqq_strategy(
    cfg: Dict,
    *,
    benchmark_core_weight: float = 0.55,
    topk: int = 30,
    max_active_weight: float = 0.08,
    max_turnover: float = 0.12,
    min_alpha_scale: float = 0.45,
) -> None:
    _set_release_grade_growth_strategy(
        cfg,
        benchmark_core_weight=float(benchmark_core_weight),
        topk=int(topk),
        max_active_weight=float(max_active_weight),
        max_turnover=float(max_turnover),
        min_alpha_scale=float(min_alpha_scale),
    )
    kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
    kwargs.update(
        {
            "max_sector_weight": 0.30,
            "feature_score_weights": {
                "$log_marketcap_q": 0.08,
                "$risk_vol_20d": -0.06,
                "$risk_beta_qqq_63d": -0.04,
                "$fmp_alpha_event_composite": 0.03,
            },
            "feature_min_percentiles": {
                "$log_marketcap_q": 0.25,
            },
            "alpha_quality_lower_excess": -0.020,
            "alpha_quality_upper_excess": 0.045,
            "min_alpha_scale": float(min_alpha_scale),
            "risk_floor": 0.85,
            "market_trend_thresh": -0.03,
            "market_trend_penalty": 0.96,
            "market_drawdown_limit": 0.10,
            "market_drawdown_penalty": 0.92,
            "crash_return_limit": 0.060,
            "crash_penalty": 0.90,
            "crash_cooldown_steps": 2,
        }
    )


def _sync_handler_kwargs(cfg: Dict) -> None:
    cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"] = cfg["data_handler_config"]


def _base_variant(cfg: Dict, exp_name: str, features: List[Tuple[str, str]]) -> Dict:
    out = copy.deepcopy(cfg)
    _set_exp_name(out, exp_name)
    _set_extra_features(out, features)
    _sync_handler_kwargs(out)
    return out


def build_variants(base_cfg: Dict) -> Dict[str, Dict]:
    variants: Dict[str, Dict] = {}

    value_lgb = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_value6_lgb_v1_topk40",
        VALUE6,
    )
    value_lgb["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        value_lgb["data_handler_config"].get("infer_processors", []),
    )
    value_lgb["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        value_lgb["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _set_model_lgb(value_lgb)
    _sync_handler_kwargs(value_lgb)
    variants["value6_lgb_v1"] = value_lgb

    label_neutral = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_value6_label_neutral_lgb_v1_topk40",
        VALUE6,
    )
    label_neutral["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        label_neutral["data_handler_config"].get("infer_processors", []),
    )
    _set_model_lgb(label_neutral)
    _sync_handler_kwargs(label_neutral)
    variants["value6_label_neutral_lgb_v1"] = label_neutral

    ranker = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_value6_ranker_v1_topk40",
        VALUE6,
    )
    ranker["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        ranker["data_handler_config"].get("infer_processors", []),
    )
    ranker["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        ranker["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _set_model_ranker(ranker)
    _sync_handler_kwargs(ranker)
    variants["value6_ranker_v1"] = ranker

    label5 = copy.deepcopy(value_lgb)
    _set_exp_name(label5, "us_sharadar_weekly_pit_value6_lgb_label5_v1_topk40")
    _set_horizon(label5, 5)
    _sync_handler_kwargs(label5)
    variants["value6_lgb_label5_v1"] = label5

    value3_factors = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_value3_factors_lgb_v1_topk40",
        VALUE3,
    )
    value3_factors["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        value3_factors["data_handler_config"].get("infer_processors", []),
    )
    value3_factors["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        value3_factors["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _set_factor_only_handler(value3_factors)
    _set_model_lgb(value3_factors)
    _sync_handler_kwargs(value3_factors)
    variants["value3_factors_lgb_v1"] = value3_factors

    value6_factors = copy.deepcopy(value_lgb)
    _set_exp_name(value6_factors, "us_sharadar_weekly_pit_value6_factors_lgb_v1_topk40")
    _set_factor_only_handler(value6_factors)
    _sync_handler_kwargs(value6_factors)
    variants["value6_factors_lgb_v1"] = value6_factors

    risk_lgb = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_value6_risk_lgb_v1_topk40",
        VALUE6 + RISK_REGIME,
    )
    risk_lgb["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        risk_lgb["data_handler_config"].get("infer_processors", []),
    )
    risk_lgb["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        risk_lgb["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _set_model_lgb(risk_lgb)
    _sync_handler_kwargs(risk_lgb)
    variants["value6_risk_lgb_v1"] = risk_lgb

    risk_lgb_benchmarkaware = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_lgb_benchmarkaware,
        "us_sharadar_weekly_pit_value6_risk_lgb_benchmarkaware_v1_topk40",
    )
    _set_benchmark_aware_strategy(risk_lgb_benchmarkaware, benchmark_core_weight=0.75, benchmark_topn=100)
    _sync_handler_kwargs(risk_lgb_benchmarkaware)
    variants["value6_risk_lgb_benchmarkaware_v1"] = risk_lgb_benchmarkaware

    risk_lgb_h5_benchmarkaware = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_lgb_h5_benchmarkaware,
        "us_sharadar_weekly_pit_value6_risk_lgb_h5_benchmarkaware_v1_topk40",
    )
    _set_horizon(risk_lgb_h5_benchmarkaware, 5)
    _set_benchmark_aware_strategy(risk_lgb_h5_benchmarkaware, benchmark_core_weight=0.75, benchmark_topn=100)
    _sync_handler_kwargs(risk_lgb_h5_benchmarkaware)
    variants["value6_risk_lgb_h5_benchmarkaware_v1"] = risk_lgb_h5_benchmarkaware

    risk_lgb_h5_benchmarkaware_lowturn = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_lgb_h5_benchmarkaware_lowturn,
        "us_sharadar_weekly_pit_value6_risk_lgb_h5_benchmarkaware_lowturn_v1_topk40",
    )
    _set_horizon(risk_lgb_h5_benchmarkaware_lowturn, 5)
    _set_benchmark_aware_strategy(
        risk_lgb_h5_benchmarkaware_lowturn,
        benchmark_core_weight=0.75,
        benchmark_topn=100,
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=140,
    )
    _sync_handler_kwargs(risk_lgb_h5_benchmarkaware_lowturn)
    variants["value6_risk_lgb_h5_benchmarkaware_lowturn_v1"] = risk_lgb_h5_benchmarkaware_lowturn

    risk_lgb_h5_benchmarkaware_lowturn_core80 = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_lgb_h5_benchmarkaware_lowturn_core80,
        "us_sharadar_weekly_pit_value6_risk_lgb_h5_benchmarkaware_lowturn_core80_v1_topk40",
    )
    _set_horizon(risk_lgb_h5_benchmarkaware_lowturn_core80, 5)
    _set_benchmark_aware_strategy(
        risk_lgb_h5_benchmarkaware_lowturn_core80,
        benchmark_core_weight=0.80,
        benchmark_topn=100,
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=140,
    )
    _sync_handler_kwargs(risk_lgb_h5_benchmarkaware_lowturn_core80)
    variants["value6_risk_lgb_h5_benchmarkaware_lowturn_core80_v1"] = risk_lgb_h5_benchmarkaware_lowturn_core80

    risk_ranker_h5_benchmarkaware_lowturn = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_ranker_h5_benchmarkaware_lowturn,
        "us_sharadar_weekly_pit_value6_risk_ranker_h5_benchmarkaware_lowturn_v1_topk40",
    )
    _set_horizon(risk_ranker_h5_benchmarkaware_lowturn, 5)
    _set_model_ranker(risk_ranker_h5_benchmarkaware_lowturn, eval_at=[40])
    _set_benchmark_aware_strategy(
        risk_ranker_h5_benchmarkaware_lowturn,
        benchmark_core_weight=0.75,
        benchmark_topn=100,
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=140,
    )
    _sync_handler_kwargs(risk_ranker_h5_benchmarkaware_lowturn)
    variants["value6_risk_ranker_h5_benchmarkaware_lowturn_v1"] = risk_ranker_h5_benchmarkaware_lowturn

    risk_ranker_h5_etfcore = copy.deepcopy(risk_lgb)
    _set_exp_name(
        risk_ranker_h5_etfcore,
        "us_sharadar_weekly_pit_value6_risk_ranker_h5_etfcore_v1_topk40",
    )
    _set_horizon(risk_ranker_h5_etfcore, 5)
    _set_model_ranker(risk_ranker_h5_etfcore, eval_at=[40])
    _set_benchmark_aware_strategy(
        risk_ranker_h5_etfcore,
        benchmark_core_weight=0.75,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(risk_ranker_h5_etfcore)
    variants["value6_risk_ranker_h5_etfcore_v1"] = risk_ranker_h5_etfcore

    risk_rank = copy.deepcopy(risk_lgb)
    _set_exp_name(risk_rank, "us_sharadar_weekly_pit_value6_risk_rank_lgb_v1_topk40")
    _add_label_rank_norm(risk_rank)
    _sync_handler_kwargs(risk_rank)
    variants["value6_risk_rank_lgb_v1"] = risk_rank

    risk_rank_lowturn = copy.deepcopy(risk_rank)
    _set_exp_name(risk_rank_lowturn, "us_sharadar_weekly_pit_value6_risk_rank_lowturn_lgb_v1_topk40")
    _set_low_turnover_strategy(risk_rank_lowturn)
    _sync_handler_kwargs(risk_rank_lowturn)
    variants["value6_risk_rank_lowturn_lgb_v1"] = risk_rank_lowturn

    sf3a_regime_ranker = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_v1_topk40",
        QUALITY_VALUE + RISK_REGIME + SECTOR_REGIME + REGIME_INTERACTIONS + SF3A,
    )
    _set_selective_feature_norm(sf3a_regime_ranker, exclude_prefixes=("MKT_",))
    _set_model_ranker(sf3a_regime_ranker)
    _set_conservative_risk_strategy(sf3a_regime_ranker)
    _sync_handler_kwargs(sf3a_regime_ranker)
    variants["qv_risk_sf3a_regime_ranker_v1"] = sf3a_regime_ranker

    sf3a_regime_ranker_benchmarkaware = copy.deepcopy(sf3a_regime_ranker)
    _set_exp_name(
        sf3a_regime_ranker_benchmarkaware,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_benchmarkaware_v1_topk40",
    )
    _set_benchmark_aware_strategy(sf3a_regime_ranker_benchmarkaware)
    _sync_handler_kwargs(sf3a_regime_ranker_benchmarkaware)
    variants["qv_risk_sf3a_regime_ranker_benchmarkaware_v1"] = sf3a_regime_ranker_benchmarkaware

    sf3a_regime_ranker_raw = copy.deepcopy(sf3a_regime_ranker)
    _set_exp_name(sf3a_regime_ranker_raw, "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_raw_v1_topk40")
    sf3a_regime_ranker_raw["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        sf3a_regime_ranker_raw["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _sync_handler_kwargs(sf3a_regime_ranker_raw)
    variants["qv_risk_sf3a_regime_ranker_raw_v1"] = sf3a_regime_ranker_raw

    sf3a_regime_ranker_h5 = copy.deepcopy(sf3a_regime_ranker)
    _set_exp_name(sf3a_regime_ranker_h5, "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_v1_topk40")
    _set_horizon(sf3a_regime_ranker_h5, 5)
    _sync_handler_kwargs(sf3a_regime_ranker_h5)
    variants["qv_risk_sf3a_regime_ranker_h5_v1"] = sf3a_regime_ranker_h5

    sf3a_regime_ranker_h5_etfcore = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfcore_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfcore,
        benchmark_core_weight=0.75,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h5_etfcore_v1"] = sf3a_regime_ranker_h5_etfcore

    sf3a_regime_ranker_h5_etfactive35 = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfactive35,
        benchmark_core_weight=0.65,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.055,
    )
    sf3a_regime_ranker_h5_etfactive35["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.35
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_v1"] = sf3a_regime_ranker_h5_etfactive35

    riskguard_score_weights = {
        "$log_marketcap_q": 0.15,
        "$risk_vol_20d": -0.20,
        "$risk_beta_spy_63d": -0.10,
        "$earn_yield_q": 0.07,
        "$fcf_yield_q": 0.07,
    }
    riskguard_min_percentiles = {"$log_marketcap_q": 0.20}

    sf3a_regime_ranker_h5_etfactive35_riskguard = copy.deepcopy(sf3a_regime_ranker_h5_etfactive35)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_riskguard,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_riskguard_v1_topk40",
    )
    riskguard_kwargs = sf3a_regime_ranker_h5_etfactive35_riskguard["port_analysis_config"]["strategy"]["kwargs"]
    riskguard_kwargs["feature_score_weights"] = copy.deepcopy(riskguard_score_weights)
    riskguard_kwargs["feature_min_percentiles"] = copy.deepcopy(riskguard_min_percentiles)
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_riskguard)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_riskguard_v1"] = (
        sf3a_regime_ranker_h5_etfactive35_riskguard
    )

    sf3a_regime_ranker_h5_etfactive35_dyn = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_dyn,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfactive35_dyn,
        benchmark_core_weight=0.65,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.055,
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.03,
        alpha_quality_upper_excess=0.04,
        min_alpha_scale=0.30,
    )
    sf3a_regime_ranker_h5_etfactive35_dyn["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.35
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_dyn)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha_v1"] = sf3a_regime_ranker_h5_etfactive35_dyn

    sf3a_regime_ranker_h5_etfactive35_dyn60 = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_dyn60,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha60_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfactive35_dyn60,
        benchmark_core_weight=0.65,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.055,
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.03,
        alpha_quality_upper_excess=0.04,
        min_alpha_scale=0.60,
    )
    sf3a_regime_ranker_h5_etfactive35_dyn60["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.35
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_dyn60)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha60_v1"] = sf3a_regime_ranker_h5_etfactive35_dyn60

    sf3a_regime_ranker_h5_etfactive35_dyn70 = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_dyn70,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha70_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfactive35_dyn70,
        benchmark_core_weight=0.65,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.055,
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.03,
        alpha_quality_upper_excess=0.04,
        min_alpha_scale=0.70,
    )
    sf3a_regime_ranker_h5_etfactive35_dyn70["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.35
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_dyn70)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha70_v1"] = sf3a_regime_ranker_h5_etfactive35_dyn70

    sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard = copy.deepcopy(sf3a_regime_ranker_h5_etfactive35_dyn70)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_v1_topk40",
    )
    dyn85_riskguard_kwargs = sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard["port_analysis_config"]["strategy"][
        "kwargs"
    ]
    dyn85_riskguard_kwargs["min_alpha_scale"] = 0.85
    dyn85_riskguard_kwargs["feature_score_weights"] = copy.deepcopy(riskguard_score_weights)
    dyn85_riskguard_kwargs["feature_min_percentiles"] = copy.deepcopy(riskguard_min_percentiles)
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_v1"] = (
        sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard
    )

    sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard_dynrisk = copy.deepcopy(
        sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard
    )
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard_dynrisk,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_dynrisk_v1_topk40",
    )
    dynrisk_kwargs = sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard_dynrisk["port_analysis_config"][
        "strategy"
    ]["kwargs"]
    dynrisk_kwargs.update(
        {
            "dynamic_risk": True,
            "market_index": "SPY",
            "risk_floor": 0.0,
            "risk_ceiling": 0.95,
            "risk_smoothing_up": 0.60,
            "risk_smoothing_down": 1.0,
            "max_risk_step_up": 0.25,
            "max_risk_step_down": 0.95,
            "market_trend_window": 63,
            "market_trend_thresh": -0.04,
            "market_trend_penalty": 0.0,
            "market_trend_boost_thresh": 0.08,
            "market_trend_boost": 1.0,
            "market_drawdown_window": 126,
            "market_drawdown_limit": 0.10,
            "market_drawdown_penalty": 0.0,
            "crash_guard": True,
            "crash_return_lookback": 5,
            "crash_return_limit": 0.06,
            "crash_penalty": 0.0,
            "crash_cooldown_steps": 2,
            "risk_min_history": 40,
        }
    )
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard_dynrisk)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_dynalpha85_riskguard_dynrisk_v1"] = (
        sf3a_regime_ranker_h5_etfactive35_dyn85_riskguard_dynrisk
    )

    sf3a_regime_ranker_h5_etfactive35_residual = copy.deepcopy(sf3a_regime_ranker_h5_etfactive35_riskguard)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_residual,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_v1_topk40",
    )
    _set_residual_label(sf3a_regime_ranker_h5_etfactive35_residual)
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_residual)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_v1"] = (
        sf3a_regime_ranker_h5_etfactive35_residual
    )

    sf3a_regime_ranker_h5_etfactive35_residual_hedged = copy.deepcopy(
        sf3a_regime_ranker_h5_etfactive35_residual
    )
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive35_residual_hedged,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_hedged_v1_topk40",
    )
    _set_hedged_benchmark_aware_strategy(sf3a_regime_ranker_h5_etfactive35_residual_hedged)
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive35_residual_hedged)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive35_residual_hedged_v1"] = (
        sf3a_regime_ranker_h5_etfactive35_residual_hedged
    )

    sf3a_regime_ranker_h5_etfactive31 = copy.deepcopy(sf3a_regime_ranker_h5)
    _set_exp_name(
        sf3a_regime_ranker_h5_etfactive31,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_etfactive31_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h5_etfactive31,
        benchmark_core_weight=0.69,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.050,
    )
    sf3a_regime_ranker_h5_etfactive31["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.32
    _sync_handler_kwargs(sf3a_regime_ranker_h5_etfactive31)
    variants["qv_risk_sf3a_regime_ranker_h5_etfactive31_v1"] = sf3a_regime_ranker_h5_etfactive31

    sf3a_regime_lgb_h5_etfcore = copy.deepcopy(sf3a_regime_ranker_h5_etfcore)
    _set_exp_name(
        sf3a_regime_lgb_h5_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_lgb_h5_etfcore_v1_topk40",
    )
    _set_model_lgb(sf3a_regime_lgb_h5_etfcore)
    _sync_handler_kwargs(sf3a_regime_lgb_h5_etfcore)
    variants["qv_risk_sf3a_regime_lgb_h5_etfcore_v1"] = sf3a_regime_lgb_h5_etfcore

    sf3a_regime_lgb_rank_h5_etfcore = copy.deepcopy(sf3a_regime_lgb_h5_etfcore)
    _set_exp_name(
        sf3a_regime_lgb_rank_h5_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_lgb_rank_h5_etfcore_v1_topk40",
    )
    _add_label_rank_norm(sf3a_regime_lgb_rank_h5_etfcore)
    _sync_handler_kwargs(sf3a_regime_lgb_rank_h5_etfcore)
    variants["qv_risk_sf3a_regime_lgb_rank_h5_etfcore_v1"] = sf3a_regime_lgb_rank_h5_etfcore

    sf3a_regime_ranker_h5_raw_etfcore = copy.deepcopy(sf3a_regime_ranker_h5_etfcore)
    _set_exp_name(
        sf3a_regime_ranker_h5_raw_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h5_raw_etfcore_v1_topk40",
    )
    sf3a_regime_ranker_h5_raw_etfcore["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        sf3a_regime_ranker_h5_raw_etfcore["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _sync_handler_kwargs(sf3a_regime_ranker_h5_raw_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h5_raw_etfcore_v1"] = sf3a_regime_ranker_h5_raw_etfcore

    sf3a_regime_ranker_h20 = copy.deepcopy(sf3a_regime_ranker)
    _set_exp_name(sf3a_regime_ranker_h20, "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h20_v1_topk40")
    _set_horizon(sf3a_regime_ranker_h20, 20)
    _sync_handler_kwargs(sf3a_regime_ranker_h20)
    variants["qv_risk_sf3a_regime_ranker_h20_v1"] = sf3a_regime_ranker_h20

    sf3a_regime_ranker_h20_etfcore = copy.deepcopy(sf3a_regime_ranker_h20)
    _set_exp_name(
        sf3a_regime_ranker_h20_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h20_etfcore_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h20_etfcore,
        benchmark_core_weight=0.80,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(sf3a_regime_ranker_h20_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h20_etfcore_v1"] = sf3a_regime_ranker_h20_etfcore

    sf3a_regime_ranker_h20_residual_etfcore = copy.deepcopy(sf3a_regime_ranker_h20_etfcore)
    _set_exp_name(
        sf3a_regime_ranker_h20_residual_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h20_residual_etfcore_v1_topk40",
    )
    _set_residual_label(sf3a_regime_ranker_h20_residual_etfcore)
    _sync_handler_kwargs(sf3a_regime_ranker_h20_residual_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h20_residual_etfcore_v1"] = (
        sf3a_regime_ranker_h20_residual_etfcore
    )

    sf3a_regime_ranker_h20_volscaled_etfcore = copy.deepcopy(sf3a_regime_ranker_h20_etfcore)
    _set_exp_name(
        sf3a_regime_ranker_h20_volscaled_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h20_volscaled_etfcore_v1_topk40",
    )
    _set_vol_scaled_label(sf3a_regime_ranker_h20_volscaled_etfcore)
    _sync_handler_kwargs(sf3a_regime_ranker_h20_volscaled_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h20_volscaled_etfcore_v1"] = (
        sf3a_regime_ranker_h20_volscaled_etfcore
    )

    sf3a_regime_ranker_h60 = copy.deepcopy(sf3a_regime_ranker)
    _set_exp_name(sf3a_regime_ranker_h60, "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h60_v1_topk40")
    _set_horizon(sf3a_regime_ranker_h60, 60)
    _sync_handler_kwargs(sf3a_regime_ranker_h60)
    variants["qv_risk_sf3a_regime_ranker_h60_v1"] = sf3a_regime_ranker_h60

    sf3a_regime_ranker_h60_etfcore = copy.deepcopy(sf3a_regime_ranker_h60)
    _set_exp_name(
        sf3a_regime_ranker_h60_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h60_etfcore_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        sf3a_regime_ranker_h60_etfcore,
        benchmark_core_weight=0.80,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.06,
        min_trade_weight=0.001,
        max_holdings=80,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(sf3a_regime_ranker_h60_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h60_etfcore_v1"] = sf3a_regime_ranker_h60_etfcore

    sf3a_regime_ranker_h60_residual_etfcore = copy.deepcopy(sf3a_regime_ranker_h60_etfcore)
    _set_exp_name(
        sf3a_regime_ranker_h60_residual_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_ranker_h60_residual_etfcore_v1_topk40",
    )
    _set_residual_label(sf3a_regime_ranker_h60_residual_etfcore)
    _sync_handler_kwargs(sf3a_regime_ranker_h60_residual_etfcore)
    variants["qv_risk_sf3a_regime_ranker_h60_residual_etfcore_v1"] = (
        sf3a_regime_ranker_h60_residual_etfcore
    )

    sf3a_regime_lgb_rank_h60_etfcore = copy.deepcopy(sf3a_regime_ranker_h60_etfcore)
    _set_exp_name(
        sf3a_regime_lgb_rank_h60_etfcore,
        "us_sharadar_weekly_pit_qv_risk_sf3a_regime_lgb_rank_h60_etfcore_v1_topk40",
    )
    _set_model_lgb(sf3a_regime_lgb_rank_h60_etfcore)
    _add_label_rank_norm(sf3a_regime_lgb_rank_h60_etfcore)
    _sync_handler_kwargs(sf3a_regime_lgb_rank_h60_etfcore)
    variants["qv_risk_sf3a_regime_lgb_rank_h60_etfcore_v1"] = sf3a_regime_lgb_rank_h60_etfcore

    stable20_factors_lgb_rank = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_stable20_factors_lgb_rank_v1_topk40",
        STABLE20_FEATURES,
    )
    stable20_factors_lgb_rank["data_handler_config"]["infer_processors"] = _remove_group_neutralize(
        stable20_factors_lgb_rank["data_handler_config"].get("infer_processors", []),
    )
    stable20_factors_lgb_rank["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
        stable20_factors_lgb_rank["data_handler_config"].get("learn_processors", []),
        fields_group="label",
    )
    _set_factor_only_handler(stable20_factors_lgb_rank)
    _set_selective_feature_norm(stable20_factors_lgb_rank, exclude_prefixes=("MKT_",))
    _set_model_lgb(stable20_factors_lgb_rank)
    _add_label_rank_norm(stable20_factors_lgb_rank)
    _set_horizon(stable20_factors_lgb_rank, 20)
    _set_stable20_strategy(stable20_factors_lgb_rank)
    _sync_handler_kwargs(stable20_factors_lgb_rank)
    variants["stable20_factors_lgb_rank_v1"] = stable20_factors_lgb_rank

    stable20_factors_ranker = copy.deepcopy(stable20_factors_lgb_rank)
    _set_exp_name(stable20_factors_ranker, "us_sharadar_weekly_pit_stable20_factors_ranker_v1_topk40")
    stable20_factors_ranker["data_handler_config"]["learn_processors"] = [
        proc
        for proc in stable20_factors_ranker["data_handler_config"].get("learn_processors", [])
        if not (isinstance(proc, dict) and _processor_class(proc) == "CSRankNorm")
    ]
    _set_model_ranker(stable20_factors_ranker)
    _sync_handler_kwargs(stable20_factors_ranker)
    variants["stable20_factors_ranker_v1"] = stable20_factors_ranker

    stable10_factors_lgb_rank = copy.deepcopy(stable20_factors_lgb_rank)
    _set_exp_name(stable10_factors_lgb_rank, "us_sharadar_weekly_pit_stable10_factors_lgb_rank_v1_topk40")
    _set_horizon(stable10_factors_lgb_rank, 10)
    _sync_handler_kwargs(stable10_factors_lgb_rank)
    variants["stable10_factors_lgb_rank_v1"] = stable10_factors_lgb_rank

    stable20_factors_lgb_rank_no13f = copy.deepcopy(stable20_factors_lgb_rank)
    _set_exp_name(
        stable20_factors_lgb_rank_no13f,
        "us_sharadar_weekly_pit_stable20_factors_lgb_rank_no13f_v1_topk40",
    )
    _set_extra_features(stable20_factors_lgb_rank_no13f, QUALITY_VALUE + STABLE_MOMENTUM)
    _sync_handler_kwargs(stable20_factors_lgb_rank_no13f)
    variants["stable20_factors_lgb_rank_no13f_v1"] = stable20_factors_lgb_rank_no13f

    stable20_factors_lgb_rank_no_mom = copy.deepcopy(stable20_factors_lgb_rank)
    _set_exp_name(
        stable20_factors_lgb_rank_no_mom,
        "us_sharadar_weekly_pit_stable20_factors_lgb_rank_no_mom_v1_topk40",
    )
    _set_extra_features(stable20_factors_lgb_rank_no_mom, QUALITY_VALUE + SELECTED_SF3A)
    _sync_handler_kwargs(stable20_factors_lgb_rank_no_mom)
    variants["stable20_factors_lgb_rank_no_mom_v1"] = stable20_factors_lgb_rank_no_mom

    stable20_capmom_lgb_rank = copy.deepcopy(stable20_factors_lgb_rank)
    _set_exp_name(
        stable20_capmom_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_lgb_rank_v1_topk40",
    )
    _set_extra_features(stable20_capmom_lgb_rank, STABLE20_CAPMOM_FEATURES)
    _sync_handler_kwargs(stable20_capmom_lgb_rank)
    variants["stable20_capmom_lgb_rank_v1"] = stable20_capmom_lgb_rank

    stable20_capmom_sizeguard_lgb_rank = copy.deepcopy(stable20_capmom_lgb_rank)
    _set_exp_name(
        stable20_capmom_sizeguard_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_sizeguard_lgb_rank_v1_topk40",
    )
    _set_sizeguard_strategy(stable20_capmom_sizeguard_lgb_rank)
    _sync_handler_kwargs(stable20_capmom_sizeguard_lgb_rank)
    variants["stable20_capmom_sizeguard_lgb_rank_v1"] = stable20_capmom_sizeguard_lgb_rank

    stable20_capmom_benchmarkaware_lgb_rank = copy.deepcopy(stable20_capmom_lgb_rank)
    _set_exp_name(
        stable20_capmom_benchmarkaware_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_benchmarkaware_lgb_rank_v1_topk40",
    )
    _set_benchmark_aware_strategy(stable20_capmom_benchmarkaware_lgb_rank)
    _sync_handler_kwargs(stable20_capmom_benchmarkaware_lgb_rank)
    variants["stable20_capmom_benchmarkaware_lgb_rank_v1"] = stable20_capmom_benchmarkaware_lgb_rank

    stable20_capmom_etfcore_lgb_rank = copy.deepcopy(stable20_capmom_lgb_rank)
    _set_exp_name(
        stable20_capmom_etfcore_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_etfcore_lgb_rank_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        stable20_capmom_etfcore_lgb_rank,
        benchmark_core_weight=0.80,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=60,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(stable20_capmom_etfcore_lgb_rank)
    variants["stable20_capmom_etfcore_lgb_rank_v1"] = stable20_capmom_etfcore_lgb_rank

    stable20_capmom_sectorneutral_etfcore_lgb_rank = _base_variant(
        base_cfg,
        "us_sharadar_weekly_pit_stable20_capmom_sectorneutral_etfcore_lgb_rank_v1_topk40",
        STABLE20_CAPMOM_FEATURES,
    )
    _set_factor_only_handler(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    _set_selective_feature_norm(stable20_capmom_sectorneutral_etfcore_lgb_rank, exclude_prefixes=("MKT_",))
    _set_model_lgb(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    _add_label_rank_norm(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    _set_horizon(stable20_capmom_sectorneutral_etfcore_lgb_rank, 20)
    _set_stable20_strategy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    _set_benchmark_aware_strategy(
        stable20_capmom_sectorneutral_etfcore_lgb_rank,
        benchmark_core_weight=0.80,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=60,
        max_weight=0.20,
        max_active_weight=0.04,
    )
    _sync_handler_kwargs(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_v1"] = (
        stable20_capmom_sectorneutral_etfcore_lgb_rank
    )

    stable60_capmom_sectorneutral_etfcore_lgb_rank = copy.deepcopy(
        stable20_capmom_sectorneutral_etfcore_lgb_rank
    )
    _set_exp_name(
        stable60_capmom_sectorneutral_etfcore_lgb_rank,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_v1_topk40",
    )
    _set_horizon(stable60_capmom_sectorneutral_etfcore_lgb_rank, 60)
    stable60_kwargs = stable60_capmom_sectorneutral_etfcore_lgb_rank["port_analysis_config"]["strategy"]["kwargs"]
    stable60_kwargs["max_turnover"] = 0.06
    stable60_kwargs["max_holdings"] = 80
    _sync_handler_kwargs(stable60_capmom_sectorneutral_etfcore_lgb_rank)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_v1"] = (
        stable60_capmom_sectorneutral_etfcore_lgb_rank
    )

    stable60_release_v2_core67_topk20 = copy.deepcopy(stable60_capmom_sectorneutral_etfcore_lgb_rank)
    _set_exp_name(
        stable60_release_v2_core67_topk20,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_topk20",
    )
    release_kwargs = stable60_release_v2_core67_topk20["port_analysis_config"]["strategy"]["kwargs"]
    release_kwargs["topk"] = 20
    release_kwargs["benchmark_core_weight"] = 0.67
    release_kwargs["max_turnover"] = 0.06
    release_kwargs["max_holdings"] = 80
    release_kwargs["max_sector_weight"] = 0.35
    _sync_handler_kwargs(stable60_release_v2_core67_topk20)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_topk20"] = (
        stable60_release_v2_core67_topk20
    )

    stable60_release_v2_core70_topk20 = copy.deepcopy(stable60_release_v2_core67_topk20)
    _set_exp_name(
        stable60_release_v2_core70_topk20,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core70_topk20",
    )
    stable60_release_v2_core70_topk20["port_analysis_config"]["strategy"]["kwargs"][
        "benchmark_core_weight"
    ] = 0.70
    _sync_handler_kwargs(stable60_release_v2_core70_topk20)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core70_topk20"] = (
        stable60_release_v2_core70_topk20
    )

    stable60_release_v3_core67_lr005_topk40 = copy.deepcopy(stable60_release_v2_core67_topk20)
    _set_exp_name(
        stable60_release_v3_core67_lr005_topk40,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v3_core67_lr005_topk40",
    )
    release_v3_kwargs = stable60_release_v3_core67_lr005_topk40["port_analysis_config"]["strategy"]["kwargs"]
    release_v3_kwargs["topk"] = 40
    release_v3_kwargs["benchmark_core_weight"] = 0.67
    release_v3_model_kwargs = stable60_release_v3_core67_lr005_topk40["task"]["model"]["kwargs"]
    release_v3_model_kwargs["learning_rate"] = 0.005
    release_v3_model_kwargs["num_boost_round"] = 4000
    release_v3_model_kwargs["early_stopping_rounds"] = 200
    _sync_handler_kwargs(stable60_release_v3_core67_lr005_topk40)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v3_core67_lr005_topk40"] = (
        stable60_release_v3_core67_lr005_topk40
    )

    stable60_growth_h60_qqq_core25_topk30 = copy.deepcopy(stable60_capmom_sectorneutral_etfcore_lgb_rank)
    _set_exp_name(
        stable60_growth_h60_qqq_core25_topk30,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_qqqexcess_core25_topk30",
    )
    _set_horizon(stable60_growth_h60_qqq_core25_topk30, 60)
    _set_benchmark_excess_label(
        stable60_growth_h60_qqq_core25_topk30,
        benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
    )
    growth_h60_kwargs = stable60_growth_h60_qqq_core25_topk30["port_analysis_config"]["strategy"]["kwargs"]
    growth_h60_kwargs["topk"] = 30
    growth_h60_kwargs["benchmark_core_weight"] = 0.25
    growth_h60_kwargs["benchmark_topn"] = 1
    growth_h60_kwargs["benchmark_tickers"] = ["QQQ"]
    growth_h60_kwargs.pop("benchmark_tickers_file", None)
    growth_h60_kwargs["benchmark_max_weight"] = 1.0
    growth_h60_kwargs["max_turnover"] = 0.15
    growth_h60_kwargs["max_holdings"] = 60
    growth_h60_kwargs["max_sector_weight"] = 0.45
    growth_h60_kwargs["max_active_weight"] = 0.12
    growth_h60_kwargs["risk_degree"] = 1.0
    _set_slow_lgb_training(stable60_growth_h60_qqq_core25_topk30)
    _sync_handler_kwargs(stable60_growth_h60_qqq_core25_topk30)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_qqqexcess_core25_topk30"] = (
        stable60_growth_h60_qqq_core25_topk30
    )

    stable20_growth_h20_qqq_core25_topk30 = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
    _set_exp_name(
        stable20_growth_h20_qqq_core25_topk30,
        "us_sharadar_weekly_pit_stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_qqqexcess_core25_topk30",
    )
    _set_horizon(stable20_growth_h20_qqq_core25_topk30, 20)
    _set_benchmark_excess_label(
        stable20_growth_h20_qqq_core25_topk30,
        benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
    )
    growth_h20_kwargs = stable20_growth_h20_qqq_core25_topk30["port_analysis_config"]["strategy"]["kwargs"]
    growth_h20_kwargs["topk"] = 30
    growth_h20_kwargs["benchmark_core_weight"] = 0.25
    growth_h20_kwargs["benchmark_topn"] = 1
    growth_h20_kwargs["benchmark_tickers"] = ["QQQ"]
    growth_h20_kwargs.pop("benchmark_tickers_file", None)
    growth_h20_kwargs["benchmark_max_weight"] = 1.0
    growth_h20_kwargs["max_turnover"] = 0.15
    growth_h20_kwargs["max_holdings"] = 60
    growth_h20_kwargs["max_sector_weight"] = 0.45
    growth_h20_kwargs["max_active_weight"] = 0.12
    growth_h20_kwargs["risk_degree"] = 1.0
    _set_slow_lgb_training(stable20_growth_h20_qqq_core25_topk30)
    _sync_handler_kwargs(stable20_growth_h20_qqq_core25_topk30)
    variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_qqqexcess_core25_topk30"] = (
        stable20_growth_h20_qqq_core25_topk30
    )

    stable20_growth_h20_abs_core25_topk30 = copy.deepcopy(stable20_growth_h20_qqq_core25_topk30)
    _set_exp_name(
        stable20_growth_h20_abs_core25_topk30,
        "us_sharadar_weekly_pit_stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_abs_core25_topk30",
    )
    _set_absolute_return_label(stable20_growth_h20_abs_core25_topk30)
    _add_label_rank_norm(stable20_growth_h20_abs_core25_topk30)
    _sync_handler_kwargs(stable20_growth_h20_abs_core25_topk30)
    variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_abs_core25_topk30"] = (
        stable20_growth_h20_abs_core25_topk30
    )

    stable20_growth_h20_residual_core25_topk30 = copy.deepcopy(stable20_growth_h20_qqq_core25_topk30)
    _set_exp_name(
        stable20_growth_h20_residual_core25_topk30,
        "us_sharadar_weekly_pit_stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_betaresqqq_core25_topk30",
    )
    _append_extra_feature(stable20_growth_h20_residual_core25_topk30, "$risk_beta_spy_63d", "RISK_BETA_SPY_63D")
    _set_residual_label(stable20_growth_h20_residual_core25_topk30, beta_feature="RISK_BETA_SPY_63D")
    _add_label_rank_norm(stable20_growth_h20_residual_core25_topk30)
    _sync_handler_kwargs(stable20_growth_h20_residual_core25_topk30)
    variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_betaresqqq_core25_topk30"] = (
        stable20_growth_h20_residual_core25_topk30
    )

    stable40_growth_h40_abs_core25_topk30 = copy.deepcopy(stable20_growth_h20_abs_core25_topk30)
    _set_exp_name(
        stable40_growth_h40_abs_core25_topk30,
        "us_sharadar_weekly_pit_stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_abs_core25_topk30",
    )
    _set_horizon(stable40_growth_h40_abs_core25_topk30, 40)
    _set_absolute_return_label(stable40_growth_h40_abs_core25_topk30)
    _add_label_rank_norm(stable40_growth_h40_abs_core25_topk30)
    _sync_handler_kwargs(stable40_growth_h40_abs_core25_topk30)
    variants["stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_abs_core25_topk30"] = (
        stable40_growth_h40_abs_core25_topk30
    )

    stable40_growth_h40_residual_core25_topk30 = copy.deepcopy(stable20_growth_h20_qqq_core25_topk30)
    _set_exp_name(
        stable40_growth_h40_residual_core25_topk30,
        "us_sharadar_weekly_pit_stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_betaresqqq_core25_topk30",
    )
    _set_horizon(stable40_growth_h40_residual_core25_topk30, 40)
    _append_extra_feature(stable40_growth_h40_residual_core25_topk30, "$risk_beta_spy_63d", "RISK_BETA_SPY_63D")
    _set_residual_label(stable40_growth_h40_residual_core25_topk30, beta_feature="RISK_BETA_SPY_63D")
    _add_label_rank_norm(stable40_growth_h40_residual_core25_topk30)
    _sync_handler_kwargs(stable40_growth_h40_residual_core25_topk30)
    variants["stable40_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h40_betaresqqq_core25_topk30"] = (
        stable40_growth_h40_residual_core25_topk30
    )

    stable5_growth_h5_qqq_core25_topk30 = copy.deepcopy(stable20_growth_h20_qqq_core25_topk30)
    _set_exp_name(
        stable5_growth_h5_qqq_core25_topk30,
        "us_sharadar_weekly_pit_stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_qqqexcess_core25_topk30",
    )
    _set_horizon(stable5_growth_h5_qqq_core25_topk30, 5)
    _set_benchmark_excess_label(
        stable5_growth_h5_qqq_core25_topk30,
        benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
    )
    _sync_handler_kwargs(stable5_growth_h5_qqq_core25_topk30)
    variants["stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_qqqexcess_core25_topk30"] = (
        stable5_growth_h5_qqq_core25_topk30
    )

    stable5_growth_h5_abs_core25_topk30 = copy.deepcopy(stable5_growth_h5_qqq_core25_topk30)
    _set_exp_name(
        stable5_growth_h5_abs_core25_topk30,
        "us_sharadar_weekly_pit_stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_abs_core25_topk30",
    )
    _set_absolute_return_label(stable5_growth_h5_abs_core25_topk30)
    _add_label_rank_norm(stable5_growth_h5_abs_core25_topk30)
    _sync_handler_kwargs(stable5_growth_h5_abs_core25_topk30)
    variants["stable5_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h5_abs_core25_topk30"] = (
        stable5_growth_h5_abs_core25_topk30
    )

    stable20_growth_h20_volscaled_core25_topk30 = copy.deepcopy(stable20_growth_h20_qqq_core25_topk30)
    _set_exp_name(
        stable20_growth_h20_volscaled_core25_topk30,
        "us_sharadar_weekly_pit_stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_volscaledqqq_core25_topk30",
    )
    _append_extra_feature(stable20_growth_h20_volscaled_core25_topk30, "$risk_vol_20d", "RISK_VOL_20D")
    _set_vol_scaled_label(
        stable20_growth_h20_volscaled_core25_topk30,
        vol_feature="RISK_VOL_20D",
        clip_abs_label=8.0,
    )
    _add_label_rank_norm(stable20_growth_h20_volscaled_core25_topk30)
    _sync_handler_kwargs(stable20_growth_h20_volscaled_core25_topk30)
    variants["stable20_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h20_volscaledqqq_core25_topk30"] = (
        stable20_growth_h20_volscaled_core25_topk30
    )

    stable60_growth_h60_volscaled_core25_topk30 = copy.deepcopy(stable60_growth_h60_qqq_core25_topk30)
    _set_exp_name(
        stable60_growth_h60_volscaled_core25_topk30,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_volscaledqqq_core25_topk30",
    )
    _append_extra_feature(stable60_growth_h60_volscaled_core25_topk30, "$risk_vol_20d", "RISK_VOL_20D")
    _set_vol_scaled_label(
        stable60_growth_h60_volscaled_core25_topk30,
        vol_feature="RISK_VOL_20D",
        clip_abs_label=8.0,
    )
    _add_label_rank_norm(stable60_growth_h60_volscaled_core25_topk30)
    _sync_handler_kwargs(stable60_growth_h60_volscaled_core25_topk30)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_volscaledqqq_core25_topk30"] = (
        stable60_growth_h60_volscaled_core25_topk30
    )

    stable60_growth_h60_residual_core25_topk30 = copy.deepcopy(stable60_growth_h60_qqq_core25_topk30)
    _set_exp_name(
        stable60_growth_h60_residual_core25_topk30,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_betaresqqq_core25_topk30",
    )
    _append_extra_feature(stable60_growth_h60_residual_core25_topk30, "$risk_beta_spy_63d", "RISK_BETA_SPY_63D")
    _set_residual_label(stable60_growth_h60_residual_core25_topk30, beta_feature="RISK_BETA_SPY_63D")
    _add_label_rank_norm(stable60_growth_h60_residual_core25_topk30)
    _sync_handler_kwargs(stable60_growth_h60_residual_core25_topk30)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_growth_v1_h60_betaresqqq_core25_topk30"] = (
        stable60_growth_h60_residual_core25_topk30
    )

    nextgen_specs = [
        ("h20_qqqexcess_lgb_rank_core35_topk25", 20, "benchmark", "lgb", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
        ("h40_qqqexcess_lgb_rank_core35_topk25", 40, "benchmark", "lgb", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
        ("h60_qqqexcess_lgb_rank_core35_topk25", 60, "benchmark", "lgb", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
        ("h40_qqqexcess_ranker_core35_topk25", 40, "benchmark", "ranker", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
        ("h40_spyexcess_lgb_rank_core35_topk25", 40, "benchmark", "lgb", "/root/.qlib/qlib_data/us_data/bench_spy.pkl"),
        ("h40_ixicexcess_lgb_rank_core35_topk25", 40, "benchmark", "lgb", "/root/.qlib/qlib_data/us_data/bench_ixic.pkl"),
        ("h40_qqqresidual_lgb_rank_core35_topk25", 40, "residual", "lgb", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
        ("h40_qqqdownside_lgb_rank_core35_topk25", 40, "downside", "lgb", "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"),
    ]
    for suffix, horizon, target_kind, model_kind, benchmark_pkl in nextgen_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_nextgen_growth_v1_{suffix}",
        )
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(cfg, benchmark_pkl=str(benchmark_pkl))
        if model_kind == "ranker":
            cfg["data_handler_config"]["learn_processors"] = [
                proc
                for proc in cfg["data_handler_config"].get("learn_processors", [])
                if not (isinstance(proc, dict) and _processor_class(proc) == "CSRankNorm")
            ]
            _set_model_ranker(cfg, eval_at=[25, 50])
        else:
            _set_model_lgb(cfg)
            _add_label_rank_norm(cfg)
            _set_slow_lgb_training(cfg, learning_rate=0.005, rounds=4000, early_stopping=200)
        if target_kind == "residual":
            _append_extra_feature(cfg, "$risk_beta_spy_63d", "RISK_BETA_SPY_63D")
            _set_residual_label(cfg, beta_feature="RISK_BETA_SPY_63D")
            _add_label_rank_norm(cfg)
        elif target_kind == "downside":
            _set_downside_adjusted_label(cfg, downside_penalty=1.0, clip_abs_label=0.50)
            _add_label_rank_norm(cfg)
        _set_nextgen_growth_strategy(cfg)
        _sync_handler_kwargs(cfg)
        variants[f"nextgen_growth_v1_{suffix}"] = cfg

    nextgen_v2_specs = [
        ("h20_qqqresidual_regime_lgb_core55_topk30", 20, "residual", 0.55, 30, 0.07, 0.12, 0.20),
        ("h40_qqqresidual_regime_lgb_core60_topk25", 40, "residual", 0.60, 25, 0.06, 0.10, 0.25),
        ("h60_qqqresidual_regime_lgb_core60_topk25", 60, "residual", 0.60, 25, 0.055, 0.08, 0.30),
        ("h40_qqqutility_regime_lgb_core60_topk25", 40, "utility", 0.60, 25, 0.06, 0.10, 0.25),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in nextgen_v2_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_nextgen_growth_v2_{suffix}",
        )
        _set_horizon(cfg, int(horizon))
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg)
        _set_model_lgb(cfg)
        _set_slow_lgb_training(cfg, learning_rate=0.005, rounds=4000, early_stopping=200)
        _set_benchmark_excess_label(cfg, benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
        if target_kind == "residual":
            _set_residual_label(cfg, beta_feature="RISK_BETA_QQQ_63D", beta_min=0.0, beta_max=2.5)
            _add_label_rank_norm(cfg)
        elif target_kind == "utility":
            _set_portfolio_utility_label(
                cfg,
                benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
                downside_penalty=0.50,
                volatility_penalty=0.05,
                clip_abs_label=0.50,
            )
            _add_label_rank_norm(cfg)
        _set_nextgen_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
        )
        kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
        kwargs.update(
            {
                "max_sector_weight": 0.35,
                "feature_score_weights": {
                    "$log_marketcap_q": 0.10,
                    "$risk_vol_20d": -0.05,
                },
                "feature_min_percentiles": {
                    "$log_marketcap_q": 0.20,
                },
                "alpha_quality_lower_excess": -0.015,
                "alpha_quality_upper_excess": 0.035,
                "min_alpha_scale": float(min_alpha_scale),
            }
        )
        _sync_handler_kwargs(cfg)
        variants[f"nextgen_growth_v2_{suffix}"] = cfg

    nextgen_v2_ranker_specs = [
        ("h40_qqqresidual_regime_ranker_core60_topk25", 40, "residual", 0.60, 25, 0.06, 0.10, 0.25),
        ("h60_qqqresidual_regime_ranker_core60_topk25", 60, "residual", 0.60, 25, 0.055, 0.08, 0.30),
        ("h40_qqqexcess_regime_ranker_core60_topk25", 40, "benchmark", 0.60, 25, 0.06, 0.10, 0.25),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in nextgen_v2_ranker_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_nextgen_growth_v2_{suffix}",
        )
        _set_horizon(cfg, int(horizon))
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg)
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_model_ranker(cfg, eval_at=[25, 50])
        _set_ranker_training(cfg, learning_rate=0.015, rounds=1200, early_stopping=120)
        _set_benchmark_excess_label(cfg, benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
        if target_kind == "residual":
            _set_residual_label(cfg, beta_feature="RISK_BETA_QQQ_63D", beta_min=0.0, beta_max=2.5)
        _set_nextgen_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
        )
        kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
        kwargs.update(
            {
                "max_sector_weight": 0.35,
                "feature_score_weights": {
                    "$log_marketcap_q": 0.10,
                    "$risk_vol_20d": -0.05,
                },
                "feature_min_percentiles": {
                    "$log_marketcap_q": 0.20,
                },
                "alpha_quality_lower_excess": -0.015,
                "alpha_quality_upper_excess": 0.035,
                "min_alpha_scale": float(min_alpha_scale),
            }
        )
        _sync_handler_kwargs(cfg)
        variants[f"nextgen_growth_v2_{suffix}"] = cfg

    score_specs = [
        ("h60_sectorneutral_composite_core20_topk30", 60, "sector_neutral", 0.20, 30),
        ("h60_betaresqqq_composite_core20_topk30", 60, "beta_residual", 0.20, 30),
        ("h40_sectorneutral_composite_core20_topk30", 40, "sector_neutral", 0.20, 30),
        ("h40_betaresqqq_composite_core20_topk30", 40, "beta_residual", 0.20, 30),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk in score_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v1_{suffix}",
        )
        _set_extra_features(cfg, COMPOSITE_SCORE_FEATURES)
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        if target_kind == "beta_residual":
            cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
                cfg["data_handler_config"].get("learn_processors", []),
                fields_group="label",
            )
            _set_residual_label(cfg, beta_feature="RISK_BETA_SPY_63D")
        _add_label_rank_norm(cfg)
        _set_feature_weighted_score_model(cfg)
        _set_nextgen_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=0.12,
            max_turnover=0.15,
        )
        cfg["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.45
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v1_{suffix}"] = cfg

    regime_score_specs = [
        ("h60_regimeqqq_core20_topk30", 60, 0.20, 30),
        ("h40_regimeqqq_core20_topk30", 40, 0.20, 30),
    ]
    for suffix, horizon, benchmark_core_weight, topk in regime_score_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v2_{suffix}",
        )
        _set_extra_features(cfg, REGIME_GROWTH_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_feature_weighted_score_model(
            cfg,
            weights=REGIME_RISK_OFF_WEIGHTS,
            regime_feature="MKT_QQQ_RET_63D",
            regime_threshold=0.0,
            risk_on_weights=REGIME_RISK_ON_WEIGHTS,
            risk_off_weights=REGIME_RISK_OFF_WEIGHTS,
        )
        _set_nextgen_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=0.12,
            max_turnover=0.15,
        )
        cfg["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.45
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v2_{suffix}"] = cfg

    regime_sleeve_specs = [
        ("h60_regimesleeve_core45_topk30", 60, 0.45, 30, 0.07, 0.10, 0.25),
        ("h40_regimesleeve_core45_topk30", 40, 0.45, 30, 0.07, 0.10, 0.25),
        ("h60_regimesleeve_core60_topk25", 60, 0.60, 25, 0.05, 0.08, 0.40),
        ("h40_regimesleeve_core60_topk25", 40, 0.60, 25, 0.05, 0.08, 0.40),
    ]
    for suffix, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in regime_sleeve_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v3_{suffix}",
        )
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_regime_sleeve_score_model(cfg)
        _set_release_grade_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v3_{suffix}"] = cfg

    ic_select_specs = [
        ("h20_qqqexcess_icselect_core45_topk30", 20, "benchmark", 0.45, 30, 0.08, 0.12, 0.20),
        ("h40_qqqexcess_icselect_core45_topk30", 40, "benchmark", 0.45, 30, 0.07, 0.10, 0.25),
        ("h40_qqqexcess_icselect_core60_topk25", 40, "benchmark", 0.60, 25, 0.05, 0.08, 0.40),
        ("h40_qqqresidual_icselect_core45_topk30", 40, "residual", 0.45, 30, 0.07, 0.10, 0.25),
        ("h60_qqqexcess_icselect_core45_topk30", 60, "benchmark", 0.45, 30, 0.06, 0.08, 0.30),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in ic_select_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v4_{suffix}",
        )
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        if target_kind == "residual":
            _set_residual_label(cfg, beta_feature="RISK_BETA_QQQ_63D", beta_min=0.0, beta_max=2.5)
        _set_ic_selected_score_model(cfg)
        _set_release_grade_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v4_{suffix}"] = cfg

    regime_ic_select_specs = [
        ("h40_qqqexcess_regimeic_core45_topk30", 40, "benchmark", 0.45, 30, 0.07, 0.10, 0.25),
        ("h40_qqqexcess_regimeic_core60_topk25", 40, "benchmark", 0.60, 25, 0.05, 0.08, 0.40),
        ("h40_qqqresidual_regimeic_core45_topk30", 40, "residual", 0.45, 30, 0.07, 0.10, 0.25),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in regime_ic_select_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v5_{suffix}",
        )
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        if target_kind == "residual":
            _set_residual_label(cfg, beta_feature="RISK_BETA_QQQ_63D", beta_min=0.0, beta_max=2.5)
        _set_ic_selected_score_model(
            cfg,
            max_features=6,
            min_selected_features=2,
            min_abs_ic=0.003,
            min_ic_days=100,
            min_worst_year_signed_ic=-0.03,
            min_recent_signed_ic=-0.005,
            recent_weight=0.65,
            regime_feature="MKT_QQQ_RET_63D_LAG1",
            risk_on_threshold=0.03,
            risk_off_threshold=-0.04,
            regime_min_ic_days=35,
        )
        _set_release_grade_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v5_{suffix}"] = cfg

    tail_regime_ic_select_specs = [
        ("h40_qqqexcess_tailregimeic_core45_topk30", 40, "benchmark", 0.45, 30, 0.07, 0.10, 0.25),
        ("h40_qqqexcess_tailregimeic_core60_topk25", 40, "benchmark", 0.60, 25, 0.05, 0.08, 0.40),
        ("h40_qqqresidual_tailregimeic_core45_topk30", 40, "residual", 0.45, 30, 0.07, 0.10, 0.25),
    ]
    for suffix, horizon, target_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in tail_regime_ic_select_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v6_{suffix}",
        )
        _set_extra_features(cfg, REGIME_GROWTH_V3_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        if target_kind == "residual":
            _set_residual_label(cfg, beta_feature="RISK_BETA_QQQ_63D", beta_min=0.0, beta_max=2.5)
        _set_ic_selected_score_model(
            cfg,
            max_features=6,
            min_selected_features=2,
            min_abs_ic=0.003,
            min_ic_days=100,
            min_worst_year_signed_ic=-0.035,
            min_recent_signed_ic=-0.005,
            recent_weight=0.55,
            regime_feature="MKT_QQQ_RET_63D_LAG1",
            risk_on_threshold=0.03,
            risk_off_threshold=-0.04,
            regime_min_ic_days=35,
            min_topq_spread=0.0,
            min_worst_year_topq_spread=-0.015,
            tail_weight=0.75,
        )
        _set_release_grade_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v6_{suffix}"] = cfg

    fmp_regime_ic_select_specs = [
        ("h40_qqqexcess_fmponly_regimeic_core45_topk30", FMP_EVENT_DIAGNOSTIC_FEATURES, 0.45, 30, 0.07, 0.10, 0.25),
        ("h40_qqqexcess_fmpcore_regimeic_core45_topk30", REGIME_GROWTH_V3_FEATURES + FMP_EVENT_CORE, 0.45, 30, 0.07, 0.10, 0.25),
    ]
    for suffix, features, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in fmp_regime_ic_select_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v7_{suffix}",
        )
        _set_extra_features(cfg, features)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, 40)
        _set_fmp_event_segments(cfg, 40)
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_ic_selected_score_model(
            cfg,
            max_features=8,
            min_selected_features=2,
            min_abs_ic=0.003,
            min_ic_days=100,
            min_worst_year_signed_ic=-0.035,
            min_recent_signed_ic=-0.005,
            recent_weight=0.60,
            regime_feature="MKT_QQQ_RET_63D_LAG1",
            risk_on_threshold=0.03,
            risk_off_threshold=-0.04,
            regime_min_ic_days=35,
            min_topq_spread=0.0,
            min_worst_year_topq_spread=-0.015,
            tail_weight=0.50,
        )
        _set_release_grade_growth_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v7_{suffix}"] = cfg

    fmp_stack_v8_specs = [
        ("h10_qqqexcess_fmpv2_stack_core85_topk30", FMP_STACK_V2_FEATURES, FMP_STACK_V2_SLEEVES, 10, 0.85, 30, 0.08, 0.10, 55),
        ("h20_qqqexcess_fmpv2_stack_core85_topk30", FMP_STACK_V2_FEATURES, FMP_STACK_V2_SLEEVES, 20, 0.85, 30, 0.08, 0.10, 55),
        ("h20_qqqexcess_fmpv2_stack_core35_topk30", FMP_STACK_V2_FEATURES, FMP_STACK_V2_SLEEVES, 20, 0.35, 30, 0.40, 0.40, 110),
        ("h40_qqqexcess_fmpv2_stack_core85_topk30", FMP_STACK_V2_FEATURES, FMP_STACK_V2_SLEEVES, 40, 0.85, 30, 0.07, 0.08, 55),
        (
            "h20_qqqexcess_fmpv2_fmponly_core90_topk25",
            FMP_EVENT_DIAGNOSTIC_FEATURES_V2,
            {"fmp_event": FMP_STACK_V2_SLEEVES["fmp_event"]},
            20,
            0.90,
            25,
            0.06,
            0.08,
            45,
        ),
    ]
    for suffix, features, sleeves, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings in fmp_stack_v8_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v8_{suffix}",
        )
        _set_extra_features(cfg, features)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_stacked_signal_score_model(
            cfg,
            sleeves=sleeves,
            max_sleeves=min(3, len(sleeves)),
            min_selected_sleeves=1,
            min_abs_ic=0.001,
            min_ic_days=60 if int(horizon) <= 10 else 80,
            min_recent_signed_ic=-0.006,
            min_worst_year_signed_ic=-0.05,
        )
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v8_{suffix}"] = cfg

    fmp_ranker_v9_specs = [
        ("h5_qqqexcess_fmpv2_ranker_core55_topk30", 5, 0.55, 30, 0.08, 0.12, 75, 0.45),
        ("h10_qqqexcess_fmpv2_ranker_core55_topk30", 10, 0.55, 30, 0.08, 0.12, 75, 0.45),
        ("h20_qqqexcess_fmpv2_ranker_core60_topk30", 20, 0.60, 30, 0.07, 0.10, 75, 0.55),
    ]
    for suffix, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings, min_alpha_scale in fmp_ranker_v9_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v9_{suffix}",
        )
        _set_extra_features(cfg, FMP_STACK_V2_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_model_ranker(cfg, eval_at=[int(topk), 100])
        _set_ranker_training(cfg, learning_rate=0.020, rounds=900, early_stopping=100)
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        kwargs = cfg["port_analysis_config"]["strategy"]["kwargs"]
        _set_qqq_overlay_dynamic_alpha(cfg, min_alpha_scale=float(min_alpha_scale))
        kwargs["max_sector_weight"] = 0.30
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v9_{suffix}"] = cfg

    fmp_ablation_v13_stack_specs = [
        (
            "h20_qqqexcess_sharadar_stack_core35_topk30",
            SHARADAR_STACK_V1_FEATURES,
            SHARADAR_STACK_V1_SLEEVES,
            20,
            0.35,
            30,
            0.40,
            0.40,
            110,
        ),
        (
            "h20_qqqexcess_fmpwhite_stack_core35_topk30",
            FMP_STACK_WHITELIST_V1_FEATURES,
            FMP_WHITELIST_V1_SLEEVES,
            20,
            0.35,
            30,
            0.40,
            0.40,
            110,
        ),
    ]
    for suffix, features, sleeves, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings in (
        fmp_ablation_v13_stack_specs
    ):
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v13_{suffix}",
        )
        _set_extra_features(cfg, features)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_stacked_signal_score_model(
            cfg,
            sleeves=sleeves,
            max_sleeves=min(3, len(sleeves)),
            min_selected_sleeves=1,
            min_abs_ic=0.001,
            min_ic_days=80,
            min_recent_signed_ic=-0.006,
            min_worst_year_signed_ic=-0.05,
        )
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v13_{suffix}"] = cfg

    fmp_ablation_v14_ranker_specs = [
        ("h5_qqqexcess_sharadar_ranker_core55_topk30", SHARADAR_STACK_V1_FEATURES, 5, 0.55, 30, 0.08, 0.12, 75, 0.45),
        (
            "h5_qqqexcess_fmpwhite_ranker_core55_topk30",
            FMP_STACK_WHITELIST_V1_FEATURES,
            5,
            0.55,
            30,
            0.08,
            0.12,
            75,
            0.45,
        ),
        ("h20_qqqexcess_sharadar_ranker_core60_topk30", SHARADAR_STACK_V1_FEATURES, 20, 0.60, 30, 0.07, 0.10, 75, 0.55),
        (
            "h20_qqqexcess_fmpwhite_ranker_core60_topk30",
            FMP_STACK_WHITELIST_V1_FEATURES,
            20,
            0.60,
            30,
            0.07,
            0.10,
            75,
            0.55,
        ),
    ]
    for suffix, features, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings, min_alpha_scale in (
        fmp_ablation_v14_ranker_specs
    ):
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v14_{suffix}",
        )
        _set_extra_features(cfg, features)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_model_ranker(cfg, eval_at=[int(topk), 100])
        _set_ranker_training(cfg, learning_rate=0.020, rounds=900, early_stopping=100)
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        _set_qqq_overlay_dynamic_alpha(cfg, min_alpha_scale=float(min_alpha_scale))
        cfg["port_analysis_config"]["strategy"]["kwargs"]["max_sector_weight"] = 0.30
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v14_{suffix}"] = cfg

    fmp_leadership_v15_stack_specs = [
        ("h20_qqqexcess_fmpleader_stack_core40_topk25", 0.40, 25, 0.40, 0.40, 110),
        ("h20_qqqexcess_fmpleader_stack_core45_topk25", 0.45, 25, 0.40, 0.40, 110),
        ("h20_qqqexcess_fmpleader_stack_core35_topk25", 0.35, 25, 0.40, 0.40, 110),
        ("h20_qqqexcess_fmpleader_stack_core40_topk30", 0.40, 30, 0.40, 0.40, 110),
    ]
    for suffix, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings in (
        fmp_leadership_v15_stack_specs
    ):
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v15_{suffix}",
        )
        _set_extra_features(cfg, FMP_LEADERSHIP_STACK_V1_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, 20)
        _set_fmp_event_segments(cfg, 20)
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_stacked_signal_score_model(
            cfg,
            sleeves=FMP_LEADERSHIP_STACK_V1_SLEEVES,
            max_sleeves=4,
            min_selected_sleeves=2,
            min_abs_ic=0.001,
            min_ic_days=80,
            min_recent_signed_ic=-0.006,
            min_worst_year_signed_ic=-0.05,
        )
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v15_{suffix}"] = cfg

    fmp_leadership_v15_regime_specs = [
        ("h20_qqqexcess_fmpleader_regime_core40_topk25", 0.40, 25, 0.40, 0.40, 110),
        ("h20_qqqexcess_fmpleader_regime_core45_topk25", 0.45, 25, 0.40, 0.40, 110),
    ]
    for suffix, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings in (
        fmp_leadership_v15_regime_specs
    ):
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v15_{suffix}",
        )
        _set_extra_features(cfg, FMP_LEADERSHIP_STACK_V1_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, 20)
        _set_fmp_event_segments(cfg, 20)
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_regime_sleeve_score_model(
            cfg,
            sleeves=FMP_LEADERSHIP_STACK_V1_SLEEVES,
            state_weights=FMP_LEADERSHIP_REGIME_STATE_WEIGHTS,
        )
        _set_high_qqq_overlay_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            max_holdings=int(max_holdings),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v15_{suffix}"] = cfg

    fmp_v10_specs = [
        ("h5_qqqexcess_fmpv2_ranker_dynrisk_core45_topk25", 5, "benchmark", "ranker", 0.45, 25, 0.10, 0.12, 0.25),
        ("h5_qqqutility_fmpv2_ranker_dynrisk_core45_topk25", 5, "utility", "ranker", 0.45, 25, 0.10, 0.12, 0.25),
        ("h10_qqqutility_fmpv2_ranker_dynrisk_core45_topk25", 10, "utility", "ranker", 0.45, 25, 0.10, 0.12, 0.30),
        ("h5_qqqutility_fmpv2_lgb_dynrisk_core45_topk25", 5, "utility", "lgb", 0.45, 25, 0.10, 0.12, 0.25),
        ("h20_qqqutility_fmpv2_stack_dynrisk_core45_topk25", 20, "utility", "stack", 0.45, 25, 0.09, 0.10, 0.35),
    ]
    for suffix, horizon, target_kind, model_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in fmp_v10_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v10_{suffix}",
        )
        _set_extra_features(cfg, FMP_STACK_V2_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        if target_kind == "utility":
            _set_portfolio_utility_label(
                cfg,
                benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
                vol_feature="RISK_VOL_20D",
                downside_penalty=0.75,
                volatility_penalty=0.10,
                clip_abs_label=0.50,
            )
        if model_kind == "ranker":
            _set_model_ranker(cfg, eval_at=[int(topk), 100])
            _set_ranker_training(cfg, learning_rate=0.015, rounds=1200, early_stopping=120)
        elif model_kind == "lgb":
            _set_model_lgb(cfg)
            _set_slow_lgb_training(cfg, learning_rate=0.006, rounds=3000, early_stopping=180)
            _add_label_rank_norm(cfg)
        elif model_kind == "stack":
            _set_stacked_signal_score_model(
                cfg,
                sleeves=FMP_STACK_V2_SLEEVES,
                max_sleeves=3,
                min_selected_sleeves=1,
                min_abs_ic=0.001,
                min_ic_days=80,
                min_recent_signed_ic=-0.006,
                min_worst_year_signed_ic=-0.05,
            )
        else:
            raise ValueError(f"unsupported v10 model kind: {model_kind}")
        _set_high_return_dynamic_qqq_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v10_{suffix}"] = cfg

    fmp_v11_specs = [
        ("h5_qqqexcess_fmpv2_ranker_mildrisk_core55_topk30", 5, "benchmark", "ranker", 0.55, 30, 0.08, 0.12, 0.45),
        ("h5_qqqutility_fmpv2_ranker_mildrisk_core55_topk30", 5, "utility", "ranker", 0.55, 30, 0.08, 0.12, 0.45),
        ("h10_qqqutility_fmpv2_ranker_mildrisk_core55_topk30", 10, "utility", "ranker", 0.55, 30, 0.08, 0.12, 0.45),
        ("h5_qqqutility_fmpv2_lgb_mildrisk_core55_topk30", 5, "utility", "lgb", 0.55, 30, 0.08, 0.12, 0.45),
    ]
    for suffix, horizon, target_kind, model_kind, benchmark_core_weight, topk, max_active_weight, max_turnover, min_alpha_scale in fmp_v11_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v11_{suffix}",
        )
        _set_extra_features(cfg, FMP_STACK_V2_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_group_neutralize(
            cfg["data_handler_config"].get("learn_processors", []),
            fields_group="label",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        if target_kind == "utility":
            _set_portfolio_utility_label(
                cfg,
                benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
                vol_feature="RISK_VOL_20D",
                downside_penalty=0.60,
                volatility_penalty=0.06,
                clip_abs_label=0.50,
            )
        if model_kind == "ranker":
            _set_model_ranker(cfg, eval_at=[int(topk), 100])
            _set_ranker_training(cfg, learning_rate=0.018, rounds=1000, early_stopping=110)
        elif model_kind == "lgb":
            _set_model_lgb(cfg)
            _set_slow_lgb_training(cfg, learning_rate=0.006, rounds=2800, early_stopping=160)
            _add_label_rank_norm(cfg)
        else:
            raise ValueError(f"unsupported v11 model kind: {model_kind}")
        _set_mild_dynamic_qqq_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v11_{suffix}"] = cfg

    fmp_v12_specs = [
        ("h5_qqqsector_fmpv2_ranker_mildrisk_core55_topk30", 5, 0.55, 30, 0.08, 0.12, 75, 0.45),
    ]
    for suffix, horizon, benchmark_core_weight, topk, max_active_weight, max_turnover, max_holdings, min_alpha_scale in fmp_v12_specs:
        cfg = copy.deepcopy(stable20_capmom_sectorneutral_etfcore_lgb_rank)
        _set_exp_name(
            cfg,
            f"us_sharadar_weekly_pit_score_growth_v12_{suffix}",
        )
        _set_extra_features(cfg, FMP_STACK_V2_FEATURES)
        _set_selective_feature_norm(cfg, exclude_prefixes=("MKT_",))
        _set_horizon(cfg, int(horizon))
        _set_fmp_event_segments(cfg, int(horizon))
        _set_benchmark_excess_label(
            cfg,
            benchmark_pkl="/root/.qlib/qlib_data/us_data/bench_qqq.pkl",
        )
        cfg["data_handler_config"]["learn_processors"] = _remove_label_rank_norm(
            cfg["data_handler_config"].get("learn_processors", [])
        )
        _set_model_ranker(cfg, eval_at=[int(topk), 100])
        _set_ranker_training(cfg, learning_rate=0.020, rounds=900, early_stopping=100)
        _set_mild_dynamic_qqq_strategy(
            cfg,
            benchmark_core_weight=float(benchmark_core_weight),
            topk=int(topk),
            max_active_weight=float(max_active_weight),
            max_turnover=float(max_turnover),
            min_alpha_scale=float(min_alpha_scale),
        )
        cfg["port_analysis_config"]["strategy"]["kwargs"]["max_holdings"] = int(max_holdings)
        _sync_handler_kwargs(cfg)
        variants[f"score_growth_v12_{suffix}"] = cfg

    stable60_release_v2_core67_dynalpha_topk20 = copy.deepcopy(stable60_release_v2_core67_topk20)
    _set_exp_name(
        stable60_release_v2_core67_dynalpha_topk20,
        "us_sharadar_weekly_pit_stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_dynalpha_topk20",
    )
    dyn_release_kwargs = stable60_release_v2_core67_dynalpha_topk20["port_analysis_config"]["strategy"]["kwargs"]
    dyn_release_kwargs["dynamic_alpha_weight"] = True
    dyn_release_kwargs["alpha_quality_window"] = 63
    dyn_release_kwargs["alpha_quality_min_history"] = 20
    dyn_release_kwargs["alpha_quality_lower_excess"] = -0.01
    dyn_release_kwargs["alpha_quality_upper_excess"] = 0.03
    dyn_release_kwargs["min_alpha_scale"] = 0.25
    dyn_release_kwargs["max_alpha_scale"] = 1.0
    _sync_handler_kwargs(stable60_release_v2_core67_dynalpha_topk20)
    variants["stable60_capmom_sectorneutral_etfcore_lgb_rank_release_v2_core67_dynalpha_topk20"] = (
        stable60_release_v2_core67_dynalpha_topk20
    )

    stable20_capmom_dynalpha_etfcore_lgb_rank = copy.deepcopy(stable20_capmom_lgb_rank)
    _set_exp_name(
        stable20_capmom_dynalpha_etfcore_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_dynalpha_etfcore_lgb_rank_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        stable20_capmom_dynalpha_etfcore_lgb_rank,
        benchmark_core_weight=0.70,
        benchmark_topn=7,
        benchmark_tickers_file="/Stock/qlib/_sfp_benchmark_tickers.txt",
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=70,
        max_weight=0.20,
        max_active_weight=0.04,
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.02,
        alpha_quality_upper_excess=0.04,
        min_alpha_scale=0.0,
    )
    _sync_handler_kwargs(stable20_capmom_dynalpha_etfcore_lgb_rank)
    variants["stable20_capmom_dynalpha_etfcore_lgb_rank_v1"] = stable20_capmom_dynalpha_etfcore_lgb_rank

    stable20_capmom_dynalpha_benchmarkaware_lgb_rank = copy.deepcopy(stable20_capmom_lgb_rank)
    _set_exp_name(
        stable20_capmom_dynalpha_benchmarkaware_lgb_rank,
        "us_sharadar_weekly_pit_stable20_capmom_dynalpha_benchmarkaware_lgb_rank_v1_topk40",
    )
    _set_benchmark_aware_strategy(
        stable20_capmom_dynalpha_benchmarkaware_lgb_rank,
        benchmark_core_weight=0.70,
        benchmark_topn=160,
        max_turnover=0.10,
        min_trade_weight=0.001,
        max_holdings=180,
        dynamic_alpha_weight=True,
        alpha_quality_window=63,
        alpha_quality_lower_excess=-0.02,
        alpha_quality_upper_excess=0.03,
        min_alpha_scale=0.0,
    )
    _sync_handler_kwargs(stable20_capmom_dynalpha_benchmarkaware_lgb_rank)
    variants["stable20_capmom_dynalpha_benchmarkaware_lgb_rank_v1"] = (
        stable20_capmom_dynalpha_benchmarkaware_lgb_rank
    )

    return variants


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build conservative US Sharadar clean candidate configs.")
    p.add_argument(
        "--base_config",
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_clean_neutral_lgb_v1_topk40.yaml",
    )
    p.add_argument("--out_dir", default="examples/benchmarks/LightGBM")
    p.add_argument("--variant", action="append", default=[], help="Only write this variant key. Repeatable.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    base_path = Path(args.base_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    base_cfg = _load_yaml(base_path)
    variants = build_variants(base_cfg)
    requested = {str(name).strip() for name in args.variant if str(name).strip()}
    missing = sorted(requested - set(variants))
    if missing:
        raise ValueError(f"unknown variants requested: {missing}")
    for name, cfg in variants.items():
        if requested and name not in requested:
            continue
        suffix = "" if re.search(r"_topk\d+$", name) else "_topk40"
        out = out_dir / f"workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_{name}{suffix}.yaml"
        if args.dry_run:
            print(out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
