#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    YAML = None
    import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if YAML is not None:
            yaml_loader = YAML(typ="safe", pure=True)
            data = yaml_loader.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config structure: {path}")
    return data


def _safe_get(d: dict, keys: Iterable[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_label_horizon(label_expr: str) -> Optional[int]:
    if not isinstance(label_expr, str):
        return None
    m = re.search(r"Ref\(\$close,\s*-(\d+)\)\s*/\s*Ref\(\$close,\s*-(\d+)\)", label_expr)
    if not m:
        return None
    n1 = int(m.group(1))
    n2 = int(m.group(2))
    if n1 <= n2:
        return None
    return n1 - n2


def _infer_label_horizon(cfg: Dict[str, Any]) -> Optional[int]:
    label_cfg = cfg.get("data_handler_config", {}).get("label", [])
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        horizons = [_parse_label_horizon(expr) for expr in label_cfg[0]]
        horizons = [int(h) for h in horizons if h is not None]
        return max(horizons) if horizons else None
    label_cfg = _safe_get(cfg, ["task", "dataset", "kwargs", "handler", "kwargs", "label"], [])
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        horizons = [_parse_label_horizon(expr) for expr in label_cfg[0]]
        horizons = [int(h) for h in horizons if h is not None]
        return max(horizons) if horizons else None
    return None


def _as_ts(v) -> pd.Timestamp:
    return pd.Timestamp(v) if v is not None else None


def _fmt_date(t: pd.Timestamp) -> str:
    return str(pd.Timestamp(t).date())


def _infer_exp_name(qlib_init: Dict[str, Any]) -> str:
    exp_manager = qlib_init.get("exp_manager", {})
    if isinstance(exp_manager, dict):
        kwargs = exp_manager.get("kwargs", {})
        name = kwargs.get("default_exp_name")
        if name:
            return str(name)
    return "us_sharadar_walkforward"


def _override_exp_name(qlib_init: Dict[str, Any], exp_name: str) -> None:
    exp_manager = qlib_init.setdefault("exp_manager", {})
    if not isinstance(exp_manager, dict):
        return
    kwargs = exp_manager.setdefault("kwargs", {})
    if isinstance(kwargs, dict):
        kwargs["default_exp_name"] = exp_name


def _override_provider_uri(qlib_init: Dict[str, Any], provider_uri: str) -> None:
    if provider_uri:
        qlib_init["provider_uri"] = provider_uri


def _processor_class(proc: Dict[str, Any]) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def _handler_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dh = cfg.get("data_handler_config")
    if isinstance(dh, dict):
        out.append(dh)
    task_dh = _safe_get(cfg, ["task", "dataset", "kwargs", "handler", "kwargs"])
    if isinstance(task_dh, dict) and all(task_dh is not item for item in out):
        out.append(task_dh)
    return out


def _override_benchmark_pkl(cfg: Dict[str, Any], benchmark_pkl: str) -> int:
    if not benchmark_pkl:
        return 0
    count = 0
    for dh in _handler_configs(cfg):
        processors = dh.get("learn_processors", []) or []
        for proc in processors:
            if isinstance(proc, dict) and _processor_class(proc) in {
                "BenchmarkExcessLabel",
                "ResidualForwardReturnLabel",
                "VolScaledExcessLabel",
                "DownsideAdjustedExcessLabel",
                "PortfolioUtilityExcessLabel",
                "DualHorizonPortfolioUtilityLabel",
            }:
                kwargs = proc.setdefault("kwargs", {})
                if isinstance(kwargs, dict):
                    kwargs["benchmark_pkl"] = benchmark_pkl
                    count += 1
    return count


def _get_trade_calendar_span(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if start is None or end is None:
        return []
    from qlib.data.data import Cal

    # Cal.calendar may return a numpy array; normalize to a plain list for simpler downstream logic.
    return list(Cal.calendar(start_time=start, end_time=end, freq="day", future=False))


def _align_to_trade_day_start(ts: pd.Timestamp) -> pd.Timestamp:
    cal = _get_trade_calendar_span(ts, ts + pd.Timedelta(days=7))
    if len(cal) == 0:
        raise ValueError(f"cannot align to trade day start near {ts}")
    return cal[0]


def _align_to_trade_day_end(ts: pd.Timestamp) -> pd.Timestamp:
    cal = _get_trade_calendar_span(ts - pd.Timedelta(days=7), ts)
    if len(cal) == 0:
        raise ValueError(f"cannot align to trade day end near {ts}")
    return cal[-1]


def _prev_trade_day(cal: List[pd.Timestamp], ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    # find first calendar entry >= ts, then step back
    lo, hi = 0, len(cal)
    while lo < hi:
        mid = (lo + hi) // 2
        if cal[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    if idx <= 0:
        raise ValueError(f"no previous trade day available for {ts} within calendar span")
    return cal[idx - 1]


def _trade_day_n_before(cal: List[pd.Timestamp], ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """
    Return the trade day that is (n-1) trading days before ts (inclusive end).

    Example: n=1 returns ts aligned to trade day end; n=63 returns 63rd day backwards.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ts_end = _align_to_trade_day_end(ts)
    # find index of ts_end
    lo, hi = 0, len(cal)
    while lo < hi:
        mid = (lo + hi) // 2
        if cal[mid] < ts_end:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    if idx >= len(cal) or cal[idx] != ts_end:
        raise ValueError(f"trade day not found in calendar: {ts_end}")
    start_idx = idx - (n - 1)
    if start_idx < 0:
        raise ValueError(f"not enough history for n={n} at {ts_end} (idx={idx})")
    return cal[start_idx]


@dataclass(frozen=True)
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp

    def as_tuple_str(self) -> Tuple[str, str]:
        return (_fmt_date(self.start), _fmt_date(self.end))


@dataclass(frozen=True)
class WalkForwardTaskSpec:
    name: str
    train: Segment
    valid: Segment
    test: Segment


def _split_test_by_year(test_start: pd.Timestamp, test_end: pd.Timestamp) -> List[Segment]:
    out: List[Segment] = []
    for year in range(test_start.year, test_end.year + 1):
        y_start = max(test_start, pd.Timestamp(f"{year}-01-01"))
        y_end = min(test_end, pd.Timestamp(f"{year}-12-31"))
        cal = _get_trade_calendar_span(y_start, y_end)
        if len(cal) < 2:
            continue
        out.append(Segment(start=cal[0], end=cal[-1]))
    return out


def _split_test_by_quarter(test_start: pd.Timestamp, test_end: pd.Timestamp) -> List[Segment]:
    """
    Split [test_start, test_end] into calendar-quarter blocks and align each block to trading days.
    """
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    if test_end < test_start:
        raise ValueError(f"invalid test range: {test_start} -> {test_end}")

    # Quarter starts (QS) between the range; include boundaries around start/end.
    q_starts = pd.date_range(test_start.normalize(), test_end.normalize(), freq="QS")
    if len(q_starts) == 0 or q_starts[0] > test_start.normalize():
        q_starts = q_starts.insert(0, test_start.normalize())

    out: List[Segment] = []
    for qs in q_starts:
        qe = (qs + pd.offsets.QuarterEnd(0)).normalize()
        q_start = max(test_start, qs)
        q_end = min(test_end, qe)
        cal = _get_trade_calendar_span(q_start, q_end)
        if len(cal) < 2:
            continue
        out.append(Segment(start=cal[0], end=cal[-1]))

    return out


def _build_walkforward_specs(
    *,
    cal: List[pd.Timestamp],
    train_start: pd.Timestamp,
    test_segments: List[Segment],
    valid_days: int,
    train_lookback_days: Optional[int],
    embargo_days: int,
) -> List[WalkForwardTaskSpec]:
    specs: List[WalkForwardTaskSpec] = []
    global_train_start = _align_to_trade_day_start(train_start)
    if embargo_days < 0:
        raise ValueError("--embargo_days must be >= 0")

    for seg in test_segments:
        test_start = seg.start
        test_end = seg.end
        pre_test = _prev_trade_day(cal, test_start)
        valid_end = (
            pre_test
            if embargo_days == 0
            else _trade_day_n_before(cal, pre_test, embargo_days + 1)
        )
        valid_start = _trade_day_n_before(cal, valid_end, valid_days)
        pre_valid = _prev_trade_day(cal, valid_start)
        train_end = (
            pre_valid
            if embargo_days == 0
            else _trade_day_n_before(cal, pre_valid, embargo_days + 1)
        )
        if train_end < global_train_start:
            raise ValueError(f"train_end before train_start for test {test_start}->{test_end}")
        if train_lookback_days is not None:
            if train_lookback_days <= 0:
                raise ValueError("--train_lookback_days must be positive")
            lookback_start = _trade_day_n_before(cal, train_end, train_lookback_days)
            train_start_eff = max(global_train_start, lookback_start)
        else:
            train_start_eff = global_train_start
        name = f"{test_start.date()}_{test_end.date()}"
        specs.append(
            WalkForwardTaskSpec(
                name=name,
                train=Segment(train_start_eff, train_end),
                valid=Segment(valid_start, valid_end),
                test=Segment(test_start, test_end),
            )
        )
    return specs


def _strip_to_signal_only(task: Dict[str, Any]) -> None:
    """
    Remove heavy records (like PortAnaRecord) to speed up walk-forward training.
    """
    recs = task.get("record")
    if not isinstance(recs, list):
        return
    keep = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if r.get("class") == "SignalRecord":
            keep.append(r)
    if keep:
        task["record"] = keep


def _apply_segments_to_task(task: Dict[str, Any], spec: WalkForwardTaskSpec) -> None:
    segs = _safe_get(task, ["dataset", "kwargs", "segments"])
    if not isinstance(segs, dict):
        raise ValueError("task.dataset.kwargs.segments missing or invalid")
    segs["train"] = spec.train.as_tuple_str()
    segs["valid"] = spec.valid.as_tuple_str()
    segs["test"] = spec.test.as_tuple_str()

    handler_kwargs = _safe_get(task, ["dataset", "kwargs", "handler", "kwargs"])
    if isinstance(handler_kwargs, dict):
        handler_kwargs["fit_start_time"] = _fmt_date(spec.train.start)
        handler_kwargs["fit_end_time"] = _fmt_date(spec.train.end)
        handler_kwargs["end_time"] = _fmt_date(spec.test.end)
        # Keep start_time unchanged; it defines the earliest available history.
        fp = handler_kwargs.get("filter_pipe")
        if isinstance(fp, list):
            for f in fp:
                if isinstance(f, dict) and "filter_end_time" in f:
                    f["filter_end_time"] = _fmt_date(spec.test.end)


def _model_class(task: Dict[str, Any]) -> str:
    return str(_safe_get(task, ["model", "class"], "") or "").split(".")[-1]


def _attach_recency_reweighter(task: Dict[str, Any], *, half_life_days: int, min_weight: float) -> bool:
    if int(half_life_days) <= 0:
        return False
    if _model_class(task) == "LGBRankerModel":
        return False
    from qlib.data.dataset.weight import RecencyReweighter

    task["reweighter"] = RecencyReweighter(
        half_life_days=int(half_life_days),
        min_weight=float(min_weight),
    )
    return True


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text or "").split(",") if x.strip()]


def _attach_regime_recency_reweighter(
    task: Dict[str, Any],
    *,
    half_life_days: int,
    min_weight: float,
    regime_feature: str,
    regime_thresholds: List[float],
    date_balance: bool,
    year_balance: bool,
    label_tail_weight: float,
    label_tail_quantile: float,
    max_weight: float,
) -> bool:
    if not str(regime_feature or "").strip():
        return False
    if _model_class(task) == "LGBRankerModel":
        return False
    from qlib.data.dataset.weight import RegimeRecencyReweighter

    task["reweighter"] = RegimeRecencyReweighter(
        half_life_days=int(half_life_days),
        min_weight=float(min_weight),
        regime_feature=str(regime_feature),
        regime_thresholds=regime_thresholds,
        date_balance=bool(date_balance),
        year_balance=bool(year_balance),
        regime_balance=True,
        label_tail_weight=float(label_tail_weight),
        label_tail_quantile=float(label_tail_quantile),
        max_weight=float(max_weight),
    )
    return True


def _stitch_preds(preds: List[Tuple[WalkForwardTaskSpec, pd.DataFrame]]) -> pd.DataFrame:
    if not preds:
        raise ValueError("no preds to stitch")

    parts = []
    for spec, df in preds:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"pred is not DataFrame for {spec.name}")
        if list(df.columns) != ["score"]:
            raise ValueError(f"unexpected pred columns for {spec.name}: {list(df.columns)}")
        dt = df.index.get_level_values("datetime")
        mask = (dt >= spec.test.start) & (dt <= spec.test.end)
        parts.append(df.loc[mask])

    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _load_pred_from_recorder(recorder) -> pd.DataFrame:
    """
    SignalRecord saves predictions as the artifact `pred.pkl`.
    Some custom pipelines may save `pred` instead; try both.
    """
    try:
        return recorder.load_object("pred.pkl")
    except Exception:
        try:
            return recorder.load_object("pred")
        except Exception as e:
            arts = []
            try:
                arts = recorder.list_artifacts()
            except Exception:
                pass
            detail = f"available_artifacts={arts}" if arts else "available_artifacts=<unavailable>"
            raise RuntimeError(f"failed to load pred artifact ({detail})") from e


def _recorder_has_pred_artifact(recorder) -> bool:
    try:
        arts = recorder.list_artifacts()
    except Exception:
        return False
    return "pred.pkl" in set(map(str, arts)) or "pred" in set(map(str, arts))


def _json_safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value


def _model_selection_diagnostics(recorder) -> Dict[str, Any]:
    try:
        model = recorder.load_object("params.pkl")
    except Exception:
        return {}
    weights = getattr(model, "selected_weights_", None)
    summary = getattr(model, "selection_summary_", None)
    sleeve_weights = getattr(model, "selected_sleeve_weights_", None)
    sleeve_summary = getattr(model, "sleeve_summary_", None)
    if (
        not isinstance(weights, dict)
        and not isinstance(summary, pd.DataFrame)
        and not isinstance(sleeve_weights, dict)
        and not isinstance(sleeve_summary, pd.DataFrame)
    ):
        return {}
    out: Dict[str, Any] = {}
    if isinstance(weights, dict):
        out["selected_weights"] = {
            str(name): float(weight)
            for name, weight in sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        }
        out["selected_feature_count"] = len(weights)
    if isinstance(sleeve_weights, dict):
        out["selected_sleeve_weights"] = {
            str(name): float(weight)
            for name, weight in sorted(sleeve_weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        }
        out["selected_sleeve_count"] = len(sleeve_weights)
    state_weights = getattr(model, "selected_weights_by_state_", None)
    if isinstance(state_weights, dict) and state_weights:
        out["selected_weights_by_state"] = {
            str(state): {
                str(name): float(weight)
                for name, weight in sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            }
            for state, weights in state_weights.items()
            if isinstance(weights, dict)
        }
    fallback_used = getattr(model, "fallback_used_", None)
    if fallback_used is not None:
        out["fallback_used"] = bool(fallback_used)
    fallback_by_state = getattr(model, "fallback_used_by_state_", None)
    if isinstance(fallback_by_state, dict) and fallback_by_state:
        out["fallback_used_by_state"] = {str(state): bool(value) for state, value in fallback_by_state.items()}
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        cols = [
            "feature",
            "coverage",
            "ic_days",
            "mean_ic",
            "recent_mean_ic",
            "same_sign_years",
            "year_count",
            "worst_year_signed_ic",
            "topq_spread",
            "recent_topq_spread",
            "worst_year_topq_spread",
            "selection_score",
        ]
        present = [col for col in cols if col in summary.columns]
        rows = []
        for row in summary.loc[:, present].head(12).to_dict(orient="records"):
            rows.append(
                {
                    str(key): (str(value) if key == "feature" else _json_safe_float(value))
                    for key, value in row.items()
                }
            )
        out["selection_summary_top"] = rows
    if isinstance(sleeve_summary, pd.DataFrame) and not sleeve_summary.empty:
        cols = [
            "sleeve",
            "coverage",
            "ic_days",
            "mean_ic",
            "recent_mean_ic",
            "same_sign_years",
            "year_count",
            "worst_year_signed_ic",
            "selection_score",
        ]
        present = [col for col in cols if col in sleeve_summary.columns]
        rows = []
        for row in sleeve_summary.loc[:, present].head(12).to_dict(orient="records"):
            rows.append(
                {
                    str(key): (str(value) if key == "sleeve" else _json_safe_float(value))
                    for key, value in row.items()
                }
            )
        out["sleeve_summary_top"] = rows
    state_summaries = getattr(model, "selection_summary_by_state_", None)
    if isinstance(state_summaries, dict) and state_summaries:
        out["selection_summary_top_by_state"] = {}
        for state, state_summary in state_summaries.items():
            if not isinstance(state_summary, pd.DataFrame) or state_summary.empty:
                continue
            present = [col for col in cols if col in state_summary.columns]
            rows = []
            for row in state_summary.loc[:, present].head(12).to_dict(orient="records"):
                rows.append(
                    {
                        str(key): (str(value) if key == "feature" else _json_safe_float(value))
                        for key, value in row.items()
                    }
                )
            out["selection_summary_top_by_state"][str(state)] = rows
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward retrain + stitched pred.pkl for US Sharadar pipeline.")
    p.add_argument("--config", required=True, help="Base workflow config YAML (qrun-style)")
    p.add_argument("--provider_uri", default=None, help="Override Qlib provider URI")
    p.add_argument("--benchmark_pkl", default=None, help="Override BenchmarkExcessLabel benchmark_pkl")
    p.add_argument("--test_start", default=None, help="Override test start date YYYY-MM-DD")
    p.add_argument("--test_end", default=None, help="Override test end date YYYY-MM-DD")
    p.add_argument(
        "--test_block",
        choices=["year", "quarter"],
        default="year",
        help="How to split the test period into walk-forward blocks",
    )
    p.add_argument("--valid_days", type=int, default=63, help="Validation length in trading days before each test block")
    p.add_argument(
        "--embargo_days",
        type=int,
        default=None,
        help="Trading-day gap between train/valid/test blocks. Defaults to the parsed label horizon when available.",
    )
    p.add_argument(
        "--train_lookback_days",
        type=int,
        default=None,
        help="Optional sliding train lookback in trading days (keeps training window size bounded)",
    )
    p.add_argument("--exp_name", default=None, help="Override MLflow experiment name")
    p.add_argument("--out_pred", required=True, help="Output path to stitched pred.pkl")
    p.add_argument("--manifest", default=None, help="Optional manifest YAML/JSON path")
    p.add_argument("--dry_run", action="store_true", help="Print planned segments and exit (no training)")
    p.add_argument("--skip_train", action="store_true", help="Skip training and stitch from existing experiment runs")
    p.add_argument("--keep_port_analysis", action="store_true", help="Keep PortAnaRecord in tasks (slower)")
    p.add_argument("--max_tasks", type=int, default=None, help="Limit number of test blocks (debug)")
    p.add_argument(
        "--recency_half_life_days",
        type=int,
        default=0,
        help="Optional exponential sample-weight half-life in observed trading days; unsupported ranker models are left unweighted.",
    )
    p.add_argument(
        "--recency_min_weight",
        type=float,
        default=0.25,
        help="Minimum sample weight when --recency_half_life_days is enabled.",
    )
    p.add_argument(
        "--regime_reweight_feature",
        default="",
        help="Optional feature-group field for regime-balanced LightGBM sample weights, e.g. MKT_QQQ_RET_63D_LAG1.",
    )
    p.add_argument(
        "--regime_reweight_thresholds",
        default="-0.04,0.03",
        help="Comma-separated regime thresholds for --regime_reweight_feature.",
    )
    p.add_argument(
        "--disable_date_balance_reweight",
        action="store_true",
        help="Disable equal-date contribution inside the regime recency reweighter.",
    )
    p.add_argument("--year_balance_reweight", action="store_true", help="Also balance training samples by calendar year.")
    p.add_argument(
        "--label_tail_reweight",
        type=float,
        default=0.0,
        help="Optional additive weight for top/bottom per-date label tails in the regime recency reweighter.",
    )
    p.add_argument(
        "--label_tail_quantile",
        type=float,
        default=0.20,
        help="Per-date label tail quantile used when --label_tail_reweight is positive.",
    )
    p.add_argument(
        "--sample_max_weight",
        type=float,
        default=5.0,
        help="Maximum normalized sample weight for regime recency reweighting.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    out_pred = Path(args.out_pred).expanduser().resolve()
    out_pred.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(cfg_path)
    if args.benchmark_pkl:
        updated = _override_benchmark_pkl(cfg, str(Path(args.benchmark_pkl).expanduser().resolve()))
        if updated <= 0:
            raise ValueError("--benchmark_pkl was provided but no benchmark-relative label processor was found")
    qlib_init = dict(cfg.get("qlib_init", {}) or {})
    _override_provider_uri(qlib_init, args.provider_uri or "")
    label_horizon_days = _infer_label_horizon(cfg)
    embargo_days = int(args.embargo_days) if args.embargo_days is not None else int(label_horizon_days or 0)

    base_task = cfg.get("task")
    if not isinstance(base_task, dict):
        raise ValueError("config missing `task`")

    segs = _safe_get(base_task, ["dataset", "kwargs", "segments"])
    if not isinstance(segs, dict):
        raise ValueError("task.dataset.kwargs.segments missing or invalid")

    base_train_start = _as_ts(segs.get("train", [None, None])[0] if isinstance(segs.get("train"), list) else segs.get("train", (None, None))[0])
    base_test = segs.get("test")
    base_test_start = _as_ts(args.test_start or (base_test[0] if isinstance(base_test, (list, tuple)) else None))
    base_test_end = _as_ts(args.test_end or (base_test[1] if isinstance(base_test, (list, tuple)) else None))
    if base_train_start is None or base_test_start is None or base_test_end is None:
        raise ValueError("unable to infer train/test ranges from config; pass --test_start/--test_end")

    exp_name_base = _infer_exp_name(qlib_init)
    exp_name = args.exp_name or f"{exp_name_base}_wf_{args.test_block}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    _override_exp_name(qlib_init, exp_name)
    import qlib
    from qlib.constant import REG_US

    qlib_init.setdefault("region", REG_US)

    qlib.init(**qlib_init)

    if args.test_block == "quarter":
        test_segments = _split_test_by_quarter(base_test_start, base_test_end)
    else:
        test_segments = _split_test_by_year(base_test_start, base_test_end)
    if args.max_tasks is not None:
        test_segments = test_segments[: args.max_tasks]
    if not test_segments:
        raise ValueError("no test segments generated (check --test_start/--test_end)")

    # Calendar used for trade-day math (needs to include train+valid+test)
    cal_start = _align_to_trade_day_start(base_train_start)
    cal_end = test_segments[-1].end
    cal = _get_trade_calendar_span(cal_start, cal_end)
    specs = _build_walkforward_specs(
        cal=cal,
        train_start=base_train_start,
        test_segments=test_segments,
        valid_days=args.valid_days,
        train_lookback_days=args.train_lookback_days,
        embargo_days=embargo_days,
    )

    print(f"experiment: {exp_name}")
    print(
        f"test_blocks: {len(specs)} "
        f"(valid_days={args.valid_days}, embargo_days={embargo_days}, label_horizon_days={label_horizon_days})"
    )
    for i, s in enumerate(specs, start=1):
        print(f"{i:02d} test={s.test.start.date()}->{s.test.end.date()} valid={s.valid.start.date()}->{s.valid.end.date()} train={s.train.start.date()}->{s.train.end.date()}")

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else out_pred.with_suffix(".manifest.json")
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": str(cfg_path),
        "experiment": exp_name,
        "provider_uri": qlib_init.get("provider_uri"),
        "mlruns_uri": (((qlib_init.get("exp_manager") or {}).get("kwargs") or {}).get("uri") or ""),
        "test_block": args.test_block,
        "valid_days": args.valid_days,
        "embargo_days": embargo_days,
        "label_horizon_days": label_horizon_days,
        "train_lookback_days": args.train_lookback_days,
        "recency_half_life_days": int(args.recency_half_life_days),
        "recency_min_weight": float(args.recency_min_weight),
        "regime_reweight_feature": str(args.regime_reweight_feature or ""),
        "regime_reweight_thresholds": str(args.regime_reweight_thresholds or ""),
        "date_balance_reweight": not bool(args.disable_date_balance_reweight),
        "year_balance_reweight": bool(args.year_balance_reweight),
        "label_tail_reweight": float(args.label_tail_reweight),
        "label_tail_quantile": float(args.label_tail_quantile),
        "sample_max_weight": float(args.sample_max_weight),
        "out_pred": str(out_pred),
        "tasks": [
            {
                "name": s.name,
                "train": list(s.train.as_tuple_str()),
                "valid": list(s.valid.as_tuple_str()),
                "test": list(s.test.as_tuple_str()),
            }
            for s in specs
        ],
    }

    if args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"dry_run: wrote manifest {manifest_path}")
        return 0

    if not args.skip_train:
        tasks = []
        recency_attached = 0
        recency_skipped_ranker = 0
        regime_attached = 0
        regime_skipped_ranker = 0
        for s in specs:
            t = copy.deepcopy(base_task)
            if not args.keep_port_analysis:
                _strip_to_signal_only(t)
            _apply_segments_to_task(t, s)
            if str(args.regime_reweight_feature or "").strip():
                if _attach_regime_recency_reweighter(
                    t,
                    half_life_days=int(args.recency_half_life_days),
                    min_weight=float(args.recency_min_weight),
                    regime_feature=str(args.regime_reweight_feature),
                    regime_thresholds=_parse_float_list(str(args.regime_reweight_thresholds)),
                    date_balance=not bool(args.disable_date_balance_reweight),
                    year_balance=bool(args.year_balance_reweight),
                    label_tail_weight=float(args.label_tail_reweight),
                    label_tail_quantile=float(args.label_tail_quantile),
                    max_weight=float(args.sample_max_weight),
                ):
                    regime_attached += 1
                elif _model_class(t) == "LGBRankerModel":
                    regime_skipped_ranker += 1
            elif int(args.recency_half_life_days) > 0:
                if _attach_recency_reweighter(
                    t,
                    half_life_days=int(args.recency_half_life_days),
                    min_weight=float(args.recency_min_weight),
                ):
                    recency_attached += 1
                elif _model_class(t) == "LGBRankerModel":
                    recency_skipped_ranker += 1
            if "reweighter" in t:
                t.setdefault("reweighter_config", repr(t["reweighter"]))
            tasks.append(t)
        if str(args.regime_reweight_feature or "").strip():
            print(
                f"regime_recency_reweighter: attached={regime_attached} "
                f"skipped_ranker={regime_skipped_ranker} "
                f"feature={str(args.regime_reweight_feature)} "
                f"thresholds={str(args.regime_reweight_thresholds)} "
                f"half_life_days={int(args.recency_half_life_days)} "
                f"min_weight={float(args.recency_min_weight):.4f} "
                f"label_tail_weight={float(args.label_tail_reweight):.4f}"
            )
        if int(args.recency_half_life_days) > 0:
            print(
                f"recency_reweighter: attached={recency_attached} "
                f"skipped_ranker={recency_skipped_ranker} "
                f"half_life_days={int(args.recency_half_life_days)} "
                f"min_weight={float(args.recency_min_weight):.4f}"
            )
        from qlib.model.trainer import TrainerR

        trainer = TrainerR(experiment_name=exp_name)
        trainer.train(tasks)

    # Load recorders and match to our planned test segments.
    from qlib.workflow import R

    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders(rtype=exp.RT_L)
    rec_map: Dict[Tuple[str, str], Any] = {}
    for r in recs:
        try:
            task_obj = r.load_object("task")
        except Exception:
            continue
        test_seg = _safe_get(task_obj, ["dataset", "kwargs", "segments", "test"])
        if isinstance(test_seg, (list, tuple)) and len(test_seg) == 2:
            key = (str(test_seg[0]), str(test_seg[1]))
            # Prefer the newest recorder for each segment (mlflow search_runs defaults to start_time DESC).
            if key in rec_map:
                continue
            # Skip recorders that don't have prediction artifacts (failed/incomplete runs).
            if not _recorder_has_pred_artifact(r):
                continue
            rec_map[key] = r

    preds: List[Tuple[WalkForwardTaskSpec, pd.DataFrame]] = []
    used = []
    missing = []
    for s in specs:
        key = s.test.as_tuple_str()
        r = rec_map.get(key)
        if r is None:
            missing.append(key)
            continue
        df = _load_pred_from_recorder(r)
        preds.append((s, df))
        row = {"test": list(key), "run_id": r.id}
        diagnostics = _model_selection_diagnostics(r)
        if diagnostics:
            row["model_selection"] = diagnostics
        used.append(row)

    if missing:
        raise RuntimeError(f"missing recorders for {len(missing)} test blocks (example={missing[0]})")

    stitched = _stitch_preds(preds)
    stitched.to_pickle(out_pred)
    manifest["used_runs"] = used
    manifest["stitched_shape"] = list(stitched.shape)
    manifest["stitched_span"] = [_fmt_date(stitched.index.get_level_values('datetime').min()), _fmt_date(stitched.index.get_level_values('datetime').max())]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"saved: {out_pred} (rows={len(stitched)})")
    print(f"saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
