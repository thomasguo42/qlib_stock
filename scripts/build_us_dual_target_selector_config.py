#!/usr/bin/env python
"""Create a dual-horizon selector workflow config from an existing US config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


LABEL_PROCESSORS = {
    "BenchmarkExcessLabel",
    "ResidualForwardReturnLabel",
    "VolScaledExcessLabel",
    "DownsideAdjustedExcessLabel",
    "PortfolioUtilityExcessLabel",
    "DualHorizonPortfolioUtilityLabel",
}

SEC_ALPHA_FEATURES = [
    "$sec_alpha_event_composite",
    "$sec_alpha_filing_freshness",
    "$sec_alpha_material_event_intensity",
    "$sec_alpha_periodic_report_intensity",
    "$sec_alpha_amendment_risk",
    "$sec_alpha_event_coverage",
]

SEC_EVENT_SLEEVE = {
    "SEC_ALPHA_EVENT_COMPOSITE": 1.00,
    "SEC_ALPHA_MATERIAL_EVENT_INTENSITY": 0.40,
    "SEC_ALPHA_AMENDMENT_RISK": 0.40,
    "SEC_ALPHA_PERIODIC_REPORT_INTENSITY": 0.20,
    "SEC_ALPHA_FILING_FRESHNESS": 0.20,
    "SEC_ALPHA_EVENT_COVERAGE": 0.10,
}


def parse_csv_ints(value: str) -> List[int]:
    out = [int(token.strip()) for token in str(value or "").split(",") if token.strip()]
    if len(out) < 2:
        raise ValueError("at least two horizons are required")
    if any(item <= 0 for item in out):
        raise ValueError("horizons must be positive")
    return out


def parse_csv_floats(value: str, expected: int) -> List[float]:
    out = [float(token.strip()) for token in str(value or "").split(",") if token.strip()]
    if len(out) != int(expected):
        raise ValueError("weights length must match horizons length")
    if any(item < 0 for item in out):
        raise ValueError("weights must be non-negative")
    total = float(sum(out))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [item / total for item in out]


def label_expr_for_horizon(horizon: int, ref_start_days: int) -> str:
    ref = max(0, int(ref_start_days))
    start_expr = "$close" if ref == 0 else f"Ref($close, -{ref})"
    return f"Ref($close, -{int(horizon) + ref})/{start_expr} - 1"


def processor_class(proc: Dict[str, Any]) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML config: {path}")
    return data


def handler_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dh = cfg.get("data_handler_config")
    if isinstance(dh, dict):
        return dh
    dh = (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("handler", {}).get("kwargs")
    if not isinstance(dh, dict):
        raise ValueError("config does not contain a data handler config")
    return dh


def ensure_sec_features(dh: Dict[str, Any], fields: Sequence[str]) -> None:
    extra_fields = dh.setdefault("extra_fields", [])
    extra_names = dh.setdefault("extra_names", [])
    existing_names = {str(name).upper() for name in extra_names}
    for field in fields:
        name = str(field).lstrip("$").upper()
        if name in existing_names:
            continue
        extra_fields.append(str(field))
        extra_names.append(name)
        existing_names.add(name)


def ensure_sec_model_sleeve(cfg: Dict[str, Any]) -> None:
    model = (cfg.get("task") or {}).get("model") or {}
    kwargs = model.get("kwargs") or {}
    sleeves = kwargs.get("sleeves")
    if not isinstance(sleeves, dict):
        return
    sleeves.setdefault("sec_event", dict(SEC_EVENT_SLEEVE))
    if "max_sleeves" in kwargs:
        try:
            kwargs["max_sleeves"] = max(int(kwargs["max_sleeves"]), len(sleeves))
        except Exception:
            kwargs["max_sleeves"] = len(sleeves)


def set_dual_target(
    cfg: Dict[str, Any],
    *,
    horizons: Sequence[int],
    weights: Sequence[float],
    benchmark_pkl: str,
    benchmark_kind: str,
    label_ref_start_days: int,
    vol_feature: str,
    downside_penalty: float,
    volatility_penalty: float,
    clip_abs_label: float,
) -> Dict[str, Any]:
    dh = handler_config(cfg)
    dh["label"] = [
        [label_expr_for_horizon(int(horizon), int(label_ref_start_days)) for horizon in horizons],
        [f"LABEL{i}" for i, _horizon in enumerate(horizons)],
    ]
    processors = dh.setdefault("learn_processors", [])
    kwargs = {
        "benchmark_pkl": str(benchmark_pkl),
        "benchmark_kind": str(benchmark_kind),
        "label_horizon_days": [int(h) for h in horizons],
        "label_weights": [float(w) for w in weights],
        "label_ref_start_days": int(label_ref_start_days),
        "fields_group": "label",
        "vol_feature": str(vol_feature),
        "downside_penalty": float(downside_penalty),
        "volatility_penalty": float(volatility_penalty),
        "scale_vol_by_horizon": True,
        "divide_by_vol": False,
        "fill_missing_vol": True,
        "drop_extra_labels": True,
    }
    if clip_abs_label > 0:
        kwargs["clip_abs_label"] = float(clip_abs_label)
    for proc in processors:
        if not isinstance(proc, dict) or processor_class(proc) not in LABEL_PROCESSORS:
            continue
        proc["class"] = "DualHorizonPortfolioUtilityLabel"
        proc["module_path"] = "qlib.contrib.data.processor"
        proc["kwargs"] = kwargs
        break
    else:
        processors.append(
            {
                "class": "DualHorizonPortfolioUtilityLabel",
                "module_path": "qlib.contrib.data.processor",
                "kwargs": kwargs,
            }
        )
    strategy_kwargs = (((cfg.get("port_analysis_config") or {}).get("strategy") or {}).get("kwargs") or {})
    if isinstance(strategy_kwargs, dict):
        strategy_kwargs["hold_thresh"] = int(max(horizons))
    return cfg


def build_config(
    *,
    base_config: Path,
    out: Path,
    horizons: Sequence[int],
    weights: Sequence[float],
    benchmark_pkl: str,
    benchmark_kind: str,
    label_ref_start_days: int,
    vol_feature: str,
    downside_penalty: float,
    volatility_penalty: float,
    clip_abs_label: float,
    include_sec_features: bool,
) -> Path:
    cfg = load_yaml(base_config)
    set_dual_target(
        cfg,
        horizons=horizons,
        weights=weights,
        benchmark_pkl=benchmark_pkl,
        benchmark_kind=benchmark_kind,
        label_ref_start_days=label_ref_start_days,
        vol_feature=vol_feature,
        downside_penalty=downside_penalty,
        volatility_penalty=volatility_penalty,
        clip_abs_label=clip_abs_label,
    )
    if include_sec_features:
        ensure_sec_features(handler_config(cfg), SEC_ALPHA_FEATURES)
        ensure_sec_model_sleeve(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch a US selector workflow YAML into a dual-target experiment.")
    p.add_argument("--base_config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", default="10,20")
    p.add_argument("--weights", default="0.35,0.65")
    p.add_argument("--benchmark_pkl", default="/root/.qlib/qlib_data/us_data/bench_qqq.pkl")
    p.add_argument("--benchmark_kind", default="return")
    p.add_argument("--label_ref_start_days", type=int, default=1)
    p.add_argument("--vol_feature", default="RISK_VOL_20D")
    p.add_argument("--downside_penalty", type=float, default=0.75)
    p.add_argument("--volatility_penalty", type=float, default=0.20)
    p.add_argument("--clip_abs_label", type=float, default=0.0)
    p.add_argument("--include_sec_features", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    horizons = parse_csv_ints(args.horizons)
    weights = parse_csv_floats(args.weights, expected=len(horizons))
    out = build_config(
        base_config=Path(args.base_config).expanduser().resolve(),
        out=Path(args.out).expanduser().resolve(),
        horizons=horizons,
        weights=weights,
        benchmark_pkl=str(args.benchmark_pkl),
        benchmark_kind=str(args.benchmark_kind),
        label_ref_start_days=int(args.label_ref_start_days),
        vol_feature=str(args.vol_feature),
        downside_penalty=float(args.downside_penalty),
        volatility_penalty=float(args.volatility_penalty),
        clip_abs_label=float(args.clip_abs_label),
        include_sec_features=bool(args.include_sec_features),
    )
    print(f"dual_target_config={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
