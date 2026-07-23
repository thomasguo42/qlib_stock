#!/usr/bin/env python
"""Build selector specialist workflow configs from an existing US Sharadar config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


RISK_FEATURES: List[Tuple[str, str]] = [
    ("$risk_beta_qqq_63d", "RISK_BETA_QQQ_63D"),
    ("$risk_vol_20d", "RISK_VOL_20D"),
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML structure: {path}")
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def handler_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg.get("data_handler_config"), dict):
        return cfg["data_handler_config"]
    return (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("handler", {}).setdefault("kwargs", {})


def set_exp_name(cfg: Dict[str, Any], name: str) -> None:
    qlib_init = cfg.setdefault("qlib_init", {})
    expm = qlib_init.setdefault("exp_manager", {})
    kwargs = expm.setdefault("kwargs", {})
    kwargs["default_exp_name"] = name
    cfg["experiment_name"] = name


def ensure_extra_features(cfg: Dict[str, Any], pairs: List[Tuple[str, str]]) -> None:
    dh = handler_config(cfg)
    fields = list(dh.get("extra_fields", []) or [])
    names = list(dh.get("extra_names", []) or [])
    name_set = {str(name).strip().upper() for name in names}
    for field, name in pairs:
        if str(name).strip().upper() not in name_set:
            fields.append(str(field))
            names.append(str(name))
            name_set.add(str(name).strip().upper())
    dh["extra_fields"] = fields
    dh["extra_names"] = names


def remove_existing_risk_filters(processors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for proc in processors:
        if not isinstance(proc, dict):
            out.append(proc)
            continue
        cls = str(proc.get("class", "")).split(".")[-1]
        if cls == "RiskTierFilter":
            continue
        out.append(proc)
    return out


def risk_filter_processor(tier: str) -> Dict[str, Any]:
    return {
        "class": "RiskTierFilter",
        "module_path": "qlib.contrib.data.processor",
        "kwargs": {
            "tier": str(tier),
            "beta_feature": "RISK_BETA_QQQ_63D",
            "vol_feature": "RISK_VOL_20D",
            "low_beta_max": 1.05,
            "low_vol_max": 0.35,
            "medium_beta_max": 1.50,
            "medium_vol_max": 0.60,
        },
    }


def add_risk_filter(cfg: Dict[str, Any], tier: str) -> None:
    dh = handler_config(cfg)
    processors = remove_existing_risk_filters(list(dh.get("learn_processors", []) or []))
    processors.append(risk_filter_processor(tier))
    dh["learn_processors"] = processors


def normalize_with_bounds(
    weights: Dict[str, float],
    *,
    floors: Dict[str, float],
    caps: Dict[str, float],
) -> Dict[str, float]:
    keys = list(weights)
    if not keys:
        return {}

    lower = {k: max(float(floors.get(k, 0.0)), 0.0) for k in keys}
    upper = {k: max(float(caps[k]), lower[k]) if k in caps else float("inf") for k in keys}
    out = {k: min(max(float(weights.get(k, 0.0)), lower[k]), upper[k]) for k in keys}

    for _ in range(20):
        total = sum(out.values())
        diff = 1.0 - total
        if abs(diff) <= 1e-12:
            break

        if diff > 0:
            adjustable = [k for k in keys if out[k] < upper[k] - 1e-12]
            if not adjustable:
                break
            finite_room = {k: upper[k] - out[k] for k in adjustable if upper[k] < float("inf")}
            if finite_room and sum(finite_room.values()) >= diff:
                denom = sum(finite_room.values())
                for k, room in finite_room.items():
                    out[k] += diff * room / denom
            else:
                preferred = [k for k in ("qqq_leadership", "momentum_regime") if k in adjustable]
                targets = preferred or adjustable
                denom = sum(max(out[k], 1e-12) for k in targets)
                for k in targets:
                    out[k] += diff * max(out[k], 1e-12) / denom
        else:
            excess = -diff
            adjustable = [k for k in keys if out[k] > lower[k] + 1e-12]
            if not adjustable:
                break
            room = {k: out[k] - lower[k] for k in adjustable}
            denom = sum(room.values())
            for k, value in room.items():
                out[k] -= excess * value / denom
                out[k] = max(out[k], lower[k])

    total = sum(out.values())
    if total > 0:
        out = {k: float(v) / total for k, v in out.items()}
    return out


def high_risk_sleeve_overlay(cfg: Dict[str, Any]) -> None:
    model = ((cfg.get("task") or {}).get("model") or {})
    kwargs = model.get("kwargs") or {}
    if str(model.get("class")) != "RegimeSleeveScoreModel" or not isinstance(kwargs, dict):
        return
    state_weights = kwargs.get("state_weights") or {}
    for state in list(state_weights):
        weights = {k: max(float(v), 0.0) for k, v in dict(state_weights.get(state) or {}).items()}
        floors = {}
        caps = {}
        if "qqq_leadership" in weights:
            floors["qqq_leadership"] = 0.45
        if "momentum_regime" in weights:
            floors["momentum_regime"] = 0.25
        if "quality_value" in weights:
            caps["quality_value"] = 0.20
        state_weights[state] = normalize_with_bounds(weights, floors=floors, caps=caps)
    kwargs["state_weights"] = state_weights


def build_high_risk_upside_config(base_cfg: Dict[str, Any], *, exp_suffix: str = "high_risk_upside") -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    ensure_extra_features(cfg, RISK_FEATURES)
    add_risk_filter(cfg, "high")
    high_risk_sleeve_overlay(cfg)
    qlib_init = cfg.get("qlib_init", {}) or {}
    old_exp = (
        ((qlib_init.get("exp_manager") or {}).get("kwargs") or {}).get("default_exp_name")
        or cfg.get("experiment_name")
        or "us_selector"
    )
    set_exp_name(cfg, f"{old_exp}_{exp_suffix}")
    cfg.setdefault("selector_specialist", {})
    cfg["selector_specialist"].update(
        {
            "kind": "high_risk_upside",
            "risk_filter_tier": "high",
            "risk_features": [name for _, name in RISK_FEATURES],
        }
    )
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build high-risk-upside selector specialist config.")
    p.add_argument("--base_config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prefix = str(args.prefix or base_path.stem)
    cfg = build_high_risk_upside_config(load_yaml(base_path))
    out_path = out_dir / f"{prefix}_high_risk_upside.yaml"
    dump_yaml(out_path, cfg)
    dh = handler_config(cfg)
    print(f"high_risk_upside={out_path}")
    print(f"extra_fields={len(dh.get('extra_fields', []) or [])}")
    print(f"learn_processors={len(dh.get('learn_processors', []) or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
