#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    return data


def _dump_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _pair_fields_names(cfg: dict) -> Tuple[List[str], List[str]]:
    dh = cfg.get("data_handler_config", {}) or {}
    fields = list(dh.get("extra_fields", []) or [])
    names = list(dh.get("extra_names", []) or [])
    if len(fields) != len(names):
        raise ValueError(
            f"extra_fields({len(fields)}) and extra_names({len(names)}) lengths differ"
        )
    return fields, names


def _classify_field(expr: str) -> str:
    e = str(expr)
    if "$insider_" in e:
        return "sf2"
    if "$inst13f_" in e:
        return "sf3a"
    return "base"


def _build_variant(fields: List[str], names: List[str], keep_groups: List[str]) -> Tuple[List[str], List[str]]:
    out_f = []
    out_n = []
    for f, n in zip(fields, names):
        grp = _classify_field(f)
        if grp in keep_groups:
            out_f.append(f)
            out_n.append(n)
    return out_f, out_n


def _set_exp_name(cfg: dict, exp_name: str):
    qlib_init = cfg.setdefault("qlib_init", {})
    expm = qlib_init.setdefault("exp_manager", {})
    kwargs = expm.setdefault("kwargs", {})
    kwargs["default_exp_name"] = exp_name


def _set_extra(cfg: dict, fields: List[str], names: List[str]):
    dh = cfg.setdefault("data_handler_config", {})
    dh["extra_fields"] = fields
    dh["extra_names"] = names


def _build_variants(base_cfg: dict) -> Dict[str, dict]:
    fields, names = _pair_fields_names(base_cfg)

    variants: Dict[str, dict] = {}
    keep_map = {
        "nocor": ["base"],
        "sf2only": ["base", "sf2"],
        "sf3aonly": ["base", "sf3a"],
        "cor_full": ["base", "sf2", "sf3a"],
    }
    for tag, groups in keep_map.items():
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
        vf, vn = _build_variant(fields, names, groups)
        _set_extra(cfg, vf, vn)

        qlib_init = cfg.get("qlib_init", {}) or {}
        expm = qlib_init.get("exp_manager", {}) or {}
        kwargs = expm.get("kwargs", {}) or {}
        old_exp = str(kwargs.get("default_exp_name", "us_sharadar_weekly_ablation"))
        _set_exp_name(cfg, f"{old_exp}_{tag}")
        variants[tag] = cfg
    return variants


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build US Sharadar COR ablation config variants.")
    p.add_argument("--base_config", required=True, help="Base YAML config path")
    p.add_argument("--out_dir", default=None, help="Output directory (default: base config dir)")
    p.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: base stem)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    base_path = Path(args.base_config).expanduser().resolve()
    if not base_path.exists():
        print(f"base_config not found: {base_path}")
        return 2

    base_cfg = _load_yaml(base_path)
    variants = _build_variants(base_cfg)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else base_path.parent
    )
    prefix = args.prefix if args.prefix else base_path.stem

    for tag, cfg in variants.items():
        out_path = out_dir / f"{prefix}_{tag}.yaml"
        _dump_yaml(out_path, cfg)
        dh = cfg.get("data_handler_config", {}) or {}
        n_fields = len(dh.get("extra_fields", []) or [])
        print(f"{tag}: {out_path} (extra_fields={n_fields})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
