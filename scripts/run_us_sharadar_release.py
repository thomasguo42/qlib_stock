#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config structure: {path}")
    return data


def _parse_experiment_info(cfg: dict) -> Tuple[str, Path]:
    qlib_init = cfg.get("qlib_init", {}) or {}
    expm = qlib_init.get("exp_manager", {}) or {}
    kwargs = expm.get("kwargs", {}) or {}
    exp_name = str(kwargs.get("default_exp_name", "")).strip()
    uri = str(kwargs.get("uri", "")).strip()
    if not exp_name:
        raise ValueError("Missing qlib_init.exp_manager.kwargs.default_exp_name in config")
    if not uri:
        uri = "/workspace/qlib/mlruns"
    return exp_name, Path(uri).expanduser().resolve()


def _read_meta_name(meta_path: Path) -> str:
    try:
        txt = meta_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    for line in txt.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _find_experiment_dir(mlruns_uri: Path, exp_name: str) -> Path:
    if not mlruns_uri.exists():
        raise FileNotFoundError(f"mlruns uri not found: {mlruns_uri}")
    for p in sorted(mlruns_uri.iterdir()):
        if not p.is_dir():
            continue
        meta = p / "meta.yaml"
        if not meta.exists():
            continue
        if _read_meta_name(meta) == exp_name:
            return p
    raise FileNotFoundError(f"Experiment '{exp_name}' not found in {mlruns_uri}")


def _list_experiment_dirs(mlruns_uri: Path) -> List[Path]:
    out: List[Path] = []
    if not mlruns_uri.exists():
        return out
    for p in sorted(mlruns_uri.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "meta.yaml").exists():
            continue
        out.append(p)
    return out


def _find_run_dir_by_id(mlruns_uri: Path, run_id: str) -> Path:
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id cannot be empty")
    for exp_dir in _list_experiment_dirs(mlruns_uri):
        cand = exp_dir / run_id
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(f"Run ID not found under any experiment in {mlruns_uri}: {run_id}")


def _load_cmd(run_dir: Path) -> str:
    fp = run_dir / "params" / "cmd-sys.argv"
    if not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _find_latest_matching_run(exp_dir: Path, config_path: Path) -> Path:
    cfg_name = config_path.name
    candidates = []
    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        cmd = _load_cmd(p)
        if not cmd:
            continue
        if cfg_name not in cmd and str(config_path) not in cmd:
            continue
        art_pred = p / "artifacts" / "pred.pkl"
        if not art_pred.exists():
            continue
        mtime = art_pred.stat().st_mtime
        candidates.append((mtime, p))
    if not candidates:
        raise FileNotFoundError(
            f"No run with pred.pkl matched config {config_path} under experiment dir {exp_dir}"
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_latest_matching_run_any_experiment(mlruns_uri: Path, config_path: Path) -> Path:
    candidates = []
    for exp_dir in _list_experiment_dirs(mlruns_uri):
        try:
            run_dir = _find_latest_matching_run(exp_dir, config_path)
            pred = run_dir / "artifacts" / "pred.pkl"
            mtime = pred.stat().st_mtime
            candidates.append((mtime, run_dir))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No run with pred.pkl matched config {config_path} under any experiment in {mlruns_uri}"
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run qrun + strict release validation for US Sharadar pipeline."
    )
    p.add_argument("--config", required=True, help="Workflow config YAML for qrun")
    p.add_argument(
        "--benchmark_pkl",
        default="/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl",
        help="Benchmark return series pickle for validator",
    )
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data", help="Qlib provider URI")
    p.add_argument("--start", default="", help="Validation start date YYYY-MM-DD")
    p.add_argument("--end", default="", help="Validation end date YYYY-MM-DD")
    p.add_argument("--qrun_bin", default="qrun", help="qrun executable")
    p.add_argument("--python_bin", default=sys.executable, help="Python executable for validator")
    p.add_argument(
        "--validator",
        default="scripts/validate_us_sharadar_pipeline.py",
        help="Validator script path",
    )
    p.add_argument(
        "--universe_auditor",
        default="scripts/audit_us_universe_integrity.py",
        help="Universe integrity auditor script path",
    )
    p.add_argument(
        "--skip_universe_integrity",
        action="store_true",
        help="Skip universe integrity audit before training/validation",
    )
    p.add_argument(
        "--integrity_reference_markets",
        default="sp500,nasdaq100",
        help="Comma-separated reference markets for universe audit",
    )
    p.add_argument(
        "--integrity_min_overlap",
        type=float,
        default=0.60,
        help="Minimum overlap ratio vs each reference market",
    )
    p.add_argument(
        "--integrity_anchors",
        default="AAPL,MSFT,NVDA,AMZN,GOOGL,META|FB,TSLA,JPM,XOM,AVGO,LLY,V,MA,HD,COST",
        help="Comma-separated anchor symbols (supports alias groups with |)",
    )
    p.add_argument("--skip_train", action="store_true", help="Skip qrun and validate latest matching run")
    p.add_argument("--skip_data_checks", action="store_true", help="Pass --skip_data_checks to validator")
    p.add_argument("--run_id", default="", help="Optional explicit MLflow run ID to validate")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2
    validator_path = Path(args.validator).expanduser().resolve()
    if not validator_path.exists():
        print(f"Validator script not found: {validator_path}")
        return 2
    auditor_path = Path(args.universe_auditor).expanduser().resolve()
    if not args.skip_universe_integrity and not auditor_path.exists():
        print(f"Universe auditor script not found: {auditor_path}")
        return 2

    cfg = _load_yaml(config_path)
    exp_name, mlruns_uri = _parse_experiment_info(cfg)

    if not args.skip_universe_integrity:
        audit_cmd = [
            args.python_bin,
            str(auditor_path),
            "--config",
            str(config_path),
            "--provider_uri",
            str(args.provider_uri),
            "--reference_markets",
            str(args.integrity_reference_markets),
            "--must_include",
            str(args.integrity_anchors),
            "--min_reference_overlap",
            str(args.integrity_min_overlap),
            "--fail_on_overlap_fail",
            "--fail_on_anchor_fail",
        ]
        if args.start:
            audit_cmd += ["--start", args.start]
        if args.end:
            audit_cmd += ["--end", args.end]
        print("== Universe Integrity Audit ==")
        print(" ".join(audit_cmd))
        rc = subprocess.call(audit_cmd)
        if rc != 0:
            print(f"universe integrity audit failed with exit code {rc}")
            return rc

    if not args.skip_train:
        train_cmd = [args.qrun_bin, str(config_path)]
        print("== Train ==")
        print(" ".join(train_cmd))
        rc = subprocess.call(train_cmd)
        if rc != 0:
            print(f"qrun failed with exit code {rc}")
            return rc

    exp_dir: Optional[Path] = None
    try:
        exp_dir = _find_experiment_dir(mlruns_uri, exp_name)
    except FileNotFoundError as e:
        print(f"warning: {e}")
        print("warning: falling back to cross-experiment run lookup")

    if args.run_id:
        try:
            if exp_dir is not None:
                run_dir = exp_dir / args.run_id
                if not run_dir.exists():
                    raise FileNotFoundError
            else:
                run_dir = _find_run_dir_by_id(mlruns_uri, args.run_id)
        except Exception:
            try:
                run_dir = _find_run_dir_by_id(mlruns_uri, args.run_id)
            except Exception as e:
                print(str(e))
                return 2
    else:
        try:
            if exp_dir is not None:
                run_dir = _find_latest_matching_run(exp_dir, config_path)
            else:
                run_dir = _find_latest_matching_run_any_experiment(mlruns_uri, config_path)
        except Exception as e:
            print(str(e))
            return 2
    pred_path = run_dir / "artifacts" / "pred.pkl"
    if not pred_path.exists():
        print(f"pred.pkl not found: {pred_path}")
        return 2

    validate_cmd = [
        args.python_bin,
        str(validator_path),
        "--config",
        str(config_path),
        "--provider_uri",
        str(args.provider_uri),
        "--pred",
        str(pred_path),
        "--benchmark_pkl",
        str(Path(args.benchmark_pkl).expanduser().resolve()),
        "--by_year",
        "--check_data_quality",
        "--fail_on_data_quality_fail",
        "--check_gates",
        "--fail_on_gate_fail",
        "--check_rolling",
        "--fail_on_rolling_fail",
        "--gate_profile",
        "release",
    ]
    if args.start:
        validate_cmd += ["--start", args.start]
    if args.end:
        validate_cmd += ["--end", args.end]
    if args.skip_data_checks:
        validate_cmd += ["--skip_data_checks"]

    print("\n== Validate ==")
    print(f"run_id={run_dir.name}")
    print(f"pred={pred_path}")
    print(" ".join(validate_cmd))
    rc = subprocess.call(validate_cmd)
    if rc != 0:
        print(f"release validation failed with exit code {rc}")
        return rc
    print("release validation PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
