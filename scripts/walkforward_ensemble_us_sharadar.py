#!/usr/bin/env python
"""Train two walk-forward US Sharadar members and emit a blended pred.pkl."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ensemble_us_sharadar_predictions import combine_predictions, load_prediction


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
        raise ValueError(f"invalid YAML config: {path}")
    return data


def _write_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _processor_class(proc: Dict) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def _handler_configs(cfg: Dict) -> List[Dict]:
    out: List[Dict] = []
    dh = cfg.get("data_handler_config")
    if isinstance(dh, dict):
        out.append(dh)
    task_dh = (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("handler", {}).get("kwargs")
    if isinstance(task_dh, dict) and all(task_dh is not item for item in out):
        out.append(task_dh)
    return out


def _override_benchmark_pkl(cfg: Dict, benchmark_pkl: str) -> int:
    if not benchmark_pkl:
        return 0
    count = 0
    for dh in _handler_configs(cfg):
        for proc in dh.get("learn_processors", []) or []:
            if not isinstance(proc, dict) or _processor_class(proc) not in LABEL_PROCESSORS:
                continue
            kwargs = proc.setdefault("kwargs", {})
            if isinstance(kwargs, dict):
                kwargs["benchmark_pkl"] = str(benchmark_pkl)
                count += 1
    return count


def _set_nested_key(obj, target_key: str, value: str) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == target_key:
                obj[key] = value
            else:
                _set_nested_key(val, target_key, value)
    elif isinstance(obj, list):
        for item in obj:
            _set_nested_key(item, target_key, value)


def _set_runtime_date_ranges(cfg: Dict, *, test_start: str, test_end: str) -> None:
    dh_cfg = cfg.get("data_handler_config")
    if isinstance(dh_cfg, dict):
        if test_end:
            dh_cfg["end_time"] = test_end
            _set_nested_key(dh_cfg.get("filter_pipe", []), "filter_end_time", test_end)

    backtest = ((cfg.get("port_analysis_config") or {}).get("backtest") or {})
    if isinstance(backtest, dict):
        if test_start:
            backtest["start_time"] = test_start
        if test_end:
            backtest["end_time"] = test_end

    seg = (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments", {})
    test = seg.get("test") if isinstance(seg, dict) else None
    if isinstance(test, list) and len(test) == 2:
        if test_start:
            test[0] = test_start
        if test_end:
            test[1] = test_end


def _exp_name(cfg: Dict) -> str:
    return str((((cfg.get("qlib_init") or {}).get("exp_manager") or {}).get("kwargs") or {}).get("default_exp_name") or "")


def _model_class(cfg: Dict) -> str:
    return str((((cfg.get("task") or {}).get("model") or {}).get("class") or "")).split(".")[-1]


def _label_horizon(cfg: Dict) -> Optional[int]:
    import re

    label = ""
    for dh in _handler_configs(cfg):
        try:
            label = str(dh.get("label", [])[0][0])
            break
        except Exception:
            continue
    m = re.search(r"Ref\(\$close,\s*-(\d+)\)\s*/\s*Ref\(\$close,\s*-(\d+)\)", label)
    if not m:
        return None
    return max(0, int(m.group(1)) - int(m.group(2)))


def _mlruns_uri(cfg: Dict) -> str:
    return str((((cfg.get("qlib_init") or {}).get("exp_manager") or {}).get("kwargs") or {}).get("uri") or "")


def _write_member_runtime_config(
    *,
    source_config: Path,
    runtime_config: Path,
    provider_uri: str,
    mlruns_uri: str,
    benchmark_pkl: str,
    test_start: str,
    test_end: str,
    override_label_benchmark_pkl: bool,
) -> Tuple[Path, Dict]:
    cfg = _load_yaml(source_config)
    qlib_init = cfg.setdefault("qlib_init", {})
    qlib_init["provider_uri"] = str(provider_uri)
    if mlruns_uri:
        expm = qlib_init.setdefault("exp_manager", {})
        kwargs = expm.setdefault("kwargs", {})
        kwargs["uri"] = str(mlruns_uri)
    if override_label_benchmark_pkl:
        updated = _override_benchmark_pkl(cfg, benchmark_pkl)
        if updated <= 0:
            raise ValueError(f"no benchmark-relative label processor found in {source_config}")
    _set_runtime_date_ranges(cfg, test_start=test_start, test_end=test_end)
    _write_yaml(runtime_config, cfg)
    return runtime_config, cfg


def _build_member_train_command(
    *,
    python_bin: str,
    walkforward_script: Path,
    config: Path,
    provider_uri: str,
    test_start: str,
    test_end: str,
    test_block: str,
    valid_days: int,
    exp_name: str,
    out_pred: Path,
    manifest: Path,
    embargo_days: Optional[int] = None,
    train_lookback_days: Optional[int] = None,
) -> List[str]:
    cmd = [
        str(python_bin),
        str(walkforward_script),
        "--config",
        str(config),
        "--provider_uri",
        str(provider_uri),
        "--test_start",
        str(test_start),
        "--test_end",
        str(test_end),
        "--test_block",
        str(test_block),
        "--valid_days",
        str(int(valid_days)),
        "--exp_name",
        str(exp_name),
        "--out_pred",
        str(out_pred),
        "--manifest",
        str(manifest),
    ]
    if embargo_days is not None:
        cmd.extend(["--embargo_days", str(int(embargo_days))])
    if train_lookback_days is not None:
        cmd.extend(["--train_lookback_days", str(int(train_lookback_days))])
    return cmd


def _read_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _date_summary(pred) -> Dict[str, object]:
    if pred.empty:
        return {"rows": 0, "dates": 0}
    dates = pred.index.get_level_values("datetime")
    return {
        "rows": int(len(pred)),
        "dates": int(dates.nunique()),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
    }


def _flatten_used_runs(member: Dict) -> List[Dict]:
    manifest = member.get("manifest") if isinstance(member.get("manifest"), dict) else {}
    rows = []
    for item in manifest.get("used_runs", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "member": member.get("name"),
                "role": member.get("role"),
                "run_id": item.get("run_id"),
                "test": item.get("test"),
                "experiment": manifest.get("experiment") or member.get("experiment"),
                "mlruns_uri": member.get("mlruns_uri") or manifest.get("mlruns_uri"),
                "model_class": member.get("model_class"),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v8/v9-style walk-forward members and blend their predictions.")
    p.add_argument("--primary_config", required=True)
    p.add_argument("--defensive_config", required=True)
    p.add_argument("--out_pred", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--work_dir", default="")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--walkforward_script", default="scripts/walkforward_train_us_sharadar.py")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--benchmark_pkl", default="")
    p.add_argument("--override_label_benchmark_pkl", action="store_true")
    p.add_argument("--mlruns_uri", default="")
    p.add_argument("--test_start", required=True)
    p.add_argument("--test_end", required=True)
    p.add_argument("--test_block", choices=["year", "quarter"], default="year")
    p.add_argument("--valid_days", type=int, default=63)
    p.add_argument("--embargo_days", type=int, default=None)
    p.add_argument("--train_lookback_days", type=int, default=None)
    p.add_argument("--primary_name", default="primary")
    p.add_argument("--defensive_name", default="defensive")
    p.add_argument("--primary_weight", type=float, default=0.70)
    p.add_argument("--defensive_weight", type=float, default=0.30)
    p.add_argument("--normalize", choices=["none", "zscore", "rank_zscore"], default="rank_zscore")
    p.add_argument("--tag", default="")
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_pred = Path(args.out_pred).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else out_pred.parent / f"{out_pred.stem}_members"
    runtime_dir = work_dir / "runtime_configs"
    pred_dir = work_dir / "preds"
    manifest_dir = work_dir / "manifests"
    log_dir = work_dir / "logs"
    for path in (runtime_dir, pred_dir, manifest_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    walkforward_script = Path(args.walkforward_script).expanduser().resolve()
    if not walkforward_script.exists():
        print(f"walk-forward script not found: {walkforward_script}", file=sys.stderr)
        return 2

    member_specs = [
        {
            "name": str(args.primary_name),
            "role": "primary",
            "source_config": Path(args.primary_config).expanduser().resolve(),
            "weight": float(args.primary_weight),
        },
        {
            "name": str(args.defensive_name),
            "role": "defensive",
            "source_config": Path(args.defensive_config).expanduser().resolve(),
            "weight": float(args.defensive_weight),
        },
    ]
    if sum(float(m["weight"]) for m in member_specs) <= 0:
        raise ValueError("member weights must sum positive")

    members = []
    for member in member_specs:
        source_config = Path(member["source_config"])
        if not source_config.exists():
            print(f"member config not found: {source_config}", file=sys.stderr)
            return 2
        runtime_config = runtime_dir / f"{member['role']}_{source_config.name}"
        runtime_config, runtime_cfg = _write_member_runtime_config(
            source_config=source_config,
            runtime_config=runtime_config,
            provider_uri=str(args.provider_uri),
            mlruns_uri=str(args.mlruns_uri or ""),
            benchmark_pkl=str(Path(args.benchmark_pkl).expanduser().resolve()) if str(args.benchmark_pkl or "").strip() else "",
            test_start=str(args.test_start),
            test_end=str(args.test_end),
            override_label_benchmark_pkl=bool(args.override_label_benchmark_pkl),
        )
        base_exp = _exp_name(runtime_cfg) or source_config.stem
        exp_name = f"{base_exp}_ensemble_{member['role']}_{tag}"
        pred_path = pred_dir / f"{member['role']}_{tag}_pred.pkl"
        member_manifest_path = manifest_dir / f"{member['role']}_{tag}_manifest.json"
        cmd = _build_member_train_command(
            python_bin=str(args.python_bin),
            walkforward_script=walkforward_script,
            config=runtime_config,
            provider_uri=str(args.provider_uri),
            test_start=str(args.test_start),
            test_end=str(args.test_end),
            test_block=str(args.test_block),
            valid_days=int(args.valid_days),
            exp_name=exp_name,
            out_pred=pred_path,
            manifest=member_manifest_path,
            embargo_days=args.embargo_days,
            train_lookback_days=args.train_lookback_days,
        )
        log_path = log_dir / f"{member['role']}_{tag}.log"
        if not args.skip_train:
            print("== Member Walk-Forward Train ==")
            print(" ".join(cmd))
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log_path.write_text(proc.stdout or "", encoding="utf-8")
            if int(proc.returncode) != 0:
                print(proc.stdout or "")
                print(f"member training failed ({member['role']}) with exit code {proc.returncode}", file=sys.stderr)
                return int(proc.returncode)
        if not pred_path.exists():
            print(f"member pred.pkl not found: {pred_path}", file=sys.stderr)
            return 2
        member_manifest = _read_manifest(member_manifest_path)
        members.append(
            {
                **member,
                "source_config": str(source_config),
                "runtime_config": str(runtime_config),
                "pred": str(pred_path),
                "manifest_path": str(member_manifest_path),
                "manifest": member_manifest,
                "log": str(log_path),
                "experiment": exp_name,
                "mlruns_uri": _mlruns_uri(runtime_cfg),
                "model_class": _model_class(runtime_cfg),
                "label_horizon_days": _label_horizon(runtime_cfg),
                "command": " ".join(cmd),
            }
        )

    pred_by_role = {member["role"]: load_prediction(Path(member["pred"])) for member in members}
    primary = pred_by_role["primary"]
    defensive = pred_by_role["defensive"]
    primary_weight = float(next(m["weight"] for m in members if m["role"] == "primary"))
    defensive_weight = float(next(m["weight"] for m in members if m["role"] == "defensive"))
    blended = combine_predictions(
        primary,
        defensive,
        gate=None,
        normal_primary_weight=primary_weight,
        normal_defensive_weight=defensive_weight,
        fallback_primary_weight=primary_weight,
        fallback_defensive_weight=defensive_weight,
        normalize=str(args.normalize),
    )
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    blended.to_pickle(out_pred)

    first_manifest = members[0].get("manifest") or {}
    manifest = {
        "ensemble": True,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "method": "static_blend",
        "normalize": str(args.normalize),
        "out_pred": str(out_pred),
        "provider_uri": str(args.provider_uri),
        "test_block": str(args.test_block),
        "valid_days": int(args.valid_days),
        "embargo_days": first_manifest.get("embargo_days"),
        "label_horizon_days": members[0].get("label_horizon_days"),
        "tasks": first_manifest.get("tasks", []),
        "weights": {str(member["name"]): float(member["weight"]) for member in members},
        "members": members,
        "used_runs": [row for member in members for row in _flatten_used_runs(member)],
        "member_predictions": {
            str(member["name"]): _date_summary(load_prediction(Path(member["pred"]))) for member in members
        },
        "combined": _date_summary(blended),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"saved: {out_pred} (rows={len(blended)})")
    print(f"saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
