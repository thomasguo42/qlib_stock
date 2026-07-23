#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse

import yaml


CONFIG_PREFIX = "workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_"
CONFIG_SUFFIX = "_topk40"
DEFAULT_VALIDATOR_BENCHMARK_PKL = "/root/.qlib/qlib_data/us_data/bench_qqq.pkl"
LABEL_PROCESSOR_TARGETS = {
    "BenchmarkExcessLabel": "benchmark_excess",
    "ResidualForwardReturnLabel": "beta_residual",
    "VolScaledExcessLabel": "vol_scaled_excess",
    "DownsideAdjustedExcessLabel": "downside_adjusted_excess",
    "PortfolioUtilityExcessLabel": "portfolio_utility_excess",
    "DualHorizonPortfolioUtilityLabel": "dual_horizon_portfolio_utility",
}


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _processor_class(proc: Dict) -> str:
    return str(proc.get("class", "")).split(".")[-1]


def _parse_label_horizon(label_expr: str) -> int:
    import re

    m = re.search(r"Ref\(\$close,\s*-(\d+)\)\s*/\s*Ref\(\$close,\s*-(\d+)\)", str(label_expr or ""))
    if not m:
        return 0
    left = int(m.group(1))
    right = int(m.group(2))
    return max(0, left - right)


def _infer_config_screen_keys(config: Path) -> List[tuple[str, int]]:
    cfg = _load_yaml(config)
    dh = cfg.get("data_handler_config") or (((cfg.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get(
        "handler", {}
    ).get("kwargs", {})
    label_expr = ""
    try:
        label_expr = str(dh.get("label", [])[0][0])
    except Exception:
        label_expr = ""
    horizon = _parse_label_horizon(label_expr)
    processors = dh.get("learn_processors", []) if isinstance(dh, dict) else []
    target_kind = "raw_return"
    has_label_group_neutralize = False
    for proc in processors or []:
        if not isinstance(proc, dict):
            continue
        cls = _processor_class(proc)
        if cls in LABEL_PROCESSOR_TARGETS:
            target_kind = LABEL_PROCESSOR_TARGETS[cls]
        if cls == "GroupNeutralize" and str((proc.get("kwargs", {}) or {}).get("fields_group", "")) == "label":
            has_label_group_neutralize = True
    keys = [(target_kind, horizon)] if horizon else []
    if has_label_group_neutralize and target_kind == "benchmark_excess":
        keys.insert(0, ("sector_neutral", horizon))
    return keys


def _load_target_screen_passes(path: Path) -> set[tuple[str, int]]:
    df_rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("screen_pass", "")).strip().lower() not in {"true", "1", "yes"}:
                continue
            try:
                df_rows.append((str(row.get("target_kind", "")).strip(), int(float(row.get("horizon_days", 0)))))
            except (TypeError, ValueError):
                continue
    return {row for row in df_rows if row[0] and row[1] > 0}


def _filter_configs_by_target_screen(configs: List[Path], screen_csv: str) -> tuple[List[Path], List[Dict[str, object]]]:
    if not str(screen_csv or "").strip():
        return configs, []
    screen_path = Path(screen_csv).expanduser().resolve()
    passes = _load_target_screen_passes(screen_path)
    rows = []
    kept = []
    for config in configs:
        keys = _infer_config_screen_keys(config)
        ok = bool(passes.intersection(keys))
        rows.append({"config": str(config), "screen_keys": keys, "screen_pass": ok})
        if ok:
            kept.append(config)
    return kept, rows


def _config_key(path: Path) -> str:
    stem = path.stem
    if stem.startswith(CONFIG_PREFIX):
        stem = stem[len(CONFIG_PREFIX) :]
    if stem.endswith(CONFIG_SUFFIX):
        stem = stem[: -len(CONFIG_SUFFIX)]
    return stem


def _collect_configs(configs: Sequence[str], globs: Sequence[str], *, max_configs: int = 0) -> List[Path]:
    paths: List[Path] = []
    for raw in configs or []:
        if not str(raw).strip():
            continue
        paths.append(Path(raw).expanduser().resolve())
    for pattern in globs or []:
        if not str(pattern).strip():
            continue
        paths.extend(Path(match) for match in glob.glob(str(Path(pattern).expanduser()), recursive=True))
    dedup: Dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved.is_file():
            dedup[str(resolved)] = resolved
    out = [dedup[key] for key in sorted(dedup)]
    if int(max_configs) > 0:
        out = out[: int(max_configs)]
    return out


def _parse_release_output(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- release_ready:"):
            parsed["release_ready"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("- release_missing_or_failed:"):
            parsed["release_missing_or_failed"] = stripped.split(":", 1)[1].strip()
            continue
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"pred", "runtime_config", "walkforward_manifest", "train_mode"}:
                parsed[key] = value
    raw_text = str(text or "")
    if "release validation PASS" in raw_text:
        parsed["release_validation"] = "PASS"
    elif "release validation diagnostics completed" in raw_text:
        parsed["release_validation"] = "DIAGNOSTICS"
    return parsed


def _add_optional_arg(cmd: List[str], flag: str, value) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([flag, text])


def _normalize_mlruns_uri(uri: str) -> str:
    text = str(uri or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    return str(Path(text).expanduser().resolve())


def _build_release_command(
    args: argparse.Namespace,
    config: Path,
    run_dir: Path,
    tag: str,
    *,
    trial_count: int = 0,
) -> Dict[str, object]:
    key = _config_key(config)
    pred_path = run_dir / "preds" / f"{key}_{tag}_pred.pkl"
    manifest_path = run_dir / "manifests" / f"{key}_{tag}_manifest.json"
    trial_registry = str(getattr(args, "trial_registry", "") or "").strip()
    if not trial_registry and not bool(getattr(args, "no_trial_registry", False)):
        trial_registry = str(run_dir / "trial_registry.jsonl")
    cmd = [
        str(args.python_bin),
        str(Path(args.release_script).expanduser().resolve()),
        "--config",
        str(config),
        "--provider_uri",
        str(Path(args.provider_uri).expanduser().resolve()),
        "--benchmark_pkl",
        str(Path(args.benchmark_pkl).expanduser().resolve()),
        "--train_mode",
        str(getattr(args, "train_mode", "walkforward")),
        "--test_block",
        str(args.test_block),
        "--valid_days",
        str(int(args.valid_days)),
        "--out_pred",
        str(pred_path),
        "--walkforward_manifest",
        str(manifest_path),
        "--walkforward_exp_name",
        f"{key}_grid_{tag}",
        "--model_quality_mode",
        str(args.model_quality_mode),
        "--training_min_best_iteration",
        str(int(args.training_min_best_iteration)),
        "--gate_profile",
        str(getattr(args, "gate_profile", "growth")),
        "--baseline_tickers",
        str(getattr(args, "baseline_tickers", "QQQ,SPY,IXIC")),
    ]
    if bool(getattr(args, "override_label_benchmark_pkl", False)):
        cmd.append("--override_label_benchmark_pkl")
    else:
        cmd.append("--preserve_label_benchmark_pkl")
    if str(getattr(args, "train_mode", "walkforward")) == "ensemble":
        for flag, attr in (
            ("--walkforward_ensemble_script", "walkforward_ensemble_script"),
            ("--ensemble_primary_config", "ensemble_primary_config"),
            ("--ensemble_defensive_config", "ensemble_defensive_config"),
            ("--ensemble_primary_name", "ensemble_primary_name"),
            ("--ensemble_defensive_name", "ensemble_defensive_name"),
            ("--ensemble_primary_weight", "ensemble_primary_weight"),
            ("--ensemble_defensive_weight", "ensemble_defensive_weight"),
            ("--ensemble_normalize", "ensemble_normalize"),
        ):
            value = getattr(args, attr, "")
            if str(value).strip():
                cmd.extend([flag, str(value)])
    _add_optional_arg(cmd, "--baseline_pkl_map", getattr(args, "baseline_pkl_map", ""))
    _add_optional_arg(cmd, "--test_start", args.test_start)
    _add_optional_arg(cmd, "--test_end", args.test_end)
    _add_optional_arg(cmd, "--start", args.start)
    _add_optional_arg(cmd, "--end", args.end)
    if args.embargo_days is not None:
        cmd.extend(["--embargo_days", str(int(args.embargo_days))])
    if args.train_lookback_days is not None:
        cmd.extend(["--train_lookback_days", str(int(args.train_lookback_days))])
    if int(getattr(args, "recency_half_life_days", 0) or 0) > 0:
        cmd.extend(
            [
                "--recency_half_life_days",
                str(int(args.recency_half_life_days)),
                "--recency_min_weight",
                str(float(getattr(args, "recency_min_weight", 0.25))),
            ]
        )
    if str(getattr(args, "regime_reweight_feature", "") or "").strip():
        cmd.extend(
            [
                "--regime_reweight_feature",
                str(args.regime_reweight_feature),
                f"--regime_reweight_thresholds={getattr(args, 'regime_reweight_thresholds', '-0.04,0.03')}",
                "--label_tail_reweight",
                str(float(getattr(args, "label_tail_reweight", 0.0))),
                "--label_tail_quantile",
                str(float(getattr(args, "label_tail_quantile", 0.20))),
                "--sample_max_weight",
                str(float(getattr(args, "sample_max_weight", 5.0))),
            ]
        )
        if bool(getattr(args, "disable_date_balance_reweight", False)):
            cmd.append("--disable_date_balance_reweight")
        if bool(getattr(args, "year_balance_reweight", False)):
            cmd.append("--year_balance_reweight")
    if args.mlruns_uri:
        cmd.extend(["--mlruns_uri", _normalize_mlruns_uri(str(args.mlruns_uri))])
    if args.skip_train:
        cmd.append("--skip_train")
    if args.allow_pred_beyond_test_segment:
        cmd.append("--allow_pred_beyond_test_segment")
    if bool(getattr(args, "collect_all_diagnostics", False)):
        cmd.append("--collect_all_diagnostics")
    if args.skip_strategy_weighted_quality:
        cmd.append("--skip_strategy_weighted_quality")
    if args.skip_rebalance_interval_quality:
        cmd.append("--skip_rebalance_interval_quality")
    if args.skip_active_risk:
        cmd.append("--skip_active_risk")
    if not args.run_preflight_audits:
        cmd.extend(["--skip_universe_integrity", "--skip_price_adjustment_integrity", "--skip_hedge_feasibility"])
    if trial_registry:
        cmd.extend(
            [
                "--trial_registry",
                str(Path(trial_registry).expanduser().resolve()),
                "--trial_id",
                f"{key}_{tag}",
                "--candidate_name",
                key,
                "--selection_reason",
                "research_grid",
            ]
        )
        if int(trial_count) > 0:
            cmd.extend(["--trial_count", str(int(trial_count))])
    return {
        "key": key,
        "config": str(config),
        "pred": str(pred_path),
        "manifest": str(manifest_path),
        "cmd": cmd,
    }


def _write_summary(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields = [
        "key",
        "config",
        "returncode",
        "status",
        "pred",
        "manifest",
        "runtime_config",
        "release_validation",
        "release_ready",
        "release_missing_or_failed",
        "log",
        "cmd",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _build_failure_map_command(args: argparse.Namespace, run_dir: Path, rows: Iterable[Dict[str, object]]) -> List[str]:
    if bool(getattr(args, "skip_failure_map", False)):
        return []
    logs: List[str] = []
    for row in rows:
        log_path = Path(str(row.get("log", "") or "")).expanduser()
        if log_path.exists() and log_path.is_file():
            logs.append(str(log_path.resolve()))
    if not logs:
        return []

    script = Path(
        str(getattr(args, "failure_map_script", "") or Path(__file__).resolve().with_name("summarize_us_sharadar_release_failures.py"))
    ).expanduser()
    out_csv = Path(str(getattr(args, "failure_map_csv", "") or run_dir / "failure_map.csv")).expanduser()
    out_md = Path(str(getattr(args, "failure_map_md", "") or run_dir / "failure_map.md")).expanduser()
    cmd = [str(args.python_bin), str(script.resolve())]
    for log in logs:
        cmd.extend(["--log", log])
    cmd.extend(["--out_csv", str(out_csv.resolve()), "--out_md", str(out_md.resolve())])
    return cmd


def _run_release_subprocess(cmd: List[str], *, timeout_seconds: int = 0) -> tuple[int, str, bool]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_seconds) if int(timeout_seconds) > 0 else None,
        )
        return int(proc.returncode), proc.stdout or "", False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        message = f"\nTIMEOUT after {int(timeout_seconds)} seconds\n"
        return 124, str(output) + message, True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a comparable walk-forward release grid for US Sharadar research configs.")
    p.add_argument("--config", action="append", default=[], help="Config YAML to run. Repeatable.")
    p.add_argument("--config_glob", action="append", default=[], help="Glob of config YAMLs to run. Repeatable.")
    p.add_argument("--max_configs", type=int, default=0, help="Optional cap after sorting/deduping configs")
    p.add_argument("--target_screen_csv", default="", help="Optional target_screen CSV from audit_us_sharadar_targets.py")
    p.add_argument(
        "--require_target_screen_pass",
        action="store_true",
        help="Skip configs whose inferred target/horizon did not pass --target_screen_csv",
    )
    p.add_argument("--out_dir", default="artifacts/research/grid", help="Grid output root")
    p.add_argument("--tag", default=None, help="Run tag; defaults to UTC timestamp")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--release_script", default="scripts/run_us_sharadar_release.py")
    p.add_argument("--train_mode", choices=["walkforward", "ensemble"], default="walkforward")
    p.add_argument("--walkforward_ensemble_script", default="scripts/walkforward_ensemble_us_sharadar.py")
    p.add_argument("--ensemble_primary_config", default="")
    p.add_argument("--ensemble_defensive_config", default="")
    p.add_argument("--ensemble_primary_name", default="primary")
    p.add_argument("--ensemble_defensive_name", default="defensive")
    p.add_argument("--ensemble_primary_weight", type=float, default=0.70)
    p.add_argument("--ensemble_defensive_weight", type=float, default=0.30)
    p.add_argument("--ensemble_normalize", choices=["none", "zscore", "rank_zscore"], default="rank_zscore")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--benchmark_pkl", default=DEFAULT_VALIDATOR_BENCHMARK_PKL)
    p.add_argument(
        "--override_label_benchmark_pkl",
        action="store_true",
        help="Also rewrite benchmark-relative label processors to --benchmark_pkl. Default preserves each config target.",
    )
    p.add_argument("--mlruns_uri", default="")
    p.add_argument("--test_start", default="")
    p.add_argument("--test_end", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--test_block", choices=["year", "quarter"], default="year")
    p.add_argument("--valid_days", type=int, default=63)
    p.add_argument("--embargo_days", type=int, default=None)
    p.add_argument("--train_lookback_days", type=int, default=None)
    p.add_argument("--recency_half_life_days", type=int, default=0)
    p.add_argument("--recency_min_weight", type=float, default=0.25)
    p.add_argument("--regime_reweight_feature", default="")
    p.add_argument("--regime_reweight_thresholds", default="-0.04,0.03")
    p.add_argument("--disable_date_balance_reweight", action="store_true")
    p.add_argument("--year_balance_reweight", action="store_true")
    p.add_argument("--label_tail_reweight", type=float, default=0.0)
    p.add_argument("--label_tail_quantile", type=float, default=0.20)
    p.add_argument("--sample_max_weight", type=float, default=5.0)
    p.add_argument("--model_quality_mode", choices=["strict", "warn", "skip"], default="strict")
    p.add_argument("--gate_profile", choices=["release", "research", "growth", "qqq_release"], default="growth")
    p.add_argument("--baseline_tickers", default="QQQ,SPY,IXIC")
    p.add_argument("--baseline_pkl_map", default="IXIC=/root/.qlib/qlib_data/us_data/bench_ixic.pkl")
    p.add_argument("--training_min_best_iteration", type=int, default=5)
    p.add_argument("--trial_registry", default="", help="Optional JSONL trial registry; defaults to <run_dir>/trial_registry.jsonl")
    p.add_argument("--no_trial_registry", action="store_true", help="Do not pass a trial registry to release validation")
    p.add_argument(
        "--collect_all_diagnostics",
        action="store_true",
        help="Ask release validation to report all strict diagnostics instead of exiting on the first failed gate.",
    )
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--allow_pred_beyond_test_segment", action="store_true")
    p.add_argument("--skip_strategy_weighted_quality", action="store_true")
    p.add_argument("--skip_rebalance_interval_quality", action="store_true")
    p.add_argument("--skip_active_risk", action="store_true")
    p.add_argument(
        "--run_preflight_audits",
        action="store_true",
        help="Run repeated universe/price/hedge preflight audits for every config. Default skips them for research grids.",
    )
    p.add_argument("--dry_run", action="store_true", help="Write commands and summary without executing")
    p.add_argument("--stop_on_fail", action="store_true", help="Stop after the first non-zero config return code")
    p.add_argument(
        "--timeout_seconds",
        type=int,
        default=0,
        help="Optional per-config timeout for train+validation subprocesses. 0 disables the timeout.",
    )
    p.add_argument(
        "--skip_failure_map",
        action="store_true",
        help="Do not write the post-run release failure-map CSV/Markdown.",
    )
    p.add_argument(
        "--failure_map_script",
        default="",
        help="Optional override for summarize_us_sharadar_release_failures.py.",
    )
    p.add_argument("--failure_map_csv", default="", help="Optional failure-map CSV path. Default: <run_dir>/failure_map.csv")
    p.add_argument("--failure_map_md", default="", help="Optional failure-map Markdown path. Default: <run_dir>/failure_map.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    configs = _collect_configs(args.config, args.config_glob, max_configs=int(args.max_configs))
    if not configs:
        print("no configs matched")
        return 2
    screen_rows: List[Dict[str, object]] = []
    if args.require_target_screen_pass:
        configs, screen_rows = _filter_configs_by_target_screen(configs, args.target_screen_csv)
        skipped = [row for row in screen_rows if not row["screen_pass"]]
        if skipped:
            print(f"target_screen_skipped={len(skipped)}")
        if not configs:
            print("no configs passed target screen")
            return 17
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir).expanduser().resolve() / tag
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "preds").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    overall_rc = 0
    for idx, config in enumerate(configs, start=1):
        plan = _build_release_command(args, config, run_dir, tag, trial_count=len(configs))
        key = str(plan["key"])
        log_path = run_dir / "logs" / f"{idx:03d}_{key}.log"
        cmd = list(plan["cmd"])
        print(f"[{idx}/{len(configs)}] {key}")
        print(" ".join(cmd))
        row: Dict[str, object] = {
            "key": key,
            "config": str(config),
            "pred": plan["pred"],
            "manifest": plan["manifest"],
            "log": str(log_path),
            "cmd": " ".join(cmd),
        }
        if args.dry_run:
            row.update({"returncode": 0, "status": "DRY_RUN"})
            log_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        else:
            returncode, stdout, timed_out = _run_release_subprocess(
                cmd,
                timeout_seconds=int(getattr(args, "timeout_seconds", 0) or 0),
            )
            log_path.write_text(stdout or "", encoding="utf-8")
            parsed = _parse_release_output(stdout or "")
            release_ready = parsed.get("release_ready", "")
            status = "TIMEOUT" if timed_out else ("PASS" if int(returncode) == 0 else "FAIL")
            if release_ready.upper() == "FAIL":
                status = "FAIL"
            row.update(
                {
                    "returncode": int(returncode),
                    "status": status,
                    "pred": parsed.get("pred", row["pred"]),
                    "manifest": parsed.get("walkforward_manifest", row["manifest"]),
                    "runtime_config": parsed.get("runtime_config", ""),
                    "release_validation": parsed.get("release_validation", ""),
                    "release_ready": release_ready,
                    "release_missing_or_failed": parsed.get("release_missing_or_failed", ""),
                }
            )
            if release_ready.upper() == "FAIL" and int(returncode) == 0 and overall_rc == 0:
                overall_rc = 17
            if int(returncode) != 0:
                overall_rc = int(returncode)
                if args.stop_on_fail:
                    rows.append(row)
                    break
        rows.append(row)
        _write_summary(run_dir / "summary.csv", rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tag": tag,
        "run_dir": str(run_dir),
        "configs": [str(path) for path in configs],
        "target_screen_csv": str(Path(args.target_screen_csv).expanduser().resolve()) if args.target_screen_csv else "",
        "target_screen_rows": [
            {
                "config": row["config"],
                "screen_keys": [[kind, horizon] for kind, horizon in row["screen_keys"]],
                "screen_pass": bool(row["screen_pass"]),
            }
            for row in screen_rows
        ],
        "rows": rows,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_summary(run_dir / "summary.csv", rows)
    if rows and not args.dry_run:
        failure_map_cmd = _build_failure_map_command(args, run_dir, rows)
        if failure_map_cmd:
            proc = subprocess.run(failure_map_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.stdout:
                print(proc.stdout.rstrip())
            if int(proc.returncode) != 0 and overall_rc == 0:
                overall_rc = int(proc.returncode)
    print(f"summary={run_dir / 'summary.csv'}")
    print(f"manifest={run_dir / 'manifest.json'}")
    return 0 if args.dry_run else int(overall_rc)


if __name__ == "__main__":
    raise SystemExit(main())
