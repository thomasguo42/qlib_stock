#!/usr/bin/env python3
"""
Run `scripts/validate_us_sharadar_pipeline.py` with a tee-style combined log.

Why:
- When piping `| tee ...`, Python may buffer stdout, and stderr (tqdm/progress bars, warnings)
  is not captured unless you do `2>&1`.
- This wrapper captures *both* stdout+stderr into a single log file while still echoing to console.

Example:
  python scripts/run_validate_logged.py \
    --config workspace/ablation_logs/.../workflow_config...gpu.yaml \
    --provider_uri /root/.qlib/qlib_data/us_data \
    --pred workspace/walkforward_runs/.../pred.pkl \
    --benchmark_pkl /root/.qlib/qlib_data/us_data/bench_etf_basket.pkl \
    --start 2022-01-01 --end 2026-02-23 \
    --by_year --check_gates --check_rolling --gate_profile release
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen


def _default_log_path(log_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"validate_{ts}.log"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run validator and tee stdout+stderr to a log file.")
    parser.add_argument("--log_dir", default="workspace/validation_logs", help="Directory for log files")
    parser.add_argument("--log_file", default=None, help="Optional explicit log filename (within log_dir)")

    # Everything after `--` is passed through to validate_us_sharadar_pipeline.py
    parser.add_argument(
        "validator_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to scripts/validate_us_sharadar_pipeline.py (prefix with -- ...)",
    )
    args = parser.parse_args(argv)

    if not args.validator_args:
        parser.error("missing validator args; pass them after `--` (see --help)")

    if args.validator_args[0] == "--":
        passthrough = args.validator_args[1:]
    else:
        passthrough = args.validator_args

    repo_root = Path(__file__).resolve().parents[1]
    log_dir = (repo_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_path = log_dir / args.log_file
    else:
        log_path = _default_log_path(log_dir)

    cmd = [sys.executable, "-u", str((repo_root / "scripts/validate_us_sharadar_pipeline.py").resolve()), *passthrough]

    # Ensure deterministic, line-oriented output when possible.
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[run_validate_logged] log: {log_path}")
    print(f"[run_validate_logged] cmd: {' '.join(cmd)}")

    with log_path.open("wb") as f:
        proc = Popen(cmd, cwd=str(repo_root), stdout=PIPE, stderr=STDOUT, env=env)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            f.write(chunk)
            f.flush()
        return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())

