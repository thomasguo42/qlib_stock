#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from typing import Any, Dict

from ruamel.yaml import YAML

import qlib
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.model.trainer import TrainerR
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.group import RollingGroup


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def _infer_exp_name(qlib_init: Dict[str, Any]) -> str:
    exp_manager = qlib_init.get("exp_manager", {})
    if isinstance(exp_manager, dict):
        kwargs = exp_manager.get("kwargs", {})
        name = kwargs.get("default_exp_name")
        if name:
            return name
    return "rolling_exp"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling training for Qlib workflows.")
    parser.add_argument("--config", required=True, help="Path to workflow YAML config")
    parser.add_argument("--rolling_step", type=int, default=20, help="Rolling step in trading days")
    parser.add_argument(
        "--rolling_type",
        choices=["exp", "slide"],
        default="slide",
        help="exp=expanding train window, slide=fixed-size windows",
    )
    parser.add_argument(
        "--trunc_days",
        type=int,
        default=6,
        help="Truncate non-test segments to avoid leakage (in trading days)",
    )
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit rolling tasks for quick runs")
    parser.add_argument("--exp_name", default=None, help="Override experiment name")
    parser.add_argument("--collect", action="store_true", help="Collect rolling results summary")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    qlib_init = cfg.get("qlib_init", {})
    exp_name = args.exp_name or _infer_exp_name(qlib_init)
    if isinstance(qlib_init.get("exp_manager"), dict):
        qlib_init["exp_manager"].setdefault("kwargs", {})["default_exp_name"] = exp_name

    qlib.init(**qlib_init)

    task = cfg["task"]
    rtype = RollingGen.ROLL_EX if args.rolling_type == "exp" else RollingGen.ROLL_SD
    rolling_gen = RollingGen(step=args.rolling_step, rtype=rtype, trunc_days=args.trunc_days)
    tasks = task_generator(task, rolling_gen)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    trainer = TrainerR(experiment_name=exp_name)
    trainer.train(tasks)

    if args.collect:
        def rec_key(recorder):
            task_cfg = recorder.load_object("task")
            return task_cfg["dataset"]["kwargs"]["segments"]["test"]

        collector = RecorderCollector(
            experiment=R.get_exp(experiment_name=exp_name),
            process_list=RollingGroup(),
            rec_key_func=rec_key,
            artifacts_key=["pred"],
        )
        result = collector()
        print(f"Collected rolling artifacts: {list(result.keys())}")
        for art_key, art_val in result.items():
            print(f"{art_key}: {list(art_val.keys())}")


if __name__ == "__main__":
    main()
