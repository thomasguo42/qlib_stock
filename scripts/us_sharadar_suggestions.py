#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.workflow import R


def _pick_scores(pred: pd.DataFrame) -> pd.Series:
    if "score" in pred.columns:
        return pred["score"]
    # fallback to first column
    return pred.iloc[:, 0]


def _latest_experiment_name(mlruns_uri: str) -> str:
    mlruns_path = Path(mlruns_uri).expanduser().resolve()
    if mlruns_path.is_file():
        # only file store is supported here; if a DB URI is used, skip
        return ""
    candidates = []
    for p in mlruns_path.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        meta = p / "meta.yaml"
        if not meta.exists():
            continue
        try:
            text = meta.read_text(encoding="utf-8")
        except Exception:
            continue
        name = ""
        creation = 0
        for line in text.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip("'\"")
            if line.startswith("creation_time:"):
                try:
                    creation = int(line.split(":", 1)[1].strip())
                except Exception:
                    creation = 0
        if name:
            candidates.append((creation, name))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def main():
    parser = argparse.ArgumentParser(description="Print top/bottom picks from latest Qlib prediction record.")
    parser.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    parser.add_argument("--experiment_name", default="us_sharadar_weekly_pit_best")
    parser.add_argument("--mlruns_uri", default="/workspace/qlib/mlruns")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--as_of", default=None, help="YYYY-MM-DD; default = latest prediction date")
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="If set, pick the latest date matching --weekday from predictions.",
    )
    parser.add_argument(
        "--weekday",
        type=int,
        default=0,
        help="0=Mon..4=Fri. Only used when --weekly is set.",
    )
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    qlib.init(
        provider_uri=args.provider_uri,
        region=REG_US,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": args.mlruns_uri,
                "default_exp_name": args.experiment_name,
            },
        },
    )

    exp_name = args.experiment_name or _latest_experiment_name(args.mlruns_uri)
    if not exp_name:
        raise RuntimeError("No experiment found in mlruns; run qrun first or pass --experiment_name.")
    rec = R.get_recorder(experiment_name=exp_name)
    pred = rec.load_object("pred.pkl")
    if pred is None or pred.empty:
        raise RuntimeError("No prediction data found in recorder.")

    scores = _pick_scores(pred)
    dt = scores.index.get_level_values("datetime")
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    elif args.weekly:
        weekday_mask = dt.weekday == args.weekday
        if not weekday_mask.any():
            raise RuntimeError("No prediction dates match requested weekday.")
        as_of = dt[weekday_mask].max()
    else:
        as_of = dt.max()
    scores = scores.xs(as_of, level="datetime", drop_level=False)

    scores = scores.sort_values(ascending=False)
    buy = scores.head(args.topk)
    sell = scores.tail(args.topk)

    print(f"As of {as_of.date()} | Top {args.topk} buy candidates")
    print(buy)
    print(f"\nAs of {as_of.date()} | Bottom {args.topk} sell candidates")
    print(sell)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_df = pd.concat([buy.rename("buy_score"), sell.rename("sell_score")], axis=1)
        out_df.to_csv(out_path)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
