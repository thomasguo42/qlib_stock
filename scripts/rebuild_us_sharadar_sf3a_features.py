#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SF3A_FIELDS = "totalvalue,percentoftotal,shrunits,shrvalue"
SF3A_WINDOWS = "20,63,252"
SF3A_PREFIX = "inst13f"
PROVENANCE_FILE = "sharadar_sf3a_features.json"


def _calendar(provider_uri: Path) -> List[pd.Timestamp]:
    cal_path = provider_uri / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar not found: {cal_path}")
    return [pd.Timestamp(x.strip()) for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _code_to_fname(symbol: str) -> str:
    try:
        from qlib.utils import code_to_fname

        return code_to_fname(symbol).lower()
    except Exception:
        return str(symbol).lower()


def _field_columns(columns: Iterable[str]) -> List[str]:
    return [c for c in columns if c != "date" and c != "ticker"]


def _overwrite_feature_bins(prepared_dir: Path, provider_uri: Path, *, prefix: str, max_workers: int) -> int:
    # max_workers is accepted for CLI symmetry; this writer is intentionally serial to avoid partial rebuild races.
    _ = max_workers
    cal = _calendar(provider_uri)
    cal_index = pd.DatetimeIndex(cal)
    features_root = provider_uri / "features"
    written = 0

    for fp in sorted(prepared_dir.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            continue

        start = df["date"].min()
        end = df["date"].max()
        mask = (cal_index >= start) & (cal_index <= end)
        out_index = cal_index[mask]
        if out_index.empty:
            continue
        start_idx = int(cal_index.get_loc(out_index[0]))
        df = df.set_index("date").reindex(out_index)

        symbol_dir = features_root / _code_to_fname(fp.stem)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for col in _field_columns(df.columns):
            if not col.startswith(f"{prefix}_"):
                continue
            values = pd.to_numeric(df[col], errors="coerce").astype("float32").to_numpy()
            out = np.hstack([start_idx, values]).astype("<f")
            out.tofile(symbol_dir / f"{col.lower()}.day.bin")
            written += 1
    return written


def _write_provenance(provider_uri: Path, payload: Dict) -> Path:
    meta_dir = provider_uri / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / PROVENANCE_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild Sharadar SF3A/13F daily features with explicit availability lag.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--raw_sf3a_dir", default="~/.qlib/sharadar/raw/sf3a")
    p.add_argument("--out_dir", default="", help="Prepared feature output directory")
    p.add_argument("--availability_lag_days", type=int, default=45)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--dump_to_qlib", action="store_true", help="Overwrite existing inst13f_* feature bins")
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    provider = Path(args.provider_uri).expanduser().resolve()
    raw_dir = Path(args.raw_sf3a_dir).expanduser().resolve()
    if not raw_dir.exists():
        print(f"raw SF3A dir not found: {raw_dir}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path("~/.qlib/sharadar/prepared").expanduser().resolve() / f"sf3a_features_rebuild_{stamp}"
    )
    repo_root = Path(__file__).resolve().parents[1]
    prepare_script = repo_root / "scripts" / "data_collector" / "sharadar" / "prepare_event_features.py"

    cmd = [
        sys.executable,
        str(prepare_script),
        "--input",
        str(raw_dir),
        "--out_dir",
        str(out_dir),
        "--ticker_col",
        "ticker",
        "--date_col",
        "calendardate",
        "--value_cols",
        SF3A_FIELDS,
        "--windows",
        SF3A_WINDOWS,
        "--aggregation_mode",
        "snapshot",
        "--prefix",
        SF3A_PREFIX,
        "--availability_lag_days",
        str(int(args.availability_lag_days)),
    ]
    if args.start:
        cmd += ["--start", args.start, "--resample_start", args.start]
    if args.end:
        cmd += ["--end", args.end, "--resample_end", args.end]
    _run(cmd, dry_run=args.dry_run)

    written_bins = 0
    if args.dump_to_qlib:
        if args.dry_run:
            print(f"dry_run: would overwrite {SF3A_PREFIX}_*.day.bin files from {out_dir}")
        else:
            written_bins = _overwrite_feature_bins(
                out_dir,
                provider,
                prefix=SF3A_PREFIX,
                max_workers=int(args.max_workers),
            )
            print(f"feature_bins_written={written_bins}")
            if written_bins <= 0:
                print(
                    "ERROR: no SF3A feature bins were written; refusing to stamp full rebuild provenance",
                    file=sys.stderr,
                )
                return 3

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "table": "SF3A",
        "mode": "full_rebuild",
        "full_rebuild": bool(args.dump_to_qlib and not args.dry_run),
        "availability_lag_days": int(args.availability_lag_days),
        "raw_sf3a_dir": str(raw_dir),
        "prepared_dir": str(out_dir),
        "provider_uri": str(provider),
        "dump_to_qlib": bool(args.dump_to_qlib),
        "feature_bins_written": int(written_bins),
    }
    if not args.dry_run:
        prov = _write_provenance(provider, payload)
        print(f"provenance: {prov}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
