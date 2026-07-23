#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sharadar_price_utils import prepare_sep_qlib_frame


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
            except Exception:
                pass
        self.flush()
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _read_last_line(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        if pos == 0:
            return ""
        block = b""
        while pos > 0 and b"\n" not in block:
            read_size = min(4096, pos)
            pos -= read_size
            f.seek(pos)
            block = f.read(read_size) + block
        lines = block.splitlines()
        if not lines:
            return ""
        return lines[-1].decode("utf-8", errors="ignore").strip()


def _load_sharadar_collector() -> type:
    here = Path(__file__).resolve()
    collector_path = here.parent / "data_collector" / "sharadar" / "collector.py"
    if not collector_path.exists():
        raise FileNotFoundError(f"Sharadar collector not found: {collector_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("sharadar_collector", collector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {collector_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "SharadarCollector")


def _write_tickers_csv_from_instruments(instruments_tsv: Path, out_csv: Path, max_tickers: Optional[int]) -> int:
    df = pd.read_csv(instruments_tsv, sep="\t", header=None, names=["ticker", "start", "end"])
    tickers = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if max_tickers is not None:
        tickers = tickers[: int(max_tickers)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(out_csv, index=False)
    return len(tickers)


def _write_tickers_csv_from_lines(lines: List[str], out_csv: Path) -> int:
    tickers = sorted(_benchmark_tickers_from_lines(lines))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(out_csv, index=False)
    return len(tickers)


def _benchmark_tickers_from_lines(lines: List[str]) -> List[str]:
    tickers: List[str] = []
    seen = set()
    for line in lines:
        ticker = str(line).strip().upper()
        if not ticker or ticker.startswith("#") or ticker in {"TICKER", "SYMBOL"} or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
    return tickers


def _read_last_date_from_csv(path: Path, date_col: str = "date") -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    try:
        s = pd.read_csv(path, usecols=[date_col], low_memory=False)[date_col]
        dt = pd.to_datetime(s, errors="coerce")
        mx = dt.max()
        if pd.isna(mx):
            return ""
        return pd.Timestamp(mx).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _detect_new_end_from_sep(raw_sep_dir: Path, tickers: List[str]) -> str:
    preferred = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM"]
    candidates = preferred + tickers[:50]
    best = ""
    for t in candidates:
        fp = raw_sep_dir / f"{t}.csv"
        d = _read_last_date_from_csv(fp, "date")
        if d and (not best or pd.Timestamp(d) > pd.Timestamp(best)):
            best = d
    return best


def _resolve_benchmark_tickers_file(path_arg: str) -> Path:
    if str(path_arg).strip():
        return Path(path_arg).expanduser().resolve()
    repo_default = Path(__file__).resolve().parents[1] / "_sfp_benchmark_tickers.txt"
    legacy_default = Path("/workspace/qlib/_sfp_benchmark_tickers.txt")
    if legacy_default.exists():
        return legacy_default
    return repo_default


def _active_market_tickers(inst_path: Path, active_end: str) -> List[str]:
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if df.empty:
        return []
    mask = df["end"].astype(str) == str(active_end)
    return df.loc[mask, "ticker"].astype(str).str.upper().str.strip().drop_duplicates().tolist()


def _market_instruments_max_end(inst_path: Path) -> str:
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if df.empty:
        return ""
    end = pd.to_datetime(df["end"], errors="coerce").dropna()
    if end.empty:
        return ""
    return pd.Timestamp(end.max()).strftime("%Y-%m-%d")


def _earliest_date(*dates: str) -> str:
    parsed = [pd.Timestamp(d) for d in dates if str(d).strip()]
    if not parsed:
        return ""
    return min(parsed).strftime("%Y-%m-%d")


def _sep_coverage_gaps(
    raw_sep_dir: Path,
    tickers: List[str],
    *,
    required_end: str,
    max_lag_days: int,
) -> Dict[str, str]:
    gaps: Dict[str, str] = {}
    required = pd.Timestamp(required_end)
    for t in tickers:
        fp = raw_sep_dir / f"{t}.csv"
        mx = _read_last_date_from_csv(fp, "date")
        if not mx:
            gaps[t] = "<missing>"
            continue
        lag_days = int((required - pd.Timestamp(mx)).days)
        if lag_days > int(max_lag_days):
            gaps[t] = mx
    return gaps


def _write_update_report(out_root: Path, stamp: str, reports: Dict[str, Dict]) -> Path:
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"update_report_{stamp}.json"
    out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _failure_count(report) -> int:
    if not isinstance(report, dict):
        return 0
    failures = report.get("failures", [])
    return len(failures) if isinstance(failures, list) else 0


def _reuse_raw_report(raw_dir: Path, tickers: List[str], date_col: str) -> Dict:
    files = 0
    missing = 0
    latest = ""
    for ticker in tickers:
        last = _read_last_date_from_csv(raw_dir / f"{ticker}.csv", date_col)
        if not last:
            missing += 1
            continue
        files += 1
        if not latest or pd.Timestamp(last) > pd.Timestamp(latest):
            latest = last
    return {
        "status": "reused_raw",
        "raw_dir": str(raw_dir),
        "date_col": date_col,
        "tickers": len(tickers),
        "files_with_dates": files,
        "missing_or_empty": missing,
        "latest_date": latest,
        "failures": [],
    }


def _feature_missing_ratio(feature_dir: Path, tickers: List[str]) -> float:
    if not tickers:
        return 0.0
    missing = 0
    for t in tickers:
        fp = feature_dir / f"{t}.csv"
        if not fp.exists() or fp.stat().st_size == 0:
            missing += 1
    return float(missing) / float(len(tickers))


def _prepare_price_delta(raw_sep_file: Path, old_last_trading_date: str) -> pd.DataFrame:
    df = pd.read_csv(raw_sep_file, low_memory=False)
    if df.empty:
        return df
    df = prepare_sep_qlib_frame(df)
    if df.empty:
        return df
    df = df[df["date"] > pd.Timestamp(old_last_trading_date)]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _prepare_sfp_qlib_frame(raw_sfp_file: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_sfp_file, low_memory=False)
    if df.empty:
        return df
    symbol = raw_sfp_file.stem.strip().upper()
    if "ticker" in df.columns:
        tickers = df["ticker"].astype(str).str.upper().str.strip().dropna()
        if not tickers.empty:
            symbol = str(tickers.iloc[0])
    df = prepare_sep_qlib_frame(df)
    if df.empty:
        return df
    df.insert(0, "symbol", symbol)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _write_sfp_qlib_files(raw_sfp_dir: Path, tickers: List[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for ticker in _benchmark_tickers_from_lines(tickers):
        raw_sfp_file = raw_sfp_dir / f"{ticker}.csv"
        if not raw_sfp_file.exists():
            continue
        df = _prepare_sfp_qlib_frame(raw_sfp_file)
        if df.empty:
            continue
        df.to_csv(out_dir / f"{ticker}.csv", index=False)
        written += 1
    return written


def _load_feature_delta(feature_file: Path, dates: List[str]) -> Optional[pd.DataFrame]:
    if not feature_file.exists() or feature_file.stat().st_size == 0:
        return None
    df = pd.read_csv(feature_file, low_memory=False)
    if df.empty or "date" not in df.columns:
        return None
    df["date"] = df["date"].astype(str)
    df = df[df["date"].isin(dates)]
    if df.empty:
        return None
    drop_cols = [c for c in ["ticker"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _build_bench_etf_basket(sfp_dir: Path, tickers: List[str]) -> pd.Series:
    prices = {}
    for t in tickers:
        fp = sfp_dir / f"{t}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        col = "closeadj" if "closeadj" in df.columns else "close"
        s = df.loc[:, ["date", col]].copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["date"]).sort_values("date").set_index("date")[col].astype("float64")
        prices[t] = s
    if not prices:
        raise RuntimeError("No SFP prices loaded; cannot build benchmark")
    px = pd.DataFrame(prices).sort_index()
    rets = px.pct_change()
    basket = rets.mean(axis=1, skipna=True)
    basket.name = "bench_etf_basket"
    return basket.dropna()


def _extend_market_instruments(inst_path: Path, old_end: str, new_end: str) -> int:
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if df.empty:
        return 0
    mask = df["end"].astype(str) == str(old_end)
    n = int(mask.sum())
    if n:
        df.loc[mask, "end"] = str(new_end)
        tmp = inst_path.with_name(f".{inst_path.name}.tmp")
        df.to_csv(tmp, sep="\t", header=False, index=False)
        tmp.replace(inst_path)
    return n


def _clamp_market_instruments_to_sep(
    inst_path: Path,
    *,
    sep_dir: Path,
    calendar_end: str,
) -> Dict[str, str]:
    """
    Ensure any ticker whose market end == calendar_end also has SEP prices through calendar_end.
    If SEP max date is earlier, clamp the market end down to that max date.
    """
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    if df.empty:
        return {}
    cal_end = str(calendar_end)
    mask = df["end"].astype(str) == cal_end
    tickers = df.loc[mask, "ticker"].astype(str).str.upper().tolist()
    if not tickers:
        return {}

    fixes: Dict[str, str] = {}
    for t in tickers:
        fp = sep_dir / f"{t}.csv"
        mx = _read_last_date_from_csv(fp, "date")
        if mx and mx < cal_end:
            fixes[t] = mx

    if fixes:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        for t, mx in fixes.items():
            df.loc[(df["ticker"] == t) & (df["end"].astype(str) == cal_end), "end"] = mx
        tmp = inst_path.with_name(f".{inst_path.name}.tmp")
        df.to_csv(tmp, sep="\t", header=False, index=False)
        tmp.replace(inst_path)
    return fixes


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _pump(src, dst):
        try:
            for line in src:
                dst.write(line)
        finally:
            try:
                src.close()
            except Exception:
                pass

    assert proc.stdout is not None
    assert proc.stderr is not None
    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _main_impl(args: argparse.Namespace) -> int:
    api_key = args.api_key.strip() or os.getenv("NDL_API_KEY", "").strip()
    if not api_key and not args.reuse_raw:
        print("ERROR: provide --api_key or set NDL_API_KEY", file=sys.stderr)
        return 2

    provider = Path(args.provider_uri).expanduser().resolve()
    cal_path = provider / "calendars" / "day.txt"
    if not cal_path.exists():
        print(f"ERROR: calendar not found: {cal_path}", file=sys.stderr)
        return 2
    old_last_trading_date = _read_last_line(cal_path)
    if not old_last_trading_date:
        print(f"ERROR: calendar empty: {cal_path}", file=sys.stderr)
        return 2

    inst_path = provider / "instruments" / f"{args.market}.txt"
    if not inst_path.exists():
        print(f"ERROR: market instruments not found: {inst_path}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir).expanduser().resolve()
    universe_dir = out_root / "universe"
    tickers_csv = universe_dir / f"{args.market}_tickers.csv"
    market_last_trading_date = _market_instruments_max_end(inst_path) or old_last_trading_date
    price_delta_start = _earliest_date(old_last_trading_date, market_last_trading_date)
    n_tickers = _write_tickers_csv_from_instruments(inst_path, tickers_csv, args.max_tickers)
    print(
        f"market={args.market} tickers={n_tickers} "
        f"old_last_trading_date={old_last_trading_date} "
        f"market_last_trading_date={market_last_trading_date} "
        f"price_delta_start={price_delta_start}"
    )
    tickers = pd.read_csv(tickers_csv)["ticker"].astype(str).str.upper().tolist()
    active_tickers = _active_market_tickers(inst_path, market_last_trading_date)
    if args.max_tickers is not None:
        active_tickers = [t for t in tickers if t in set(active_tickers)]

    etf_txt = _resolve_benchmark_tickers_file(args.benchmark_tickers_file)
    etf_lines = etf_txt.read_text(encoding="utf-8").splitlines() if etf_txt.exists() else []
    etf_tickers = _benchmark_tickers_from_lines(etf_lines)
    if not etf_tickers:
        print(f"warning: benchmark tickers file missing or empty: {etf_txt}", file=sys.stderr)
        if args.fail_on_update_gaps:
            print("ERROR: benchmark ticker list is required with --fail_on_update_gaps", file=sys.stderr)
            return 3
    etf_csv = universe_dir / "_sfp_benchmark_tickers.csv"
    _write_tickers_csv_from_lines(etf_lines, etf_csv)

    raw_sep_dir = out_root / "raw" / "sep"
    raw_sf2_dir = out_root / "raw" / "sf2"
    raw_sf3a_dir = out_root / "raw" / "sf3a"
    raw_sfp_dir = out_root / "raw" / "sfp"

    update_reports: Dict[str, Dict] = {}

    if args.reuse_raw:
        print("reuse_raw=1: skipping Sharadar API downloads and using existing raw files")
        update_reports["SEP"] = _reuse_raw_report(raw_sep_dir, tickers, "date")
        update_reports["SFP"] = _reuse_raw_report(raw_sfp_dir, etf_tickers, "date")
        update_reports["SF2"] = _reuse_raw_report(raw_sf2_dir, tickers, "filingdate")
        update_reports["SF3A"] = _reuse_raw_report(raw_sf3a_dir, tickers, "calendardate")
    else:
        SharadarCollector = _load_sharadar_collector()
        collector = SharadarCollector(api_key=api_key, out_dir=str(out_root))

        update_reports["SEP"] = collector.update_sep(
            tickers_file=str(tickers_csv),
            sep_dir=str(raw_sep_dir),
            days_back=int(args.days_back_sep),
            return_report=True,
        )
        update_reports["SFP"] = collector.update_sfp(
            tickers_file=str(etf_csv),
            sfp_dir=str(raw_sfp_dir),
            days_back=int(args.days_back_sfp),
            return_report=True,
        )
        update_reports["SF2"] = collector.update_table_for_tickers(
            "SF2",
            tickers_file=str(tickers_csv),
            out_dir=str(raw_sf2_dir),
            date_field="filingdate",
            days_back=int(args.days_back_sf2),
            fallback_start="2016-01-01",
            max_tickers=args.max_tickers,
            return_report=True,
        )
        update_reports["SF3A"] = collector.update_table_for_tickers(
            "SF3A",
            tickers_file=str(tickers_csv),
            out_dir=str(raw_sf3a_dir),
            date_field="calendardate",
            days_back=int(args.days_back_sf3a),
            fallback_start="2016-01-01",
            max_tickers=args.max_tickers,
            return_report=True,
        )

    if args.fail_on_update_gaps:
        failed_tables = {name: _failure_count(report) for name, report in update_reports.items()}
        failed_tables = {name: count for name, count in failed_tables.items() if count > int(args.max_update_failures)}
        if failed_tables:
            print(
                "ERROR: update failures exceed threshold: "
                + ", ".join([f"{name}={count}" for name, count in sorted(failed_tables.items())])
                + f" allowed_per_table={int(args.max_update_failures)}",
                file=sys.stderr,
            )
            return 3

    new_end = _detect_new_end_from_sep(raw_sep_dir, tickers)
    if not new_end:
        print(f"ERROR: failed to detect new end date from SEP under {raw_sep_dir}", file=sys.stderr)
        return 2
    print(f"detected_new_end={new_end}")
    if args.fail_on_update_gaps:
        coverage_tickers = active_tickers or tickers
        sep_gaps = _sep_coverage_gaps(
            raw_sep_dir,
            coverage_tickers,
            required_end=new_end,
            max_lag_days=int(args.max_sep_lag_days),
        )
        if len(sep_gaps) > int(args.max_sep_coverage_gaps):
            sample = ", ".join([f"{k}->{v}" for k, v in list(sorted(sep_gaps.items()))[:10]])
            print(
                "ERROR: SEP coverage gaps exceed threshold: "
                f"gaps={len(sep_gaps)} allowed={int(args.max_sep_coverage_gaps)} "
                f"required_end={new_end} sample={sample}",
                file=sys.stderr,
            )
            return 3

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = _write_update_report(out_root, stamp, update_reports)
    print(f"update_report={report_path}")
    repo_root = Path(__file__).resolve().parents[1]
    prep_sf2_dir = out_root / "prepared" / f"sf2_features_warmup_{stamp}"
    prep_sf3a_dir = out_root / "prepared" / f"sf3a_features_warmup_{stamp}"
    delta_dir = out_root / "prepared" / f"qlib_delta_{stamp}"
    delta_dir.mkdir(parents=True, exist_ok=True)

    warmup_anchor = price_delta_start or old_last_trading_date
    warmup_sf2_start = (pd.Timestamp(warmup_anchor) - pd.Timedelta(days=int(args.warmup_sf2_days))).strftime(
        "%Y-%m-%d"
    )
    warmup_sf3a_start = (pd.Timestamp(warmup_anchor) - pd.Timedelta(days=int(args.warmup_sf3a_days))).strftime(
        "%Y-%m-%d"
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "data_collector" / "sharadar" / "prepare_event_features.py"),
            "--input",
            str(raw_sf2_dir),
            "--out_dir",
            str(prep_sf2_dir),
            "--ticker_col",
            "ticker",
            "--date_col",
            "filingdate",
            "--value_cols",
            "transactionshares,transactionvalue,sharesownedbeforetransaction,sharesownedfollowingtransaction",
            "--windows",
            "5,20,63",
            "--aggregation_mode",
            "event",
            "--prefix",
            "insider",
            "--start",
            warmup_sf2_start,
            "--resample_start",
            warmup_sf2_start,
            "--resample_end",
            new_end,
        ],
        dry_run=args.dry_run,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "data_collector" / "sharadar" / "prepare_event_features.py"),
            "--input",
            str(raw_sf3a_dir),
            "--out_dir",
            str(prep_sf3a_dir),
            "--ticker_col",
            "ticker",
            "--date_col",
            "calendardate",
            "--value_cols",
            "totalvalue,percentoftotal,shrunits,shrvalue",
            "--windows",
            "20,63,252",
            "--aggregation_mode",
            "snapshot",
            "--prefix",
            "inst13f",
            "--availability_lag_days",
            str(int(args.sf3a_availability_lag_days)),
            "--start",
            warmup_sf3a_start,
            "--resample_start",
            warmup_sf3a_start,
            "--resample_end",
            new_end,
        ],
        dry_run=args.dry_run,
    )

    if args.fail_on_update_gaps and not args.dry_run:
        sf2_missing = _feature_missing_ratio(prep_sf2_dir, tickers)
        sf3a_missing = _feature_missing_ratio(prep_sf3a_dir, tickers)
        if sf2_missing > float(args.max_missing_sf2_feature_ratio):
            print(
                "ERROR: missing SF2 feature files exceed threshold: "
                f"{sf2_missing:.2%} > {float(args.max_missing_sf2_feature_ratio):.2%}",
                file=sys.stderr,
            )
            return 3
        if sf3a_missing > float(args.max_missing_sf3a_feature_ratio):
            print(
                "ERROR: missing SF3A feature files exceed threshold: "
                f"{sf3a_missing:.2%} > {float(args.max_missing_sf3a_feature_ratio):.2%}",
                file=sys.stderr,
            )
            return 3

        sfp_gaps = _sep_coverage_gaps(
            raw_sfp_dir,
            etf_tickers,
            required_end=new_end,
            max_lag_days=int(args.max_sfp_lag_days),
        )
        if len(sfp_gaps) > int(args.max_sfp_coverage_gaps):
            sample = ", ".join([f"{k}->{v}" for k, v in list(sorted(sfp_gaps.items()))[:10]])
            print(
                "ERROR: SFP benchmark coverage gaps exceed threshold: "
                f"gaps={len(sfp_gaps)} allowed={int(args.max_sfp_coverage_gaps)} sample={sample}",
                file=sys.stderr,
            )
            return 3

    written = 0
    for t in tickers:
        raw_sep_file = raw_sep_dir / f"{t}.csv"
        if not raw_sep_file.exists():
            continue
        df_price = _prepare_price_delta(raw_sep_file, price_delta_start or old_last_trading_date)
        if df_price.empty:
            continue
        dates = df_price["date"].astype(str).tolist()

        df_ins = _load_feature_delta(prep_sf2_dir / f"{t}.csv", dates)
        df_i13 = _load_feature_delta(prep_sf3a_dir / f"{t}.csv", dates)

        merged = df_price
        if df_ins is not None:
            merged = merged.merge(df_ins, on="date", how="left")
        if df_i13 is not None:
            merged = merged.merge(df_i13, on="date", how="left")

        for c in merged.columns:
            if c == "date":
                continue
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype("float32")

        out_file = delta_dir / f"{t}.csv"
        merged.to_csv(out_file, index=False)
        written += 1

    if written == 0:
        print("No delta files produced (no new trading dates); skipping dump_update.")
        if args.fail_on_update_gaps:
            print("ERROR: no delta files produced while --fail_on_update_gaps is enabled", file=sys.stderr)
            return 3
    else:
        print(f"delta_files_written={written} delta_dir={delta_dir}")
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "dump_bin.py"),
                "dump_update",
                "--data_path",
                str(delta_dir),
                "--qlib_dir",
                str(provider),
                "--freq",
                "day",
                "--date_field_name",
                "date",
                "--file_suffix",
                ".csv",
                "--exclude_fields",
                "symbol",
                "--max_workers",
                "16",
            ],
            dry_run=args.dry_run,
        )

    if etf_tickers:
        sfp_delta_dir = out_root / "prepared" / f"qlib_sfp_benchmark_{stamp}"
        sfp_written = _write_sfp_qlib_files(raw_sfp_dir, etf_tickers, sfp_delta_dir)
        print(f"sfp_benchmark_files_written={sfp_written} sfp_delta_dir={sfp_delta_dir}")
        if sfp_written == 0:
            if args.fail_on_update_gaps:
                print("ERROR: no SFP benchmark files prepared while --fail_on_update_gaps is enabled", file=sys.stderr)
                return 3
        else:
            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "dump_bin.py"),
                    "dump_update",
                    "--data_path",
                    str(sfp_delta_dir),
                    "--qlib_dir",
                    str(provider),
                    "--freq",
                    "day",
                    "--date_field_name",
                    "date",
                    "--symbol_field_name",
                    "symbol",
                    "--file_suffix",
                    ".csv",
                    "--exclude_fields",
                    "symbol",
                    "--max_workers",
                    "8",
                ],
                dry_run=args.dry_run,
            )

        bench = _build_bench_etf_basket(raw_sfp_dir, etf_tickers)
        out_bench = provider / "bench_etf_basket.pkl"
        print(f"bench_etf_basket: start={bench.index.min().date()} end={bench.index.max().date()} rows={len(bench)}")
        if not args.dry_run:
            bench.to_pickle(out_bench)
            print(f"saved: {out_bench}")

    new_last_trading_date = _read_last_line(cal_path)
    print(f"calendar_updated: {old_last_trading_date} -> {new_last_trading_date}")
    if not args.dry_run and new_last_trading_date:
        current_market_end = _market_instruments_max_end(inst_path)
        if new_last_trading_date != current_market_end:
            n_ext = _extend_market_instruments(inst_path, current_market_end, new_last_trading_date)
            print(f"market_instruments_extended: file={inst_path} rows_updated={n_ext}")
        fixes = _clamp_market_instruments_to_sep(inst_path, sep_dir=raw_sep_dir, calendar_end=new_last_trading_date)
        if fixes:
            items = ", ".join([f"{k}->{v}" for k, v in sorted(fixes.items())])
            print(f"market_instruments_clamped_to_sep: {items}")
        if args.fail_on_update_gaps and len(fixes) > int(args.max_clamped_symbols):
            print(
                "ERROR: clamped instruments exceed threshold: "
                f"clamped={len(fixes)} allowed={int(args.max_clamped_symbols)}",
                file=sys.stderr,
            )
            return 4
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Catch up US Sharadar Qlib dataset via Nasdaq Data Link API.")
    p.add_argument("--provider_uri", default=os.getenv("QLIB_PROVIDER_URI", "/root/.qlib/qlib_data/us_data"))
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--api_key", default="", help="Optional; defaults to env NDL_API_KEY")
    p.add_argument("--out_dir", default="~/.qlib/sharadar")
    p.add_argument(
        "--benchmark_tickers_file",
        default="",
        help="Benchmark ETF ticker list. Defaults to repo _sfp_benchmark_tickers.txt.",
    )
    p.add_argument("--max_tickers", type=int, default=None, help="For smoke runs only")
    p.add_argument("--days_back_sep", type=int, default=10)
    p.add_argument("--days_back_sfp", type=int, default=10)
    p.add_argument("--days_back_sf2", type=int, default=60)
    p.add_argument("--days_back_sf3a", type=int, default=400)
    p.add_argument(
        "--reuse_raw",
        action="store_true",
        help="Skip API downloads and continue from existing raw files under --out_dir.",
    )
    p.add_argument("--warmup_sf2_days", type=int, default=70)
    p.add_argument("--warmup_sf3a_days", type=int, default=420)
    p.add_argument(
        "--sf3a_availability_lag_days",
        type=int,
        default=45,
        help="Calendar-day lag applied to SF3A calendardate before daily snapshot features are built.",
    )
    p.add_argument(
        "--fail_on_update_gaps",
        action="store_true",
        help="Return non-zero if SEP coverage, delta output, or instrument clamping exceed thresholds.",
    )
    p.add_argument("--max_update_failures", type=int, default=0, help="Allowed per-table collector failures")
    p.add_argument("--max_sep_lag_days", type=int, default=0, help="Allowed SEP lag in calendar days for active symbols")
    p.add_argument("--max_sep_coverage_gaps", type=int, default=0, help="Allowed number of active symbols breaching SEP lag")
    p.add_argument("--max_sfp_lag_days", type=int, default=1, help="Allowed SFP benchmark lag in calendar days")
    p.add_argument("--max_sfp_coverage_gaps", type=int, default=0, help="Allowed benchmark ETFs breaching SFP lag")
    p.add_argument(
        "--max_missing_sf2_feature_ratio",
        type=float,
        default=0.25,
        help="Allowed missing SF2 prepared feature-file ratio when strict update gaps are enabled",
    )
    p.add_argument(
        "--max_missing_sf3a_feature_ratio",
        type=float,
        default=0.10,
        help="Allowed missing SF3A prepared feature-file ratio when strict update gaps are enabled",
    )
    p.add_argument("--max_clamped_symbols", type=int, default=0, help="Allowed number of symbols clamped after update")
    p.add_argument("--log_dir", default="~/.qlib/sharadar/logs")
    p.add_argument("--log_file", default="", help="Optional explicit log file path")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file).expanduser().resolve() if str(args.log_file).strip() else (
        Path(args.log_dir).expanduser().resolve() / f"update_us_sharadar_data_{stamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_fp:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, log_fp)
        sys.stderr = _Tee(old_err, log_fp)
        try:
            print(f"log_file={log_path}", file=sys.stderr)
            print(f"cmd={' '.join([sys.executable, *sys.argv])}", file=sys.stderr)
            return _main_impl(args)
        except Exception:
            print("FATAL: update_us_sharadar_data failed with exception:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1
        finally:
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    raise SystemExit(main())
