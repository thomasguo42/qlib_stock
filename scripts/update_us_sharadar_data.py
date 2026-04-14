#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
    tickers = sorted({x.strip().upper() for x in lines if x.strip()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(out_csv, index=False)
    return len(tickers)


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


def _prepare_price_delta(raw_sep_file: Path, old_last_trading_date: str) -> pd.DataFrame:
    df = pd.read_csv(raw_sep_file, low_memory=False)
    if df.empty:
        return df
    if "date" not in df.columns:
        return df.iloc[0:0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if "closeadj" not in df.columns or "closeunadj" not in df.columns:
        return df.iloc[0:0]
    df["factor"] = df["closeadj"] / df["closeunadj"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.loc[df["closeunadj"] == 0, "factor"] = np.nan
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] * df["factor"]
    keep = ["date", "open", "high", "low", "close", "volume", "factor"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    df = df.loc[:, keep].dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df["date"] > pd.Timestamp(old_last_trading_date)]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


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
    if not api_key:
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
    n_tickers = _write_tickers_csv_from_instruments(inst_path, tickers_csv, args.max_tickers)
    print(f"market={args.market} tickers={n_tickers} old_last_trading_date={old_last_trading_date}")
    tickers = pd.read_csv(tickers_csv)["ticker"].astype(str).str.upper().tolist()

    etf_txt = Path("/workspace/qlib/_sfp_benchmark_tickers.txt")
    etf_lines = etf_txt.read_text(encoding="utf-8").splitlines() if etf_txt.exists() else []
    etf_csv = universe_dir / "_sfp_benchmark_tickers.csv"
    _write_tickers_csv_from_lines(etf_lines, etf_csv)

    SharadarCollector = _load_sharadar_collector()
    collector = SharadarCollector(api_key=api_key, out_dir=str(out_root))

    raw_sep_dir = out_root / "raw" / "sep"
    raw_sf2_dir = out_root / "raw" / "sf2"
    raw_sf3a_dir = out_root / "raw" / "sf3a"
    raw_sfp_dir = out_root / "raw" / "sfp"

    collector.update_sep(
        tickers_file=str(tickers_csv),
        sep_dir=str(raw_sep_dir),
        days_back=int(args.days_back_sep),
    )
    collector.update_sfp(
        tickers_file=str(etf_csv),
        sfp_dir=str(raw_sfp_dir),
        days_back=int(args.days_back_sfp),
    )
    collector.update_table_for_tickers(
        "SF2",
        tickers_file=str(tickers_csv),
        out_dir=str(raw_sf2_dir),
        date_field="filingdate",
        days_back=int(args.days_back_sf2),
        fallback_start="2016-01-01",
        max_tickers=args.max_tickers,
    )
    collector.update_table_for_tickers(
        "SF3A",
        tickers_file=str(tickers_csv),
        out_dir=str(raw_sf3a_dir),
        date_field="calendardate",
        days_back=int(args.days_back_sf3a),
        fallback_start="2016-01-01",
        max_tickers=args.max_tickers,
    )

    new_end = _detect_new_end_from_sep(raw_sep_dir, tickers)
    if not new_end:
        print(f"ERROR: failed to detect new end date from SEP under {raw_sep_dir}", file=sys.stderr)
        return 2
    print(f"detected_new_end={new_end}")

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    prep_sf2_dir = out_root / "prepared" / f"sf2_features_warmup_{stamp}"
    prep_sf3a_dir = out_root / "prepared" / f"sf3a_features_warmup_{stamp}"
    delta_dir = out_root / "prepared" / f"qlib_delta_{stamp}"
    delta_dir.mkdir(parents=True, exist_ok=True)

    warmup_sf2_start = (pd.Timestamp(old_last_trading_date) - pd.Timedelta(days=int(args.warmup_sf2_days))).strftime(
        "%Y-%m-%d"
    )
    warmup_sf3a_start = (pd.Timestamp(old_last_trading_date) - pd.Timedelta(days=int(args.warmup_sf3a_days))).strftime(
        "%Y-%m-%d"
    )

    _run(
        [
            sys.executable,
            "scripts/data_collector/sharadar/prepare_event_features.py",
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
            "scripts/data_collector/sharadar/prepare_event_features.py",
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
            "--start",
            warmup_sf3a_start,
            "--resample_start",
            warmup_sf3a_start,
            "--resample_end",
            new_end,
        ],
        dry_run=args.dry_run,
    )

    written = 0
    for t in tickers:
        raw_sep_file = raw_sep_dir / f"{t}.csv"
        if not raw_sep_file.exists():
            continue
        df_price = _prepare_price_delta(raw_sep_file, old_last_trading_date)
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
    else:
        print(f"delta_files_written={written} delta_dir={delta_dir}")
        _run(
            [
                sys.executable,
                "scripts/dump_bin.py",
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

    if etf_lines:
        bench = _build_bench_etf_basket(raw_sfp_dir, [x.strip().upper() for x in etf_lines if x.strip()])
        out_bench = provider / "bench_etf_basket.pkl"
        print(f"bench_etf_basket: start={bench.index.min().date()} end={bench.index.max().date()} rows={len(bench)}")
        if not args.dry_run:
            bench.to_pickle(out_bench)
            print(f"saved: {out_bench}")

    new_last_trading_date = _read_last_line(cal_path)
    print(f"calendar_updated: {old_last_trading_date} -> {new_last_trading_date}")
    if not args.dry_run and new_last_trading_date:
        if new_last_trading_date != old_last_trading_date:
            n_ext = _extend_market_instruments(inst_path, old_last_trading_date, new_last_trading_date)
            print(f"market_instruments_extended: file={inst_path} rows_updated={n_ext}")
        fixes = _clamp_market_instruments_to_sep(inst_path, sep_dir=raw_sep_dir, calendar_end=new_last_trading_date)
        if fixes:
            items = ", ".join([f"{k}->{v}" for k, v in sorted(fixes.items())])
            print(f"market_instruments_clamped_to_sep: {items}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Catch up US Sharadar Qlib dataset via Nasdaq Data Link API.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--api_key", default="", help="Optional; defaults to env NDL_API_KEY")
    p.add_argument("--out_dir", default="~/.qlib/sharadar")
    p.add_argument("--max_tickers", type=int, default=None, help="For smoke runs only")
    p.add_argument("--days_back_sep", type=int, default=10)
    p.add_argument("--days_back_sfp", type=int, default=10)
    p.add_argument("--days_back_sf2", type=int, default=60)
    p.add_argument("--days_back_sf3a", type=int, default=400)
    p.add_argument("--warmup_sf2_days", type=int, default=70)
    p.add_argument("--warmup_sf3a_days", type=int, default=420)
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
