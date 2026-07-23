#!/usr/bin/env python
"""Build conservative daily features from SEC submissions filing history."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


RAW_ROOT = "/root/.qlib/sec/raw"
PREPARED_ROOT = "/root/.qlib/sec/prepared"
PROVENANCE_FILE = "sec_event_features.json"
DEFAULT_WINDOWS = "5,20,63,126"
DEFAULT_PREFIX = "sec"
STALE_EVENT_DAYS = 9999.0
DEFAULT_SEC_USER_AGENT = "qlib-stock-selector/1.0 research"
MATERIAL_8K_ITEMS = {"1.01", "2.02", "5.02", "7.01", "8.01"}
RESERVED_SYMBOLS = {"CIK", "TICKER", "SYMBOL"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def parse_windows(value: str) -> List[int]:
    windows = [int(x) for x in parse_csv_list(value)]
    windows = [w for w in windows if w > 0]
    if not windows:
        raise ValueError("windows must contain at least one positive integer")
    return sorted(set(windows))


def canonical_symbol(value: str) -> str:
    return str(value or "").strip().upper()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_symbol_from_path(path: Path) -> str:
    stem = path.stem
    if stem.upper().startswith("CIK"):
        return ""
    return canonical_symbol(stem)


def _candidate_raw_paths(raw_root: Path, symbol: str, cik: Optional[str] = None) -> List[Path]:
    symbol = canonical_symbol(symbol)
    out = [
        raw_root / "submissions" / f"{symbol}.json",
        raw_root / f"{symbol}.json",
    ]
    if cik:
        cik10 = f"{int(str(cik).lstrip('0') or '0'):010d}"
        out.extend([raw_root / "submissions" / f"CIK{cik10}.json", raw_root / f"CIK{cik10}.json"])
    return out


def find_raw_submission(raw_root: Path, symbol: str, cik: Optional[str] = None) -> Optional[Path]:
    for path in _candidate_raw_paths(raw_root, symbol, cik):
        if path.exists():
            return path
    return None


def raw_symbols(raw_root: Path) -> List[str]:
    symbols = set()
    for base in (raw_root / "submissions", raw_root):
        if not base.exists():
            continue
        for fp in base.glob("*.json"):
            symbol = _raw_symbol_from_path(fp)
            if symbol and symbol not in RESERVED_SYMBOLS:
                symbols.add(symbol)
    return sorted(symbols)


def _records_from_submission_payload(payload: Mapping, symbol: str) -> List[Dict]:
    filings = payload.get("filings") if isinstance(payload, Mapping) else None
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if recent is None and isinstance(payload, Mapping) and isinstance(payload.get("form"), list):
        recent = payload
    if isinstance(recent, list):
        rows = [dict(row) for row in recent if isinstance(row, Mapping)]
    elif isinstance(recent, Mapping):
        keys = [key for key, value in recent.items() if isinstance(value, list)]
        max_len = max((len(recent[key]) for key in keys), default=0)
        rows = []
        for i in range(max_len):
            row = {}
            for key in keys:
                values = recent.get(key) or []
                row[key] = values[i] if i < len(values) else None
            rows.append(row)
    else:
        rows = []
    for row in rows:
        row["symbol"] = symbol
    return rows


def _archive_submission_paths(raw_root: Path, symbol: str, cik: Optional[str] = None) -> List[Path]:
    symbol = canonical_symbol(symbol)
    out = []
    for base_name in [symbol]:
        if base_name:
            out.extend(sorted((raw_root / "submissions_archives" / base_name).glob("*.json")))
    if cik:
        cik10 = f"{int(str(cik).lstrip('0') or '0'):010d}"
        out.extend(sorted((raw_root / "submissions_archives" / f"CIK{cik10}").glob("*.json")))
    seen = set()
    unique = []
    for path in out:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_submission_events(raw_root: Path, symbols: Sequence[str], cik_map: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    rows: List[Dict] = []
    for symbol in symbols:
        symbol = canonical_symbol(symbol)
        if not symbol or symbol in RESERVED_SYMBOLS:
            continue
        fp = find_raw_submission(raw_root, symbol, (cik_map or {}).get(symbol))
        if fp is None:
            continue
        payload = _load_json(fp)
        rows.extend(_records_from_submission_payload(payload, symbol))
        for archive_fp in _archive_submission_paths(raw_root, symbol, (cik_map or {}).get(symbol)):
            rows.extend(_records_from_submission_payload(_load_json(archive_fp), symbol))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _availability_date(series: pd.Series, lag_days: int) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    parsed = parsed.dt.tz_convert(None).dt.normalize()
    if int(lag_days) != 0:
        parsed = parsed + pd.Timedelta(days=int(lag_days))
    return parsed


def _items_count(value: object) -> float:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0.0
    return float(len([token for token in re.split(r"[,; ]+", text) if token.strip()]))


def _has_material_8k_item(value: object) -> float:
    text = str(value or "").strip()
    tokens = {token.strip() for token in re.split(r"[,; ]+", text) if token.strip()}
    return 1.0 if bool(tokens & MATERIAL_8K_ITEMS) else 0.0


def _is_truthy(value: object) -> float:
    text = str(value or "").strip().lower()
    return 1.0 if text in {"1", "true", "t", "yes", "y"} else 0.0


def make_filing_events(raw_root: Path, symbols: Sequence[str], lag_days: int, prefix: str, cik_map: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    df = load_submission_events(raw_root, symbols, cik_map=cik_map)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    date_source = df["acceptanceDateTime"] if "acceptanceDateTime" in df.columns else pd.Series(pd.NaT, index=df.index)
    if "filingDate" in df.columns:
        date_source = date_source.where(date_source.notna(), df["filingDate"])
    df["date"] = _availability_date(date_source, lag_days)
    if "form" not in df.columns:
        df["form"] = ""
    if "items" not in df.columns:
        df["items"] = ""
    if "isXBRL" not in df.columns:
        df["isXBRL"] = 0
    forms = df["form"].astype(str).str.upper().str.strip()
    items = df["items"]
    is_8k = forms.isin({"8-K", "8-K/A", "6-K", "6-K/A"})
    is_10q = forms.isin({"10-Q", "10-Q/A", "10-QT", "10-QT/A"})
    is_10k = forms.isin({"10-K", "10-K/A", "10-KT", "10-KT/A"})
    is_proxy = forms.str.contains("14A", regex=False)
    is_registration = forms.str.startswith(("S-1", "F-1", "S-3", "F-3"))
    is_ownership = forms.isin({"3", "3/A", "4", "4/A", "5", "5/A"})
    is_amend = forms.str.endswith("/A")
    item_count = items.map(_items_count)
    material_8k = is_8k.astype(float) * items.map(_has_material_8k_item)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(canonical_symbol),
            "date": df["date"],
            f"{prefix}_filing_count": 1.0,
            f"{prefix}_8k_count": is_8k.astype("float32"),
            f"{prefix}_10q_count": is_10q.astype("float32"),
            f"{prefix}_10k_count": is_10k.astype("float32"),
            f"{prefix}_proxy_count": is_proxy.astype("float32"),
            f"{prefix}_registration_count": is_registration.astype("float32"),
            f"{prefix}_ownership_count": is_ownership.astype("float32"),
            f"{prefix}_amend_count": is_amend.astype("float32"),
            f"{prefix}_xbrl_count": df["isXBRL"].map(_is_truthy).astype("float32"),
            f"{prefix}_material_8k_count": material_8k.astype("float32"),
            f"{prefix}_8k_item_count": item_count.where(is_8k, 0.0).astype("float32"),
        }
    )
    return out.dropna(subset=["symbol", "date"])


def _date_range(start: str, end: str, events: Sequence[pd.DataFrame]) -> pd.DatetimeIndex:
    if start:
        lo = pd.Timestamp(start)
    else:
        mins = [pd.to_datetime(df["date"], errors="coerce").dropna().min() for df in events if not df.empty and "date" in df.columns]
        lo = pd.Timestamp(min(mins)) if mins else pd.Timestamp.utcnow().normalize()
    if end:
        hi = pd.Timestamp(end)
    else:
        maxs = [pd.to_datetime(df["date"], errors="coerce").dropna().max() for df in events if not df.empty and "date" in df.columns]
        hi = pd.Timestamp(max(maxs)) if maxs else pd.Timestamp.utcnow().normalize()
    if hi < lo:
        raise ValueError(f"end {hi.date()} is earlier than start {lo.date()}")
    return pd.date_range(lo.normalize(), hi.normalize(), freq="D")


def _days_since_event(index: pd.DatetimeIndex, has_event: pd.Series) -> pd.Series:
    event_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    mask = has_event.reindex(index).fillna(False).astype(bool)
    if bool(mask.any()):
        event_dates.loc[mask] = index[mask.to_numpy()]
    latest = event_dates.ffill()
    today = pd.Series(index, index=index, dtype="datetime64[ns]")
    days = (today - latest).dt.days.astype(float)
    return days.where(latest.notna(), STALE_EVENT_DAYS).clip(lower=0.0, upper=STALE_EVENT_DAYS)


def _freshness(days_since: pd.Series, window: int) -> pd.Series:
    days = pd.to_numeric(days_since, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = np.exp(-days.fillna(STALE_EVENT_DAYS).astype(float) / float(max(1, int(window))))
    return pd.Series(out, index=days_since.index, dtype=float).where(days < STALE_EVENT_DAYS, 0.0)


def expected_feature_columns(prefix: str, windows: Sequence[int]) -> List[str]:
    base_counts = [
        f"{prefix}_filing_count",
        f"{prefix}_8k_count",
        f"{prefix}_10q_count",
        f"{prefix}_10k_count",
        f"{prefix}_proxy_count",
        f"{prefix}_registration_count",
        f"{prefix}_ownership_count",
        f"{prefix}_amend_count",
        f"{prefix}_xbrl_count",
        f"{prefix}_material_8k_count",
        f"{prefix}_8k_item_count",
    ]
    cols: List[str] = []
    for col in base_counts:
        cols.append(f"{col}_daily")
        for w in windows:
            cols.append(f"{col}_{int(w)}d_sum")
    cols.append(f"{prefix}_filing_days_since_latest")
    for w in windows:
        cols.append(f"{prefix}_filing_{int(w)}d_freshness")
    cols.extend(
        [
            f"{prefix}_alpha_filing_freshness",
            f"{prefix}_alpha_material_event_intensity",
            f"{prefix}_alpha_periodic_report_intensity",
            f"{prefix}_alpha_amendment_risk",
            f"{prefix}_alpha_event_coverage",
            f"{prefix}_alpha_event_composite",
        ]
    )
    return cols


def event_features_for_symbol(daily: pd.DataFrame, *, date_index: pd.DatetimeIndex, prefix: str, windows: Sequence[int]) -> pd.DataFrame:
    frame = daily.set_index("date").sort_index().reindex(date_index).fillna(0.0)
    out = pd.DataFrame(index=date_index)
    base_cols = [col for col in daily.columns if col not in {"symbol", "date"}]
    for col in base_cols:
        values = pd.to_numeric(frame.get(col), errors="coerce").fillna(0.0).astype(float)
        out[f"{col}_daily"] = values.astype("float32")
        for w in windows:
            out[f"{col}_{int(w)}d_sum"] = values.rolling(int(w), min_periods=1).sum().astype("float32")
    filing_daily = pd.to_numeric(out.get(f"{prefix}_filing_count_daily"), errors="coerce").fillna(0.0)
    days_since = _days_since_event(date_index, filing_daily > 0)
    out[f"{prefix}_filing_days_since_latest"] = days_since.astype("float32")
    for w in windows:
        out[f"{prefix}_filing_{int(w)}d_freshness"] = _freshness(days_since, int(w)).astype("float32")
    return out


def add_alpha_features(frame: pd.DataFrame, *, prefix: str, windows: Sequence[int]) -> pd.DataFrame:
    out = frame.copy()
    long_window = 63 if 63 in windows else max(windows)
    freshness = pd.to_numeric(out.get(f"{prefix}_filing_{long_window}d_freshness"), errors="coerce").fillna(0.0)
    material = pd.to_numeric(out.get(f"{prefix}_material_8k_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    items = pd.to_numeric(out.get(f"{prefix}_8k_item_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    tenq = pd.to_numeric(out.get(f"{prefix}_10q_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    tenk = pd.to_numeric(out.get(f"{prefix}_10k_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    amend = pd.to_numeric(out.get(f"{prefix}_amend_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    filing = pd.to_numeric(out.get(f"{prefix}_filing_count_{long_window}d_sum"), errors="coerce").fillna(0.0)
    out[f"{prefix}_alpha_filing_freshness"] = freshness.astype("float32")
    out[f"{prefix}_alpha_material_event_intensity"] = np.log1p(material + 0.25 * items).astype("float32")
    out[f"{prefix}_alpha_periodic_report_intensity"] = np.log1p(tenq + tenk).astype("float32")
    out[f"{prefix}_alpha_amendment_risk"] = (-np.log1p(amend)).astype("float32")
    out[f"{prefix}_alpha_event_coverage"] = np.log1p(filing).clip(0.0, 10.0).astype("float32")
    components = [
        out[f"{prefix}_alpha_filing_freshness"],
        0.20 * out[f"{prefix}_alpha_material_event_intensity"],
        0.10 * out[f"{prefix}_alpha_periodic_report_intensity"],
        0.15 * out[f"{prefix}_alpha_amendment_risk"],
    ]
    out[f"{prefix}_alpha_event_composite"] = pd.concat(components, axis=1).mean(axis=1).fillna(0.0).astype("float32")
    return out


def build_feature_frames(
    raw_root: Path,
    *,
    start: str,
    end: str,
    windows: Sequence[int],
    availability_lag_days: int,
    prefix: str,
    symbols: Optional[Sequence[str]] = None,
    cik_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    symbol_list = sorted({canonical_symbol(s) for s in (symbols or raw_symbols(raw_root)) if canonical_symbol(s)})
    if not symbol_list:
        return {}
    events = make_filing_events(raw_root, symbol_list, availability_lag_days, prefix, cik_map=cik_map)
    date_index = _date_range(start, end, [events])
    expected_cols = expected_feature_columns(prefix, windows)
    frames: Dict[str, pd.DataFrame] = {}
    for symbol in symbol_list:
        s = events[events["symbol"] == symbol] if not events.empty else pd.DataFrame()
        if s.empty:
            out = pd.DataFrame(0.0, index=date_index, columns=expected_cols)
        else:
            daily = s.groupby(["symbol", "date"], as_index=False).sum(numeric_only=True)
            out = event_features_for_symbol(daily, date_index=date_index, prefix=prefix, windows=windows)
            out = add_alpha_features(out, prefix=prefix, windows=windows)
            out = out.reindex(columns=expected_cols, fill_value=0.0).fillna(0.0)
        out = out.reset_index().rename(columns={"index": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        frames[symbol] = out
    return frames


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


def _overwrite_feature_bins(prepared_dir: Path, provider_uri: Path, *, prefix: str) -> int:
    cal = pd.DatetimeIndex(_calendar(provider_uri))
    features_root = provider_uri / "features"
    written = 0
    for fp in sorted(prepared_dir.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            continue
        mask = (cal >= df["date"].min()) & (cal <= df["date"].max())
        out_index = cal[mask]
        if out_index.empty:
            continue
        start_idx = int(cal.get_loc(out_index[0]))
        df = df.set_index("date").reindex(out_index).fillna(0.0)
        symbol_dir = features_root / _code_to_fname(fp.stem)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for col in df.columns:
            if not str(col).startswith(f"{prefix}_"):
                continue
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32").to_numpy()
            out = np.hstack([start_idx, values]).astype("<f")
            out.tofile(symbol_dir / f"{col.lower()}.day.bin")
            written += 1
    return written


def extract_cik_from_secfilings(value: object) -> str:
    match = re.search(r"CIK=0*([0-9]+)", str(value or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def load_cik_map(tickers_csv: Path) -> Dict[str, str]:
    if not tickers_csv.exists():
        return {}
    df = pd.read_csv(tickers_csv, usecols=lambda c: c in {"ticker", "secfilings"}, low_memory=False)
    if "ticker" not in df.columns or "secfilings" not in df.columns:
        return {}
    df["symbol"] = df["ticker"].map(canonical_symbol)
    df["cik"] = df["secfilings"].map(extract_cik_from_secfilings)
    df = df[(df["symbol"] != "") & (df["cik"] != "")]
    return dict(df.drop_duplicates("symbol", keep="last").set_index("symbol")["cik"])


def download_missing_submissions(
    *,
    raw_root: Path,
    symbols: Sequence[str],
    cik_map: Mapping[str, str],
    user_agent: str,
    sleep_seconds: float = 0.11,
    download_archives: bool = True,
) -> int:
    out_dir = raw_root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    headers = {"User-Agent": str(user_agent or DEFAULT_SEC_USER_AGENT)}
    for symbol in symbols:
        symbol = canonical_symbol(symbol)
        cik = cik_map.get(symbol)
        if not symbol or not cik:
            continue
        fp = find_raw_submission(raw_root, symbol, cik)
        if fp is not None:
            payload = fp.read_text(encoding="utf-8")
        else:
            cik10 = f"{int(str(cik).lstrip('0') or '0'):010d}"
            url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    payload = resp.read().decode("utf-8")
            except Exception as exc:
                print(f"WARNING: failed to download SEC submissions for {symbol} CIK{cik10}: {exc}", file=sys.stderr)
                continue
            (out_dir / f"{symbol}.json").write_text(payload, encoding="utf-8")
            written += 1
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
        if download_archives:
            try:
                submission = json.loads(payload)
            except Exception:
                submission = {}
            files = ((submission.get("filings") or {}).get("files") or []) if isinstance(submission, Mapping) else []
            archive_dir = raw_root / "submissions_archives" / symbol
            for file_row in files:
                name = str((file_row or {}).get("name") or "").strip()
                if not name or "/" in name:
                    continue
                archive_path = archive_dir / name
                if archive_path.exists():
                    continue
                archive_url = f"https://data.sec.gov/submissions/{name}"
                archive_req = urllib.request.Request(archive_url, headers=headers)
                try:
                    with urllib.request.urlopen(archive_req, timeout=30) as resp:
                        archive_payload = resp.read().decode("utf-8")
                except Exception as exc:
                    print(f"WARNING: failed to download SEC archive for {symbol} {name}: {exc}", file=sys.stderr)
                    continue
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path.write_text(archive_payload, encoding="utf-8")
                written += 1
                if sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))
    return written


def write_provenance(provider_uri: Path, payload: Mapping) -> Path:
    meta = provider_uri / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    out = meta / PROVENANCE_FILE
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PIT-safe daily features from free SEC submissions history.")
    p.add_argument("--raw_root", default=RAW_ROOT)
    p.add_argument("--out_dir", default="")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--tickers_csv", default="/root/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to raw JSON symbol files.")
    p.add_argument("--symbols_file", default="", help="Optional newline-delimited symbol list")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--windows", default=DEFAULT_WINDOWS)
    p.add_argument("--availability_lag_days", type=int, default=1)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--download_missing", action="store_true")
    p.add_argument("--skip_archive_download", action="store_true", help="Do not fetch older SEC submissions archive files")
    p.add_argument("--sec_user_agent", default=DEFAULT_SEC_USER_AGENT)
    p.add_argument("--dump_to_qlib", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists() and not args.download_missing:
        print(f"ERROR: raw_root not found: {raw_root}", file=sys.stderr)
        return 2
    windows = parse_windows(args.windows)
    requested_symbols = parse_csv_list(args.symbols)
    if args.symbols_file:
        symbol_path = Path(args.symbols_file).expanduser().resolve()
        file_symbols = [line.strip() for line in symbol_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        requested_symbols = sorted({*requested_symbols, *file_symbols})
    cik_map = load_cik_map(Path(args.tickers_csv).expanduser().resolve())
    if args.download_missing and args.dry_run:
        symbols_for_download = requested_symbols or sorted(cik_map.keys())
        print(f"dry_run: would download up to {len(symbols_for_download)} SEC submissions into {raw_root}")
        downloaded = 0
    elif args.download_missing:
        symbols_for_download = requested_symbols or sorted(cik_map.keys())
        downloaded = download_missing_submissions(
            raw_root=raw_root,
            symbols=symbols_for_download,
            cik_map=cik_map,
            user_agent=str(args.sec_user_agent),
            download_archives=not bool(args.skip_archive_download),
        )
    else:
        downloaded = 0
    if not raw_root.exists():
        print(f"ERROR: raw_root not found after download: {raw_root}", file=sys.stderr)
        return 2
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path(PREPARED_ROOT).expanduser().resolve() / f"sec_event_features_{stamp}"
    )
    frames = build_feature_frames(
        raw_root,
        start=str(args.start),
        end=str(end),
        windows=windows,
        availability_lag_days=int(args.availability_lag_days),
        prefix=str(args.prefix),
        symbols=requested_symbols or None,
        cik_map=cik_map,
    )
    if not frames:
        print("ERROR: no SEC feature frames were built", file=sys.stderr)
        return 3
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for symbol, frame in frames.items():
            frame.to_csv(out_dir / f"{symbol}.csv", index=False)
    feature_cols = sorted({c for frame in frames.values() for c in frame.columns if c != "date"})
    alpha_cols = sorted(c for c in feature_cols if str(c).startswith(f"{args.prefix}_alpha_"))
    provider = Path(args.provider_uri).expanduser().resolve()
    written_bins = 0
    if args.dump_to_qlib:
        if args.dry_run:
            print(f"dry_run: would overwrite {args.prefix}_*.day.bin files from {out_dir}")
        else:
            written_bins = _overwrite_feature_bins(out_dir, provider, prefix=str(args.prefix))
            if written_bins <= 0:
                print("ERROR: no SEC feature bins were written", file=sys.stderr)
                return 4
    payload = {
        "created_utc": utc_now_iso(),
        "raw_root": str(raw_root),
        "prepared_dir": str(out_dir),
        "provider_uri": str(provider),
        "tickers_csv": str(Path(args.tickers_csv).expanduser().resolve()),
        "start": str(args.start),
        "end": str(end),
        "availability_lag_days": int(args.availability_lag_days),
        "windows": windows,
        "prefix": str(args.prefix),
        "tickers": int(len(frames)),
        "download_missing": bool(args.download_missing),
        "download_archives": bool(args.download_missing and not args.skip_archive_download),
        "downloaded_submissions": int(downloaded),
        "feature_columns": feature_cols,
        "feature_column_count": int(len(feature_cols)),
        "directional_feature_version": 1,
        "directional_feature_columns": alpha_cols,
        "dump_to_qlib": bool(args.dump_to_qlib),
        "feature_bins_written": int(written_bins),
        "pit_safety": "features are aligned to filing acceptance or filing date plus availability_lag_days",
        "source": "SEC company submissions JSON",
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        prov = write_provenance(provider, payload) if args.dump_to_qlib else out_dir / "sec_event_features_manifest.json"
        if not args.dump_to_qlib:
            prov.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"prepared_dir={out_dir}")
        print(f"manifest={prov}")
    print(f"tickers={len(frames)} feature_columns={len(feature_cols)} rows={sum(len(f) for f in frames.values())} bins={written_bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
