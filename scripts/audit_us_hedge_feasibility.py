#!/usr/bin/env python
import argparse
import sys
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


COMMON_INVERSE_ETFS = {
    "DOG",
    "DXD",
    "PSQ",
    "QID",
    "RWM",
    "SDS",
    "SH",
    "SJB",
    "SPXU",
    "SQQQ",
    "TBF",
    "TBT",
    "TWM",
    "UDN",
}


def _coerce_ticker_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace(",", "\n").splitlines()
    elif isinstance(value, IterableABC):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        ticker = str(item).strip().upper()
        if not ticker or ticker.startswith("#") or ticker in {"TICKER", "SYMBOL"} or ticker in seen:
            continue
        out.append(ticker)
        seen.add(ticker)
    return out


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _safe_get(obj: Dict, keys: Iterable[str], default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _tickers_from_file(path_value: Optional[str]) -> List[str]:
    if not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists():
        return []
    return _coerce_ticker_list(path.read_text(encoding="utf-8").splitlines())


def _strategy_hedge_tickers(cfg: Dict) -> Tuple[List[str], str]:
    kwargs = _safe_get(cfg, ["port_analysis_config", "strategy", "kwargs"], {}) or {}
    tickers = _coerce_ticker_list(kwargs.get("hedge_tickers"))
    file_value = str(kwargs.get("hedge_tickers_file") or "").strip()
    tickers.extend(_tickers_from_file(file_value))
    return _coerce_ticker_list(tickers), file_value


def _normalize_close_frame(raw) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    data = raw
    if isinstance(data, pd.DataFrame):
        if "$close" in data.columns:
            data = data["$close"]
        elif data.shape[1] == 1:
            data = data.iloc[:, 0]
        else:
            return pd.DataFrame()
    if not isinstance(data, pd.Series) or data.empty or not isinstance(data.index, pd.MultiIndex):
        return pd.DataFrame()
    names = list(data.index.names)
    dt_level = "datetime" if "datetime" in names else 0
    inst_level = "instrument" if "instrument" in names else 1
    series = pd.to_numeric(data, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = series.unstack(inst_level).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.columns = pd.Index(frame.columns.astype(str).str.upper())
    return frame


def _load_qlib_close(
    provider_uri: str,
    tickers: List[str],
    *,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    if not tickers:
        return pd.DataFrame(), ""
    try:
        import qlib
        from qlib.constant import REG_US
        from qlib.data import D

        qlib.init(provider_uri=provider_uri, region=REG_US)
        raw = D.features(tickers, ["$close"], start_time=start, end_time=end)
        return _normalize_close_frame(raw), ""
    except Exception as exc:
        return pd.DataFrame(), repr(exc)


def _raw_sfp_summary(raw_sfp_dir: Path, ticker: str) -> Dict[str, object]:
    path = raw_sfp_dir / f"{ticker.upper()}.csv"
    if not path.exists():
        return {"raw_exists": False, "raw_rows": 0, "raw_start": "", "raw_end": ""}
    try:
        df = pd.read_csv(path, usecols=lambda c: str(c).lower() == "date")
        dates = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values()
        if dates.empty:
            return {"raw_exists": True, "raw_rows": 0, "raw_start": "", "raw_end": ""}
        return {
            "raw_exists": True,
            "raw_rows": int(len(dates)),
            "raw_start": dates.iloc[0].strftime("%Y-%m-%d"),
            "raw_end": dates.iloc[-1].strftime("%Y-%m-%d"),
        }
    except Exception:
        return {"raw_exists": True, "raw_rows": 0, "raw_start": "", "raw_end": ""}


def _qlib_ticker_summary(close: pd.DataFrame, ticker: str, *, min_history_days: int, max_missing_ratio: float) -> Dict[str, object]:
    ticker = ticker.upper()
    if close.empty or ticker not in close.columns:
        return {
            "qlib_rows": 0,
            "qlib_start": "",
            "qlib_end": "",
            "missing_ratio": 1.0,
            "qlib_ok": False,
        }
    series = pd.to_numeric(close[ticker], errors="coerce").replace([np.inf, -np.inf], np.nan)
    total = int(len(series))
    valid = series.dropna()
    missing_ratio = 1.0 if total <= 0 else float(series.isna().mean())
    ok = len(valid) >= int(min_history_days) and missing_ratio <= float(max_missing_ratio)
    return {
        "qlib_rows": int(len(valid)),
        "qlib_start": "" if valid.empty else valid.index.min().strftime("%Y-%m-%d"),
        "qlib_end": "" if valid.empty else valid.index.max().strftime("%Y-%m-%d"),
        "missing_ratio": missing_ratio,
        "qlib_ok": bool(ok),
    }


def _audit_rows(
    tickers: List[str],
    *,
    raw_sfp_dir: Path,
    qlib_close: pd.DataFrame,
    min_history_days: int,
    max_missing_ratio: float,
    require_inverse: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        row = {"ticker": ticker}
        row.update(_raw_sfp_summary(raw_sfp_dir, ticker))
        row.update(
            _qlib_ticker_summary(
                qlib_close,
                ticker,
                min_history_days=int(min_history_days),
                max_missing_ratio=float(max_missing_ratio),
            )
        )
        row["inverse_ok"] = (ticker in COMMON_INVERSE_ETFS) if require_inverse else True
        row["usable"] = bool(row["qlib_ok"] and row["inverse_ok"])
        rows.append(row)
    return rows


def _print_rows(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("(no hedge tickers configured)")
        return
    headers = [
        "ticker",
        "raw_exists",
        "raw_rows",
        "raw_start",
        "raw_end",
        "qlib_rows",
        "qlib_start",
        "qlib_end",
        "missing_ratio",
        "inverse_ok",
        "usable",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        vals = []
        for key in headers:
            value = row.get(key, "")
            if key == "missing_ratio":
                value = f"{float(value):.4f}"
            vals.append(str(value))
        print(" | ".join(vals))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether configured hedge ETFs are usable in qlib.")
    parser.add_argument("--config", default="", help="Optional workflow config containing strategy hedge settings")
    parser.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    parser.add_argument("--raw_sfp_dir", default="~/.qlib/sharadar/raw/sfp")
    parser.add_argument("--hedge_tickers", default="", help="Comma/newline separated hedge tickers")
    parser.add_argument("--hedge_tickers_file", default="", help="Optional hedge ticker file")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--min_history_days", type=int, default=252)
    parser.add_argument("--max_missing_ratio", type=float, default=0.05)
    parser.add_argument("--min_available_hedges", type=int, default=1)
    parser.add_argument("--require_inverse", action="store_true", help="Require configured tickers to be known inverse ETFs")
    parser.add_argument("--allow_cash_only", action="store_true", help="Pass if no usable hedge is present")
    parser.add_argument("--fail_on_no_hedge", action="store_true", help="Return non-zero if no configured usable hedge passes")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg: Dict = {}
    cfg_file_tickers: List[str] = []
    cfg_hedge_file = ""
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}", file=sys.stderr)
            return 2
        cfg = _load_yaml(cfg_path)
        cfg_file_tickers, cfg_hedge_file = _strategy_hedge_tickers(cfg)

    tickers = []
    tickers.extend(cfg_file_tickers)
    tickers.extend(_coerce_ticker_list(args.hedge_tickers))
    tickers.extend(_tickers_from_file(args.hedge_tickers_file))
    tickers = _coerce_ticker_list(tickers)
    hedge_file = args.hedge_tickers_file or cfg_hedge_file

    print("== Hedge Feasibility Audit ==")
    print(f"provider_uri={Path(args.provider_uri).expanduser().resolve()}")
    print(f"raw_sfp_dir={Path(args.raw_sfp_dir).expanduser().resolve()}")
    if hedge_file:
        print(f"hedge_tickers_file={Path(hedge_file).expanduser()}")
    print(f"hedge_tickers={','.join(tickers) if tickers else ''}")
    print(f"short_supported=False")
    print("cash_supported=True")

    if not tickers:
        print("RESULT: PASS (cash-only)" if args.allow_cash_only else "RESULT: FAIL (no hedge tickers)")
        return 0 if args.allow_cash_only and not args.fail_on_no_hedge else 1

    qlib_close, qlib_error = _load_qlib_close(
        str(args.provider_uri),
        tickers,
        start=args.start or None,
        end=args.end or None,
    )
    if qlib_error:
        print(f"qlib_error={qlib_error}")
    rows = _audit_rows(
        tickers,
        raw_sfp_dir=Path(args.raw_sfp_dir).expanduser().resolve(),
        qlib_close=qlib_close,
        min_history_days=int(args.min_history_days),
        max_missing_ratio=float(args.max_missing_ratio),
        require_inverse=bool(args.require_inverse),
    )
    _print_rows(rows)
    usable = [row["ticker"] for row in rows if row.get("usable")]
    required = max(1, int(args.min_available_hedges))
    ok = len(usable) >= required
    if ok:
        print(f"RESULT: PASS usable_hedges={','.join(usable)}")
        return 0
    if args.allow_cash_only and not args.fail_on_no_hedge:
        print("RESULT: PASS (cash-only; no usable hedge data)")
        return 0
    print(f"RESULT: FAIL usable_hedges={len(usable)} required={required}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
