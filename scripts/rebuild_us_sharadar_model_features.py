#!/usr/bin/env python
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROVENANCE_FILE = "sharadar_model_features.json"
MARKET_ETFS = ["SPY", "QQQ", "IWM", "IXIC"]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
SECTORS = [
    "technology",
    "financial_services",
    "consumer_cyclical",
    "healthcare",
    "industrials",
    "communication_services",
    "consumer_defensive",
    "energy",
    "utilities",
    "real_estate",
    "basic_materials",
]
SECTOR_TO_ETF = {
    "technology": "XLK",
    "financial_services": "XLF",
    "consumer_cyclical": "XLY",
    "healthcare": "XLV",
    "industrials": "XLI",
    "communication_services": "XLC",
    "consumer_defensive": "XLP",
    "energy": "XLE",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "basic_materials": "XLB",
}
RISK_FEATURES = [
    "risk_ret_20d",
    "risk_ret_63d",
    "risk_ret_252d",
    "risk_vol_20d",
    "risk_vol_63d",
    "risk_dvol_20d",
    "risk_beta_spy_63d",
    "risk_beta_spy_252d",
    "risk_beta_qqq_63d",
    "risk_beta_qqq_252d",
    "risk_relret_spy_63d",
    "risk_relret_spy_252d",
    "risk_relret_qqq_63d",
    "risk_relret_qqq_252d",
]
REGIME_FEATURES = [
    "mkt_spy_ret_20d",
    "mkt_spy_ret_63d",
    "mkt_spy_vol_20d",
    "mkt_spy_dd_63d",
    "mkt_qqq_ret_63d",
    "mkt_qqq_vol_20d",
    "mkt_iwm_ret_63d",
    "mkt_iwm_vol_20d",
]
LAGGED_REGIME_FEATURES = [
    "mkt_spy_ret_20d_lag1",
    "mkt_spy_ret_63d_lag1",
    "mkt_spy_ret_252d_lag1",
    "mkt_spy_vol_20d_lag1",
    "mkt_spy_vol_63d_lag1",
    "mkt_spy_dd_63d_lag1",
    "mkt_spy_dd_126d_lag1",
    "mkt_qqq_ret_20d_lag1",
    "mkt_qqq_ret_63d_lag1",
    "mkt_qqq_ret_252d_lag1",
    "mkt_qqq_vol_20d_lag1",
    "mkt_qqq_vol_63d_lag1",
    "mkt_qqq_dd_63d_lag1",
    "mkt_qqq_dd_126d_lag1",
    "mkt_iwm_ret_20d_lag1",
    "mkt_iwm_ret_63d_lag1",
    "mkt_iwm_vol_20d_lag1",
    "mkt_ixic_ret_63d_lag1",
    "mkt_ixic_vol_20d_lag1",
]
MARKET_BREADTH_FEATURES = [
    "mkt_breadth_ret20_pos_lag1",
    "mkt_breadth_ret63_pos_lag1",
    "mkt_breadth_ret63_above_spy_lag1",
    "mkt_breadth_ret252_pos_lag1",
    "mkt_dispersion_ret63_lag1",
]
SECTOR_REGIME_FEATURES = [f"mkt_{etf.lower()}_ret_63d" for etf in SECTOR_ETFS]
REGIME_INTERACTION_FEATURES = [
    "regime_beta_spy63_ret63",
    "regime_beta_spy63_dd63",
    "regime_vol20_spyvol20",
    "regime_relret63_spyret63",
    "regime_sector_relret63",
    "regime_sector_relret63_beta63",
]
META_FEATURES = ["meta_scalemarketcap", "meta_scalerevenue"] + [f"meta_sector_{s}" for s in SECTORS]
FEATURE_FIELDS = (
    RISK_FEATURES
    + REGIME_FEATURES
    + LAGGED_REGIME_FEATURES
    + MARKET_BREADTH_FEATURES
    + SECTOR_REGIME_FEATURES
    + REGIME_INTERACTION_FEATURES
    + META_FEATURES
)


def _calendar(provider_uri: Path) -> List[pd.Timestamp]:
    cal_path = provider_uri / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar not found: {cal_path}")
    return [pd.Timestamp(x.strip()) for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_market_tickers(provider_uri: Path, market: str, max_tickers: Optional[int]) -> List[str]:
    inst_path = provider_uri / "instruments" / f"{market}.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"instrument file not found: {inst_path}")
    df = pd.read_csv(inst_path, sep="\t", header=None, names=["ticker", "start", "end"])
    tickers = sorted(df["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist())
    if max_tickers is not None and int(max_tickers) > 0:
        tickers = tickers[: int(max_tickers)]
    return tickers


def _code_to_fname(symbol: str) -> str:
    try:
        from qlib.utils import code_to_fname

        return code_to_fname(symbol).lower()
    except Exception:
        return str(symbol).lower()


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _scale_value(value: object) -> float:
    text = str(value or "").strip()
    m = re.match(r"^\s*(\d+)", text)
    return float(m.group(1)) if m else np.nan


def _price_series(path: Path, calendar: pd.DatetimeIndex, *, ffill: bool) -> pd.Series:
    if not path.exists():
        return pd.Series(index=calendar, dtype="float32")
    usecols = lambda c: c in {"date", "closeadj", "close"}  # noqa: E731
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if df.empty or "date" not in df.columns:
        return pd.Series(index=calendar, dtype="float32")
    value_col = "closeadj" if "closeadj" in df.columns else "close"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")[value_col].sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).reindex(calendar)
    if ffill:
        s = s.ffill()
    return s.astype("float32")


def _ret(close: pd.Series, window: int) -> pd.Series:
    return (close / close.shift(int(window)) - 1.0).replace([np.inf, -np.inf], np.nan).astype("float32")


def _vol(ret: pd.Series, window: int) -> pd.Series:
    return (ret.rolling(int(window), min_periods=max(5, int(window) // 3)).std() * np.sqrt(252.0)).astype("float32")


def _drawdown(close: pd.Series, window: int) -> pd.Series:
    peak = close.rolling(int(window), min_periods=max(5, int(window) // 3)).max()
    return (close / peak - 1.0).replace([np.inf, -np.inf], np.nan).astype("float32")


def _lag1(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").shift(1).replace([np.inf, -np.inf], np.nan).astype("float32")


def _rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, window: int) -> pd.Series:
    window = int(window)
    cov = stock_ret.rolling(window, min_periods=max(20, window // 3)).cov(market_ret)
    var = market_ret.rolling(window, min_periods=max(20, window // 3)).var()
    return (cov / (var + 1e-12)).replace([np.inf, -np.inf], np.nan).astype("float32")


def _market_feature_frame(
    raw_sfp_dir: Path,
    calendar: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, int]]:
    out = pd.DataFrame(index=calendar)
    closes: Dict[str, pd.Series] = {}
    close_counts: Dict[str, int] = {}
    for etf in MARKET_ETFS + SECTOR_ETFS:
        close = _price_series(raw_sfp_dir / f"{etf}.csv", calendar, ffill=True)
        closes[etf] = close
        close_counts[etf] = int(close.notna().sum())

    spy = closes["SPY"]
    spy_ret_1d = spy.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    daily_returns: Dict[str, pd.Series] = {"SPY": spy_ret_1d.astype("float32")}
    out["mkt_spy_ret_20d"] = _ret(spy, 20)
    out["mkt_spy_ret_63d"] = _ret(spy, 63)
    out["mkt_spy_vol_20d"] = _vol(spy_ret_1d, 20)
    out["mkt_spy_dd_63d"] = _drawdown(spy, 63)
    for etf in ["QQQ", "IWM"]:
        ret_1d = closes[etf].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        daily_returns[etf] = ret_1d.astype("float32")
        out[f"mkt_{etf.lower()}_ret_63d"] = _ret(closes[etf], 63)
        out[f"mkt_{etf.lower()}_vol_20d"] = _vol(ret_1d, 20)
    for etf in MARKET_ETFS:
        lower = etf.lower()
        close = closes[etf]
        ret_1d = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        daily_returns[etf] = ret_1d.astype("float32")
        out[f"mkt_{lower}_ret_20d_lag1"] = _lag1(_ret(close, 20))
        out[f"mkt_{lower}_ret_63d_lag1"] = _lag1(_ret(close, 63))
        if etf in {"SPY", "QQQ"}:
            out[f"mkt_{lower}_ret_252d_lag1"] = _lag1(_ret(close, 252))
            out[f"mkt_{lower}_vol_63d_lag1"] = _lag1(_vol(ret_1d, 63))
            out[f"mkt_{lower}_dd_63d_lag1"] = _lag1(_drawdown(close, 63))
            out[f"mkt_{lower}_dd_126d_lag1"] = _lag1(_drawdown(close, 126))
        out[f"mkt_{lower}_vol_20d_lag1"] = _lag1(_vol(ret_1d, 20))
    for etf in SECTOR_ETFS:
        out[f"mkt_{etf.lower()}_ret_63d"] = _ret(closes[etf], 63)
    return out.astype("float32"), daily_returns, close_counts


def _stock_breadth_feature_frame(
    raw_sep_dir: Path,
    tickers: Iterable[str],
    calendar: pd.DatetimeIndex,
    *,
    spy_ret_63d: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    tickers = list(tickers)
    closes: Dict[str, pd.Series] = {}
    for ticker in tickers:
        close = _price_series(raw_sep_dir / f"{ticker}.csv", calendar, ffill=True)
        if close.notna().sum() >= 80:
            closes[str(ticker).upper()] = close
    out = pd.DataFrame(index=calendar)
    counts = {"tickers_requested": int(len(tickers)), "tickers_used": int(len(closes))}
    if not closes:
        for field in MARKET_BREADTH_FEATURES:
            out[field] = np.nan
        return out.astype("float32"), counts

    close_frame = pd.DataFrame(closes, index=calendar).astype("float32")
    ret20 = close_frame / close_frame.shift(20) - 1.0
    ret63 = close_frame / close_frame.shift(63) - 1.0
    ret252 = close_frame / close_frame.shift(252) - 1.0

    def ratio(mask: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
        denom = valid.sum(axis=1).replace(0, np.nan)
        return mask.where(valid, False).sum(axis=1) / denom

    valid20 = ret20.notna()
    valid63 = ret63.notna()
    valid252 = ret252.notna()
    out["mkt_breadth_ret20_pos_lag1"] = _lag1(ratio(ret20 > 0.0, valid20))
    out["mkt_breadth_ret63_pos_lag1"] = _lag1(ratio(ret63 > 0.0, valid63))
    if spy_ret_63d is None:
        spy_ret_63d = pd.Series(np.nan, index=calendar, dtype="float32")
    out["mkt_breadth_ret63_above_spy_lag1"] = _lag1(ratio(ret63.gt(spy_ret_63d, axis=0), valid63))
    out["mkt_breadth_ret252_pos_lag1"] = _lag1(ratio(ret252 > 0.0, valid252))
    out["mkt_dispersion_ret63_lag1"] = _lag1(ret63.std(axis=1, skipna=True))
    return out[MARKET_BREADTH_FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32"), counts


def _metadata_rows(tickers_csv: Path) -> Dict[str, Dict[str, float]]:
    if not tickers_csv.exists():
        return {}
    cols = ["table", "ticker", "sector", "scalemarketcap", "scalerevenue"]
    df = pd.read_csv(tickers_csv, usecols=lambda c: c in cols, low_memory=False)
    if "table" in df.columns:
        df = df[df["table"].astype(str).str.upper() == "SEP"]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates("ticker", keep="last")
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        vals = {
            "meta_scalemarketcap": _scale_value(row.get("scalemarketcap")),
            "meta_scalerevenue": _scale_value(row.get("scalerevenue")),
        }
        sector = _slug(row.get("sector"))
        for known in SECTORS:
            vals[f"meta_sector_{known}"] = 1.0 if sector == known else 0.0
        out[str(row["ticker"])] = vals
    return out


def _stock_feature_frame(
    close: pd.Series,
    volume: pd.Series,
    spy_ret: pd.Series,
    qqq_ret: pd.Series,
    market_features: pd.DataFrame,
    meta: Dict[str, float],
) -> pd.DataFrame:
    out = market_features.copy()
    ret_1d = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).astype("float32")
    out["risk_ret_20d"] = _ret(close, 20)
    out["risk_ret_63d"] = _ret(close, 63)
    out["risk_ret_252d"] = _ret(close, 252)
    out["risk_vol_20d"] = _vol(ret_1d, 20)
    out["risk_vol_63d"] = _vol(ret_1d, 63)
    out["risk_dvol_20d"] = np.log1p((close * volume).rolling(20, min_periods=5).mean()).astype("float32")
    out["risk_beta_spy_63d"] = _rolling_beta(ret_1d, spy_ret, 63)
    out["risk_beta_spy_252d"] = _rolling_beta(ret_1d, spy_ret, 252)
    out["risk_beta_qqq_63d"] = _rolling_beta(ret_1d, qqq_ret, 63)
    out["risk_beta_qqq_252d"] = _rolling_beta(ret_1d, qqq_ret, 252)
    out["risk_relret_spy_63d"] = (out["risk_ret_63d"] - market_features["mkt_spy_ret_63d"]).astype("float32")
    out["risk_relret_spy_252d"] = (out["risk_ret_252d"] - _ret_from_series(spy_ret, 252)).astype("float32")
    out["risk_relret_qqq_63d"] = (out["risk_ret_63d"] - market_features["mkt_qqq_ret_63d"]).astype("float32")
    out["risk_relret_qqq_252d"] = (out["risk_ret_252d"] - _ret_from_series(qqq_ret, 252)).astype("float32")
    out["regime_beta_spy63_ret63"] = (out["risk_beta_spy_63d"] * market_features["mkt_spy_ret_63d"]).astype("float32")
    out["regime_beta_spy63_dd63"] = (out["risk_beta_spy_63d"] * market_features["mkt_spy_dd_63d"]).astype("float32")
    out["regime_vol20_spyvol20"] = (out["risk_vol_20d"] * market_features["mkt_spy_vol_20d"]).astype("float32")
    out["regime_relret63_spyret63"] = (out["risk_relret_spy_63d"] * market_features["mkt_spy_ret_63d"]).astype("float32")
    sector_relret = pd.Series(np.nan, index=out.index, dtype="float32")
    for sector, etf in SECTOR_TO_ETF.items():
        if meta.get(f"meta_sector_{sector}") == 1.0:
            sector_relret = (
                market_features[f"mkt_{etf.lower()}_ret_63d"] - market_features["mkt_spy_ret_63d"]
            ).astype("float32")
            break
    out["regime_sector_relret63"] = sector_relret
    out["regime_sector_relret63_beta63"] = (sector_relret * out["risk_beta_spy_63d"]).astype("float32")
    for col in META_FEATURES:
        out[col] = np.float32(meta.get(col, np.nan))
    return out[FEATURE_FIELDS].replace([np.inf, -np.inf], np.nan).astype("float32")


def _ret_from_series(ret_1d: pd.Series, window: int) -> pd.Series:
    gross = (1.0 + ret_1d.astype(float)).replace([np.inf, -np.inf], np.nan)
    return (gross.rolling(int(window), min_periods=max(20, int(window) // 3)).apply(np.prod, raw=True) - 1.0).astype("float32")


def _read_stock_price_volume(path: Path, calendar: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    if not path.exists():
        return pd.Series(index=calendar, dtype="float32"), pd.Series(index=calendar, dtype="float32")
    usecols = lambda c: c in {"date", "closeadj", "close", "volume"}  # noqa: E731
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if df.empty or "date" not in df.columns:
        return pd.Series(index=calendar, dtype="float32"), pd.Series(index=calendar, dtype="float32")
    close_col = "closeadj" if "closeadj" in df.columns else "close"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    return (
        df[close_col].replace([np.inf, -np.inf], np.nan).reindex(calendar).astype("float32"),
        df["volume"].replace([np.inf, -np.inf], np.nan).reindex(calendar).astype("float32"),
    )


def _overwrite_feature_bins(prepared_dir: Path, provider_uri: Path) -> int:
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
        start = df["date"].min()
        end = df["date"].max()
        out_index = cal[(cal >= start) & (cal <= end)]
        if out_index.empty:
            continue
        start_idx = int(cal.get_loc(out_index[0]))
        df = df.set_index("date").reindex(out_index)
        symbol_dir = features_root / _code_to_fname(fp.stem)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for col in FEATURE_FIELDS:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").astype("float32").to_numpy()
            out = np.hstack([start_idx, values]).astype("<f")
            out.tofile(symbol_dir / f"{col}.day.bin")
            written += 1
    return written


def _write_provenance(provider_uri: Path, payload: Dict) -> Path:
    meta_dir = provider_uri / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / PROVENANCE_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Sharadar stock risk/regime/static model features into qlib bins.")
    p.add_argument("--provider_uri", default="/root/.qlib/qlib_data/us_data")
    p.add_argument("--raw_sep_dir", default="~/.qlib/sharadar/raw/sep")
    p.add_argument("--raw_sfp_dir", default="~/.qlib/sharadar/raw/sfp")
    p.add_argument("--tickers_csv", default="~/.qlib/sharadar/raw/tickers.csv")
    p.add_argument("--market", default="pit_mrq_large_idx")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--out_dir", default="")
    p.add_argument("--dump_to_qlib", action="store_true")
    p.add_argument("--max_tickers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    provider = Path(args.provider_uri).expanduser().resolve()
    raw_sep_dir = Path(args.raw_sep_dir).expanduser().resolve()
    raw_sfp_dir = Path(args.raw_sfp_dir).expanduser().resolve()
    tickers_csv = Path(args.tickers_csv).expanduser().resolve()
    if not raw_sep_dir.exists():
        print(f"raw SEP dir not found: {raw_sep_dir}", file=sys.stderr)
        return 2
    if not raw_sfp_dir.exists():
        print(f"raw SFP dir not found: {raw_sfp_dir}", file=sys.stderr)
        return 2

    full_calendar = pd.DatetimeIndex(_calendar(provider))
    output_calendar = full_calendar
    if args.start:
        output_calendar = output_calendar[output_calendar >= pd.Timestamp(args.start)]
    if args.end:
        output_calendar = output_calendar[output_calendar <= pd.Timestamp(args.end)]
    if len(output_calendar) == 0:
        print("empty calendar after start/end filters", file=sys.stderr)
        return 2

    tickers = _read_market_tickers(provider, args.market, args.max_tickers)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path("~/.qlib/sharadar/prepared").expanduser().resolve() / f"model_features_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    market_features, market_daily_returns, market_close_counts = _market_feature_frame(raw_sfp_dir, full_calendar)
    spy_ret = market_daily_returns.get("SPY", pd.Series(np.nan, index=full_calendar, dtype="float32"))
    qqq_ret = market_daily_returns.get("QQQ", pd.Series(np.nan, index=full_calendar, dtype="float32"))
    breadth_features, breadth_counts = _stock_breadth_feature_frame(
        raw_sep_dir,
        tickers,
        full_calendar,
        spy_ret_63d=market_features.get("mkt_spy_ret_63d"),
    )
    market_features = market_features.join(breadth_features, how="left")
    meta_by_ticker = _metadata_rows(tickers_csv)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider_uri": str(provider),
                    "market": args.market,
                    "tickers": len(tickers),
                    "calendar_days": len(output_calendar),
                    "feature_calendar_days": len(full_calendar),
                    "raw_sep_dir": str(raw_sep_dir),
                    "raw_sfp_dir": str(raw_sfp_dir),
                    "tickers_csv": str(tickers_csv),
                    "out_dir": str(out_dir),
                    "feature_fields": FEATURE_FIELDS,
                    "market_etf_close_counts": market_close_counts,
                    "breadth_source_counts": breadth_counts,
                },
                indent=2,
            )
        )
        return 0

    written_csv = 0
    skipped_no_price = 0
    for i, ticker in enumerate(tickers, start=1):
        close, volume = _read_stock_price_volume(raw_sep_dir / f"{ticker}.csv", full_calendar)
        if close.reindex(output_calendar).notna().sum() < 30:
            skipped_no_price += 1
            continue
        features = _stock_feature_frame(
            close,
            volume,
            spy_ret,
            qqq_ret,
            market_features,
            meta_by_ticker.get(ticker, {}),
        ).reindex(output_calendar).reset_index()
        features = features.rename(columns={"index": "date"})
        features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
        features.to_csv(out_dir / f"{ticker}.csv", index=False)
        written_csv += 1
        if i % 100 == 0:
            print(f"prepared_tickers={i}/{len(tickers)}")

    written_bins = 0
    if args.dump_to_qlib:
        written_bins = _overwrite_feature_bins(out_dir, provider)
        print(f"feature_bins_written={written_bins}")
        if written_bins <= 0:
            print("ERROR: no model feature bins were written", file=sys.stderr)
            return 3

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": "risk_regime_static_features",
        "provider_uri": str(provider),
        "market": str(args.market),
        "raw_sep_dir": str(raw_sep_dir),
        "raw_sfp_dir": str(raw_sfp_dir),
        "tickers_csv": str(tickers_csv),
        "calendar_start": str(output_calendar[0].date()),
        "calendar_end": str(output_calendar[-1].date()),
        "feature_calendar_start": str(full_calendar[0].date()),
        "feature_calendar_end": str(full_calendar[-1].date()),
        "tickers": int(len(tickers)),
        "prepared_csv_files": int(written_csv),
        "skipped_no_price": int(skipped_no_price),
        "prepared_dir": str(out_dir),
        "dump_to_qlib": bool(args.dump_to_qlib),
        "feature_bins_written": int(written_bins),
        "feature_fields": list(FEATURE_FIELDS),
        "market_etfs": list(MARKET_ETFS),
        "sector_etfs": list(SECTOR_ETFS),
        "market_etf_close_counts": market_close_counts,
        "breadth_source_counts": breadth_counts,
        "static_metadata_source": "current_tickers_table",
    }
    prov = _write_provenance(provider, payload)
    print(f"prepared_csv_files={written_csv}")
    print(f"skipped_no_price={skipped_no_price}")
    print(f"provenance: {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
