# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import fire
import numpy as np
import pandas as pd
import requests
from loguru import logger


BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_ticker_list(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "ticker" in df.columns:
        tickers = df["ticker"].astype(str).tolist()
    else:
        tickers = df.iloc[:, 0].astype(str).tolist()
    return sorted(set([t.strip().upper() for t in tickers if str(t).strip()]))


def _extract_meta_cursor(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    meta = payload.get("meta") or payload.get("datatable", {}).get("meta") or {}
    if isinstance(meta, dict):
        return meta.get("next_cursor_id", "") or ""
    return ""


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


def _read_last_date_from_csv(path: Path) -> str:
    line = _read_last_line(path)
    if not line or line.lower().startswith("ticker,"):
        return ""
    parts = line.split(",")
    if len(parts) < 2:
        return ""
    return parts[1].strip()


class SharadarCollector:
    """
    Collector for Sharadar (Nasdaq Data Link) tables used in Qlib.
    """

    def __init__(
        self,
        api_key: str,
        out_dir: str = "~/.qlib/sharadar",
        per_page: int = 10000,
        sleep: float = 0.2,
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.per_page = per_page
        self.sleep = sleep
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def _get(self, table: str, params: Dict[str, str]) -> dict:
        url = f"{BASE_URL}/{table}.json"
        params = dict(params)
        params["api_key"] = self.api_key
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:200]}")
        return resp.json()

    def _paginate_to_csv(
        self,
        table: str,
        out_file: Path,
        params: Dict[str, str],
    ) -> None:
        cursor = ""
        page = 1
        first_write = True
        _ensure_dir(out_file.parent)
        while True:
            page_params = dict(params)
            page_params["qopts.per_page"] = str(self.per_page)
            if cursor:
                page_params["qopts.cursor_id"] = cursor
            payload = self._get(table, page_params)
            dt = payload.get("datatable", {})
            columns = [c["name"] for c in dt.get("columns", [])]
            rows = dt.get("data", [])
            if first_write:
                with out_file.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(rows)
                first_write = False
            else:
                with out_file.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
            logger.info(f"{table}: wrote page {page} rows={len(rows)} -> {out_file}")
            cursor = _extract_meta_cursor(payload)
            if not cursor or not rows:
                break
            page += 1
            if self.sleep:
                time.sleep(self.sleep)

    # ----------------------------
    # Downloads
    # ----------------------------
    def download_tickers(self, out_file: str = None) -> str:
        """
        Download SHARADAR/TICKERS to CSV.
        """
        if out_file is None:
            out_file = str(self.out_dir.joinpath("raw/tickers.csv"))
        out_path = Path(out_file).expanduser().resolve()
        logger.info(f"Downloading TICKERS -> {out_path}")
        self._paginate_to_csv("TICKERS", out_path, params={})
        return str(out_path)

    def download_sep(
        self,
        tickers_file: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        out_dir: str = None,
    ) -> str:
        """
        Download SHARADAR/SEP (equity prices) per ticker to CSV.
        """
        if out_dir is None:
            out_dir = str(self.out_dir.joinpath("raw/sep"))
        out_dir = str(Path(out_dir).expanduser().resolve())
        _ensure_dir(Path(out_dir))
        tickers = _read_ticker_list(Path(tickers_file))
        logger.info(f"Downloading SEP for {len(tickers)} tickers -> {out_dir}")
        for idx, ticker in enumerate(tickers, 1):
            params = {"ticker": ticker, "date.gte": start}
            if end:
                params["date.lte"] = end
            out_file = Path(out_dir).joinpath(f"{ticker}.csv")
            if out_file.exists():
                logger.info(f"Skip existing SEP file: {out_file}")
                continue
            try:
                self._paginate_to_csv("SEP", out_file, params=params)
            except Exception as e:
                logger.warning(f"SEP download failed for {ticker}: {e}")
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return out_dir

    def update_sep(
        self,
        tickers_file: str,
        sep_dir: str = None,
        days_back: int = 5,
        fallback_start: str = "2000-01-01",
    ) -> str:
        """
        Incrementally update SEP files by pulling the last N days per ticker.
        """
        if sep_dir is None:
            sep_dir = str(self.out_dir.joinpath("raw/sep"))
        sep_dir = Path(sep_dir).expanduser().resolve()
        _ensure_dir(sep_dir)
        tickers = _read_ticker_list(Path(tickers_file))
        logger.info(f"Updating SEP for {len(tickers)} tickers -> {sep_dir}")
        for idx, ticker in enumerate(tickers, 1):
            out_file = sep_dir.joinpath(f"{ticker}.csv")
            last_date = _read_last_date_from_csv(out_file)
            if not last_date:
                params = {"ticker": ticker, "date.gte": fallback_start}
                try:
                    self._paginate_to_csv("SEP", out_file, params=params)
                except Exception as e:
                    logger.warning(f"SEP update failed for {ticker}: {e}")
                continue
            try:
                start_date = (pd.Timestamp(last_date) - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
            except Exception:
                start_date = last_date
            tmp_file = sep_dir.joinpath(f".{ticker}.update.tmp.csv")
            params = {"ticker": ticker, "date.gte": start_date}
            try:
                self._paginate_to_csv("SEP", tmp_file, params=params)
            except Exception as e:
                logger.warning(f"SEP update failed for {ticker}: {e}")
                if tmp_file.exists():
                    tmp_file.unlink()
                continue
            if tmp_file.exists() and tmp_file.stat().st_size > 0:
                try:
                    df = pd.read_csv(tmp_file)
                    if "date" in df.columns:
                        df = df[df["date"] > last_date]
                    if not df.empty:
                        df.to_csv(out_file, mode="a", header=False, index=False)
                except Exception as e:
                    logger.warning(f"SEP merge failed for {ticker}: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return str(sep_dir)

    def download_sfp(
        self,
        tickers_file: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        out_dir: str = None,
    ) -> str:
        """
        Download SHARADAR/SFP (fund prices) per ticker to CSV.
        """
        if out_dir is None:
            out_dir = str(self.out_dir.joinpath("raw/sfp"))
        out_dir = str(Path(out_dir).expanduser().resolve())
        _ensure_dir(Path(out_dir))
        tickers = _read_ticker_list(Path(tickers_file))
        logger.info(f"Downloading SFP for {len(tickers)} tickers -> {out_dir}")
        for idx, ticker in enumerate(tickers, 1):
            params = {"ticker": ticker, "date.gte": start}
            if end:
                params["date.lte"] = end
            out_file = Path(out_dir).joinpath(f"{ticker}.csv")
            if out_file.exists():
                logger.info(f"Skip existing SFP file: {out_file}")
                continue
            try:
                self._paginate_to_csv("SFP", out_file, params=params)
            except Exception as e:
                logger.warning(f"SFP download failed for {ticker}: {e}")
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return out_dir

    def download_sf1(
        self,
        tickers_file: str,
        dimension: str = "MRQ",
        out_file: str = None,
    ) -> str:
        """
        Download SHARADAR/SF1 (fundamentals PIT) to a single CSV file.
        """
        if out_file is None:
            out_file = str(self.out_dir.joinpath(f"raw/sf1_{dimension}.csv"))
        out_path = Path(out_file).expanduser().resolve()
        tickers = _read_ticker_list(Path(tickers_file))
        logger.info(f"Downloading SF1 dimension={dimension} for {len(tickers)} tickers -> {out_path}")
        # Download in batches by ticker to keep requests smaller
        batch_size = 200
        _ensure_dir(out_path.parent)
        first_write = True
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            params = {"ticker": ",".join(batch), "dimension": dimension}
            try:
                payload = self._get("SF1", {**params, "qopts.per_page": str(self.per_page)})
            except Exception as e:
                logger.warning(f"SF1 batch failed ({i}-{i+batch_size}): {e}")
                continue
            dt = payload.get("datatable", {})
            columns = [c["name"] for c in dt.get("columns", [])]
            rows = dt.get("data", [])
            if first_write:
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(rows)
                first_write = False
            else:
                with out_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
            logger.info(f"SF1 batch {i}-{i+len(batch)} rows={len(rows)}")
            if self.sleep:
                time.sleep(self.sleep)
        return str(out_path)

    # ----------------------------
    # Universe + Qlib prep
    # ----------------------------
    def build_universe(
        self,
        tickers_file: str,
        out_file: str = None,
        include_exchanges: str = "NYSE,NASDAQ,NYSEARCA,NYSEMKT,NYSEAMERICAN,BATS,IEXG",
        include_categories: str = "",
        exclude_keywords: str = "ETF,ETN,Fund,Closed-End,Closed End,CEF,Warrant,Unit,Preferred,ADR",
    ) -> str:
        """
        Build a default US common-stock universe from TICKERS.
        """
        df = pd.read_csv(tickers_file)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["exchange"] = df["exchange"].astype(str)
        df["category"] = df["category"].astype(str)
        df["currency"] = df["currency"].astype(str)
        df["location"] = df["location"].astype(str)

        include_ex = [x.strip() for x in include_exchanges.split(",") if x.strip()]
        include_cat = [x.strip() for x in include_categories.split(",") if x.strip()]
        exclude_kw = [x.strip().lower() for x in exclude_keywords.split(",") if x.strip()]

        mask = df["exchange"].isin(include_ex)
        mask &= df["currency"].str.upper().eq("USD")
        loc = df["location"].astype(str)
        mask &= loc.str.contains("U.S.A", case=False, na=False) | loc.str.contains("United States", case=False, na=False) | loc.str.upper().isin(["USA", "US", "U.S."])
        if include_cat:
            mask &= df["category"].isin(include_cat)
        if exclude_kw:
            lower_cat = df["category"].str.lower()
            for kw in exclude_kw:
                mask &= ~lower_cat.str.contains(kw)

        u = df.loc[mask, ["ticker", "name", "exchange", "category", "firstpricedate", "lastpricedate", "isdelisted"]]
        if out_file is None:
            out_file = str(self.out_dir.joinpath("universe/us_common_stocks.csv"))
        out_path = Path(out_file).expanduser().resolve()
        _ensure_dir(out_path.parent)
        u.to_csv(out_path, index=False)
        logger.info(f"Universe saved -> {out_path} rows={len(u)}")
        return str(out_path)

    def prepare_qlib_csv(
        self,
        sep_dir: str,
        out_dir: str = None,
    ) -> str:
        """
        Convert SEP raw CSVs to Qlib-friendly CSVs with dividend-adjusted prices.
        Output columns: date, open, high, low, close, volume, factor
        """
        if out_dir is None:
            out_dir = str(self.out_dir.joinpath("prepared/qlib_csv"))
        out_path = Path(out_dir).expanduser().resolve()
        _ensure_dir(out_path)
        sep_dir = Path(sep_dir).expanduser().resolve()
        files = sorted(sep_dir.glob("*.csv"))
        logger.info(f"Preparing Qlib CSVs from {len(files)} SEP files -> {out_path}")
        for idx, fp in enumerate(files, 1):
            df = pd.read_csv(fp)
            if df.empty:
                continue
            df["factor"] = df["closeadj"] / df["closeunadj"]
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.loc[df["closeunadj"] == 0, "factor"] = np.nan
            df["open"] = df["open"] * df["factor"]
            df["high"] = df["high"] * df["factor"]
            df["low"] = df["low"] * df["factor"]
            df["close"] = df["close"] * df["factor"]
            df = df.loc[:, ["date", "open", "high", "low", "close", "volume", "factor"]]
            df = df.dropna(subset=["date", "open", "high", "low", "close"])
            df = df.sort_values("date")
            out_file = out_path.joinpath(fp.name)
            df.to_csv(out_file, index=False)
            if idx % 500 == 0:
                logger.info(f"Prepared {idx}/{len(files)}")
        return str(out_path)


if __name__ == "__main__":
    fire.Fire(SharadarCollector)
