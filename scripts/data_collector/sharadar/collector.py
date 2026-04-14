# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fire
import numpy as np
import pandas as pd
import requests
import yaml
from loguru import logger


API_ROOT = "https://data.nasdaq.com/api/v3"
BASE_URL = f"{API_ROOT}/datatables/SHARADAR"
DATABASE_META_URL = f"{API_ROOT}/databases/SHARADAR/metadata.json"
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


def _read_last_date_from_csv_col(path: Path, date_col: str) -> str:
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


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _upsert_csv_range(
    out_file: Path,
    df_new: pd.DataFrame,
    *,
    date_col: str,
    start_date: str,
) -> None:
    if df_new is None or df_new.empty:
        return
    df_new = df_new.copy()
    df_new[date_col] = pd.to_datetime(df_new[date_col], errors="coerce")
    df_new = df_new.dropna(subset=[date_col]).sort_values(date_col)

    if not out_file.exists() or out_file.stat().st_size == 0:
        df_new[date_col] = df_new[date_col].dt.strftime("%Y-%m-%d")
        _atomic_write_csv(df_new, out_file)
        return

    df_old = pd.read_csv(out_file, low_memory=False)
    if date_col in df_old.columns:
        df_old[date_col] = pd.to_datetime(df_old[date_col], errors="coerce")
        df_old = df_old.dropna(subset=[date_col])
        df_keep = df_old[df_old[date_col] < pd.Timestamp(start_date)]
    else:
        df_keep = df_old.iloc[0:0]

    merged = pd.concat([df_keep, df_new], ignore_index=True, sort=False)
    merged = merged.drop_duplicates()
    if date_col in merged.columns:
        merged = merged.sort_values(date_col)
        merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    _atomic_write_csv(merged, out_file)


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _parse_key_value_csv(value: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if value is None:
        return out
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if not isinstance(value, str):
        raise TypeError(f"filters must be string or dict, got {type(value)}")
    if not value.strip():
        return out
    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid filter '{item}', expected key=value")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid filter '{item}', empty key")
        out[key] = val
    return out


def _extract_table_names(obj: Any) -> List[str]:
    names = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
            continue
        if not isinstance(cur, dict):
            continue
        for k, v in cur.items():
            lk = str(k).lower()
            if lk in {"datatable_code", "table", "table_code"} and isinstance(v, str):
                if v and v.upper() == v and v.replace("_", "").isalnum():
                    names.add(v)
            if lk == "name" and isinstance(v, str):
                if v and v.upper() == v and v.replace("_", "").isalnum():
                    names.add(v)
            if isinstance(v, (dict, list)):
                stack.append(v)
    return sorted(names)


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

    def _request_json(self, url: str, params: Dict[str, str]) -> dict:
        params = dict(params)
        params["api_key"] = self.api_key
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:200]}")
        return resp.json()

    def _get(self, table: str, params: Dict[str, str]) -> dict:
        return self._request_json(f"{BASE_URL}/{table}.json", params=params)

    def _get_table_metadata(self, table: str) -> dict:
        return self._request_json(f"{BASE_URL}/{table}/metadata.json", params={})

    def _normalize_columns(self, columns: Optional[str]) -> str:
        cols = _parse_csv_list(columns)
        return ",".join(cols)

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

    def describe_table(self, table: str, out_file: str = None) -> str:
        """
        Fetch metadata + sample schema for a SHARADAR datatable.
        """
        t = str(table).strip().upper()
        if not t:
            raise ValueError("table cannot be empty")
        if out_file is None:
            out_file = str(self.out_dir.joinpath(f"schema/{t}.json"))
        out_path = Path(out_file).expanduser().resolve()
        _ensure_dir(out_path.parent)

        result = {
            "table": t,
            "retrieved_at_utc": pd.Timestamp.utcnow().isoformat(),
            "metadata_ok": False,
            "sample_ok": False,
            "metadata_error": "",
            "sample_error": "",
            "metadata": {},
            "sample_columns": [],
            "sample_rows": 0,
        }

        try:
            md = self._get_table_metadata(t)
            result["metadata_ok"] = True
            result["metadata"] = md
        except Exception as e:
            result["metadata_error"] = str(e)

        try:
            sample = self._get(t, params={"qopts.per_page": "1"})
            dt = sample.get("datatable", {})
            cols = [c["name"] for c in dt.get("columns", [])]
            rows = dt.get("data", [])
            result["sample_ok"] = True
            result["sample_columns"] = cols
            result["sample_rows"] = len(rows)
        except Exception as e:
            result["sample_error"] = str(e)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=False)
        logger.info(f"Saved schema snapshot -> {out_path}")
        return str(out_path)

    def discover_tables(
        self,
        tables: str = "",
        out_file: str = None,
        verify_access: bool = True,
    ) -> str:
        """
        Discover and optionally verify table accessibility.

        If `tables` is provided, it is treated as comma-separated candidates.
        Otherwise we try metadata endpoints and then fall back to common SHARADAR tables.
        """
        names = set([x.upper() for x in _parse_csv_list(tables)])
        if not names:
            for url in [f"{BASE_URL}/metadata.json", DATABASE_META_URL]:
                try:
                    payload = self._request_json(url, params={})
                    names.update(_extract_table_names(payload))
                except Exception as e:
                    logger.warning(f"Table discovery endpoint failed: {url} ({e})")
            # Fallback candidates for Sharadar CORE US (COR) bundle users.
            names.update(
                {
                    "TICKERS",
                    "SEP",
                    "SFP",
                    "SF1",
                    "SF2",
                    "SF3",
                    "SF3A",
                    "SF3B",
                    "DAILY",
                    "ACTIONS",
                    "EVENTS",
                    "SP500",
                    "INDICATORS",
                }
            )

        rows = []
        for name in sorted(names):
            ok = None
            cols = []
            err = ""
            if verify_access:
                try:
                    sample = self._get(name, {"qopts.per_page": "1"})
                    dt = sample.get("datatable", {})
                    cols = [c["name"] for c in dt.get("columns", [])]
                    ok = True
                except Exception as e:
                    ok = False
                    err = str(e)
            rows.append(
                {
                    "table": name,
                    "accessible": "" if ok is None else ("1" if ok else "0"),
                    "n_columns": len(cols),
                    "sample_columns": ",".join(cols[:30]),
                    "error": err,
                }
            )

        if out_file is None:
            out_file = str(self.out_dir.joinpath("schema/discovered_tables.csv"))
        out_path = Path(out_file).expanduser().resolve()
        _ensure_dir(out_path.parent)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info(f"Saved table discovery -> {out_path} rows={len(rows)}")
        return str(out_path)

    def download_table(
        self,
        table: str,
        out_file: str = None,
        filters: str = "",
        columns: str = "",
        skip_existing: bool = False,
    ) -> str:
        """
        Download any SHARADAR datatable to CSV.
        - filters: comma-separated key=value pairs (e.g. ticker=AAPL,date.gte=2020-01-01)
        - columns: comma-separated qopts.columns
        """
        t = str(table).strip().upper()
        if not t:
            raise ValueError("table cannot be empty")
        if out_file is None:
            out_file = str(self.out_dir.joinpath(f"raw/{t.lower()}.csv"))
        out_path = Path(out_file).expanduser().resolve()
        if skip_existing and out_path.exists():
            logger.info(f"Skip existing table file: {out_path}")
            return str(out_path)

        params = _parse_key_value_csv(filters)
        cols = self._normalize_columns(columns)
        if cols:
            params["qopts.columns"] = cols
        logger.info(f"Downloading table {t} -> {out_path}")
        self._paginate_to_csv(t, out_path, params=params)
        return str(out_path)

    def download_table_for_tickers(
        self,
        table: str,
        tickers_file: str,
        ticker_field: str = "ticker",
        start: str = "",
        end: str = "",
        date_field: str = "date",
        out_dir: str = None,
        filters: str = "",
        columns: str = "",
        skip_existing: bool = True,
        max_tickers: int = None,
    ) -> str:
        """
        Download any SHARADAR table per ticker (one CSV per symbol).
        Useful for extending beyond SEP/SFP (e.g. insider/institution tables).
        """
        t = str(table).strip().upper()
        if out_dir is None:
            out_dir = str(self.out_dir.joinpath(f"raw/{t.lower()}"))
        out_path = Path(out_dir).expanduser().resolve()
        _ensure_dir(out_path)

        tickers = _read_ticker_list(Path(tickers_file))
        if max_tickers is not None:
            tickers = tickers[: int(max_tickers)]
        common_filters = _parse_key_value_csv(filters)
        cols = self._normalize_columns(columns)
        logger.info(f"Downloading {t} for {len(tickers)} tickers -> {out_path}")

        for idx, ticker in enumerate(tickers, 1):
            fp = out_path.joinpath(f"{ticker}.csv")
            if skip_existing and fp.exists():
                continue
            params = dict(common_filters)
            params[str(ticker_field)] = ticker
            if start:
                params[f"{date_field}.gte"] = str(start)
            if end:
                params[f"{date_field}.lte"] = str(end)
            if cols:
                params["qopts.columns"] = cols
            try:
                self._paginate_to_csv(t, fp, params=params)
            except Exception as e:
                logger.warning(f"{t} download failed for {ticker}: {e}")
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return str(out_path)

    def download_from_map(
        self,
        map_file: str,
        tickers_file: str = "",
        start: str = "",
        end: str = "",
        only: str = "",
        skip_existing: bool = True,
        max_tickers: int = None,
        report_file: str = None,
    ) -> str:
        """
        Download tables from a bundle map YAML.

        YAML format example:
          tables:
            - name: equity_prices
              table: SEP
              mode: per_ticker
              ticker_field: ticker
              date_field: date
              out_dir: raw/sep
              filters: "dimension=MRQ"
        """
        map_path = Path(map_file).expanduser().resolve()
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")
        with map_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML structure: {map_path}")
        entries = cfg.get("tables", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"No `tables` list found in map: {map_path}")

        only_set = set([x.lower() for x in _parse_csv_list(only)])
        report = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("table") or "<unknown>")
            if only_set and name.lower() not in only_set:
                continue
            table = str(item.get("table", "")).strip().upper()
            mode = str(item.get("mode", "full")).strip().lower()
            if not table:
                report.append({"name": name, "table": "", "status": "ERROR", "detail": "missing table"})
                continue

            try:
                if mode == "full":
                    out_file = item.get("out_file")
                    if out_file:
                        out_file = str(self.out_dir.joinpath(str(out_file)))
                    self.download_table(
                        table=table,
                        out_file=out_file,
                        filters=str(item.get("filters", "")),
                        columns=str(item.get("columns", "")),
                        skip_existing=skip_existing,
                    )
                elif mode == "per_ticker":
                    if not tickers_file:
                        raise ValueError(f"tickers_file required for per_ticker mode ({name})")
                    out_dir = item.get("out_dir")
                    if out_dir:
                        out_dir = str(self.out_dir.joinpath(str(out_dir)))
                    entry_max_tickers = item.get("max_tickers", None)
                    if entry_max_tickers is None or str(entry_max_tickers).strip() == "":
                        entry_max_tickers = max_tickers
                    else:
                        entry_max_tickers = int(entry_max_tickers)
                    self.download_table_for_tickers(
                        table=table,
                        tickers_file=tickers_file,
                        ticker_field=str(item.get("ticker_field", "ticker")),
                        date_field=str(item.get("date_field", "date")),
                        start=start,
                        end=end,
                        out_dir=out_dir,
                        filters=str(item.get("filters", "")),
                        columns=str(item.get("columns", "")),
                        skip_existing=skip_existing,
                        max_tickers=entry_max_tickers,
                    )
                else:
                    raise ValueError(f"unsupported mode: {mode}")
                report.append({"name": name, "table": table, "status": "OK", "detail": mode})
            except Exception as e:
                logger.warning(f"Map download failed for {name}({table}): {e}")
                report.append({"name": name, "table": table, "status": "ERROR", "detail": str(e)})

        if report_file is None:
            report_file = str(self.out_dir.joinpath("reports/bundle_download_report.csv"))
        report_path = Path(report_file).expanduser().resolve()
        _ensure_dir(report_path.parent)
        pd.DataFrame(report).to_csv(report_path, index=False)
        logger.info(f"Saved bundle download report -> {report_path}")
        return str(report_path)

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
            last_date = _read_last_date_from_csv_col(out_file, "date") or _read_last_date_from_csv(out_file)
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
                    df_new = pd.read_csv(tmp_file, low_memory=False)
                    _upsert_csv_range(out_file, df_new, date_col="date", start_date=start_date)
                except Exception as e:
                    logger.warning(f"SEP merge failed for {ticker}: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return str(sep_dir)

    def update_sfp(
        self,
        tickers_file: str,
        sfp_dir: str = None,
        days_back: int = 5,
        fallback_start: str = "2000-01-01",
    ) -> str:
        """
        Incrementally update SFP files by pulling the last N days per ticker.
        """
        if sfp_dir is None:
            sfp_dir = str(self.out_dir.joinpath("raw/sfp"))
        sfp_dir = Path(sfp_dir).expanduser().resolve()
        _ensure_dir(sfp_dir)
        tickers = _read_ticker_list(Path(tickers_file))
        logger.info(f"Updating SFP for {len(tickers)} tickers -> {sfp_dir}")
        for idx, ticker in enumerate(tickers, 1):
            out_file = sfp_dir.joinpath(f"{ticker}.csv")
            last_date = _read_last_date_from_csv_col(out_file, "date") or _read_last_date_from_csv(out_file)
            if not last_date:
                params = {"ticker": ticker, "date.gte": fallback_start}
                try:
                    self._paginate_to_csv("SFP", out_file, params=params)
                except Exception as e:
                    logger.warning(f"SFP update failed for {ticker}: {e}")
                continue
            try:
                start_date = (pd.Timestamp(last_date) - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
            except Exception:
                start_date = last_date
            tmp_file = sfp_dir.joinpath(f".{ticker}.update.tmp.csv")
            params = {"ticker": ticker, "date.gte": start_date}
            try:
                self._paginate_to_csv("SFP", tmp_file, params=params)
            except Exception as e:
                logger.warning(f"SFP update failed for {ticker}: {e}")
                if tmp_file.exists():
                    tmp_file.unlink()
                continue
            if tmp_file.exists() and tmp_file.stat().st_size > 0:
                try:
                    df_new = pd.read_csv(tmp_file, low_memory=False)
                    _upsert_csv_range(out_file, df_new, date_col="date", start_date=start_date)
                except Exception as e:
                    logger.warning(f"SFP merge failed for {ticker}: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 200 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return str(sfp_dir)

    def update_table_for_tickers(
        self,
        table: str,
        tickers_file: str,
        *,
        out_dir: str = None,
        ticker_field: str = "ticker",
        date_field: str = "date",
        days_back: int = 30,
        fallback_start: str = "2010-01-01",
        filters: str = "",
        columns: str = "",
        max_tickers: int = None,
    ) -> str:
        """
        Incrementally update a per-ticker datatable by overwriting rows >= (last_date - days_back).
        This is safer than pure append because it can absorb late corrections.
        """
        t = str(table).strip().upper()
        if not t:
            raise ValueError("table cannot be empty")
        if out_dir is None:
            out_dir = str(self.out_dir.joinpath(f"raw/{t.lower()}"))
        out_path = Path(out_dir).expanduser().resolve()
        _ensure_dir(out_path)

        tickers = _read_ticker_list(Path(tickers_file))
        if max_tickers is not None:
            tickers = tickers[: int(max_tickers)]

        common_filters = _parse_key_value_csv(filters)
        cols = self._normalize_columns(columns)

        logger.info(
            f"Updating {t} for {len(tickers)} tickers -> {out_path} (date_field={date_field}, days_back={days_back})"
        )
        for idx, ticker in enumerate(tickers, 1):
            out_file = out_path.joinpath(f"{ticker}.csv")
            last_date = _read_last_date_from_csv_col(out_file, date_field)
            if not last_date:
                start_date = fallback_start
            else:
                try:
                    start_date = (pd.Timestamp(last_date) - pd.Timedelta(days=int(days_back))).strftime("%Y-%m-%d")
                except Exception:
                    start_date = last_date

            tmp_file = out_path.joinpath(f".{ticker}.update.tmp.csv")
            params = dict(common_filters)
            params[str(ticker_field)] = ticker
            params[f"{date_field}.gte"] = start_date
            if cols:
                params["qopts.columns"] = cols
            try:
                self._paginate_to_csv(t, tmp_file, params=params)
            except Exception as e:
                logger.warning(f"{t} update failed for {ticker}: {e}")
                if tmp_file.exists():
                    tmp_file.unlink()
                continue

            if tmp_file.exists() and tmp_file.stat().st_size > 0:
                try:
                    df_new = pd.read_csv(tmp_file, low_memory=False)
                    _upsert_csv_range(out_file, df_new, date_col=date_field, start_date=start_date)
                except Exception as e:
                    logger.warning(f"{t} merge failed for {ticker}: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            if self.sleep:
                time.sleep(self.sleep)
            if idx % 200 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} tickers")
        return str(out_path)

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
            cursor = ""
            page = 1
            while True:
                params = {
                    "ticker": ",".join(batch),
                    "dimension": dimension,
                    "qopts.per_page": str(self.per_page),
                }
                if cursor:
                    params["qopts.cursor_id"] = cursor
                try:
                    payload = self._get("SF1", params)
                except Exception as e:
                    logger.warning(f"SF1 batch failed ({i}-{i+batch_size}) page={page}: {e}")
                    break

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

                logger.info(f"SF1 batch {i}-{i+len(batch)} page={page} rows={len(rows)}")
                cursor = _extract_meta_cursor(payload)
                if not cursor or not rows:
                    break
                page += 1
                if self.sleep:
                    time.sleep(self.sleep)
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
