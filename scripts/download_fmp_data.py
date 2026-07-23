#!/usr/bin/env python
"""Download and audit selected FMP external-data endpoints.

The script intentionally reads the API key from an environment variable and
never writes it to manifests, URLs, or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


BASE_URL = "https://financialmodelingprep.com/stable"
DEFAULT_OUT_ROOT = "/root/.qlib/fmp"
PROVENANCE_VERSION = 1


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    symbol_scoped: bool = True
    default_params: Mapping[str, Any] = None  # type: ignore[assignment]
    date_params: Tuple[str, str] = ("from", "to")

    def params(self) -> Dict[str, Any]:
        return dict(self.default_params or {})


ENDPOINTS: Dict[str, EndpointSpec] = {
    "analyst_estimates_annual": EndpointSpec(
        "analyst-estimates",
        default_params={"period": "annual", "page": 0, "limit": 1000},
    ),
    "analyst_estimates_quarter": EndpointSpec(
        "analyst-estimates",
        default_params={"period": "quarter", "page": 0, "limit": 1000},
    ),
    "earnings": EndpointSpec("earnings"),
    "earnings_calendar": EndpointSpec("earnings-calendar", symbol_scoped=False),
    "grades": EndpointSpec("grades"),
    "grades_historical": EndpointSpec("grades-historical", default_params={"page": 0, "limit": 1000}),
    "price_target_summary": EndpointSpec("price-target-summary"),
    "price_target_consensus": EndpointSpec("price-target-consensus"),
    "price_target_news": EndpointSpec("price-target-news", default_params={"page": 0, "limit": 1000}),
}


DATE_COL_CANDIDATES = (
    "date",
    "publishedDate",
    "acceptedDate",
    "fillingDate",
    "filingDate",
    "reportDate",
    "epsDate",
    "calendarDate",
    "period",
    "fiscalDateEnding",
    "updatedFromDate",
    "updatedToDate",
)

HEADER_SYMBOLS = {"TICKER", "SYMBOL"}

PIT_REQUIRED_DATASETS = {
    "analyst_estimates_annual",
    "analyst_estimates_quarter",
    "earnings",
    "earnings_calendar",
}

PIT_ASOF_COL_CANDIDATES = (
    "publishedDate",
    "acceptedDate",
    "updatedDate",
    "lastUpdated",
    "filingDate",
    "fillingDate",
    "date",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_csv_list(value: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(value or "").replace("\n", ",").split(","):
        token = item.strip()
        if not token:
            continue
        if token not in seen:
            out.append(token)
            seen.add(token)
    return out


def canonical_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def fmp_request_symbol(symbol: str) -> str:
    # FMP commonly uses BRK-B/BF-B style class-share symbols, while qlib
    # instruments may use BRK.B/BF.B. Keep qlib symbol names in files/manifests.
    return canonical_symbol(symbol).replace(".", "-")


def safe_file_stem(value: str) -> str:
    cleaned = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "unknown"


def load_symbols(symbols: str = "", symbols_file: str = "") -> List[str]:
    raw = parse_csv_list(symbols)
    if symbols_file:
        path = Path(symbols_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"symbols file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "," in stripped:
                parts = parse_csv_list(stripped)
            else:
                parts = [stripped.split()[0]]
            if parts and canonical_symbol(parts[0]) in HEADER_SYMBOLS:
                continue
            raw.extend(parts)
    out: List[str] = []
    seen = set()
    for symbol in raw:
        s = canonical_symbol(symbol)
        if s and s not in HEADER_SYMBOLS and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def schema_hash(columns: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(c) for c in columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "historical", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        if payload:
            return [payload]
    return []


def build_url(path: str, params: Mapping[str, Any], api_key: str, *, base_url: str = BASE_URL) -> str:
    clean_params = {k: v for k, v in params.items() if v is not None and str(v) != ""}
    clean_params["apikey"] = api_key
    query = urllib.parse.urlencode(clean_params)
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}?{query}"


def redact_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    redacted = [(k, "<redacted>" if k.lower() == "apikey" else v) for k, v in query]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(redacted)))


def fetch_json(url: str, *, timeout: int, retries: int, sleep: float) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(int(retries) + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "qlib-fmp-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= int(retries):
                break
            time.sleep(float(sleep) * (attempt + 1))
    assert last_error is not None
    raise last_error


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dataframe_from_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(list(records))


def date_span(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    df = dataframe_from_records(records)
    spans: Dict[str, Dict[str, str]] = {}
    if df.empty:
        return spans
    for col in df.columns:
        if col not in DATE_COL_CANDIDATES:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        parsed = parsed.dropna()
        if parsed.empty:
            continue
        spans[col] = {
            "min": pd.Timestamp(parsed.min()).strftime("%Y-%m-%d"),
            "max": pd.Timestamp(parsed.max()).strftime("%Y-%m-%d"),
        }
    return spans


def audit_records(dataset: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    df = dataframe_from_records(records)
    columns = [str(c) for c in df.columns] if not df.empty else []
    date_spans = date_span(records)
    raw_asof_cols = [c for c in columns if c in PIT_ASOF_COL_CANDIDATES]
    asof_cols = list(raw_asof_cols)
    if dataset.startswith("analyst_estimates"):
        # In FMP's analyst-estimates endpoint, ``date`` is the fiscal period
        # endpoint date, not proof that the estimate snapshot was known then.
        asof_cols = [c for c in asof_cols if c != "date"]
    fiscal_cols = [c for c in columns if c.lower() in {"fiscalyear", "fiscalperiod", "period", "calendarDate".lower()}]
    warnings: List[str] = []
    if dataset in PIT_REQUIRED_DATASETS and not asof_cols:
        warnings.append(
            "no obvious as-of/publication timestamp column; do not use this dataset for revision features without further PIT proof"
        )
    if dataset.startswith("analyst_estimates") and not any(c in columns for c in ("numAnalystsRevenue", "numAnalystsEps")):
        warnings.append("analyst-count columns not found in sample")
    if dataset.startswith("analyst_estimates") and not any(c in columns for c in ("revenueAvg", "epsAvg", "ebitdaAvg", "netIncomeAvg")):
        warnings.append("estimate-value columns not found in sample")
    if dataset in {"earnings", "earnings_calendar"} and not any(c.lower() in {"epsactual", "epsestimated", "revenueactual", "revenueestimated"} for c in columns):
        warnings.append("expected earnings estimate/actual columns not found in sample")

    return {
        "rows": int(len(records)),
        "columns": columns,
        "column_count": int(len(columns)),
        "schema_hash": schema_hash(columns) if columns else "",
        "date_spans": date_spans,
        "asof_candidate_columns": asof_cols,
        "raw_date_candidate_columns": raw_asof_cols,
        "fiscal_candidate_columns": fiscal_cols,
        "warnings": warnings,
    }


def dataset_params(
    dataset: str,
    spec: EndpointSpec,
    *,
    symbol: str = "",
    start: str = "",
    end: str = "",
    limit: int = 1000,
) -> Dict[str, Any]:
    params = spec.params()
    if spec.symbol_scoped:
        params["symbol"] = fmp_request_symbol(symbol)
    else:
        from_param, to_param = spec.date_params
        params[from_param] = start
        params[to_param] = end
    if "limit" in params:
        params["limit"] = int(limit)
    return params


def download_dataset(
    dataset: str,
    spec: EndpointSpec,
    *,
    api_key: str,
    out_root: Path,
    symbols: Sequence[str],
    start: str,
    end: str,
    limit: int,
    timeout: int,
    retries: int,
    sleep: float,
    pages: int = 1,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets: Sequence[str] = symbols if spec.symbol_scoped else [""]
    for symbol in targets:
        params = dataset_params(dataset, spec, symbol=symbol, start=start, end=end, limit=limit)
        url = build_url(spec.path, params, api_key)
        target_name = safe_file_stem(symbol or f"{start}_{end}")
        raw_path = out_root / "raw" / dataset / f"{target_name}.json"
        started = utc_now_iso()
        status = "DRY_RUN"
        error = ""
        payload: Any = []
        records: List[Dict[str, Any]] = []
        request_urls: List[str] = []
        pages_fetched = 0
        try:
            if not dry_run:
                if "page" in params and int(pages) > 1:
                    records = []
                    seen = set()
                    for page in range(int(pages)):
                        page_params = dict(params)
                        page_params["page"] = page
                        page_url = build_url(spec.path, page_params, api_key)
                        request_urls.append(redact_url(page_url))
                        page_payload = fetch_json(page_url, timeout=timeout, retries=retries, sleep=sleep)
                        page_records = records_from_payload(page_payload)
                        pages_fetched += 1
                        added = 0
                        for rec in page_records:
                            fp = json.dumps(rec, sort_keys=True, default=str)
                            if fp in seen:
                                continue
                            seen.add(fp)
                            records.append(rec)
                            added += 1
                        if sleep:
                            time.sleep(float(sleep))
                        if not page_records or added == 0:
                            break
                    payload = records
                else:
                    request_urls.append(redact_url(url))
                    payload = fetch_json(url, timeout=timeout, retries=retries, sleep=sleep)
                    records = records_from_payload(payload)
                    pages_fetched = 1
                write_json(raw_path, payload)
                status = "OK"
            audit = audit_records(dataset, records)
        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            audit = audit_records(dataset, [])
            if not dry_run:
                write_json(raw_path, {"error": error, "url": redact_url(url), "created_utc": utc_now_iso()})
        rows.append(
            {
                "dataset": dataset,
                "path": spec.path,
                "symbol": symbol,
                "params": {k: v for k, v in params.items() if k.lower() != "apikey"},
                "raw_path": str(raw_path),
                "status": status,
                "error": error,
                "request_url": redact_url(url),
                "request_urls": request_urls,
                "pages_fetched": pages_fetched,
                "started_utc": started,
                "finished_utc": utc_now_iso(),
                "audit": audit,
            }
        )
        if sleep and not dry_run:
            time.sleep(float(sleep))
    return rows


def write_audit_csv(out: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for entry in entries:
        audit = entry.get("audit") if isinstance(entry.get("audit"), dict) else {}
        spans = audit.get("date_spans") if isinstance(audit, dict) else {}
        rows.append(
            {
                "dataset": entry.get("dataset", ""),
                "symbol": entry.get("symbol", ""),
                "status": entry.get("status", ""),
                "rows": audit.get("rows", 0) if isinstance(audit, dict) else 0,
                "column_count": audit.get("column_count", 0) if isinstance(audit, dict) else 0,
                "schema_hash": audit.get("schema_hash", "") if isinstance(audit, dict) else "",
                "asof_candidate_columns": ",".join(audit.get("asof_candidate_columns", [])) if isinstance(audit, dict) else "",
                "date_spans": json.dumps(spans, sort_keys=True),
                "warnings": "; ".join(audit.get("warnings", [])) if isinstance(audit, dict) else "",
                "error": entry.get("error", ""),
                "raw_path": entry.get("raw_path", ""),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download selected FMP raw endpoint data and write an audit manifest.")
    p.add_argument("--symbols", default="AAPL,MSFT,NVDA,TSLA,JPM,XOM,COST,AVGO")
    p.add_argument("--symbols_file", default="")
    p.add_argument("--max_symbols", type=int, default=0, help="Optional cap after de-duplication; useful for audits")
    p.add_argument("--datasets", default="analyst_estimates_annual,analyst_estimates_quarter,earnings,grades_historical,grades,price_target_summary,price_target_consensus")
    p.add_argument("--start", default="2022-01-01", help="Start date for date-range endpoints such as earnings_calendar")
    p.add_argument("--end", default="", help="End date for date-range endpoints; default is today UTC")
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--manifest_out", default="", help="Optional explicit manifest JSON path")
    p.add_argument("--audit_csv", default="", help="Optional explicit audit CSV path")
    p.add_argument("--api_key_env", default="FMP_API_KEY")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--pages", type=int, default=1, help="Max pages for paginated endpoints with a page parameter")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--base_url", default=BASE_URL)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global BASE_URL
    BASE_URL = str(args.base_url).rstrip("/")

    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key and not args.dry_run:
        print(f"ERROR: set {args.api_key_env} in the environment", file=sys.stderr)
        return 2

    symbols = load_symbols(args.symbols, args.symbols_file)
    if int(args.max_symbols) > 0:
        symbols = symbols[: int(args.max_symbols)]
    datasets = parse_csv_list(args.datasets)
    unknown = [d for d in datasets if d not in ENDPOINTS]
    if unknown:
        print(f"ERROR: unknown datasets: {unknown}; available={sorted(ENDPOINTS)}", file=sys.stderr)
        return 3
    if not symbols and any(ENDPOINTS[d].symbol_scoped for d in datasets):
        print("ERROR: at least one symbol is required for symbol-scoped datasets", file=sys.stderr)
        return 4

    out_root = Path(args.out_root).expanduser().resolve()
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else (
        out_root / "metadata" / f"fmp_download_manifest_{stamp}.json"
    )
    audit_csv_path = Path(args.audit_csv).expanduser().resolve() if args.audit_csv else (
        out_root / "metadata" / f"fmp_download_audit_{stamp}.csv"
    )

    entries: List[Dict[str, Any]] = []
    for dataset in datasets:
        entries.extend(
            download_dataset(
                dataset,
                ENDPOINTS[dataset],
                api_key=api_key,
                out_root=out_root,
                symbols=symbols,
                start=str(args.start),
                end=str(end),
                limit=int(args.limit),
                timeout=int(args.timeout),
                retries=int(args.retries),
                sleep=float(args.sleep),
                pages=max(1, int(args.pages)),
                dry_run=bool(args.dry_run),
            )
        )

    manifest = {
        "created_utc": utc_now_iso(),
        "provenance_version": PROVENANCE_VERSION,
        "base_url": BASE_URL,
        "api_key_env": str(args.api_key_env),
        "api_key_written": False,
        "symbols": symbols,
        "datasets": datasets,
        "start": str(args.start),
        "end": str(end),
        "out_root": str(out_root),
        "entries": entries,
    }
    if not args.dry_run:
        write_json(manifest_path, manifest)
        write_audit_csv(audit_csv_path, entries)
        print(f"manifest={manifest_path}")
        print(f"audit_csv={audit_csv_path}")
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))

    ok = sum(1 for e in entries if e["status"] == "OK")
    errors = sum(1 for e in entries if e["status"] == "ERROR")
    total_rows = sum(int((e.get("audit") or {}).get("rows", 0)) for e in entries)
    print(f"datasets={len(datasets)} symbols={len(symbols)} requests={len(entries)} ok={ok} errors={errors} total_rows={total_rows}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
