#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.us_sharadar_release_checks import release_decision, strategy_feasibility_check_rows

SF3A_PROVENANCE_FILE = "sharadar_sf3a_features.json"
SF1_PIT_PROVENANCE_FILE = "sharadar_sf1_pit.json"
SF1_RATIO_PROVENANCE_FILE = "sharadar_sf1_ratio_features.json"
MODEL_FEATURE_PROVENANCE_FILE = "sharadar_model_features.json"
FMP_FEATURE_PROVENANCE_FILE = "fmp_event_features.json"
SEC_FEATURE_PROVENANCE_FILE = "sec_event_features.json"
MODEL_FEATURE_PREFIXES = ("risk_", "mkt_", "meta_")
FMP_FEATURE_PREFIX = "fmp_"
SEC_FEATURE_PREFIX = "sec_"
SF1_RATIO_FIELDS = {
    "roe_q",
    "roa_q",
    "ebitda_margin_q",
    "fcf_margin_q",
    "leverage_q",
    "cash_assets_q",
    "capex_assets_q",
    "asset_turn_q",
    "div_yield_px_q",
    "earn_yield_q",
    "book_px_q",
    "fcf_yield_q",
    "marketcap_q",
    "log_marketcap_q",
}

BENCHMARK_TICKER_ALIASES = {
    "ICIC": "IXIC",
    "^IXIC": "IXIC",
    "NASDAQ": "IXIC",
    "NASDAQCOMPOSITE": "IXIC",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_pickle_compat(path: Path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if not str(getattr(e, "name", "")).startswith("numpy._core"):
            raise
        import importlib

        sys.modules.setdefault("numpy._core", np.core)
        for name in ("numeric", "multiarray", "umath"):
            sys.modules.setdefault(f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}"))
        return pd.read_pickle(path)


def _safe_get(d: dict, keys: Iterable[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _config_uses_prefix(data_handler_config: dict, prefix: str) -> bool:
    fields = []
    for key in ("extra_fields", "extra_names", "pit_fields"):
        val = data_handler_config.get(key, [])
        if isinstance(val, list):
            fields.extend([str(x) for x in val])
    prefix = prefix.lower()
    return any(prefix in x.lower() for x in fields)


def _config_uses_pit(data_handler_config: dict) -> bool:
    pit_fields = data_handler_config.get("pit_fields", [])
    if isinstance(pit_fields, list) and len(pit_fields) > 0:
        return True
    fields = []
    for key in ("extra_fields", "label"):
        val = data_handler_config.get(key, [])
        if isinstance(val, list):
            for item in val:
                if isinstance(item, (list, tuple)):
                    fields.extend(str(x) for x in item)
                else:
                    fields.append(str(item))
    lowered = [x.lower() for x in fields]
    return any("p($$" in x or "$$" in x for x in lowered)


def _config_uses_sf1_ratio_features(data_handler_config: dict) -> bool:
    fields = []
    for key in ("extra_fields", "extra_names"):
        val = data_handler_config.get(key, [])
        if isinstance(val, list):
            fields.extend(str(x).strip().lower().lstrip("$") for x in val)
    return any(field in SF1_RATIO_FIELDS for field in fields)


def _field_names_from_expressions(expressions: Iterable[object]) -> List[str]:
    names: List[str] = []
    for expr in expressions or []:
        text = str(expr)
        names.extend(m.group(1).strip().lower() for m in re.finditer(r"\$([A-Za-z0-9_]+)", text))
    return names


def _config_model_feature_names(data_handler_config: dict) -> List[str]:
    fields = data_handler_config.get("extra_fields", [])
    if not isinstance(fields, list):
        return []
    names = sorted(
        {
            name
            for name in _field_names_from_expressions(fields)
            if any(name.startswith(prefix) for prefix in MODEL_FEATURE_PREFIXES)
        }
    )
    return names


def _config_fmp_feature_names(data_handler_config: dict) -> List[str]:
    fields = data_handler_config.get("extra_fields", [])
    if not isinstance(fields, list):
        return []
    return sorted({name for name in _field_names_from_expressions(fields) if name.startswith(FMP_FEATURE_PREFIX)})


def _config_sec_feature_names(data_handler_config: dict) -> List[str]:
    fields = data_handler_config.get("extra_fields", [])
    if not isinstance(fields, list):
        return []
    return sorted({name for name in _field_names_from_expressions(fields) if name.startswith(SEC_FEATURE_PREFIX)})


def _config_uses_static_metadata_features(data_handler_config: dict) -> bool:
    return any(name.startswith("meta_") for name in _config_model_feature_names(data_handler_config))


def _market_etf_from_model_feature(field: str) -> Optional[str]:
    field = str(field).lower()
    if not field.startswith("mkt_"):
        return None
    rest = field[4:]
    if "_" not in rest:
        return None
    prefix = rest.split("_", 1)[0]
    if prefix in {"breadth", "dispersion"}:
        return None
    return prefix.upper()


def _parse_label_horizon(label_expr: str) -> Optional[int]:
    if not isinstance(label_expr, str):
        return None
    # common pattern: Ref($close, -11)/Ref($close, -1) - 1
    import re

    m = re.search(r"Ref\(\$close,\s*-(\d+)\)\s*/\s*Ref\(\$close,\s*-(\d+)\)", label_expr)
    if not m:
        return None
    n1 = int(m.group(1))
    n2 = int(m.group(2))
    if n1 <= n2:
        return None
    return n1 - n2


def _parse_label_horizons(label_cfg: object) -> List[int]:
    if not (isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list)):
        return []
    out = []
    for expr in label_cfg[0]:
        horizon = _parse_label_horizon(expr)
        if horizon is not None:
            out.append(int(horizon))
    return out


def _as_ts(val) -> pd.Timestamp:
    return pd.Timestamp(val) if val is not None else None


def _date_span_summary(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return "<missing>"
    return f"{start.date()} -> {end.date()} ({(end - start).days} days)"


def _get_calendar_span(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if start is None or end is None:
        return []
    from qlib.data.data import Cal

    return list(Cal.calendar(start_time=start, end_time=end, freq="day", future=False))


def _count_trade_days_between(calendar: List[pd.Timestamp], left_end: pd.Timestamp, right_start: pd.Timestamp) -> int:
    if left_end is None or right_start is None:
        return 0
    left_end = pd.Timestamp(left_end)
    right_start = pd.Timestamp(right_start)
    return sum(1 for dt in calendar if left_end < pd.Timestamp(dt) < right_start)


def _required_embargo_days(args: argparse.Namespace, label_horizon: Optional[int]) -> int:
    if args.embargo_days is not None:
        return max(0, int(args.embargo_days))
    return max(0, int(label_horizon or 0))


def _provenance_check_rows(provider_uri: str, data_handler_config: dict, args: argparse.Namespace) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    meta_dir = Path(provider_uri).expanduser().resolve() / "metadata"

    if args.require_sf3a_provenance and _config_uses_prefix(data_handler_config, "inst13f"):
        path = meta_dir / SF3A_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("sf3a_provenance_present", prov is not None, str(path)))
        if prov is not None:
            full_rebuild = bool(prov.get("full_rebuild"))
            lag_days = prov.get("availability_lag_days")
            lag_ok = isinstance(lag_days, (int, float)) and int(lag_days) >= int(args.sf3a_min_availability_lag_days)
            rows.append(("sf3a_full_rebuild", full_rebuild, f"full_rebuild={full_rebuild}"))
            rows.append(
                (
                    "sf3a_availability_lag",
                    lag_ok,
                    f"{lag_days} >= {int(args.sf3a_min_availability_lag_days)} calendar_days",
                )
            )

    if args.require_sf1_pit_provenance and _config_uses_pit(data_handler_config):
        path = meta_dir / SF1_PIT_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("sf1_pit_provenance_present", prov is not None, str(path)))
        if prov is not None:
            date_col = str(prov.get("date_col", ""))
            date_offset_days = prov.get("date_offset_days")
            dump_to_qlib = bool(prov.get("dump_to_qlib"))
            offset_ok = isinstance(date_offset_days, int) and int(date_offset_days) == 0
            rows.append(("sf1_pit_uses_publication_date", date_col.lower() == "datekey", f"date_col={date_col}"))
            rows.append(("sf1_pit_no_extra_datekey_offset", offset_ok, f"date_offset_days={date_offset_days}"))
            rows.append(("sf1_pit_dumped_to_qlib", dump_to_qlib, f"dump_to_qlib={dump_to_qlib}"))
    if args.require_sf1_pit_provenance and _config_uses_sf1_ratio_features(data_handler_config):
        path = meta_dir / SF1_RATIO_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("sf1_ratio_provenance_present", prov is not None, str(path)))
        if prov is not None:
            date_col = str(prov.get("date_col", ""))
            date_offset_days = prov.get("date_offset_days")
            dump_to_qlib = bool(prov.get("dump_to_qlib"))
            ratio_fields = set(map(str, prov.get("ratio_fields", []) or []))
            fields_ok = SF1_RATIO_FIELDS.issubset(ratio_fields)
            offset_ok = isinstance(date_offset_days, int) and int(date_offset_days) == 0
            rows.append(("sf1_ratio_uses_publication_date", date_col.lower() == "datekey", f"date_col={date_col}"))
            rows.append(("sf1_ratio_no_extra_datekey_offset", offset_ok, f"date_offset_days={date_offset_days}"))
            rows.append(("sf1_ratio_dumped_to_qlib", dump_to_qlib, f"dump_to_qlib={dump_to_qlib}"))
            rows.append(("sf1_ratio_fields_complete", fields_ok, f"fields={len(ratio_fields)}/{len(SF1_RATIO_FIELDS)}"))

    model_feature_names = _config_model_feature_names(data_handler_config)
    if getattr(args, "require_model_feature_provenance", False) and model_feature_names:
        path = meta_dir / MODEL_FEATURE_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("model_feature_provenance_present", prov is not None, str(path)))
        if prov is not None:
            dump_to_qlib = bool(prov.get("dump_to_qlib"))
            feature_fields = {str(x).lower() for x in prov.get("feature_fields", []) or []}
            missing_fields = sorted(set(model_feature_names) - feature_fields)
            prepared_files = prov.get("prepared_csv_files")
            bins_written = prov.get("feature_bins_written")
            rows.append(("model_feature_dumped_to_qlib", dump_to_qlib, f"dump_to_qlib={dump_to_qlib}"))
            rows.append(
                (
                    "model_feature_fields_complete",
                    len(missing_fields) == 0,
                    f"required={len(model_feature_names)}, missing={missing_fields[:8]}",
                )
            )
            rows.append(
                (
                    "model_feature_prepared_files",
                    isinstance(prepared_files, int) and int(prepared_files) > 0,
                    f"prepared_csv_files={prepared_files}",
                )
            )
            rows.append(
                (
                    "model_feature_bins_written",
                    isinstance(bins_written, int) and int(bins_written) > 0,
                    f"feature_bins_written={bins_written}",
                )
            )
            counts = prov.get("market_etf_close_counts", {}) or {}
            required_etfs = sorted(
                {
                    etf
                    for etf in (_market_etf_from_model_feature(name) for name in model_feature_names)
                    if etf is not None
                }
            )
            missing_etfs = [etf for etf in required_etfs if int(counts.get(etf, 0) or 0) <= 0]
            if required_etfs:
                rows.append(
                    (
                        "model_feature_market_etf_sources",
                        len(missing_etfs) == 0,
                        f"required_etfs={required_etfs}, missing_or_empty={missing_etfs}",
                    )
                )
            uses_meta = any(name.startswith("meta_") for name in model_feature_names)
            meta_allowed = bool(getattr(args, "allow_static_metadata_features", False))
            rows.append(
                (
                    "model_feature_static_metadata_allowed",
                    (not uses_meta) or meta_allowed,
                    f"uses_meta={uses_meta}, allow_static_metadata_features={meta_allowed}",
                )
            )

    fmp_feature_names = _config_fmp_feature_names(data_handler_config)
    if getattr(args, "require_fmp_feature_provenance", False) and fmp_feature_names:
        path = meta_dir / FMP_FEATURE_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("fmp_feature_provenance_present", prov is not None, str(path)))
        if prov is not None:
            dump_to_qlib = bool(prov.get("dump_to_qlib"))
            lag_days = prov.get("availability_lag_days")
            lag_ok = isinstance(lag_days, (int, float)) and int(lag_days) >= int(
                getattr(args, "fmp_feature_min_availability_lag_days", 1)
            )
            feature_columns = {str(x).lower() for x in prov.get("feature_columns", []) or []}
            missing_fields = sorted(set(fmp_feature_names) - feature_columns)
            tickers = prov.get("tickers")
            bins_written = prov.get("feature_bins_written")
            excluded = {str(x) for x in prov.get("excluded_datasets", []) or []}
            required_exclusions = {
                "analyst_estimates_annual",
                "analyst_estimates_quarter",
                "price_target_summary",
                "price_target_consensus",
            }
            directional_required = sorted(name for name in fmp_feature_names if name.startswith("fmp_alpha_"))
            rows.append(("fmp_feature_dumped_to_qlib", dump_to_qlib, f"dump_to_qlib={dump_to_qlib}"))
            rows.append(
                (
                    "fmp_feature_availability_lag",
                    lag_ok,
                    f"{lag_days} >= {int(getattr(args, 'fmp_feature_min_availability_lag_days', 1))} calendar_days",
                )
            )
            rows.append(
                (
                    "fmp_feature_fields_complete",
                    len(missing_fields) == 0,
                    f"required={len(fmp_feature_names)}, missing={missing_fields[:8]}",
                )
            )
            rows.append(
                (
                    "fmp_feature_tickers",
                    isinstance(tickers, int) and int(tickers) > 0,
                    f"tickers={tickers}",
                )
            )
            rows.append(
                (
                    "fmp_feature_bins_written",
                    isinstance(bins_written, int) and int(bins_written) > 0,
                    f"feature_bins_written={bins_written}",
                )
            )
            rows.append(
                (
                    "fmp_feature_current_only_inputs_excluded",
                    required_exclusions.issubset(excluded),
                    f"excluded={sorted(excluded)}",
                )
            )
            if directional_required:
                try:
                    directional_version = int(prov.get("directional_feature_version", 0) or 0)
                except (TypeError, ValueError):
                    directional_version = 0
                directional_columns = {str(x).lower() for x in prov.get("directional_feature_columns", []) or []}
                missing_directional = sorted(set(directional_required) - directional_columns)
                classification = prov.get("dataset_classification", {}) or {}
                safe_class = set(map(str, classification.get("pit_safe_event_history", []) or []))
                excluded_class = set(map(str, classification.get("excluded_current_only_or_unproven_asof", []) or []))
                rows.append(
                    (
                        "fmp_directional_feature_version",
                        directional_version >= 2,
                        f"{directional_version} >= 2",
                    )
                )
                rows.append(
                    (
                        "fmp_directional_fields_listed",
                        len(missing_directional) == 0,
                        f"required={len(directional_required)}, missing={missing_directional[:8]}",
                    )
                )
                rows.append(
                    (
                        "fmp_dataset_classification_present",
                        bool(safe_class) and required_exclusions.issubset(excluded_class),
                        f"pit_safe={sorted(safe_class)}, excluded={sorted(excluded_class)}",
                    )
                )

    sec_feature_names = _config_sec_feature_names(data_handler_config)
    if getattr(args, "require_sec_feature_provenance", False) and sec_feature_names:
        path = meta_dir / SEC_FEATURE_PROVENANCE_FILE
        prov = _read_json(path)
        rows.append(("sec_feature_provenance_present", prov is not None, str(path)))
        if prov is not None:
            dump_to_qlib = bool(prov.get("dump_to_qlib"))
            lag_days = prov.get("availability_lag_days")
            lag_ok = isinstance(lag_days, (int, float)) and int(lag_days) >= int(
                getattr(args, "sec_feature_min_availability_lag_days", 1)
            )
            feature_columns = {str(x).lower() for x in prov.get("feature_columns", []) or []}
            missing_fields = sorted(set(sec_feature_names) - feature_columns)
            tickers = prov.get("tickers")
            bins_written = prov.get("feature_bins_written")
            directional_required = sorted(name for name in sec_feature_names if name.startswith("sec_alpha_"))
            directional_columns = {str(x).lower() for x in prov.get("directional_feature_columns", []) or []}
            missing_directional = sorted(set(directional_required) - directional_columns)
            rows.append(("sec_feature_dumped_to_qlib", dump_to_qlib, f"dump_to_qlib={dump_to_qlib}"))
            rows.append(
                (
                    "sec_feature_availability_lag",
                    lag_ok,
                    f"{lag_days} >= {int(getattr(args, 'sec_feature_min_availability_lag_days', 1))} calendar_days",
                )
            )
            rows.append(
                (
                    "sec_feature_fields_complete",
                    len(missing_fields) == 0,
                    f"required={len(sec_feature_names)}, missing={missing_fields[:8]}",
                )
            )
            rows.append(
                (
                    "sec_feature_tickers",
                    isinstance(tickers, int) and int(tickers) > 0,
                    f"tickers={tickers}",
                )
            )
            rows.append(
                (
                    "sec_feature_bins_written",
                    isinstance(bins_written, int) and int(bins_written) > 0,
                    f"feature_bins_written={bins_written}",
                )
            )
            if directional_required:
                try:
                    directional_version = int(prov.get("directional_feature_version", 0) or 0)
                except (TypeError, ValueError):
                    directional_version = 0
                rows.append(("sec_directional_feature_version", directional_version >= 1, f"{directional_version} >= 1"))
                rows.append(
                    (
                        "sec_directional_fields_listed",
                        len(missing_directional) == 0,
                        f"required={len(directional_required)}, missing={missing_directional[:8]}",
                    )
                )
    return rows


def _manifest_check_rows(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    label_horizon: Optional[int],
    pred: Optional[pd.DataFrame],
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    if not manifest_path.exists():
        return [("walkforward_manifest_present", False, str(manifest_path))]
    manifest = _read_json(manifest_path)
    if manifest is None:
        return [("walkforward_manifest_valid_json", False, str(manifest_path))]

    tasks = manifest.get("tasks", [])
    tasks_ok = isinstance(tasks, list) and len(tasks) > 0
    rows.append(("walkforward_manifest_tasks_present", tasks_ok, f"tasks={len(tasks) if isinstance(tasks, list) else 'invalid'}"))
    if not tasks_ok:
        return rows

    req_embargo = _required_embargo_days(args, label_horizon)
    manifest_embargo = manifest.get("embargo_days")
    try:
        manifest_embargo_int = int(manifest_embargo)
        manifest_embargo_ok = manifest_embargo_int >= req_embargo
        manifest_embargo_detail = f"{manifest_embargo_int} >= required={req_embargo}"
    except Exception:
        manifest_embargo_ok = False
        manifest_embargo_detail = f"missing/invalid embargo_days={manifest_embargo!r}"
    rows.append(("walkforward_manifest_embargo_recorded", manifest_embargo_ok, manifest_embargo_detail))

    parsed = []
    for i, task in enumerate(tasks):
        try:
            train = task["train"]
            valid = task["valid"]
            test = task["test"]
            parsed.append(
                (
                    i,
                    pd.Timestamp(train[0]),
                    pd.Timestamp(train[1]),
                    pd.Timestamp(valid[0]),
                    pd.Timestamp(valid[1]),
                    pd.Timestamp(test[0]),
                    pd.Timestamp(test[1]),
                )
            )
        except Exception:
            rows.append((f"walkforward_task_{i}_segments_parse", False, str(task)))
    if not parsed:
        return rows

    cal = _get_calendar_span(min(x[1] for x in parsed), max(x[6] for x in parsed))
    train_valid_gaps = []
    valid_test_gaps = []
    temporal_ok = True
    for _, train_start, train_end, valid_start, valid_end, test_start, test_end in parsed:
        temporal_ok = temporal_ok and train_start <= train_end < valid_start <= valid_end < test_start <= test_end
        train_valid_gaps.append(_count_trade_days_between(cal, train_end, valid_start))
        valid_test_gaps.append(_count_trade_days_between(cal, valid_end, test_start))

    rows.append(("walkforward_tasks_temporal_order", temporal_ok, f"tasks={len(parsed)}"))
    rows.append(
        (
            "walkforward_train_valid_embargo_min",
            min(train_valid_gaps) >= req_embargo,
            f"min_gap={min(train_valid_gaps)} >= required={req_embargo}",
        )
    )
    rows.append(
        (
            "walkforward_valid_test_embargo_min",
            min(valid_test_gaps) >= req_embargo,
            f"min_gap={min(valid_test_gaps)} >= required={req_embargo}",
        )
    )
    windows = sorted((x[5].normalize(), x[6].normalize()) for x in parsed)
    overlap_count = sum(1 for i in range(1, len(windows)) if windows[i][0] <= windows[i - 1][1])
    rows.append(("walkforward_test_windows_non_overlapping", overlap_count == 0, f"overlaps={overlap_count}"))

    if pred is not None:
        score = _extract_score_series(pred)
        if isinstance(score.index, pd.MultiIndex):
            dt_level = "datetime" if "datetime" in score.index.names else 0
            pred_dates = pd.DatetimeIndex(score.index.get_level_values(dt_level)).normalize().unique()
            uncovered = [
                d for d in pred_dates if not any(start <= pd.Timestamp(d) <= end for start, end in windows)
            ]
            empty_windows = [
                (start, end)
                for start, end in windows
                if not any(start <= pd.Timestamp(d) <= end for d in pred_dates)
            ]
            rows.append(
                (
                    "walkforward_pred_dates_within_manifest_tests",
                    len(uncovered) == 0,
                    f"uncovered_dates={len(uncovered)}",
                )
            )
            rows.append(
                (
                    "walkforward_manifest_tests_have_predictions",
                    len(empty_windows) == 0,
                    f"empty_test_windows={len(empty_windows)}",
                )
            )
    return rows


def _read_mlflow_meta_name(meta_path: Path) -> str:
    try:
        txt = meta_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    for line in txt.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _normalize_mlruns_uri(uri: str) -> str:
    text = str(uri or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    return str(Path(text).expanduser().resolve())


def _mlruns_local_path(uri: str) -> Optional[Path]:
    text = str(uri or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    if not parsed.scheme:
        return Path(text).expanduser().resolve()
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return None


def _mlruns_uri_from_config(cfg: dict, cfg_path: Path) -> str:
    uri = _safe_get(cfg, ["qlib_init", "exp_manager", "kwargs", "uri"], "")
    if uri:
        return _normalize_mlruns_uri(str(uri))
    return str((cfg_path.resolve().parents[3] / "mlruns").resolve())


def _find_mlflow_experiment_dir(mlruns_uri: Path, exp_name: str) -> Optional[Path]:
    if not mlruns_uri.exists():
        return None
    for child in sorted(mlruns_uri.iterdir()):
        if not child.is_dir():
            continue
        meta = child / "meta.yaml"
        if meta.exists() and _read_mlflow_meta_name(meta) == exp_name:
            return child
    return None


def _read_metric_points(metric_path: Path) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    try:
        lines = metric_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return points
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            points.append((int(parts[2]), float(parts[1])))
        except Exception:
            continue
    return points


def _metric_higher_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return lowered.startswith("ndcg") or lowered.startswith("auc") or "ndcg" in lowered


def _best_metric_iteration(run_dir: Path) -> Optional[Tuple[str, int, float, int, float]]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return None
    metric_names = sorted(p.name for p in metrics_dir.iterdir() if p.is_file() and p.name.endswith(".valid"))
    if not metric_names:
        return None
    if "l2.valid" in metric_names:
        metric_name = "l2.valid"
    else:
        metric_name = metric_names[0]
    points = _read_metric_points(metrics_dir / metric_name)
    if not points:
        return None
    return _best_metric_iteration_from_points(metric_name, points)


def _best_metric_iteration_from_points(
    metric_name: str,
    points: List[Tuple[int, float]],
) -> Optional[Tuple[str, int, float, int, float]]:
    if not points:
        return None
    higher = _metric_higher_is_better(metric_name)
    best_step, best_val = max(points, key=lambda x: x[1]) if higher else min(points, key=lambda x: x[1])
    last_step, last_val = points[-1]
    # MLflow steps are zero-based LightGBM callback iterations; user logs print one-based iterations.
    return metric_name, int(best_step) + 1, float(best_val), int(last_step) + 1, float(last_val)


def _best_metric_iteration_from_mlflow(client, run_id: str) -> Optional[Tuple[str, int, float, int, float]]:
    run = client.get_run(run_id)
    metric_names = sorted(str(name) for name in run.data.metrics.keys() if str(name).endswith(".valid"))
    if not metric_names:
        return None
    metric_name = "l2.valid" if "l2.valid" in metric_names else metric_names[0]
    history = client.get_metric_history(run_id, metric_name)
    points = [(int(point.step), float(point.value)) for point in history]
    return _best_metric_iteration_from_points(metric_name, points)


def _training_diagnostic_rows(
    manifest_path: Path,
    *,
    cfg: dict,
    cfg_path: Path,
    min_best_iteration: int,
) -> List[Tuple[str, bool, str]]:
    model_class = str(_safe_get(cfg, ["task", "model", "class"], "") or "").split(".")[-1]
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if model_class in {
        "FeatureWeightedScoreModel",
        "ICSelectedScoreModel",
        "RegimeSleeveScoreModel",
        "StackedSignalScoreModel",
    } and not (isinstance(manifest, dict) and bool(manifest.get("ensemble"))):
        return [
            (
                "training_non_iterative_model",
                True,
                f"model={model_class}; best-iteration diagnostics not applicable",
            )
        ]
    if not manifest_path.exists():
        return [("training_manifest_present", False, str(manifest_path))]
    if manifest is None:
        return [("training_manifest_valid_json", False, str(manifest_path))]
    exp_name = str(manifest.get("experiment") or "").strip()
    used_runs = manifest.get("used_runs", [])
    if bool(manifest.get("ensemble")):
        return _ensemble_training_diagnostic_rows(
            used_runs,
            default_mlruns_uri=_mlruns_uri_from_config(cfg, cfg_path),
            min_best_iteration=int(min_best_iteration),
        )
    if not exp_name or not isinstance(used_runs, list) or not used_runs:
        return [
            (
                "training_manifest_used_runs_present",
                False,
                f"experiment={exp_name!r}, used_runs={len(used_runs) if isinstance(used_runs, list) else 'invalid'}",
            )
        ]

    mlruns_uri = _mlruns_uri_from_config(cfg, cfg_path)
    mlruns_path = _mlruns_local_path(mlruns_uri)
    exp_dir: Optional[Path] = None
    mlflow_client = None
    mlflow_exp_id = None
    if mlruns_path is not None:
        exp_dir = _find_mlflow_experiment_dir(mlruns_path, exp_name)
        if exp_dir is None:
            return [("training_experiment_dir_found", False, f"{exp_name} under {mlruns_uri}")]
    else:
        try:
            import mlflow

            mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlruns_uri)
            exp = mlflow_client.get_experiment_by_name(exp_name)
            if exp is None:
                return [("training_experiment_found", False, f"{exp_name} under {mlruns_uri}")]
            mlflow_exp_id = str(exp.experiment_id)
        except Exception as e:
            return [("training_experiment_found", False, f"{exp_name} under {mlruns_uri}: {e!r}")]

    diagnostics = []
    missing = []
    for item in used_runs:
        run_id = str((item or {}).get("run_id") or "").strip() if isinstance(item, dict) else ""
        if not run_id:
            missing.append("<missing_run_id>")
            continue
        if exp_dir is not None:
            best = _best_metric_iteration(exp_dir / run_id)
        else:
            try:
                run = mlflow_client.get_run(run_id)
                if mlflow_exp_id is not None and str(run.info.experiment_id) != str(mlflow_exp_id):
                    best = None
                else:
                    best = _best_metric_iteration_from_mlflow(mlflow_client, run_id)
            except Exception:
                best = None
        if best is None:
            missing.append(run_id)
            continue
        metric_name, best_iter, best_val, last_iter, last_val = best
        diagnostics.append((run_id, metric_name, best_iter, best_val, last_iter, last_val))

    rows: List[Tuple[str, bool, str]] = []
    rows.append(
        (
            "training_metrics_present",
            len(missing) == 0 and len(diagnostics) == len(used_runs),
            f"metrics={len(diagnostics)}/{len(used_runs)}, missing={missing[:3]}",
        )
    )
    if diagnostics:
        best_iters = [x[2] for x in diagnostics]
        min_iter = min(best_iters)
        median_iter = float(np.median(best_iters))
        weak = [f"{run_id}:{best_iter}" for run_id, _, best_iter, _, _, _ in diagnostics if best_iter < min_best_iteration]
        rows.append(
            (
                "training_best_iteration_min",
                min_iter >= int(min_best_iteration),
                f"min={min_iter}, median={median_iter:.1f}, threshold={int(min_best_iteration)}, weak={weak[:5]}",
            )
        )
    return rows


def _ensemble_training_diagnostic_rows(
    used_runs: object,
    *,
    default_mlruns_uri: str,
    min_best_iteration: int,
) -> List[Tuple[str, bool, str]]:
    if not isinstance(used_runs, list) or not used_runs:
        return [("training_ensemble_used_runs_present", False, f"used_runs={len(used_runs) if isinstance(used_runs, list) else 'invalid'}")]

    non_iterative = {
        "FeatureWeightedScoreModel",
        "ICSelectedScoreModel",
        "RegimeSleeveScoreModel",
        "StackedSignalScoreModel",
    }
    rows: List[Tuple[str, bool, str]] = []
    diagnostics = []
    missing = []
    skipped = []
    exp_cache: Dict[Tuple[str, str], Optional[Path]] = {}
    client_cache = {}
    exp_id_cache = {}

    for item in used_runs:
        if not isinstance(item, dict):
            missing.append("<invalid_run>")
            continue
        run_id = str(item.get("run_id") or "").strip()
        member = str(item.get("member") or item.get("role") or "?")
        model_class = str(item.get("model_class") or "").split(".")[-1]
        if not run_id:
            missing.append(f"{member}:<missing_run_id>")
            continue
        if model_class in non_iterative:
            skipped.append(f"{member}:{run_id}:{model_class}")
            continue

        exp_name = str(item.get("experiment") or "").strip()
        mlruns_uri = str(item.get("mlruns_uri") or default_mlruns_uri or "").strip()
        if not exp_name or not mlruns_uri:
            missing.append(f"{member}:{run_id}:missing_experiment_or_uri")
            continue
        mlruns_path = _mlruns_local_path(mlruns_uri)
        best = None
        if mlruns_path is not None:
            cache_key = (str(mlruns_path), exp_name)
            if cache_key not in exp_cache:
                exp_cache[cache_key] = _find_mlflow_experiment_dir(mlruns_path, exp_name)
            exp_dir = exp_cache[cache_key]
            if exp_dir is None:
                missing.append(f"{member}:{run_id}:experiment_not_found")
                continue
            best = _best_metric_iteration(exp_dir / run_id)
        else:
            try:
                if mlruns_uri not in client_cache:
                    import mlflow

                    client_cache[mlruns_uri] = mlflow.tracking.MlflowClient(tracking_uri=mlruns_uri)
                client = client_cache[mlruns_uri]
                exp_cache_key = (mlruns_uri, exp_name)
                if exp_cache_key not in exp_id_cache:
                    exp = client.get_experiment_by_name(exp_name)
                    exp_id_cache[exp_cache_key] = None if exp is None else str(exp.experiment_id)
                exp_id = exp_id_cache[exp_cache_key]
                if exp_id is None:
                    missing.append(f"{member}:{run_id}:experiment_not_found")
                    continue
                run = client.get_run(run_id)
                if str(run.info.experiment_id) != str(exp_id):
                    missing.append(f"{member}:{run_id}:experiment_mismatch")
                    continue
                best = _best_metric_iteration_from_mlflow(client, run_id)
            except Exception:
                best = None
        if best is None:
            missing.append(f"{member}:{run_id}:metrics_missing")
            continue
        metric_name, best_iter, best_val, last_iter, last_val = best
        diagnostics.append((member, run_id, metric_name, best_iter, best_val, last_iter, last_val))

    rows.append(
        (
            "training_ensemble_non_iterative_runs",
            True,
            f"skipped={len(skipped)}",
        )
    )
    rows.append(
        (
            "training_metrics_present",
            len(missing) == 0 and (len(diagnostics) + len(skipped)) == len(used_runs),
            f"metrics={len(diagnostics)}, skipped_non_iterative={len(skipped)}, total={len(used_runs)}, missing={missing[:3]}",
        )
    )
    if diagnostics:
        best_iters = [x[3] for x in diagnostics]
        min_iter = min(best_iters)
        median_iter = float(np.median(best_iters))
        weak = [f"{member}:{run_id}:{best_iter}" for member, run_id, _, best_iter, _, _, _ in diagnostics if best_iter < min_best_iteration]
        rows.append(
            (
                "training_best_iteration_min",
                min_iter >= int(min_best_iteration),
                f"min={min_iter}, median={median_iter:.1f}, threshold={int(min_best_iteration)}, weak={weak[:5]}",
            )
        )
    return rows


def _count_active_instruments(inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], date: pd.Timestamp) -> int:
    total = 0
    for spans in inst_spans.values():
        for s, e in spans:
            if s <= date <= e:
                total += 1
                break
    return total


def _sample_month_starts(calendar: List[pd.Timestamp], max_months: int) -> List[pd.Timestamp]:
    if len(calendar) == 0:
        return []
    ser = pd.Series(calendar)
    grouped = ser.groupby(ser.dt.to_period("M")).first().sort_values()
    if max_months is not None and max_months > 0:
        grouped = grouped.iloc[:max_months]
    return grouped.tolist()


def _sample_instruments(inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], n: int, seed: int) -> List[str]:
    inst = sorted(inst_spans.keys())
    if not inst:
        return []
    if n is None or n <= 0 or n >= len(inst):
        return inst
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inst), size=n, replace=False)
    return [inst[i] for i in sorted(idx)]


def _sample_active_instruments(
    inst_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    date: pd.Timestamp,
    n: int,
    seed: int,
) -> List[str]:
    date = pd.Timestamp(date)
    inst = sorted(
        symbol
        for symbol, spans in inst_spans.items()
        if any(pd.Timestamp(start) <= date <= pd.Timestamp(end) for start, end in spans)
    )
    if not inst:
        return []
    if n is None or n <= 0 or n >= len(inst):
        return inst
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inst), size=n, replace=False)
    return [inst[i] for i in sorted(idx)]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join([str(x) for x in col]) for col in df.columns]
    return df


def _missing_ratio(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.isna().mean().sort_values(ascending=False)


def _max_constant_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return 0
    run = 1
    max_run = 1
    for i in range(1, len(vals)):
        if np.isclose(vals[i], vals[i - 1], rtol=1e-10, atol=1e-12):
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    max_run = max(max_run, run)
    return max_run


def _pit_staleness_days(df: pd.DataFrame, field: str) -> pd.Series:
    if df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    series = df[field]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if not isinstance(series.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    wide = series.unstack("instrument")
    res = {}
    for inst in wide.columns:
        vals = wide[inst].values
        res[inst] = _max_constant_run(vals)
    return pd.Series(res)


def _run_backtest(
    pred: pd.DataFrame,
    strategy_cfg: dict,
    start_time,
    end_time,
    benchmark,
    account=10000000,
    exchange_kwargs=None,
    cost_mult=1.0,
    deal_price="close",
):
    from qlib.backtest import backtest as normal_backtest

    merged_exchange_kwargs = {
        "limit_threshold": None,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }
    if isinstance(exchange_kwargs, dict):
        merged_exchange_kwargs.update(exchange_kwargs)
    if deal_price is not None:
        merged_exchange_kwargs["deal_price"] = deal_price
    for key in ("open_cost", "close_cost"):
        val = merged_exchange_kwargs.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            merged_exchange_kwargs[key] = float(val) * float(cost_mult)

    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    backtest_cfg = {
        "start_time": start_time,
        "end_time": end_time,
        "account": account,
        "benchmark": benchmark,
        "exchange_kwargs": merged_exchange_kwargs,
    }
    portfolio_metric_dict, _ = normal_backtest(
        executor=executor_cfg, strategy=strategy_cfg, **backtest_cfg
    )
    report, _ = portfolio_metric_dict["1day"]
    return report


def _slice_report(report: pd.DataFrame, start_time, end_time) -> pd.DataFrame:
    if report is None or report.empty:
        return pd.DataFrame()
    out = report.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    return out[(out.index >= start) & (out.index <= end)].copy()


def _warmup_start_for_window(
    calendar: List[pd.Timestamp],
    start_time,
    warmup_days: int,
) -> pd.Timestamp:
    start = pd.Timestamp(start_time).normalize()
    if int(warmup_days) <= 0:
        return start
    cal = pd.DatetimeIndex(calendar).normalize()
    if cal.empty:
        return start
    cal = pd.DatetimeIndex(sorted(cal.unique()))
    idx = int(np.searchsorted(cal.values, np.datetime64(start), side="left"))
    if idx >= len(cal):
        idx = len(cal) - 1
    if pd.Timestamp(cal[idx]).normalize() > start and idx > 0:
        idx -= 1
    warm_idx = max(0, idx - int(warmup_days))
    return pd.Timestamp(cal[warm_idx]).normalize()


def _run_backtest_eval_window(
    pred: pd.DataFrame,
    strategy_cfg: dict,
    eval_start,
    eval_end,
    benchmark,
    *,
    calendar: List[pd.Timestamp],
    warmup_days: int = 0,
    account=10000000,
    exchange_kwargs=None,
    cost_mult=1.0,
    deal_price="close",
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    warmup_start = _warmup_start_for_window(calendar, eval_start, int(warmup_days))
    report = _run_backtest(
        pred,
        strategy_cfg,
        warmup_start,
        eval_end,
        benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
        cost_mult=cost_mult,
        deal_price=deal_price,
    )
    return _slice_report(report, eval_start, eval_end), warmup_start


def _summarize_report(report: pd.DataFrame) -> Dict[str, float]:
    from qlib.contrib.evaluate import risk_analysis

    strat = risk_analysis(report["return"] - report["cost"], freq="1day")
    bench = risk_analysis(report["bench"], freq="1day")
    gross_excess = risk_analysis(report["return"] - report["bench"], freq="1day")
    excess = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="1day")
    out = {
        "ann_return": float(strat.loc["annualized_return", "risk"]),
        "ir": float(strat.loc["information_ratio", "risk"]),
        "mdd": float(strat.loc["max_drawdown", "risk"]),
        "bench_ann_return": float(bench.loc["annualized_return", "risk"]),
        "gross_excess_ann_return": float(gross_excess.loc["annualized_return", "risk"]),
        "excess_ann_return": float(excess.loc["annualized_return", "risk"]),
        "excess_ir": float(excess.loc["information_ratio", "risk"]),
    }
    if "turnover" in report.columns:
        out["avg_turnover"] = float(report["turnover"].mean())
    if "cost" in report.columns and "return" in report.columns:
        out["avg_cost"] = float(report["cost"].mean())
    out["n_days"] = int(len(report))
    return out


def _format_float(val: Optional[float], digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "n/a"
    return f"{val:.{digits}f}"


def _fmt_opt(val) -> str:
    return "<n/a>" if val is None else str(val)


def _summarize_benchmark(benchmark) -> str:
    if isinstance(benchmark, pd.Series):
        if benchmark.empty:
            return "series(len=0)"
        idx = pd.DatetimeIndex(benchmark.index)
        return f"series(len={len(benchmark)}, span={idx.min().date()}->{idx.max().date()})"
    if isinstance(benchmark, list):
        return "list(" + ",".join([str(x) for x in benchmark]) + ")"
    return str(benchmark)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


def _finite_float_or_nan(val) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _report_net_return_series(report: pd.DataFrame) -> pd.Series:
    if not isinstance(report, pd.DataFrame) or "return" not in report.columns:
        return pd.Series(dtype=float)
    ret = pd.to_numeric(report["return"], errors="coerce")
    if "cost" in report.columns:
        cost = pd.to_numeric(report["cost"], errors="coerce").reindex(ret.index).fillna(0.0)
    else:
        cost = 0.0
    out = (ret - cost).replace([np.inf, -np.inf], np.nan)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _summarize_return_series(returns: pd.Series) -> Dict[str, float]:
    from qlib.contrib.evaluate import risk_analysis

    ret = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = {"ann_return": float("nan"), "ir": float("nan"), "mdd": float("nan"), "n_days": int(len(ret))}
    if ret.empty:
        return out
    risk = risk_analysis(ret, freq="1day")
    out["ann_return"] = _finite_float_or_nan(risk.loc["annualized_return", "risk"])
    out["ir"] = _finite_float_or_nan(risk.loc["information_ratio", "risk"])
    out["mdd"] = _finite_float_or_nan(risk.loc["max_drawdown", "risk"])
    return out


def _aligned_strategy_baseline_frame(report: pd.DataFrame, baseline_returns: pd.Series) -> pd.DataFrame:
    strat = _report_net_return_series(report)
    bench = pd.to_numeric(baseline_returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bench.index = pd.DatetimeIndex(bench.index).normalize()
    bench = bench[~bench.index.duplicated(keep="last")].sort_index()
    frame = pd.DataFrame({"strategy": strat, "baseline": bench.reindex(strat.index)})
    return frame


def _baseline_metrics_from_frame(frame: pd.DataFrame) -> Dict[str, float]:
    missing_ratio = float(frame["baseline"].isna().mean()) if len(frame) else float("nan")
    aligned = frame.dropna(subset=["strategy", "baseline"])
    strategy_metrics = _summarize_return_series(aligned["strategy"])
    baseline_metrics = _summarize_return_series(aligned["baseline"])
    excess_metrics = _summarize_return_series(aligned["strategy"] - aligned["baseline"])
    strategy_mdd = _safe_float(strategy_metrics.get("mdd"))
    baseline_mdd = _safe_float(baseline_metrics.get("mdd"))
    mdd_gap = (
        abs(strategy_mdd) - abs(baseline_mdd)
        if strategy_mdd is not None and baseline_mdd is not None
        else float("nan")
    )
    return {
        "n_days": int(len(aligned)),
        "missing_ratio": missing_ratio,
        "strategy_ann_return": strategy_metrics.get("ann_return", float("nan")),
        "strategy_ir": strategy_metrics.get("ir", float("nan")),
        "strategy_mdd": strategy_metrics.get("mdd", float("nan")),
        "baseline_ann_return": baseline_metrics.get("ann_return", float("nan")),
        "baseline_ir": baseline_metrics.get("ir", float("nan")),
        "baseline_mdd": baseline_metrics.get("mdd", float("nan")),
        "excess_ann_return": excess_metrics.get("ann_return", float("nan")),
        "excess_ir": excess_metrics.get("ir", float("nan")),
        "mdd_gap": mdd_gap,
    }


def _baseline_period_metrics(report: pd.DataFrame, baseline_returns: pd.Series) -> Dict[str, float]:
    return _baseline_metrics_from_frame(_aligned_strategy_baseline_frame(report, baseline_returns))


def _rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    gross = (1.0 + returns).where(returns.notna())
    min_periods = max(2, min(int(window), max(2, int(window) // 3)))
    return gross.rolling(int(window), min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _evaluate_baseline_regime_gates(
    report: pd.DataFrame,
    baseline_returns: pd.Series,
    *,
    ticker: str,
    return_window: int,
    vol_window: int,
    strong_return_threshold: float,
    weak_return_threshold: float,
    min_days: int,
    min_up_day_excess_ann: float,
    min_down_day_excess_ann: float,
    min_strong_excess_ann: float,
    min_weak_excess_ann: float,
    min_high_vol_excess_ann: float,
    min_low_vol_excess_ann: float,
) -> Tuple[List[Tuple[str, str, Dict[str, float]]], List[Tuple[str, bool, str]]]:
    frame = _aligned_strategy_baseline_frame(report, baseline_returns)
    aligned = frame.dropna(subset=["strategy", "baseline"]).copy()
    if aligned.empty:
        return [], [(f"baseline_regime_{ticker}_available", False, "no aligned baseline returns")]

    baseline = aligned["baseline"]
    trailing_return = _rolling_compound_return(baseline, int(return_window))
    trailing_vol = baseline.rolling(int(vol_window), min_periods=max(2, min(int(vol_window), 5))).std(ddof=0)
    high_vol_cut = trailing_vol.quantile(0.75) if trailing_vol.notna().any() else np.nan
    low_vol_cut = trailing_vol.quantile(0.25) if trailing_vol.notna().any() else np.nan

    regime_defs = [
        ("up_days", baseline > 0.0, min_up_day_excess_ann),
        ("down_days", baseline < 0.0, min_down_day_excess_ann),
        (f"{int(return_window)}d_strong", trailing_return > float(strong_return_threshold), min_strong_excess_ann),
        (f"{int(return_window)}d_weak", trailing_return < float(weak_return_threshold), min_weak_excess_ann),
        (f"vol{int(vol_window)}_top_quartile", trailing_vol >= high_vol_cut, min_high_vol_excess_ann),
        (f"vol{int(vol_window)}_bottom_quartile", trailing_vol <= low_vol_cut, min_low_vol_excess_ann),
    ]

    rows: List[Tuple[str, str, Dict[str, float]]] = []
    checks: List[Tuple[str, bool, str]] = []
    for regime_name, mask, threshold in regime_defs:
        mask = pd.Series(mask, index=aligned.index).fillna(False)
        metrics = _baseline_metrics_from_frame(aligned.loc[mask])
        rows.append((ticker, regime_name, metrics))
        n_days = int(metrics.get("n_days") or 0)
        excess = _safe_float(metrics.get("excess_ann_return"))
        enough_days = n_days >= int(min_days)
        ok = enough_days and excess is not None and excess >= float(threshold)
        checks.append(
            (
                f"baseline_regime_{ticker}_{regime_name}_excess_ann",
                ok,
                f"{_format_float(excess)} >= {_format_float(threshold)} (n_days={n_days} >= {int(min_days)})",
            )
        )
    return rows, checks


def _load_qlib_ticker_returns(ticker: str, start_time, end_time, *, field: str = "$close") -> pd.Series:
    from qlib.data import D

    ticker = _coerce_ticker_list(ticker)[0] if _coerce_ticker_list(ticker) else str(ticker).strip().upper()
    features = D.features([ticker], [field], start_time=start_time, end_time=end_time)
    frame = _normalize_feature_frame(features, [field])
    if frame.empty or field not in frame.columns:
        return pd.Series(dtype=float, name=ticker)
    close = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if isinstance(close.index, pd.MultiIndex):
        if "datetime" in close.index.names:
            close.index = pd.DatetimeIndex(close.index.get_level_values("datetime")).normalize()
        else:
            close.index = pd.DatetimeIndex(close.index.get_level_values(0)).normalize()
        close = close.groupby(level=0).last()
    else:
        close.index = pd.DatetimeIndex(close.index).normalize()
    close = close.sort_index()
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret.name = ticker
    return ret


def _load_benchmark_return_pkl(path: Path, ticker: str, start_time, end_time) -> pd.Series:
    obj = _read_pickle_compat(path.expanduser().resolve())
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"baseline pkl must be a Series or single-column DataFrame: {path}")
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise ValueError(f"baseline pkl must be a pandas Series: {path}")
    ret = pd.to_numeric(obj, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ret.index = pd.DatetimeIndex(ret.index).normalize()
    ret = ret[~ret.index.duplicated(keep="last")].sort_index()
    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    ret = ret[(ret.index >= start) & (ret.index <= end)].dropna()
    ret.name = str(ticker).upper()
    return ret


def _default_baseline_pkl_path(ticker: str) -> Path:
    return Path("/root/.qlib/qlib_data/us_data") / f"bench_{str(ticker).strip().lower()}.pkl"


def _load_external_baseline_returns(
    ticker: str,
    start_time,
    end_time,
    *,
    pkl_map: Optional[Dict[str, Path]] = None,
) -> Tuple[pd.Series, str]:
    ticker = _coerce_ticker_list(ticker)[0] if _coerce_ticker_list(ticker) else str(ticker).strip().upper()
    pkl_map = pkl_map or {}
    if ticker in pkl_map:
        path = pkl_map[ticker]
        return _load_benchmark_return_pkl(path, ticker, start_time, end_time), f"pkl:{path.expanduser().resolve()}"
    default_pkl = _default_baseline_pkl_path(ticker)
    try:
        returns = _load_qlib_ticker_returns(ticker, start_time, end_time)
    except Exception:
        if default_pkl.exists():
            return _load_benchmark_return_pkl(default_pkl, ticker, start_time, end_time), f"pkl:{default_pkl.resolve()}"
        raise
    if returns.empty and default_pkl.exists():
        return _load_benchmark_return_pkl(default_pkl, ticker, start_time, end_time), f"pkl:{default_pkl.resolve()}"
    return returns, "qlib"


def _extract_score_series(pred: pd.DataFrame) -> pd.Series:
    if isinstance(pred, pd.Series):
        return pred
    if isinstance(pred, pd.DataFrame):
        if pred.shape[1] == 0:
            return pd.Series(dtype=float)
        return pred.iloc[:, 0]
    return pd.Series(dtype=float)


def _evaluate_strict_data_quality(
    pred: pd.DataFrame,
    *,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    bt_calendar: List[pd.Timestamp],
    benchmark,
    topk: int,
    min_daily_scores: int,
    min_daily_coverage: float,
    max_nan_score_ratio: float,
    max_missing_close_ratio: float,
    max_missing_volume_ratio: float,
    max_benchmark_nan_ratio: float,
) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    score = _extract_score_series(pred)
    if score.empty:
        checks.append(("pred_has_scores", False, "prediction score series is empty"))
        return checks
    if not isinstance(score.index, pd.MultiIndex) or score.index.nlevels < 2:
        checks.append(("pred_multiindex", False, "prediction score index must be MultiIndex(datetime, instrument)"))
        return checks

    dt_level = "datetime" if "datetime" in score.index.names else 0
    dt_index = pd.DatetimeIndex(score.index.get_level_values(dt_level))
    in_range = (dt_index >= bt_start) & (dt_index <= bt_end)
    score = score[in_range]
    if score.empty:
        checks.append(("pred_non_empty_in_backtest", False, f"no scores in {bt_start.date()}->{bt_end.date()}"))
        return checks

    dup_cnt = int(score.index.duplicated(keep=False).sum())
    checks.append(("pred_no_duplicate_index", dup_cnt == 0, f"duplicate_rows={dup_cnt}"))

    nan_ratio = float(score.isna().mean())
    checks.append(
        (
            "pred_nan_ratio",
            nan_ratio <= max_nan_score_ratio,
            f"{_format_float(nan_ratio)} <= {_format_float(max_nan_score_ratio)}",
        )
    )

    daily_non_na = score.dropna().groupby(level=dt_level).size()
    cal_idx = pd.DatetimeIndex(bt_calendar)
    covered_days = int((daily_non_na.reindex(cal_idx, fill_value=0) >= int(min_daily_scores)).sum())
    total_days = int(len(cal_idx))
    coverage = float(covered_days) / float(total_days) if total_days > 0 else 0.0
    checks.append(
        (
            "pred_daily_coverage",
            coverage >= min_daily_coverage,
            f"{covered_days}/{total_days}={_format_float(coverage)} >= {_format_float(min_daily_coverage)} (min_daily_scores={min_daily_scores})",
        )
    )

    selected_pairs = []
    selected_instruments = set()
    for dt, day_score in score.dropna().groupby(level=dt_level):
        if isinstance(day_score.index, pd.MultiIndex):
            day_inst = day_score.droplevel(dt_level)
        else:
            day_inst = day_score
        top = day_inst.sort_values(ascending=False).head(topk)
        for inst in top.index.tolist():
            selected_pairs.append((pd.Timestamp(dt), inst))
            selected_instruments.add(inst)

    selected_pairs = sorted(set(selected_pairs))
    if not selected_pairs:
        checks.append(("topk_data_quality_pairs", False, "no valid top-k prediction pairs found"))
    else:
        from qlib.data import D

        px = D.features(sorted(selected_instruments), ["$close", "$volume"], start_time=bt_start, end_time=bt_end)
        close = px["$close"] if "$close" in px.columns else px.iloc[:, 0]
        volume = px["$volume"] if "$volume" in px.columns else pd.Series(index=close.index, dtype=float)

        if isinstance(close.index, pd.MultiIndex) and "instrument" in close.index.names and "datetime" in close.index.names:
            if close.index.names[0] == "instrument":
                pair_index = pd.MultiIndex.from_tuples(
                    [(inst, dt) for dt, inst in selected_pairs], names=["instrument", "datetime"]
                )
            else:
                pair_index = pd.MultiIndex.from_tuples(selected_pairs, names=["datetime", "instrument"])
        else:
            pair_index = pd.MultiIndex.from_tuples(selected_pairs, names=["datetime", "instrument"])

        close_sel = close.reindex(pair_index)
        volume_sel = volume.reindex(pair_index)
        close_missing = float(close_sel.isna().mean())
        volume_missing = float(volume_sel.isna().mean())
        checks.append(
            (
                "topk_close_missing_ratio",
                close_missing <= max_missing_close_ratio,
                f"{_format_float(close_missing)} <= {_format_float(max_missing_close_ratio)} over {len(pair_index)} top-k pairs",
            )
        )
        checks.append(
            (
                "topk_volume_missing_ratio",
                volume_missing <= max_missing_volume_ratio,
                f"{_format_float(volume_missing)} <= {_format_float(max_missing_volume_ratio)} over {len(pair_index)} top-k pairs",
            )
        )

    if isinstance(benchmark, pd.Series):
        b = benchmark.copy()
        b.index = pd.DatetimeIndex(b.index)
        bench_nan_ratio = float(b.reindex(cal_idx).isna().mean()) if len(cal_idx) > 0 else 0.0
        checks.append(
            (
                "benchmark_nan_ratio",
                bench_nan_ratio <= max_benchmark_nan_ratio,
                f"{_format_float(bench_nan_ratio)} <= {_format_float(max_benchmark_nan_ratio)}",
            )
        )

    return checks


def _normalize_datetime_instrument_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.MultiIndex):
        return series
    names = list(series.index.names)
    if "datetime" in names and "instrument" in names:
        series = series.reorder_levels(["datetime", "instrument"])
        series.index = series.index.set_names(["datetime", "instrument"])
        return series.sort_index()
    if series.index.nlevels >= 2:
        series.index = series.index.set_names(["datetime", "instrument"] + names[2:])
    return series.sort_index()


def _strategy_feature_control_maps(strategy_cfg: Optional[Dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    strategy_cfg = strategy_cfg or {}

    def normalize(raw, *, lower=None, upper=None) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out = {}
        for field, value in raw.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            if lower is not None:
                v = max(float(lower), v)
            if upper is not None:
                v = min(float(upper), v)
            out[str(field)] = v
        return out

    return (
        normalize(strategy_cfg.get("feature_score_weights")),
        normalize(strategy_cfg.get("feature_min_percentiles"), lower=0.0, upper=1.0),
    )


def _normalize_feature_frame(features, fields: List[str]) -> pd.DataFrame:
    if isinstance(features, pd.Series):
        features = features.to_frame(fields[0] if len(fields) == 1 else "feature")
    if not isinstance(features, pd.DataFrame) or features.empty:
        return pd.DataFrame()
    out = features.copy()
    if len(fields) == 1 and fields[0] not in out.columns and out.shape[1] == 1:
        out = out.rename(columns={out.columns[0]: fields[0]})
    if isinstance(out.index, pd.MultiIndex) and list(out.index.names) == ["instrument", "datetime"]:
        out = out.reorder_levels(["datetime", "instrument"]).sort_index()
    cols = [field for field in fields if field in out.columns]
    return out[cols].sort_index() if cols else pd.DataFrame()


def _cs_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((numeric - float(numeric.mean())) / std).fillna(0.0).astype(float)


def _demote_masked_scores(scores: pd.Series, mask: pd.Series) -> pd.Series:
    adjusted = scores.copy()
    mask = mask.reindex(adjusted.index).fillna(False).astype(bool)
    if adjusted.empty or not bool(mask.any()):
        return adjusted
    finite = adjusted.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return adjusted
    spread = float(finite.max() - finite.min())
    if not math.isfinite(spread):
        spread = 1.0
    floor = float(finite.min() - max(1.0, spread))
    step = max(1e-9, max(1.0, spread) * 1e-9)
    for inst in adjusted.index[mask.to_numpy()]:
        adjusted.loc[inst] = floor
        floor -= step
    return adjusted


def _apply_strategy_score_controls(
    score: pd.Series,
    features: pd.DataFrame,
    *,
    feature_score_weights: Dict[str, float],
    feature_min_percentiles: Dict[str, float],
) -> pd.Series:
    if score.empty or (not feature_score_weights and not feature_min_percentiles) or features.empty:
        return score
    panel = pd.DataFrame({"score": score}).join(features.reindex(score.index), how="left")

    def adjust_day(group: pd.DataFrame) -> pd.Series:
        base = pd.to_numeric(group["score"], errors="coerce")
        adjusted = _cs_zscore(base) if feature_score_weights else base.astype(float).copy()
        for field, weight in feature_score_weights.items():
            if field in group.columns:
                adjusted = adjusted + float(weight) * _cs_zscore(group[field])
        for field, min_pct in feature_min_percentiles.items():
            if field in group.columns:
                pct = pd.to_numeric(group[field], errors="coerce").rank(pct=True)
                adjusted = _demote_masked_scores(adjusted, pct.isna() | (pct < float(min_pct)))
        return adjusted.where(base.notna(), np.nan)

    adjusted = pd.concat([adjust_day(group) for _, group in panel.groupby(level="datetime", sort=True)]).sort_index()
    adjusted.name = score.name
    return adjusted.reindex(score.index)


def _benchmark_forward_return(
    benchmark,
    *,
    label_horizon_days: int,
    label_ref_start_days: int,
) -> Optional[pd.Series]:
    if not isinstance(benchmark, pd.Series):
        return None
    bench = benchmark.copy()
    bench.index = pd.DatetimeIndex(bench.index)
    bench = bench.sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    ret = bench.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    gross = 1.0 + ret
    cp = gross.cumprod()
    start = max(0, int(label_ref_start_days))
    horizon = max(1, int(label_horizon_days))
    return cp.shift(-(start + horizon)) / cp.shift(-start) - 1.0


def _infer_label_ref_start_days(data_handler_config: dict) -> int:
    processors = data_handler_config.get("learn_processors", [])
    if not isinstance(processors, list):
        return 1
    for proc in processors:
        if not isinstance(proc, dict):
            continue
        if str(proc.get("class", "")).split(".")[-1] not in {
            "BenchmarkExcessLabel",
            "ResidualForwardReturnLabel",
            "VolScaledExcessLabel",
            "DownsideAdjustedExcessLabel",
            "PortfolioUtilityExcessLabel",
            "DualHorizonPortfolioUtilityLabel",
        }:
            continue
        kwargs = proc.get("kwargs", {}) or {}
        try:
            return max(0, int(kwargs.get("label_ref_start_days", 1)))
        except Exception:
            return 1
    return 1


def _load_prediction_label_frame(
    pred: pd.DataFrame,
    *,
    label_expr: str,
    benchmark,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    label_horizon_days: int,
    label_ref_start_days: int,
    strategy_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    score = _extract_score_series(pred)
    score = _normalize_datetime_instrument_index(score)
    if not isinstance(score.index, pd.MultiIndex):
        return pd.DataFrame(columns=["score", "label"])
    dt_index = pd.DatetimeIndex(score.index.get_level_values("datetime"))
    score = score[(dt_index >= bt_start) & (dt_index <= bt_end)].dropna()
    if score.empty:
        return pd.DataFrame(columns=["score", "label"])

    from qlib.data import D

    instruments = sorted(set(score.index.get_level_values("instrument")))
    feature_weights, feature_mins = _strategy_feature_control_maps(strategy_cfg)
    control_fields = list(dict.fromkeys([*feature_weights.keys(), *feature_mins.keys()]))
    fields = list(dict.fromkeys([label_expr, *control_fields]))
    raw_features = D.features(instruments, fields, start_time=bt_start, end_time=bt_end)
    feature_frame = _normalize_feature_frame(raw_features, fields)
    if feature_frame.empty:
        label = pd.Series(dtype=float)
    elif label_expr in feature_frame.columns:
        label = feature_frame[label_expr]
    else:
        label = feature_frame.iloc[:, 0]
    label = _normalize_datetime_instrument_index(label.astype(float))

    if control_fields:
        score = _apply_strategy_score_controls(
            score,
            feature_frame.reindex(columns=control_fields),
            feature_score_weights=feature_weights,
            feature_min_percentiles=feature_mins,
        )

    bench_fwd = _benchmark_forward_return(
        benchmark,
        label_horizon_days=label_horizon_days,
        label_ref_start_days=label_ref_start_days,
    )
    if bench_fwd is not None and not bench_fwd.empty:
        label_dates = pd.DatetimeIndex(label.index.get_level_values("datetime"))
        bench_vals = bench_fwd.reindex(label_dates).to_numpy()
        label = label - bench_vals

    frame = pd.concat([score.rename("score"), label.rename("label")], axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def _model_quality_metrics(
    frame: pd.DataFrame,
    *,
    topk: int,
    min_daily_count: int,
    recent_days: Optional[int] = None,
) -> Dict[str, float]:
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return {"days": 0, "rows": 0}
    dt_level = "datetime" if "datetime" in frame.index.names else 0
    dates = pd.DatetimeIndex(frame.index.get_level_values(dt_level)).normalize()
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    if recent_days is not None and int(recent_days) > 0:
        keep_dates = set(unique_dates[-int(recent_days) :])
        frame = frame[dates.isin(keep_dates)]
    if frame.empty:
        return {"days": 0, "rows": 0}

    ic_vals = []
    topk_vals = []
    spread_vals = []
    used_days = 0
    for _, day in frame.groupby(level=dt_level, sort=True):
        day = day.dropna(subset=["score", "label"])
        if len(day) < int(min_daily_count):
            continue
        used_days += 1
        ic = day["score"].corr(day["label"], method="spearman")
        if isinstance(ic, (int, float)) and math.isfinite(float(ic)):
            ic_vals.append(float(ic))
        k = min(max(1, int(topk)), len(day))
        topk_vals.append(float(day.nlargest(k, "score")["label"].mean()))
        q = max(1, int(math.ceil(len(day) * 0.20)))
        hi = day.nlargest(q, "score")["label"].mean()
        lo = day.nsmallest(q, "score")["label"].mean()
        spread_vals.append(float(hi - lo))

    ic_ser = pd.Series(ic_vals, dtype=float)
    out: Dict[str, float] = {
        "days": float(used_days),
        "rows": float(len(frame)),
        "mean_ic": float(ic_ser.mean()) if not ic_ser.empty else float("nan"),
        "pos_ic_rate": float((ic_ser > 0).mean()) if not ic_ser.empty else float("nan"),
        "topk_mean_label": float(np.mean(topk_vals)) if topk_vals else float("nan"),
        "topq_minus_bottomq": float(np.mean(spread_vals)) if spread_vals else float("nan"),
    }
    ic_std = float(ic_ser.std(ddof=1)) if len(ic_ser) > 1 else float("nan")
    out["icir"] = float(out["mean_ic"] / ic_std * np.sqrt(252)) if math.isfinite(ic_std) and ic_std > 0 else float("nan")
    return out


def _model_quality_by_year(
    frame: pd.DataFrame,
    *,
    topk: int,
    min_daily_count: int,
) -> List[Tuple[str, Dict[str, float]]]:
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return []
    dt_level = "datetime" if "datetime" in frame.index.names else 0
    dates = pd.DatetimeIndex(frame.index.get_level_values(dt_level))
    rows: List[Tuple[str, Dict[str, float]]] = []
    for year in sorted(dates.year.unique()):
        mask = dates.year == int(year)
        metrics = _model_quality_metrics(
            frame.loc[mask],
            topk=topk,
            min_daily_count=min_daily_count,
            recent_days=None,
        )
        rows.append((str(int(year)), metrics))
    return rows


def _filter_model_frame_by_weekday(frame: pd.DataFrame, weekday: int) -> pd.DataFrame:
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return frame.iloc[0:0]
    dt_level = "datetime" if "datetime" in frame.index.names else 0
    dates = pd.DatetimeIndex(frame.index.get_level_values(dt_level))
    return frame.loc[dates.weekday == int(weekday)]


def _rebalance_trade_dates(calendar: List[pd.Timestamp], weekday: int) -> List[pd.Timestamp]:
    cal = pd.DatetimeIndex(calendar).normalize()
    out: List[pd.Timestamp] = []
    for dt in cal:
        week_start = dt - pd.Timedelta(days=dt.weekday())
        target = week_start + pd.Timedelta(days=int(weekday))
        idx = np.searchsorted(cal.values, np.datetime64(target), side="left")
        if idx < len(cal) and pd.Timestamp(cal[idx]).normalize() == dt:
            out.append(pd.Timestamp(dt))
    return out


def _rebalance_signal_dates(
    calendar: List[pd.Timestamp],
    weekday: int,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> List[pd.Timestamp]:
    cal = pd.DatetimeIndex(calendar).normalize()
    trade_dates = set(_rebalance_trade_dates(list(cal), weekday))
    start = pd.Timestamp(start).normalize() if start is not None else None
    end = pd.Timestamp(end).normalize() if end is not None else None
    out: List[pd.Timestamp] = []
    for i, dt in enumerate(cal):
        trade_dt = pd.Timestamp(dt)
        if trade_dt not in trade_dates or i <= 0:
            continue
        if start is not None and trade_dt < start:
            continue
        if end is not None and trade_dt > end:
            continue
        out.append(pd.Timestamp(cal[i - 1]))
    return out


def _strategy_trade_signal_pairs(
    calendar: List[pd.Timestamp],
    *,
    rebalance_weekday: Optional[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    signal_shift: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Pair simulator execution dates with the signal date used by qlib strategies."""
    cal = pd.DatetimeIndex(calendar).normalize()
    if cal.empty:
        return []
    cal = pd.DatetimeIndex(sorted(cal.unique()))
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    shift = max(0, int(signal_shift))
    if rebalance_weekday is not None:
        trade_dates = _rebalance_trade_dates(list(cal), int(rebalance_weekday))
    else:
        trade_dates = [pd.Timestamp(dt) for dt in cal]
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for trade_dt in trade_dates:
        trade_dt = pd.Timestamp(trade_dt).normalize()
        if trade_dt < start or trade_dt > end:
            continue
        idx = int(np.searchsorted(cal.values, np.datetime64(trade_dt), side="left"))
        if idx >= len(cal) or pd.Timestamp(cal[idx]).normalize() != trade_dt:
            continue
        signal_idx = idx - shift
        if signal_idx < 0:
            continue
        out.append((trade_dt, pd.Timestamp(cal[signal_idx]).normalize()))
    return out


def _filter_model_frame_by_dates(frame: pd.DataFrame, keep_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return frame.iloc[0:0]
    keep = {pd.Timestamp(dt).normalize() for dt in keep_dates}
    dt_level = "datetime" if "datetime" in frame.index.names else 0
    dates = pd.DatetimeIndex(frame.index.get_level_values(dt_level)).normalize()
    return frame.loc[dates.isin(keep)]


def _filter_prediction_by_dates(pred: pd.DataFrame, keep_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    if pred.empty or not isinstance(pred.index, pd.MultiIndex):
        return pred.iloc[0:0]
    keep = {pd.Timestamp(dt).normalize() for dt in keep_dates}
    dt_level = "datetime" if "datetime" in pred.index.names else 0
    dates = pd.DatetimeIndex(pred.index.get_level_values(dt_level)).normalize()
    return pred.loc[dates.isin(keep)]


def _load_ensemble_source_dates(
    pred: pd.DataFrame,
    *,
    manifest_path: Optional[Path] = None,
    gate_csv_path: Optional[Path] = None,
    primary_name: str = "primary",
    defensive_name: str = "defensive",
) -> Tuple[Dict[str, List[pd.Timestamp]], List[Tuple[str, bool, str]]]:
    rows: List[Tuple[str, bool, str]] = []
    if pred.empty or not isinstance(pred.index, pd.MultiIndex):
        rows.append(("ensemble_source_pred_index", False, "prediction is empty or not MultiIndex"))
        return {}, rows
    dt_level = "datetime" if "datetime" in pred.index.names else 0
    pred_dates = pd.DatetimeIndex(pred.index.get_level_values(dt_level)).normalize()
    all_dates = pd.DatetimeIndex(sorted(pred_dates.unique()))

    manifest = _read_json(manifest_path) if manifest_path is not None and manifest_path.exists() else None
    if manifest_path is not None:
        rows.append(
            (
                "ensemble_manifest_loaded",
                manifest is not None,
                str(manifest_path.expanduser().resolve()) if manifest is not None else "missing_or_invalid",
            )
        )
    if gate_csv_path is None and isinstance(manifest, dict):
        raw_gate = ((manifest.get("confidence_gate") or {}).get("gate_report_csv") or "")
        if raw_gate:
            gate_csv_path = Path(raw_gate).expanduser()

    if gate_csv_path is None or not gate_csv_path.exists():
        rows.append(("ensemble_gate_loaded", False, "gate csv missing; treating all dates as primary"))
        return {primary_name: list(all_dates)}, rows

    try:
        gate = pd.read_csv(gate_csv_path)
    except Exception as e:
        rows.append(("ensemble_gate_loaded", False, repr(e)))
        return {primary_name: list(all_dates)}, rows

    date_col = "datetime" if "datetime" in gate.columns else gate.columns[0]
    if "fallback" not in gate.columns:
        rows.append(("ensemble_gate_fallback_column", False, "fallback column missing"))
        return {primary_name: list(all_dates)}, rows
    gate_dates = pd.DatetimeIndex(pd.to_datetime(gate[date_col], errors="coerce")).normalize()
    fallback = gate["fallback"]
    if fallback.dtype != bool:
        fallback = fallback.astype(str).str.lower().isin({"true", "1", "yes", "y"})
    fallback_by_date = pd.Series(fallback.to_numpy(dtype=bool), index=gate_dates).dropna()
    fallback_by_date = fallback_by_date[~fallback_by_date.index.duplicated(keep="last")]
    is_fallback = pd.Series(all_dates, index=all_dates).map(fallback_by_date).fillna(False).astype(bool)
    sources = {
        primary_name: [pd.Timestamp(dt) for dt in all_dates[~is_fallback.to_numpy()]],
        defensive_name: [pd.Timestamp(dt) for dt in all_dates[is_fallback.to_numpy()]],
    }
    rows.append(
        (
            "ensemble_gate_loaded",
            True,
            f"{gate_csv_path.expanduser().resolve()} dates={len(gate)} fallback_dates={len(sources[defensive_name])}",
        )
    )
    return sources, rows


def _source_aware_prediction_label_frame(
    pred: pd.DataFrame,
    *,
    label_expr: str,
    benchmark,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    label_ref_start_days: int,
    strategy_cfg: Optional[Dict],
    source_dates: Dict[str, List[pd.Timestamp]],
    source_horizons: Dict[str, int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source, dates in source_dates.items():
        if not dates:
            continue
        horizon = int(source_horizons.get(source) or 0)
        if horizon <= 0:
            continue
        source_pred = _filter_prediction_by_dates(pred, dates)
        if source_pred.empty:
            continue
        frame = _load_prediction_label_frame(
            source_pred,
            label_expr=label_expr,
            benchmark=benchmark,
            bt_start=bt_start,
            bt_end=bt_end,
            label_horizon_days=horizon,
            label_ref_start_days=label_ref_start_days,
            strategy_cfg=strategy_cfg,
        )
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source"] = source
        frame["source_horizon_days"] = float(horizon)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["score", "label", "source", "source_horizon_days"])
    return pd.concat(frames).sort_index()


def _print_ensemble_source_summary(source_dates: Dict[str, List[pd.Timestamp]]) -> None:
    if not source_dates:
        print("(no ensemble source dates)")
        return
    print("source | dates | start | end | ratio")
    print("--- | --- | --- | --- | ---")
    total = sum(len(v) for v in source_dates.values())
    for source, dates in source_dates.items():
        idx = pd.DatetimeIndex(dates)
        ratio = len(idx) / total if total else float("nan")
        start = str(idx.min().date()) if len(idx) else ""
        end = str(idx.max().date()) if len(idx) else ""
        print(f"{source} | {len(idx)} | {start} | {end} | {_format_float(ratio)}")


def _print_model_quality_by_source(frame: pd.DataFrame, *, topk: int, min_daily_count: int) -> None:
    if frame.empty or "source" not in frame.columns:
        print("(no source-aware model quality rows)")
        return
    print("source | days | rows | mean_ic | pos_ic_rate | topk_mean_label | topq_minus_bottomq")
    print("--- | --- | --- | --- | --- | --- | ---")
    for source, source_frame in frame.groupby("source", sort=True):
        metrics = _model_quality_metrics(source_frame.drop(columns=["source"], errors="ignore"), topk=topk, min_daily_count=min_daily_count)
        print(
            " | ".join(
                [
                    str(source),
                    str(int(metrics.get("days", 0) or 0)),
                    str(int(metrics.get("rows", 0) or 0)),
                    _format_float(metrics.get("mean_ic")),
                    _format_float(metrics.get("pos_ic_rate")),
                    _format_float(metrics.get("topk_mean_label")),
                    _format_float(metrics.get("topq_minus_bottomq")),
                ]
            )
        )


def _model_quality_gate_rows(
    full: Dict[str, float],
    recent: Dict[str, float],
    args: argparse.Namespace,
    yearly: Optional[List[Tuple[str, Dict[str, float]]]] = None,
    prefix: str = "model_quality",
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []

    full_days = int(full.get("days", 0) or 0)
    recent_days = int(recent.get("days", 0) or 0)
    rows.append(
        (
            f"{prefix}_full_days",
            full_days >= int(args.model_quality_min_days),
            f"{full_days} >= {int(args.model_quality_min_days)}",
        )
    )
    rows.append(
        (
            f"{prefix}_recent_days",
            recent_days >= int(args.model_quality_min_recent_days),
            f"{recent_days} >= {int(args.model_quality_min_recent_days)}",
        )
    )

    checks = [
        (f"{prefix}_full_mean_ic", full.get("mean_ic"), args.model_quality_min_full_mean_ic),
        (f"{prefix}_full_topq_spread", full.get("topq_minus_bottomq"), args.model_quality_min_full_topq_spread),
        (f"{prefix}_recent_mean_ic", recent.get("mean_ic"), args.model_quality_min_recent_mean_ic),
        (f"{prefix}_recent_pos_ic_rate", recent.get("pos_ic_rate"), args.model_quality_min_recent_pos_ic_rate),
        (f"{prefix}_recent_topk_label", recent.get("topk_mean_label"), args.model_quality_min_recent_topk_mean_label),
        (f"{prefix}_recent_topq_spread", recent.get("topq_minus_bottomq"), args.model_quality_min_recent_topq_spread),
    ]
    for name, value, threshold in checks:
        val = _safe_float(value)
        ok = val is not None and val >= float(threshold)
        rows.append((name, ok, f"{_format_float(val)} >= {_format_float(float(threshold))}"))

    if yearly is not None:
        min_year_days = int(getattr(args, "model_quality_min_year_days", 120))
        min_positive_years = int(getattr(args, "model_quality_min_positive_years", 0))
        min_worst_year_mean_ic = float(getattr(args, "model_quality_min_worst_year_mean_ic", -1.0))
        min_worst_year_topk_mean_label = float(
            getattr(args, "model_quality_min_worst_year_topk_mean_label", -1.0)
        )
        min_worst_year_topq_spread = float(getattr(args, "model_quality_min_worst_year_topq_spread", -1.0))

        eligible = []
        skipped = 0
        for period, metrics in yearly:
            days = int(metrics.get("days", 0) or 0)
            if days < min_year_days:
                skipped += 1
                continue
            eligible.append((period, metrics))
        if min_positive_years > 0:
            positive_years = sum(1 for _, metrics in eligible if _safe_float(metrics.get("mean_ic")) is not None and metrics.get("mean_ic") >= 0)
            cap_positive_years = not bool(getattr(args, "strict_positive_year_count", False))
            required_positive_years = (
                min(min_positive_years, len(eligible))
                if cap_positive_years and eligible
                else min_positive_years
            )
            cap_detail = (
                f", configured_min_positive_years={min_positive_years}"
                if required_positive_years != min_positive_years
                else ""
            )
            rows.append(
                (
                    f"{prefix}_positive_ic_years",
                    positive_years >= required_positive_years,
                    f"{positive_years} >= {required_positive_years} (eligible_years={len(eligible)}, min_year_days={min_year_days}, skipped_short_years={skipped}{cap_detail})",
                )
            )
        if eligible:
            year_mean_ics = [_safe_float(metrics.get("mean_ic")) for _, metrics in eligible]
            year_topk_labels = [_safe_float(metrics.get("topk_mean_label")) for _, metrics in eligible]
            year_spreads = [_safe_float(metrics.get("topq_minus_bottomq")) for _, metrics in eligible]
            year_mean_ics = [v for v in year_mean_ics if v is not None]
            year_topk_labels = [v for v in year_topk_labels if v is not None]
            year_spreads = [v for v in year_spreads if v is not None]
            worst_mean_ic = min(year_mean_ics) if year_mean_ics else None
            worst_topk_label = min(year_topk_labels) if year_topk_labels else None
            worst_spread = min(year_spreads) if year_spreads else None
        else:
            worst_mean_ic = None
            worst_topk_label = None
            worst_spread = None
        if min_worst_year_mean_ic > -1.0:
            rows.append(
                (
                    f"{prefix}_worst_year_mean_ic",
                    worst_mean_ic is not None and worst_mean_ic >= min_worst_year_mean_ic,
                    f"{_format_float(worst_mean_ic)} >= {_format_float(min_worst_year_mean_ic)}",
                )
            )
        if min_worst_year_topk_mean_label > -1.0:
            rows.append(
                (
                    f"{prefix}_worst_year_topk_label",
                    worst_topk_label is not None and worst_topk_label >= min_worst_year_topk_mean_label,
                    f"{_format_float(worst_topk_label)} >= {_format_float(min_worst_year_topk_mean_label)}",
                )
            )
        if min_worst_year_topq_spread > -1.0:
            rows.append(
                (
                    f"{prefix}_worst_year_topq_spread",
                    worst_spread is not None and worst_spread >= min_worst_year_topq_spread,
                    f"{_format_float(worst_spread)} >= {_format_float(min_worst_year_topq_spread)}",
                )
            )
    return rows


def _print_model_quality_summary(label: str, metrics: Dict[str, float]) -> None:
    keys = ["days", "rows", "mean_ic", "icir", "pos_ic_rate", "topk_mean_label", "topq_minus_bottomq"]
    vals = []
    for key in keys:
        val = metrics.get(key)
        vals.append(f"{key}={_format_float(val)}" if key not in {"days", "rows"} else f"{key}={int(val or 0)}")
    _print_kv(label, ", ".join(vals))


def _print_model_quality_yearly(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    if not rows:
        print("(no yearly model-quality rows)")
        return
    headers = ["year", "days", "rows", "mean_ic", "icir", "pos_ic_rate", "topk_mean_label", "topq_minus_bottomq"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for year, metrics in rows:
        print(
            " | ".join(
                [
                    year,
                    str(int(metrics.get("days", 0) or 0)),
                    str(int(metrics.get("rows", 0) or 0)),
                    _format_float(metrics.get("mean_ic")),
                    _format_float(metrics.get("icir")),
                    _format_float(metrics.get("pos_ic_rate")),
                    _format_float(metrics.get("topk_mean_label")),
                    _format_float(metrics.get("topq_minus_bottomq")),
                ]
            )
        )


def _load_strategy_sector_map(strategy_cfg: Dict) -> Dict[str, str]:
    path = str(strategy_cfg.get("sector_map_csv") or "").strip()
    if not path:
        return {}
    fp = Path(path).expanduser()
    if not fp.exists():
        return {}
    ticker_col = str(strategy_cfg.get("sector_ticker_col") or "ticker")
    sector_col = str(strategy_cfg.get("sector_col") or "sector")
    try:
        df = pd.read_csv(fp, usecols=lambda c: c in {ticker_col, sector_col}, low_memory=False)
    except Exception:
        return {}
    if ticker_col not in df.columns or sector_col not in df.columns:
        return {}
    tickers = df[ticker_col].astype(str).str.upper().str.strip()
    sectors = df[sector_col].astype(str).str.strip()
    sectors = sectors.mask(sectors.eq("") | sectors.str.lower().isin({"nan", "none"}), "__UNKNOWN__")
    sector_map = pd.Series(sectors.values, index=tickers.values)
    sector_map = sector_map[~sector_map.index.duplicated(keep="last")]
    return sector_map.to_dict()


def _apply_sector_cap_to_score(score: pd.Series, strategy_cfg: Dict, sector_map: Dict[str, str]) -> pd.Series:
    if score.empty or not sector_map:
        return score
    max_count = strategy_cfg.get("max_sector_count")
    if max_count is None and strategy_cfg.get("max_sector_weight") is not None:
        try:
            topk = max(1, int(strategy_cfg.get("topk", 40)))
            max_count = int(math.floor(float(strategy_cfg["max_sector_weight"]) * topk))
        except Exception:
            max_count = None
    if max_count is None:
        return score
    max_count = max(1, int(max_count))
    sorted_score = score.dropna().sort_values(ascending=False)
    if sorted_score.empty:
        return score
    spread = float(sorted_score.max() - sorted_score.min())
    if not math.isfinite(spread):
        spread = 1.0
    floor = float(sorted_score.min() - max(1.0, spread))
    step = max(1e-9, max(1.0, spread) * 1e-9)
    counts: Dict[str, int] = {}
    adjusted = sorted_score.copy()
    for inst in sorted_score.index:
        sector = sector_map.get(str(inst).upper().strip(), "__UNKNOWN__")
        count = counts.get(sector, 0)
        if count < max_count:
            counts[sector] = count + 1
            continue
        adjusted.loc[inst] = floor
        floor -= step
    return adjusted.reindex(score.index)


def _active_risk_score_weights(scores: pd.Series, *, topk: int) -> pd.Series:
    top = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False).head(int(topk))
    if top.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(top), index=top.index, dtype=float)


def _coerce_ticker_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace(",", "\n").splitlines()
    elif isinstance(value, IterableABC):
        raw = list(value)
    else:
        raw = [value]
    tickers: List[str] = []
    seen = set()
    for item in raw:
        ticker = str(item).strip().upper()
        ticker = BENCHMARK_TICKER_ALIASES.get(ticker, ticker)
        if not ticker or ticker.startswith("#") or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
    return tickers


def _parse_ticker_path_map(value) -> Dict[str, Path]:
    if value is None:
        return {}
    items = value if isinstance(value, IterableABC) and not isinstance(value, str) else str(value).replace(",", "\n").splitlines()
    out: Dict[str, Path] = {}
    for item in items:
        text = str(item).strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        ticker, path = text.split("=", 1)
        ticker = _coerce_ticker_list(ticker)
        if not ticker:
            continue
        out[ticker[0]] = Path(path.strip()).expanduser()
    return out


def _strategy_benchmark_tickers(strategy_cfg: Dict) -> List[str]:
    tickers = _coerce_ticker_list(strategy_cfg.get("benchmark_tickers"))
    path_val = str(strategy_cfg.get("benchmark_tickers_file") or "").strip()
    if path_val:
        path = Path(path_val).expanduser()
        if path.exists():
            tickers.extend(_coerce_ticker_list(path.read_text(encoding="utf-8").splitlines()))
    return _coerce_ticker_list(tickers)


def _strategy_is_hedged(strategy_class: str) -> bool:
    return "Hedged" in str(strategy_class)


def _strategy_hedge_tickers(strategy_cfg: Dict) -> List[str]:
    tickers = _coerce_ticker_list(strategy_cfg.get("hedge_tickers"))
    path_val = str(strategy_cfg.get("hedge_tickers_file") or "").strip()
    if path_val:
        path = Path(path_val).expanduser()
        if path.exists():
            tickers.extend(_coerce_ticker_list(path.read_text(encoding="utf-8").splitlines()))
    return _coerce_ticker_list(tickers)


def _equal_weight_benchmark_tickers(
    strategy_cfg: Dict,
    *,
    available: Optional[Iterable[str]] = None,
) -> pd.Series:
    tickers = _strategy_benchmark_tickers(strategy_cfg)
    if available is not None:
        available_set = {str(x).upper() for x in available}
        tickers = [ticker for ticker in tickers if ticker in available_set]
    if not tickers:
        return pd.Series(dtype=float)
    topn = int(strategy_cfg.get("benchmark_topn") or 0)
    if topn > 0:
        tickers = tickers[:topn]
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def _equal_weight_hedge_tickers(
    strategy_cfg: Dict,
    *,
    available: Optional[Iterable[str]] = None,
) -> pd.Series:
    tickers = _strategy_hedge_tickers(strategy_cfg)
    if available is not None:
        available_set = {str(x).upper() for x in available}
        tickers = [ticker for ticker in tickers if ticker in available_set]
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def _trailing_weighted_excess_from_close(
    alpha_weights: pd.Series,
    benchmark_weights: pd.Series,
    close_history: Optional[pd.Series],
    trade_dt: Optional[pd.Timestamp],
    *,
    window: int,
    min_history: int,
) -> Optional[float]:
    from qlib.contrib.strategy.benchmark_aware import normalize_long_weights

    alpha = normalize_long_weights(alpha_weights)
    bench = normalize_long_weights(benchmark_weights)
    if alpha.empty or bench.empty or close_history is None or trade_dt is None:
        return None
    close = pd.to_numeric(close_history, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if close.empty or not isinstance(close.index, pd.MultiIndex):
        return None
    close = _normalize_datetime_instrument_index(close)
    dt_level = "datetime" if "datetime" in close.index.names else 0
    inst_level = "instrument" if "instrument" in close.index.names else 1
    dates = pd.DatetimeIndex(close.index.get_level_values(dt_level)).normalize()
    needed = set(alpha.index.astype(str)).union(set(bench.index.astype(str)))
    inst = close.index.get_level_values(inst_level).astype(str)
    close = close[(dates < pd.Timestamp(trade_dt).normalize()) & pd.Series(inst, index=close.index).isin(needed)]
    if close.empty:
        return None
    close_wide = close.unstack(inst_level).sort_index().tail(max(2, int(window)))
    close_wide = close_wide.dropna(how="all")
    if len(close_wide) < max(2, int(min_history)):
        return None

    first = close_wide.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
    last = close_wide.apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
    ret = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret[(first.reindex(ret.index) > 0) & (last.reindex(ret.index) > 0)]
    if ret.empty:
        return None
    alpha = normalize_long_weights(alpha.reindex(ret.index).dropna())
    bench = normalize_long_weights(bench.reindex(ret.index).dropna())
    if alpha.empty or bench.empty or float(alpha.sum()) < 0.50 or float(bench.sum()) < 0.50:
        return None
    alpha_ret = float((alpha * ret.reindex(alpha.index)).sum())
    bench_ret = float((bench * ret.reindex(bench.index)).sum())
    if not math.isfinite(alpha_ret) or not math.isfinite(bench_ret):
        return None
    return alpha_ret - bench_ret


def _dynamic_benchmark_core_weight(
    strategy_cfg: Dict,
    candidates: pd.Series,
    benchmark_weights: pd.Series,
    *,
    topk: int,
    trade_dt: Optional[pd.Timestamp],
    close_history: Optional[pd.Series],
) -> float:
    from qlib.contrib.strategy.benchmark_aware import rank_score_weights

    base_core = float(strategy_cfg.get("benchmark_core_weight", 0.40))
    if not bool(strategy_cfg.get("dynamic_alpha_weight", False)):
        return base_core
    alpha_top = pd.to_numeric(candidates, errors="coerce").dropna().sort_values(ascending=False).head(int(topk))
    alpha_weights = rank_score_weights(
        alpha_top,
        topn=int(topk),
        method=str(strategy_cfg.get("weighting", "rank")),
        temperature=float(strategy_cfg.get("temperature", 1.0)),
    )
    excess = _trailing_weighted_excess_from_close(
        alpha_weights,
        benchmark_weights,
        close_history,
        trade_dt,
        window=int(strategy_cfg.get("alpha_quality_window") or 63),
        min_history=int(strategy_cfg.get("alpha_quality_min_history") or 20),
    )
    if excess is None or not math.isfinite(excess):
        return base_core
    lower = float(strategy_cfg.get("alpha_quality_lower_excess", -0.03))
    upper = float(strategy_cfg.get("alpha_quality_upper_excess", 0.03))
    lower, upper = min(lower, upper), max(lower, upper)
    min_scale = float(np.clip(float(strategy_cfg.get("min_alpha_scale", 0.0)), 0.0, 1.0))
    max_scale = float(np.clip(float(strategy_cfg.get("max_alpha_scale", 1.0)), min_scale, 1.0))
    if upper <= lower + 1e-12:
        scale = max_scale if excess >= upper else min_scale
    elif excess <= lower:
        scale = min_scale
    elif excess >= upper:
        scale = max_scale
    else:
        scale = min_scale + (float(excess) - lower) / (upper - lower) * (max_scale - min_scale)
    alpha_weight = (1.0 - base_core) * scale
    return float(np.clip(1.0 - alpha_weight, 0.0, 1.0))


def _close_history_for_instrument(
    close_history: Optional[pd.Series],
    instrument: str,
    trade_dt: pd.Timestamp,
) -> pd.Series:
    if close_history is None or not isinstance(close_history, pd.Series) or close_history.empty:
        return pd.Series(dtype=float)
    if not isinstance(close_history.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    close = _normalize_datetime_instrument_index(
        pd.to_numeric(close_history, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    )
    if close.empty or not isinstance(close.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    dt_level = "datetime" if "datetime" in close.index.names else 0
    inst_level = "instrument" if "instrument" in close.index.names else 1
    dates = pd.DatetimeIndex(close.index.get_level_values(dt_level)).normalize()
    inst = pd.Index(close.index.get_level_values(inst_level).astype(str).str.upper())
    target = str(instrument).upper().strip()
    mask = (inst == target) & (dates < pd.Timestamp(trade_dt).normalize())
    out = close.loc[mask]
    if out.empty:
        return pd.Series(dtype=float)
    out.index = dates[mask]
    out = pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out > 0].sort_index()
    return out[~out.index.duplicated(keep="last")]


def _strategy_dynamic_hedge_weight(
    strategy_cfg: Dict,
    close_history: Optional[pd.Series],
    trade_dt: Optional[pd.Timestamp],
    state: Optional[Dict[str, float]] = None,
) -> float:
    mode = str(strategy_cfg.get("hedge_mode", "long_hedge") or "off").strip().lower()
    state = state if state is not None else {}
    if mode not in {"long_hedge", "long", "inverse_etf"}:
        state["last_hedge_weight"] = 0.0
        return 0.0

    min_w = float(np.clip(float(strategy_cfg.get("hedge_min_weight", 0.0)), 0.0, 1.0))
    max_w = float(np.clip(float(strategy_cfg.get("hedge_max_weight", 0.35)), min_w, 1.0))
    base_w = float(np.clip(float(strategy_cfg.get("hedge_weight", 0.0)), min_w, max_w))
    smoothed = float(state.get("hedge_smoothed", base_w))
    target = base_w

    market_index = str(strategy_cfg.get("market_index") or "").strip().upper()
    if market_index and trade_dt is not None:
        close = _close_history_for_instrument(close_history, market_index, pd.Timestamp(trade_dt))
        min_history = max(2, int(strategy_cfg.get("hedge_min_history") or 40))
        if len(close) >= min_history:
            trend_thresh_raw = strategy_cfg.get("hedge_trend_thresh", -0.04)
            if trend_thresh_raw is not None:
                trend_window_n = max(2, int(strategy_cfg.get("hedge_trend_window") or 63))
                trend = close.tail(trend_window_n)
                if len(trend) >= min(min_history, trend_window_n):
                    trend_ret = float(trend.iloc[-1] / trend.iloc[0] - 1.0)
                    if math.isfinite(trend_ret) and trend_ret < float(trend_thresh_raw):
                        target = max(target, max_w)

            drawdown_limit_raw = strategy_cfg.get("hedge_drawdown_limit", 0.10)
            if drawdown_limit_raw is not None:
                drawdown_window_n = max(2, int(strategy_cfg.get("hedge_drawdown_window") or 126))
                dd_close = close.tail(drawdown_window_n)
                if len(dd_close) >= min(min_history, drawdown_window_n):
                    drawdown = float((dd_close / dd_close.cummax() - 1.0).min())
                    if math.isfinite(drawdown) and drawdown <= -abs(float(drawdown_limit_raw)):
                        target = max(target, max_w)

            crash_limit_raw = strategy_cfg.get("hedge_crash_return_limit", 0.06)
            if crash_limit_raw is not None:
                lookback = max(1, int(strategy_cfg.get("hedge_crash_return_lookback") or 5))
                if len(close) >= lookback + 1:
                    recent = close.tail(lookback + 1)
                    recent_return = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
                    if math.isfinite(recent_return) and recent_return <= -abs(float(crash_limit_raw)):
                        target = max(target, max_w)

    target = float(np.clip(target, min_w, max_w))
    if target > smoothed and strategy_cfg.get("hedge_max_step_up") is not None:
        target = min(target, smoothed + max(0.0, float(strategy_cfg.get("hedge_max_step_up"))))
    elif target < smoothed and strategy_cfg.get("hedge_max_step_down") is not None:
        target = max(target, smoothed - max(0.0, float(strategy_cfg.get("hedge_max_step_down"))))
    smoothing = float(np.clip(float(strategy_cfg.get("hedge_smoothing", 1.0)), 0.0, 1.0))
    smoothing_up = float(np.clip(float(strategy_cfg.get("hedge_smoothing_up", smoothing)), 0.0, 1.0))
    smoothing_down = float(np.clip(float(strategy_cfg.get("hedge_smoothing_down", smoothing)), 0.0, 1.0))
    alpha = smoothing_up if target >= smoothed else smoothing_down
    smoothed = float(np.clip((1.0 - alpha) * smoothed + alpha * target, min_w, max_w))
    state["hedge_smoothed"] = smoothed
    state["last_hedge_weight"] = smoothed
    return smoothed


def _apply_strategy_hedge_overlay(
    weights: pd.Series,
    *,
    strategy_cfg: Dict,
    strategy_class: str,
    available_hedges: Optional[Iterable[str]] = None,
    trade_dt: Optional[pd.Timestamp] = None,
    close_history: Optional[pd.Series] = None,
    hedge_state: Optional[Dict[str, float]] = None,
) -> pd.Series:
    from qlib.contrib.strategy.benchmark_aware import normalize_long_weights

    base = normalize_long_weights(weights)
    if base.empty or not _strategy_is_hedged(strategy_class):
        return base
    hedge_weight = _strategy_dynamic_hedge_weight(
        strategy_cfg,
        close_history,
        trade_dt,
        state=hedge_state,
    )
    if hedge_weight <= 1e-12:
        return base
    hedge = _equal_weight_hedge_tickers(strategy_cfg, available=available_hedges)
    if hedge.empty:
        if hedge_state is not None:
            hedge_state["last_hedge_weight"] = 0.0
        return base
    target = base.mul(1.0 - hedge_weight).add(hedge.mul(hedge_weight), fill_value=0.0)
    return normalize_long_weights(target)


def _strategy_dynamic_risk_degree(
    strategy_cfg: Dict,
    close_history: Optional[pd.Series],
    trade_dt: pd.Timestamp,
    base_risk_degree: float,
    state: Dict[str, float],
) -> float:
    if not bool(strategy_cfg.get("dynamic_risk", False)):
        return float(base_risk_degree)
    market_index = str(strategy_cfg.get("market_index") or "").strip().upper()
    if not market_index:
        return float(base_risk_degree)

    risk_floor = max(0.0, float(strategy_cfg.get("risk_floor", 0.0)))
    risk_ceiling = float(strategy_cfg.get("risk_ceiling", base_risk_degree))
    risk_ceiling = float(np.clip(risk_ceiling, risk_floor, 1.0))
    smoothed = float(state.get("smoothed", np.clip(base_risk_degree, risk_floor, risk_ceiling)))
    close = _close_history_for_instrument(close_history, market_index, trade_dt)
    min_history = max(2, int(strategy_cfg.get("risk_min_history") or 20))
    if len(close) < min_history:
        state["smoothed"] = smoothed
        return smoothed

    target = float(base_risk_degree)
    trend_window_n = max(2, int(strategy_cfg.get("market_trend_window") or 63))
    trend = close.tail(trend_window_n)
    if len(trend) >= min(min_history, trend_window_n):
        trend_ret = float(trend.iloc[-1] / trend.iloc[0] - 1.0)
        if math.isfinite(trend_ret):
            if trend_ret < float(strategy_cfg.get("market_trend_thresh", -0.04)):
                target *= float(np.clip(float(strategy_cfg.get("market_trend_penalty", 0.50)), 0.0, 1.0))
            elif strategy_cfg.get("market_trend_boost_thresh") is not None and trend_ret > float(
                strategy_cfg.get("market_trend_boost_thresh")
            ):
                target *= max(0.0, float(strategy_cfg.get("market_trend_boost", 1.0)))

    if strategy_cfg.get("market_drawdown_limit") is not None:
        dd_window_n = max(2, int(strategy_cfg.get("market_drawdown_window") or 126))
        dd_close = close.tail(dd_window_n)
        if len(dd_close) >= min(min_history, dd_window_n):
            drawdown = float((dd_close / dd_close.cummax() - 1.0).min())
            if math.isfinite(drawdown) and drawdown <= -abs(float(strategy_cfg.get("market_drawdown_limit"))):
                target *= float(np.clip(float(strategy_cfg.get("market_drawdown_penalty", 1.0)), 0.0, 1.0))

    cooldown = int(state.get("cooldown", 0) or 0)
    if bool(strategy_cfg.get("crash_guard", False)) and strategy_cfg.get("crash_return_limit") is not None:
        lookback = max(1, int(strategy_cfg.get("crash_return_lookback") or 5))
        if len(close) >= lookback + 1:
            recent = close.tail(lookback + 1)
            recent_return = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
            if math.isfinite(recent_return) and recent_return <= -abs(float(strategy_cfg.get("crash_return_limit"))):
                cooldown = max(cooldown, max(0, int(strategy_cfg.get("crash_cooldown_steps") or 0)))
        if cooldown > 0:
            target *= float(np.clip(float(strategy_cfg.get("crash_penalty", 1.0)), 0.0, 1.0))
            cooldown = max(0, cooldown - 1)
    state["cooldown"] = float(cooldown)

    target = float(np.clip(target, risk_floor, risk_ceiling))
    if target > smoothed and strategy_cfg.get("max_risk_step_up") is not None:
        target = min(target, smoothed + max(0.0, float(strategy_cfg.get("max_risk_step_up"))))
    elif target < smoothed and strategy_cfg.get("max_risk_step_down") is not None:
        target = max(target, smoothed - max(0.0, float(strategy_cfg.get("max_risk_step_down"))))
    smoothing = float(np.clip(float(strategy_cfg.get("risk_smoothing", 1.0)), 0.0, 1.0))
    smoothing_up = float(np.clip(float(strategy_cfg.get("risk_smoothing_up", smoothing)), 0.0, 1.0))
    smoothing_down = float(np.clip(float(strategy_cfg.get("risk_smoothing_down", smoothing)), 0.0, 1.0))
    alpha = smoothing_up if target >= smoothed else smoothing_down
    smoothed = float(np.clip((1.0 - alpha) * smoothed + alpha * target, risk_floor, risk_ceiling))
    state["smoothed"] = smoothed
    return smoothed


def _strategy_portfolio_weights_for_day(
    day_score: pd.Series,
    *,
    strategy_cfg: Dict,
    strategy_class: str,
    topk: int,
    marketcap: Optional[pd.Series] = None,
    benchmark_weights: Optional[pd.Series] = None,
    current_weights: Optional[pd.Series] = None,
    trade_dt: Optional[pd.Timestamp] = None,
    close_history: Optional[pd.Series] = None,
    available_hedges: Optional[Iterable[str]] = None,
    hedge_state: Optional[Dict[str, float]] = None,
) -> pd.Series:
    from qlib.contrib.strategy.benchmark_aware import (
        benchmark_weights_from_marketcap,
        build_benchmark_aware_weights,
        limit_turnover_toward_target,
        normalize_long_weights,
        rank_score_weights,
    )

    score = pd.to_numeric(day_score, errors="coerce").dropna()
    if score.empty:
        return pd.Series(dtype=float)
    topk = max(1, int(topk))
    is_benchmark_aware = "BenchmarkAware" in str(strategy_class)

    if is_benchmark_aware:
        strategy_benchmark_topn = max(1, int(strategy_cfg.get("benchmark_topn") or 500))
        liquidity_buffer = max(1, int(strategy_cfg.get("liquidity_buffer") or 3))
        cand_n = min(len(score), max(topk, 1) * liquidity_buffer)
        candidates = score.sort_values(ascending=False).head(cand_n)
        if benchmark_weights is not None:
            bench_source = normalize_long_weights(pd.to_numeric(benchmark_weights, errors="coerce"))
            strategy_bench = normalize_long_weights(bench_source.sort_values(ascending=False).head(strategy_benchmark_topn))
        elif _strategy_benchmark_tickers(strategy_cfg):
            strategy_bench = _equal_weight_benchmark_tickers(strategy_cfg)
        else:
            mc = pd.Series(dtype=float) if marketcap is None else pd.to_numeric(marketcap, errors="coerce")
            strategy_bench = benchmark_weights_from_marketcap(mc, topn=strategy_benchmark_topn)
        if strategy_bench.empty:
            return pd.Series(dtype=float)
        benchmark_core_weight = _dynamic_benchmark_core_weight(
            strategy_cfg,
            candidates,
            strategy_bench,
            topk=topk,
            trade_dt=trade_dt,
            close_history=close_history,
        )
        weights = normalize_long_weights(
            build_benchmark_aware_weights(
                candidates,
                strategy_bench,
                topk=topk,
                benchmark_core_weight=benchmark_core_weight,
                alpha_weighting=str(strategy_cfg.get("weighting", "rank")),
                alpha_temperature=float(strategy_cfg.get("temperature", 1.0)),
                max_weight=strategy_cfg.get("max_weight"),
                max_benchmark_weight=strategy_cfg.get("benchmark_max_weight"),
                max_active_weight=strategy_cfg.get("max_active_weight"),
            )
        )
        if (
            strategy_cfg.get("max_turnover") is not None
            or float(strategy_cfg.get("min_trade_weight") or 0.0) > 0
            or float(strategy_cfg.get("min_position_weight") or 0.0) > 0
            or strategy_cfg.get("max_holdings") is not None
        ):
            weights = limit_turnover_toward_target(
                weights,
                current_weights,
                max_turnover=strategy_cfg.get("max_turnover"),
                min_trade_weight=float(strategy_cfg.get("min_trade_weight") or 0.0),
                min_position_weight=float(strategy_cfg.get("min_position_weight") or 0.0),
                max_positions=strategy_cfg.get("max_holdings"),
            )
        weights = _apply_strategy_hedge_overlay(
            weights,
            strategy_cfg=strategy_cfg,
            strategy_class=strategy_class,
            available_hedges=available_hedges,
            trade_dt=trade_dt,
            close_history=close_history,
            hedge_state=hedge_state,
        )
        return normalize_long_weights(weights)

    if "ScoreWeighted" in str(strategy_class):
        liquidity_buffer = max(1, int(strategy_cfg.get("liquidity_buffer") or 3))
        cand_n = min(len(score), topk * liquidity_buffer)
        candidates = score.sort_values(ascending=False).head(cand_n).head(topk)
        weights = rank_score_weights(
            candidates,
            topn=topk,
            method=str(strategy_cfg.get("weighting", "rank")),
            temperature=float(strategy_cfg.get("temperature", 1.0)),
        )
        max_weight = strategy_cfg.get("max_weight")
        if max_weight is not None and not weights.empty:
            weights = weights.clip(upper=max(0.0, float(max_weight)))
        return normalize_long_weights(weights)

    return _active_risk_score_weights(score, topk=topk)


def _active_risk_metrics_for_predictions(
    pred: pd.DataFrame,
    *,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    bt_calendar: List[pd.Timestamp],
    strategy_cfg: Dict,
    strategy_class: str,
    rebalance_weekday: Optional[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    from qlib.contrib.strategy.benchmark_aware import active_weight_metrics, benchmark_weights_from_marketcap, normalize_long_weights
    from qlib.data import D

    score = _normalize_datetime_instrument_index(_extract_score_series(pred))
    if not isinstance(score.index, pd.MultiIndex) or score.empty:
        return pd.DataFrame()
    trade_signal_pairs = _strategy_trade_signal_pairs(
        bt_calendar,
        rebalance_weekday=rebalance_weekday,
        start=bt_start,
        end=bt_end,
        signal_shift=int(getattr(args, "strategy_signal_shift", 1)),
    )
    if not trade_signal_pairs:
        return pd.DataFrame()
    signal_dates = [signal_dt for _, signal_dt in trade_signal_pairs]
    signal_start = min(signal_dates)
    signal_end = max(signal_dates)
    dates = pd.DatetimeIndex(score.index.get_level_values("datetime")).normalize()
    score = score[(dates >= signal_start) & (dates <= signal_end)].dropna()
    if score.empty:
        return pd.DataFrame()

    benchmark_tickers = _strategy_benchmark_tickers(strategy_cfg) if "BenchmarkAware" in str(strategy_class) else []
    hedge_tickers = _strategy_hedge_tickers(strategy_cfg) if _strategy_is_hedged(strategy_class) else []
    hedge_market = str(strategy_cfg.get("market_index") or "").strip().upper() if hedge_tickers else ""
    instruments = sorted(
        set(score.index.get_level_values("instrument").astype(str))
        .union(benchmark_tickers)
        .union(hedge_tickers)
    )
    if hedge_market:
        instruments = sorted(set(instruments).union({hedge_market}))
    feature_weights, feature_mins = _strategy_feature_control_maps(strategy_cfg)
    control_fields = list(dict.fromkeys([*feature_weights.keys(), *feature_mins.keys()]))
    marketcap_field = str(args.active_risk_marketcap_field)
    benchmark_availability_field = "$close"
    needs_close_history = (
        benchmark_tickers
        or hedge_tickers
        or bool(strategy_cfg.get("dynamic_alpha_weight", False))
        or bool(hedge_market)
    )
    fields = list(dict.fromkeys([marketcap_field, *control_fields, *([benchmark_availability_field] if needs_close_history else [])]))
    feature_start = min(signal_start, bt_start)
    if bool(strategy_cfg.get("dynamic_alpha_weight", False)):
        feature_start = pd.Timestamp(feature_start) - pd.Timedelta(
            days=max(180, int(strategy_cfg.get("alpha_quality_window") or 63) * 3)
        )
    if hedge_tickers:
        hedge_lookback = max(
            int(strategy_cfg.get("hedge_trend_window") or 63),
            int(strategy_cfg.get("hedge_drawdown_window") or 126),
            int(strategy_cfg.get("hedge_crash_return_lookback") or 5) + 1,
            int(strategy_cfg.get("hedge_min_history") or 40),
        )
        feature_start = min(
            pd.Timestamp(feature_start),
            pd.Timestamp(feature_start) - pd.Timedelta(days=max(180, hedge_lookback * 3)),
        )
    raw_features = D.features(instruments, fields, start_time=feature_start, end_time=bt_end)
    features = _normalize_feature_frame(raw_features, fields)
    if features.empty or (not benchmark_tickers and marketcap_field not in features.columns):
        return pd.DataFrame()

    if control_fields:
        score = _apply_strategy_score_controls(
            score,
            features.reindex(columns=control_fields),
            feature_score_weights=feature_weights,
            feature_min_percentiles=feature_mins,
        )

    sector_map = _load_strategy_sector_map(strategy_cfg)
    topk = max(1, int(strategy_cfg.get("topk") or args.active_risk_topk or 40))
    metric_benchmark_topn = max(1, int(args.active_risk_benchmark_topn))

    rows = []
    prev_port = pd.Series(dtype=float)
    hedge_state: Dict[str, float] = {}
    dt_level = "datetime" if "datetime" in score.index.names else 0
    score_by_date = {
        pd.Timestamp(dt).normalize(): day_score.droplevel(dt_level)
        for dt, day_score in score.groupby(level=dt_level, sort=True)
    }
    for trade_dt, signal_dt in trade_signal_pairs:
        day_score = score_by_date.get(signal_dt)
        if day_score is None or day_score.empty:
            continue
        day_score.index = day_score.index.astype(str)
        day_score = _apply_sector_cap_to_score(day_score, strategy_cfg, sector_map)

        day_features = _feature_frame_for_date(features, signal_dt)
        if day_features.empty:
            continue
        marketcap = (
            pd.to_numeric(day_features[marketcap_field], errors="coerce")
            if marketcap_field in day_features.columns
            else pd.Series(dtype=float)
        )
        if benchmark_tickers:
            available_benchmark = benchmark_tickers
            if benchmark_availability_field in day_features.columns:
                close = pd.to_numeric(day_features[benchmark_availability_field], errors="coerce")
                available_benchmark = close.replace([np.inf, -np.inf], np.nan).dropna().index.astype(str).tolist()
            metric_bench = _equal_weight_benchmark_tickers(strategy_cfg, available=available_benchmark)
            explicit_bench = metric_bench
        else:
            metric_bench = benchmark_weights_from_marketcap(marketcap, topn=metric_benchmark_topn)
            explicit_bench = None
        if metric_bench.empty:
            continue
        available_hedges = None
        if hedge_tickers:
            available_hedges = hedge_tickers
            if benchmark_availability_field in day_features.columns:
                close = pd.to_numeric(day_features[benchmark_availability_field], errors="coerce")
                available_hedges = close.replace([np.inf, -np.inf], np.nan).dropna().index.astype(str).tolist()

        port = _strategy_portfolio_weights_for_day(
            day_score,
            strategy_cfg=strategy_cfg,
            strategy_class=strategy_class,
            topk=topk,
            marketcap=marketcap,
            benchmark_weights=explicit_bench,
            current_weights=prev_port,
            trade_dt=trade_dt,
            close_history=features[benchmark_availability_field] if benchmark_availability_field in features.columns else None,
            available_hedges=available_hedges,
            hedge_state=hedge_state,
        )
        port = normalize_long_weights(port)
        if port.empty:
            continue
        prev_port = port
        metrics = active_weight_metrics(port, metric_bench, sector_map=sector_map)
        if benchmark_tickers:
            core_tickers = {str(t).upper() for t in metric_bench.index.astype(str)}
            hedge_set = {str(t).upper() for t in hedge_tickers}
            port_upper = pd.Series(port.to_numpy(dtype=float), index=pd.Index(port.index.astype(str).str.upper()))
            try:
                core_target = float(strategy_cfg.get("benchmark_core_weight", metric_bench.sum()))
            except (TypeError, ValueError):
                core_target = float(metric_bench.sum())
            core_target = float(np.clip(core_target, 0.0, 1.0))
            core_weight = float(port_upper.reindex(sorted(core_tickers)).fillna(0.0).sum())
            stock_mask = [inst not in core_tickers and inst not in hedge_set for inst in port_upper.index]
            stock_weights = port_upper.loc[stock_mask]
            stock_weights = pd.to_numeric(stock_weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            stock_weights = stock_weights[stock_weights > 0]
            metrics.update(
                {
                    "core_target_weight": core_target,
                    "core_weight_held": core_weight,
                    "core_weight_abs_gap": abs(core_weight - core_target),
                    "stock_overlay_weight": float(stock_weights.sum()) if not stock_weights.empty else 0.0,
                    "max_single_stock_weight": float(stock_weights.max()) if not stock_weights.empty else 0.0,
                    "stock_names": float(len(stock_weights)),
                }
            )
            if sector_map and not stock_weights.empty:
                stock_sectors = pd.Series(
                    [sector_map.get(str(inst).upper().strip(), "__UNKNOWN__") for inst in stock_weights.index],
                    index=stock_weights.index,
                )
                sector_stock = stock_weights.groupby(stock_sectors).sum()
                metrics["max_sector_stock_weight"] = (
                    float(sector_stock.max()) if not sector_stock.empty else float("nan")
                )
        metrics["datetime"] = trade_dt
        metrics["signal_datetime"] = signal_dt
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def _summarize_active_risk(metrics: pd.DataFrame, recent_rebalances: Optional[int] = None) -> Dict[str, float]:
    if metrics.empty:
        return {"days": 0}
    frame = metrics.tail(int(recent_rebalances)) if recent_rebalances is not None and int(recent_rebalances) > 0 else metrics
    out = {
        "days": float(len(frame)),
        "mean_active_share": float(frame["active_share"].mean()),
        "p95_active_share": float(frame["active_share"].quantile(0.95)),
        "mean_max_abs_active_weight": float(frame["max_abs_active_weight"].mean()),
        "mean_benchmark_weight_held": float(frame["benchmark_weight_held"].mean()),
        "min_benchmark_weight_held": float(frame["benchmark_weight_held"].min()),
        "mean_portfolio_names": float(frame["portfolio_names"].mean()),
    }
    if "max_abs_sector_active_weight" in frame.columns:
        out["mean_max_abs_sector_active_weight"] = float(frame["max_abs_sector_active_weight"].mean())
    if "core_weight_abs_gap" in frame.columns:
        out["mean_core_target_weight"] = float(pd.to_numeric(frame["core_target_weight"], errors="coerce").mean())
        out["mean_core_weight_held"] = float(pd.to_numeric(frame["core_weight_held"], errors="coerce").mean())
        out["mean_core_weight_abs_gap"] = float(pd.to_numeric(frame["core_weight_abs_gap"], errors="coerce").mean())
        out["mean_stock_overlay_weight"] = float(pd.to_numeric(frame["stock_overlay_weight"], errors="coerce").mean())
        out["mean_max_single_stock_weight"] = float(pd.to_numeric(frame["max_single_stock_weight"], errors="coerce").mean())
        out["mean_stock_names"] = float(pd.to_numeric(frame["stock_names"], errors="coerce").mean())
    if "max_sector_stock_weight" in frame.columns:
        out["mean_max_sector_stock_weight"] = float(pd.to_numeric(frame["max_sector_stock_weight"], errors="coerce").mean())
    return out


def _active_risk_gate_rows(
    full: Dict[str, float],
    recent: Dict[str, float],
    args: argparse.Namespace,
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    full_days = int(full.get("days", 0) or 0)
    recent_days = int(recent.get("days", 0) or 0)
    rows.append(
        (
            "active_risk_full_rebalances",
            full_days >= int(args.active_risk_min_rebalances),
            f"{full_days} >= {int(args.active_risk_min_rebalances)}",
        )
    )
    rows.append(
        (
            "active_risk_recent_rebalances",
            recent_days >= min(int(args.active_risk_recent_rebalances), full_days),
            f"{recent_days} >= {min(int(args.active_risk_recent_rebalances), full_days)}",
        )
    )
    if full.get("mean_core_weight_abs_gap") is not None:
        checks = [
            (
                "active_risk_mean_core_weight_gap",
                full.get("mean_core_weight_abs_gap"),
                args.active_risk_max_mean_core_weight_gap,
                "<=",
            ),
            (
                "active_risk_recent_core_weight_gap",
                recent.get("mean_core_weight_abs_gap"),
                args.active_risk_max_recent_core_weight_gap,
                "<=",
            ),
            (
                "active_risk_mean_core_weight_held",
                full.get("mean_core_weight_held"),
                args.active_risk_min_mean_benchmark_coverage,
                ">=",
            ),
            (
                "active_risk_recent_core_weight_held",
                recent.get("mean_core_weight_held"),
                args.active_risk_min_recent_benchmark_coverage,
                ">=",
            ),
            (
                "active_risk_mean_stock_overlay_weight",
                full.get("mean_stock_overlay_weight"),
                args.active_risk_max_mean_stock_overlay_weight,
                "<=",
            ),
            (
                "active_risk_mean_max_single_stock_weight",
                full.get("mean_max_single_stock_weight"),
                args.active_risk_max_mean_single_stock_weight,
                "<=",
            ),
            (
                "active_risk_mean_stock_names",
                full.get("mean_stock_names"),
                args.active_risk_max_mean_portfolio_names,
                "<=",
            ),
        ]
        if full.get("mean_max_sector_stock_weight") is not None:
            sector_stock_threshold = getattr(args, "active_risk_max_mean_sector_stock_weight", None)
            if sector_stock_threshold is None:
                sector_stock_threshold = args.active_risk_max_mean_sector_active_weight
            checks.append(
                (
                    "active_risk_mean_sector_stock_weight",
                    full.get("mean_max_sector_stock_weight"),
                    sector_stock_threshold,
                    "<=",
                )
            )
    else:
        checks = [
            ("active_risk_mean_active_share", full.get("mean_active_share"), args.active_risk_max_mean_active_share, "<="),
            (
                "active_risk_recent_mean_active_share",
                recent.get("mean_active_share"),
                args.active_risk_max_recent_mean_active_share,
                "<=",
            ),
            (
                "active_risk_mean_max_abs_active_weight",
                full.get("mean_max_abs_active_weight"),
                args.active_risk_max_mean_abs_active_weight,
                "<=",
            ),
            (
                "active_risk_mean_benchmark_coverage",
                full.get("mean_benchmark_weight_held"),
                args.active_risk_min_mean_benchmark_coverage,
                ">=",
            ),
            (
                "active_risk_recent_benchmark_coverage",
                recent.get("mean_benchmark_weight_held"),
                args.active_risk_min_recent_benchmark_coverage,
                ">=",
            ),
            (
                "active_risk_mean_portfolio_names",
                full.get("mean_portfolio_names"),
                args.active_risk_max_mean_portfolio_names,
                "<=",
            ),
        ]
        if full.get("mean_max_abs_sector_active_weight") is not None:
            checks.append(
                (
                    "active_risk_mean_sector_active_weight",
                    full.get("mean_max_abs_sector_active_weight"),
                    args.active_risk_max_mean_sector_active_weight,
                    "<=",
                )
            )
    for name, value, threshold, op in checks:
        val = _safe_float(value)
        threshold = float(threshold)
        ok = val is not None and (val <= threshold if op == "<=" else val >= threshold)
        rows.append((name, ok, f"{_format_float(val)} {op} {_format_float(threshold)}"))
    return rows


def _print_active_risk_summary(label: str, metrics: Dict[str, float]) -> None:
    keys = [
        "days",
        "mean_active_share",
        "p95_active_share",
        "mean_max_abs_active_weight",
        "mean_benchmark_weight_held",
        "min_benchmark_weight_held",
        "mean_core_target_weight",
        "mean_core_weight_held",
        "mean_core_weight_abs_gap",
        "mean_stock_overlay_weight",
        "mean_max_single_stock_weight",
        "mean_stock_names",
        "mean_max_sector_stock_weight",
        "mean_max_abs_sector_active_weight",
        "mean_portfolio_names",
    ]
    vals = []
    for key in keys:
        if key not in metrics:
            continue
        val = metrics.get(key)
        vals.append(f"{key}={int(val or 0)}" if key == "days" else f"{key}={_format_float(val)}")
    _print_kv(label, ", ".join(vals))


def _feature_frame_for_date(features: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame) or features.empty or not isinstance(features.index, pd.MultiIndex):
        return pd.DataFrame()
    dt_level = "datetime" if "datetime" in features.index.names else 0
    try:
        day = features.xs(pd.Timestamp(dt).normalize(), level=dt_level)
    except KeyError:
        return pd.DataFrame()
    if isinstance(day.index, pd.MultiIndex):
        level = "instrument" if "instrument" in day.index.names else day.index.nlevels - 1
        day = day.groupby(level=level).last()
    day.index = day.index.astype(str)
    return day


def _label_series_for_date(label: pd.Series, dt: pd.Timestamp) -> pd.Series:
    if not isinstance(label.index, pd.MultiIndex) or label.empty:
        return pd.Series(dtype=float)
    dt_level = "datetime" if "datetime" in label.index.names else 0
    try:
        day = label.xs(pd.Timestamp(dt).normalize(), level=dt_level)
    except KeyError:
        return pd.Series(dtype=float)
    if isinstance(day.index, pd.MultiIndex):
        level = "instrument" if "instrument" in day.index.names else day.index.nlevels - 1
        day = day.groupby(level=level).last()
    day.index = day.index.astype(str)
    return pd.to_numeric(day, errors="coerce")


def _strategy_weighted_label_frame(
    pred: pd.DataFrame,
    *,
    label_expr: str,
    benchmark,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    label_horizon_days: int,
    label_ref_start_days: int,
    bt_calendar: List[pd.Timestamp],
    strategy_cfg: Dict,
    strategy_class: str,
    rebalance_weekday: Optional[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    from qlib.data import D

    score = _normalize_datetime_instrument_index(_extract_score_series(pred))
    if not isinstance(score.index, pd.MultiIndex) or score.empty:
        return pd.DataFrame()
    trade_signal_pairs = _strategy_trade_signal_pairs(
        bt_calendar,
        rebalance_weekday=rebalance_weekday,
        start=bt_start,
        end=bt_end,
        signal_shift=int(getattr(args, "strategy_signal_shift", 1)),
    )
    if not trade_signal_pairs:
        return pd.DataFrame()
    signal_dates = [signal_dt for _, signal_dt in trade_signal_pairs]
    signal_start = min(signal_dates)
    signal_end = max(signal_dates)
    dates = pd.DatetimeIndex(score.index.get_level_values("datetime")).normalize()
    score = score[(dates >= signal_start) & (dates <= signal_end)].dropna()
    if score.empty:
        return pd.DataFrame()

    feature_weights, feature_mins = _strategy_feature_control_maps(strategy_cfg)
    control_fields = list(dict.fromkeys([*feature_weights.keys(), *feature_mins.keys()]))
    is_benchmark_aware = "BenchmarkAware" in str(strategy_class)
    benchmark_tickers = _strategy_benchmark_tickers(strategy_cfg) if is_benchmark_aware else []
    hedge_tickers = _strategy_hedge_tickers(strategy_cfg) if _strategy_is_hedged(strategy_class) else []
    dynamic_risk_market = (
        str(strategy_cfg.get("market_index") or "").strip().upper()
        if bool(strategy_cfg.get("dynamic_risk", False))
        else ""
    )
    hedge_market = str(strategy_cfg.get("market_index") or "").strip().upper() if hedge_tickers else ""
    instruments = sorted(
        set(score.index.get_level_values("instrument").astype(str))
        .union(benchmark_tickers)
        .union(hedge_tickers)
    )
    if dynamic_risk_market:
        instruments = sorted(set(instruments).union({dynamic_risk_market}))
    if hedge_market:
        instruments = sorted(set(instruments).union({hedge_market}))
    benchmark_weight_field = str(strategy_cfg.get("benchmark_weight_field") or "").strip() or None
    marketcap_field = (
        args.strategy_weighted_marketcap_field
        or strategy_cfg.get("benchmark_marketcap_field")
        or getattr(args, "active_risk_marketcap_field", "$marketcap_q")
        or "$marketcap_q"
    )
    weight_fields: List[str] = []
    if is_benchmark_aware and not benchmark_tickers:
        weight_fields.append(str(benchmark_weight_field or marketcap_field))
    close_history_field = "$close"
    close_history_fields = (
        [close_history_field]
        if benchmark_tickers
        or hedge_tickers
        or bool(strategy_cfg.get("dynamic_alpha_weight", False))
        or bool(dynamic_risk_market)
        or bool(hedge_market)
        else []
    )
    fields = list(dict.fromkeys([label_expr, *weight_fields, *control_fields, *close_history_fields]))
    feature_start = min(signal_start, bt_start)
    if bool(strategy_cfg.get("dynamic_alpha_weight", False)):
        feature_start = pd.Timestamp(feature_start) - pd.Timedelta(
            days=max(180, int(strategy_cfg.get("alpha_quality_window") or 63) * 3)
        )
    if hedge_tickers:
        hedge_lookback = max(
            int(strategy_cfg.get("hedge_trend_window") or 63),
            int(strategy_cfg.get("hedge_drawdown_window") or 126),
            int(strategy_cfg.get("hedge_crash_return_lookback") or 5) + 1,
            int(strategy_cfg.get("hedge_min_history") or 40),
        )
        feature_start = min(
            pd.Timestamp(feature_start),
            pd.Timestamp(feature_start) - pd.Timedelta(days=max(180, hedge_lookback * 3)),
        )
    raw_features = D.features(instruments, fields, start_time=feature_start, end_time=bt_end)
    features = _normalize_feature_frame(raw_features, fields)
    if features.empty or label_expr not in features.columns:
        return pd.DataFrame()

    label = _normalize_datetime_instrument_index(pd.to_numeric(features[label_expr], errors="coerce"))
    bench_fwd = _benchmark_forward_return(
        benchmark,
        label_horizon_days=label_horizon_days,
        label_ref_start_days=label_ref_start_days,
    )
    if bench_fwd is not None and not bench_fwd.empty:
        label_dates = pd.DatetimeIndex(label.index.get_level_values("datetime"))
        label = label - bench_fwd.reindex(label_dates).to_numpy()

    if control_fields:
        score = _apply_strategy_score_controls(
            score,
            features.reindex(columns=control_fields),
            feature_score_weights=feature_weights,
            feature_min_percentiles=feature_mins,
        )

    sector_map = _load_strategy_sector_map(strategy_cfg)
    topk = max(1, int(args.strategy_weighted_topk) if int(args.strategy_weighted_topk) > 0 else int(strategy_cfg.get("topk", 40)))

    rows = []
    prev_port = pd.Series(dtype=float)
    hedge_state: Dict[str, float] = {}
    dt_level = "datetime" if "datetime" in score.index.names else 0
    score_by_date = {
        pd.Timestamp(dt).normalize(): day_score.droplevel(dt_level)
        for dt, day_score in score.groupby(level=dt_level, sort=True)
    }
    for trade_dt, signal_dt in trade_signal_pairs:
        day_score = score_by_date.get(signal_dt)
        if day_score is None or day_score.empty:
            continue
        day_score.index = day_score.index.astype(str)
        day_score = _apply_sector_cap_to_score(day_score, strategy_cfg, sector_map)

        day_features = _feature_frame_for_date(features, signal_dt)
        marketcap = None
        explicit_bench = None
        if is_benchmark_aware:
            if day_features.empty:
                continue
            if benchmark_tickers:
                available_benchmark = day_features.index
                if close_history_field in day_features.columns:
                    close = pd.to_numeric(day_features[close_history_field], errors="coerce")
                    available_benchmark = close.replace([np.inf, -np.inf], np.nan).dropna().index
                explicit_bench = _equal_weight_benchmark_tickers(strategy_cfg, available=available_benchmark)
                if explicit_bench.empty:
                    continue
            elif benchmark_weight_field and benchmark_weight_field in day_features.columns:
                explicit_bench = pd.to_numeric(day_features[benchmark_weight_field], errors="coerce")
            elif str(marketcap_field) in day_features.columns:
                marketcap = pd.to_numeric(day_features[str(marketcap_field)], errors="coerce")
            else:
                continue
        available_hedges = None
        if hedge_tickers:
            if day_features.empty:
                continue
            available_hedges = hedge_tickers
            if close_history_field in day_features.columns:
                close = pd.to_numeric(day_features[close_history_field], errors="coerce")
                available_hedges = close.replace([np.inf, -np.inf], np.nan).dropna().index.astype(str).tolist()

        port = _strategy_portfolio_weights_for_day(
            day_score,
            strategy_cfg=strategy_cfg,
            strategy_class=strategy_class,
            topk=topk,
            marketcap=marketcap,
            benchmark_weights=explicit_bench,
            current_weights=prev_port,
            trade_dt=trade_dt,
            close_history=features[close_history_field] if close_history_field in features.columns else None,
            available_hedges=available_hedges,
            hedge_state=hedge_state,
        )
        if port.empty:
            continue
        prev_port = port

        day_label = _label_series_for_date(label, signal_dt).reindex(port.index)
        valid_label = day_label.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_label.empty:
            continue
        covered_weight = float(port.reindex(valid_label.index).fillna(0.0).sum())
        if not math.isfinite(covered_weight) or covered_weight <= 1e-12:
            continue
        eval_weights = port.reindex(valid_label.index).fillna(0.0) / covered_weight
        rows.append(
            {
                "datetime": trade_dt,
                "signal_datetime": signal_dt,
                "weighted_label": float((eval_weights * valid_label).sum()),
                "selected_mean_label": float(valid_label.mean()),
                "label_weight_coverage": covered_weight,
                "portfolio_names": float((port > 0).sum()),
                "label_names": float(len(valid_label)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def _strategy_weighted_quality_metrics(frame: pd.DataFrame, recent_rebalances: Optional[int] = None) -> Dict[str, float]:
    if frame.empty:
        return {"rebalances": 0}
    data = frame.tail(int(recent_rebalances)) if recent_rebalances is not None and int(recent_rebalances) > 0 else frame
    label = pd.to_numeric(data["weighted_label"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if label.empty:
        return {"rebalances": 0}
    std = float(label.std(ddof=1)) if len(label) > 1 else float("nan")
    out = {
        "rebalances": float(len(label)),
        "mean_weighted_label": float(label.mean()),
        "median_weighted_label": float(label.median()),
        "positive_label_rate": float((label > 0).mean()),
        "mean_label_weight_coverage": float(pd.to_numeric(data["label_weight_coverage"], errors="coerce").mean()),
        "min_label_weight_coverage": float(pd.to_numeric(data["label_weight_coverage"], errors="coerce").min()),
        "mean_portfolio_names": float(pd.to_numeric(data["portfolio_names"], errors="coerce").mean()),
    }
    out["weighted_label_ir"] = float(out["mean_weighted_label"] / std * np.sqrt(52)) if math.isfinite(std) and std > 0 else float("nan")
    return out


def _strategy_weighted_quality_by_year(frame: pd.DataFrame) -> List[Tuple[str, Dict[str, float]]]:
    if frame.empty:
        return []
    dates = pd.DatetimeIndex(frame.index)
    rows: List[Tuple[str, Dict[str, float]]] = []
    for year in sorted(dates.year.unique()):
        rows.append((str(int(year)), _strategy_weighted_quality_metrics(frame.loc[dates.year == int(year)])))
    return rows


def _strategy_weighted_quality_gate_rows(
    full: Dict[str, float],
    recent: Dict[str, float],
    args: argparse.Namespace,
    *,
    yearly: Optional[List[Tuple[str, Dict[str, float]]]] = None,
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    full_n = int(full.get("rebalances", 0) or 0)
    recent_n = int(recent.get("rebalances", 0) or 0)
    rows.append(
        (
            "strategy_weighted_full_rebalances",
            full_n >= int(args.strategy_weighted_min_rebalances),
            f"{full_n} >= {int(args.strategy_weighted_min_rebalances)}",
        )
    )
    recent_required = min(int(args.strategy_weighted_min_recent_rebalances), full_n)
    rows.append(
        (
            "strategy_weighted_recent_rebalances",
            recent_n >= recent_required,
            f"{recent_n} >= {recent_required}",
        )
    )
    checks = [
        (
            "strategy_weighted_full_mean_label",
            full.get("mean_weighted_label"),
            args.strategy_weighted_min_full_mean_label,
            ">=",
        ),
        (
            "strategy_weighted_recent_mean_label",
            recent.get("mean_weighted_label"),
            args.strategy_weighted_min_recent_mean_label,
            ">=",
        ),
        (
            "strategy_weighted_positive_label_rate",
            full.get("positive_label_rate"),
            args.strategy_weighted_min_positive_label_rate,
            ">=",
        ),
        (
            "strategy_weighted_recent_positive_label_rate",
            recent.get("positive_label_rate"),
            args.strategy_weighted_min_recent_positive_label_rate,
            ">=",
        ),
        (
            "strategy_weighted_mean_label_weight_coverage",
            full.get("mean_label_weight_coverage"),
            args.strategy_weighted_min_mean_label_weight_coverage,
            ">=",
        ),
        (
            "strategy_weighted_recent_label_weight_coverage",
            recent.get("mean_label_weight_coverage"),
            args.strategy_weighted_min_recent_label_weight_coverage,
            ">=",
        ),
    ]
    for name, value, threshold, op in checks:
        val = _safe_float(value)
        threshold = float(threshold)
        ok = val is not None and (val >= threshold if op == ">=" else val <= threshold)
        rows.append((name, ok, f"{_format_float(val)} {op} {_format_float(threshold)}"))

    if yearly is not None:
        min_year_rebalances = int(args.strategy_weighted_min_year_rebalances)
        eligible = [(year, metrics) for year, metrics in yearly if int(metrics.get("rebalances", 0) or 0) >= min_year_rebalances]
        skipped = len(yearly) - len(eligible)
        configured_min_positive_years = int(args.strategy_weighted_min_positive_years)
        min_positive_years = (
            min(configured_min_positive_years, len(eligible))
            if not bool(getattr(args, "strict_positive_year_count", False))
            else configured_min_positive_years
        )
        positive_years = sum(
            1
            for _, metrics in eligible
            if _safe_float(metrics.get("mean_weighted_label")) is not None and float(metrics["mean_weighted_label"]) >= 0
        )
        detail = (
            f"{positive_years} >= {min_positive_years} "
            f"(eligible_years={len(eligible)}, min_year_rebalances={min_year_rebalances}, skipped_short_years={skipped}"
        )
        if min_positive_years != configured_min_positive_years:
            detail += f", configured_min_positive_years={configured_min_positive_years}"
        detail += ")"
        rows.append(
            (
                "strategy_weighted_positive_years",
                positive_years >= min_positive_years,
                detail,
            )
        )
        worst_year = None
        if eligible:
            vals = [_safe_float(metrics.get("mean_weighted_label")) for _, metrics in eligible]
            vals = [v for v in vals if v is not None]
            worst_year = min(vals) if vals else None
        threshold = float(args.strategy_weighted_min_worst_year_mean_label)
        rows.append(
            (
                "strategy_weighted_worst_year_mean_label",
                worst_year is not None and worst_year >= threshold,
                f"{_format_float(worst_year)} >= {_format_float(threshold)}",
            )
        )
    return rows


def _print_strategy_weighted_quality_summary(label: str, metrics: Dict[str, float]) -> None:
    keys = [
        "rebalances",
        "mean_weighted_label",
        "median_weighted_label",
        "weighted_label_ir",
        "positive_label_rate",
        "mean_label_weight_coverage",
        "min_label_weight_coverage",
        "mean_portfolio_names",
    ]
    vals = []
    for key in keys:
        val = metrics.get(key)
        vals.append(f"{key}={int(val or 0)}" if key == "rebalances" else f"{key}={_format_float(val)}")
    _print_kv(label, ", ".join(vals))


def _print_strategy_weighted_quality_yearly(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    if not rows:
        print("(no yearly strategy-weighted quality rows)")
        return
    headers = [
        "year",
        "rebalances",
        "mean_weighted_label",
        "weighted_label_ir",
        "positive_label_rate",
        "mean_label_weight_coverage",
        "mean_portfolio_names",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for year, metrics in rows:
        print(
            " | ".join(
                [
                    year,
                    str(int(metrics.get("rebalances", 0) or 0)),
                    _format_float(metrics.get("mean_weighted_label")),
                    _format_float(metrics.get("weighted_label_ir")),
                    _format_float(metrics.get("positive_label_rate")),
                    _format_float(metrics.get("mean_label_weight_coverage")),
                    _format_float(metrics.get("mean_portfolio_names")),
                ]
            )
        )


def _benchmark_interval_return(benchmark, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    if not isinstance(benchmark, pd.Series):
        return None
    bench = benchmark.copy()
    bench.index = pd.DatetimeIndex(bench.index).normalize()
    bench = bench.sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    ret = pd.to_numeric(bench, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end <= start:
        return None
    window = ret[(ret.index > start) & (ret.index <= end)].dropna()
    if window.empty:
        return None
    return float((1.0 + window).prod() - 1.0)


def _annualized_interval_return(returns: pd.Series, holding_days: pd.Series) -> float:
    ret = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    days = pd.to_numeric(holding_days, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = ret.notna() & days.notna() & (days > 0)
    if not bool(valid.any()):
        return float("nan")
    gross = (1.0 + ret.loc[valid]).clip(lower=1e-12)
    total_days = float(days.loc[valid].sum())
    if not math.isfinite(total_days) or total_days <= 0:
        return float("nan")
    return float(gross.prod() ** (252.0 / total_days) - 1.0)


def _rebalance_interval_quality_frame(
    pred: pd.DataFrame,
    *,
    benchmark,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
    bt_calendar: List[pd.Timestamp],
    strategy_cfg: Dict,
    strategy_class: str,
    rebalance_weekday: Optional[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    from qlib.contrib.strategy.benchmark_aware import benchmark_weights_from_marketcap, normalize_long_weights
    from qlib.data import D

    score = _normalize_datetime_instrument_index(_extract_score_series(pred))
    if not isinstance(score.index, pd.MultiIndex) or score.empty:
        return pd.DataFrame()
    trade_signal_pairs = _strategy_trade_signal_pairs(
        bt_calendar,
        rebalance_weekday=rebalance_weekday,
        start=bt_start,
        end=bt_end,
        signal_shift=int(getattr(args, "strategy_signal_shift", 1)),
    )
    if len(trade_signal_pairs) < 2:
        return pd.DataFrame()
    signal_dates_for_scores = [signal_dt for _, signal_dt in trade_signal_pairs]
    signal_start = min(signal_dates_for_scores)
    signal_end = max(signal_dates_for_scores)
    dates = pd.DatetimeIndex(score.index.get_level_values("datetime")).normalize()
    score = score[(dates >= signal_start) & (dates <= signal_end)].dropna()
    if score.empty:
        return pd.DataFrame()

    feature_weights, feature_mins = _strategy_feature_control_maps(strategy_cfg)
    control_fields = list(dict.fromkeys([*feature_weights.keys(), *feature_mins.keys()]))
    is_benchmark_aware = "BenchmarkAware" in str(strategy_class)
    benchmark_tickers = _strategy_benchmark_tickers(strategy_cfg) if is_benchmark_aware else []
    hedge_tickers = _strategy_hedge_tickers(strategy_cfg) if _strategy_is_hedged(strategy_class) else []
    dynamic_risk_market = (
        str(strategy_cfg.get("market_index") or "").strip().upper()
        if bool(strategy_cfg.get("dynamic_risk", False))
        else ""
    )
    hedge_market = str(strategy_cfg.get("market_index") or "").strip().upper() if hedge_tickers else ""
    instruments = sorted(
        set(score.index.get_level_values("instrument").astype(str))
        .union(benchmark_tickers)
        .union(hedge_tickers)
    )
    if dynamic_risk_market:
        instruments = sorted(set(instruments).union({dynamic_risk_market}))
    if hedge_market:
        instruments = sorted(set(instruments).union({hedge_market}))
    benchmark_weight_field = str(strategy_cfg.get("benchmark_weight_field") or "").strip() or None
    marketcap_field = (
        args.rebalance_interval_marketcap_field
        or strategy_cfg.get("benchmark_marketcap_field")
        or getattr(args, "active_risk_marketcap_field", "$marketcap_q")
        or "$marketcap_q"
    )
    price_field = str(args.rebalance_interval_price_field or "").strip()
    if not price_field:
        deal_price = str(args.rebalance_interval_deal_price or "close").strip().lower()
        price_field = f"${deal_price}" if deal_price in {"open", "close"} else "$close"
    weight_fields: List[str] = []
    if is_benchmark_aware and not benchmark_tickers:
        weight_fields.append(str(benchmark_weight_field or marketcap_field))
    close_history_field = "$close"
    close_history_fields = (
        [close_history_field]
        if benchmark_tickers
        or hedge_tickers
        or bool(strategy_cfg.get("dynamic_alpha_weight", False))
        or bool(dynamic_risk_market)
        or bool(hedge_market)
        else []
    )
    fields = list(dict.fromkeys([price_field, *weight_fields, *control_fields, *close_history_fields]))
    trade_dates = [trade_dt for trade_dt, _ in trade_signal_pairs]
    feature_start = min(min(trade_dates), signal_start)
    if bool(strategy_cfg.get("dynamic_alpha_weight", False)):
        feature_start = pd.Timestamp(feature_start) - pd.Timedelta(
            days=max(180, int(strategy_cfg.get("alpha_quality_window") or 63) * 3)
        )
    if dynamic_risk_market:
        dynamic_risk_lookback = max(
            int(strategy_cfg.get("market_trend_window") or 63),
            int(strategy_cfg.get("market_drawdown_window") or 126),
            int(strategy_cfg.get("crash_return_lookback") or 5) + 1,
            int(strategy_cfg.get("risk_min_history") or 20),
        )
        feature_start = min(
            pd.Timestamp(feature_start),
            pd.Timestamp(feature_start) - pd.Timedelta(days=max(180, dynamic_risk_lookback * 3)),
        )
    if hedge_tickers:
        hedge_lookback = max(
            int(strategy_cfg.get("hedge_trend_window") or 63),
            int(strategy_cfg.get("hedge_drawdown_window") or 126),
            int(strategy_cfg.get("hedge_crash_return_lookback") or 5) + 1,
            int(strategy_cfg.get("hedge_min_history") or 40),
        )
        feature_start = min(
            pd.Timestamp(feature_start),
            pd.Timestamp(feature_start) - pd.Timedelta(days=max(180, hedge_lookback * 3)),
        )
    raw_features = D.features(instruments, fields, start_time=feature_start, end_time=trade_dates[-1])
    features = _normalize_feature_frame(raw_features, fields)
    if features.empty or price_field not in features.columns:
        return pd.DataFrame()

    if control_fields:
        score = _apply_strategy_score_controls(
            score,
            features.reindex(columns=control_fields),
            feature_score_weights=feature_weights,
            feature_min_percentiles=feature_mins,
        )

    sector_map = _load_strategy_sector_map(strategy_cfg)
    topk = max(
        1,
        int(args.rebalance_interval_topk)
        if int(args.rebalance_interval_topk) > 0
        else int(strategy_cfg.get("topk", 40)),
    )
    try:
        risk_degree = float(strategy_cfg.get("risk_degree", 1.0))
    except (TypeError, ValueError):
        risk_degree = 1.0
    if not math.isfinite(risk_degree):
        risk_degree = 1.0
    risk_state = {
        "smoothed": float(
            np.clip(
                risk_degree,
                max(0.0, float(strategy_cfg.get("risk_floor", 0.0))),
                float(strategy_cfg.get("risk_ceiling", risk_degree)),
            )
        ),
        "cooldown": 0.0,
    }
    strategy_benchmark_topn = max(1, int(strategy_cfg.get("benchmark_topn") or 500))
    dt_level = "datetime" if "datetime" in score.index.names else 0
    score_by_date = {
        pd.Timestamp(dt).normalize(): day_score.droplevel(dt_level)
        for dt, day_score in score.groupby(level=dt_level, sort=True)
    }

    rows = []
    prev_port = pd.Series(dtype=float)
    hedge_state: Dict[str, float] = {}
    for (dt, signal_dt), (next_dt, _) in zip(trade_signal_pairs[:-1], trade_signal_pairs[1:]):
        day_score = score_by_date.get(signal_dt)
        if day_score is None or day_score.empty:
            continue
        day_score = pd.to_numeric(day_score, errors="coerce").dropna()
        day_score.index = day_score.index.astype(str)
        day_score = _apply_sector_cap_to_score(day_score, strategy_cfg, sector_map)

        weight_features = _feature_frame_for_date(features, signal_dt)
        entry_features = _feature_frame_for_date(features, dt)
        next_features = _feature_frame_for_date(features, next_dt)
        if weight_features.empty or entry_features.empty or next_features.empty:
            continue
        entry = pd.to_numeric(entry_features[price_field], errors="coerce")
        exit_ = pd.to_numeric(next_features[price_field], errors="coerce")
        inst_ret = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        inst_ret.index = inst_ret.index.astype(str)
        allowed_return_index = set(day_score.index.astype(str)).union(benchmark_tickers).union(hedge_tickers)
        inst_ret = inst_ret[inst_ret.index.astype(str).isin(allowed_return_index)]
        if inst_ret.empty:
            continue

        marketcap = None
        explicit_bench = None
        if is_benchmark_aware:
            if benchmark_tickers:
                if close_history_field in weight_features.columns:
                    benchmark_available_source = pd.to_numeric(weight_features[close_history_field], errors="coerce")
                elif price_field in weight_features.columns:
                    benchmark_available_source = pd.to_numeric(weight_features[price_field], errors="coerce")
                else:
                    benchmark_available_source = pd.Series(dtype=float)
                available_benchmark = (
                    benchmark_available_source.replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .index.astype(str)
                )
                explicit_bench = _equal_weight_benchmark_tickers(strategy_cfg, available=available_benchmark)
                if explicit_bench.empty:
                    continue
            elif benchmark_weight_field and benchmark_weight_field in weight_features.columns:
                explicit_bench = pd.to_numeric(weight_features[benchmark_weight_field], errors="coerce")
            elif str(marketcap_field) in weight_features.columns:
                marketcap = pd.to_numeric(weight_features[str(marketcap_field)], errors="coerce")
            else:
                continue
        available_hedges = None
        if hedge_tickers:
            if close_history_field in weight_features.columns:
                hedge_available_source = pd.to_numeric(weight_features[close_history_field], errors="coerce")
            elif price_field in weight_features.columns:
                hedge_available_source = pd.to_numeric(weight_features[price_field], errors="coerce")
            else:
                hedge_available_source = pd.Series(dtype=float)
            available_hedges = (
                hedge_available_source.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .index.astype(str)
                .tolist()
            )

        port = normalize_long_weights(
            _strategy_portfolio_weights_for_day(
                day_score,
                strategy_cfg=strategy_cfg,
                strategy_class=strategy_class,
                topk=topk,
                marketcap=marketcap,
                benchmark_weights=explicit_bench,
                current_weights=prev_port,
                trade_dt=dt,
                close_history=features[close_history_field] if close_history_field in features.columns else None,
                available_hedges=available_hedges,
                hedge_state=hedge_state,
            )
        )
        if port.empty:
            continue
        valid_port_ret = inst_ret.reindex(port.index).dropna()
        covered_weight = float(port.reindex(valid_port_ret.index).fillna(0.0).sum())
        if not math.isfinite(covered_weight) or covered_weight <= 1e-12:
            continue
        eval_port = port.reindex(valid_port_ret.index).fillna(0.0) / covered_weight
        gross_portfolio_return = float((eval_port * valid_port_ret).sum())
        effective_risk_degree = _strategy_dynamic_risk_degree(
            strategy_cfg,
            features[close_history_field] if close_history_field in features.columns else None,
            dt,
            risk_degree,
            risk_state,
        )
        portfolio_return = float(effective_risk_degree * gross_portfolio_return)
        drifted_port = port.reindex(valid_port_ret.index).fillna(0.0) * (1.0 + valid_port_ret)
        drifted_port = drifted_port.replace([np.inf, -np.inf], np.nan).dropna()
        prev_port = normalize_long_weights(drifted_port) if not drifted_port.empty else port

        proxy_return = float("nan")
        raw_proxy_return = float("nan")
        proxy_coverage = float("nan")
        if explicit_bench is not None:
            proxy_bench = normalize_long_weights(explicit_bench.sort_values(ascending=False).head(strategy_benchmark_topn))
        elif marketcap is not None:
            proxy_bench = benchmark_weights_from_marketcap(marketcap, topn=strategy_benchmark_topn)
        else:
            proxy_bench = pd.Series(dtype=float)
        if not proxy_bench.empty:
            valid_proxy_ret = inst_ret.reindex(proxy_bench.index).dropna()
            proxy_coverage = float(proxy_bench.reindex(valid_proxy_ret.index).fillna(0.0).sum())
            if math.isfinite(proxy_coverage) and proxy_coverage > 1e-12:
                eval_proxy = proxy_bench.reindex(valid_proxy_ret.index).fillna(0.0) / proxy_coverage
                raw_proxy_return = float((eval_proxy * valid_proxy_ret).sum())
                proxy_return = float(effective_risk_degree * raw_proxy_return)

        bench_return = _benchmark_interval_return(benchmark, dt, next_dt)
        if bench_return is None:
            continue
        holding_days = max(1, _count_trade_days_between(bt_calendar, dt, next_dt) + 1)
        rows.append(
            {
                "datetime": dt,
                "signal_datetime": signal_dt,
                "exit_datetime": next_dt,
                "holding_days": float(holding_days),
                "portfolio_return": portfolio_return,
                "benchmark_return": float(bench_return),
                "excess_return": portfolio_return - float(bench_return),
                "proxy_benchmark_return": proxy_return,
                "raw_proxy_benchmark_return": raw_proxy_return,
                "active_vs_proxy_return": portfolio_return - proxy_return if math.isfinite(proxy_return) else float("nan"),
                "proxy_vs_benchmark_return": proxy_return - float(bench_return) if math.isfinite(proxy_return) else float("nan"),
                "raw_proxy_vs_benchmark_return": raw_proxy_return - float(bench_return) if math.isfinite(raw_proxy_return) else float("nan"),
                "risk_degree": effective_risk_degree,
                "hedge_weight": float(hedge_state.get("last_hedge_weight", 0.0)),
                "return_weight_coverage": covered_weight,
                "proxy_return_weight_coverage": proxy_coverage,
                "portfolio_names": float((port > 0).sum()),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def _rebalance_interval_quality_metrics(
    frame: pd.DataFrame,
    recent_rebalances: Optional[int] = None,
) -> Dict[str, float]:
    if frame.empty:
        return {"rebalances": 0}
    data = frame.tail(int(recent_rebalances)) if recent_rebalances is not None and int(recent_rebalances) > 0 else frame
    data = data.copy()
    ret = pd.to_numeric(data["excess_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ret = ret.dropna()
    if ret.empty:
        return {"rebalances": 0}
    holding_days = pd.to_numeric(data.reindex(ret.index)["holding_days"], errors="coerce")
    out = {
        "rebalances": float(len(ret)),
        "ann_excess_return": _annualized_interval_return(ret, holding_days),
        "mean_excess_return": float(ret.mean()),
        "median_excess_return": float(ret.median()),
        "positive_excess_rate": float((ret > 0).mean()),
        "ann_portfolio_return": _annualized_interval_return(data.reindex(ret.index)["portfolio_return"], holding_days),
        "ann_benchmark_return": _annualized_interval_return(data.reindex(ret.index)["benchmark_return"], holding_days),
        "ann_active_vs_proxy_return": _annualized_interval_return(
            data.reindex(ret.index)["active_vs_proxy_return"],
            holding_days,
        ),
        "ann_proxy_vs_benchmark_return": _annualized_interval_return(
            data.reindex(ret.index)[
                "raw_proxy_vs_benchmark_return"
                if "raw_proxy_vs_benchmark_return" in data.columns
                else "proxy_vs_benchmark_return"
            ],
            holding_days,
        ),
        "mean_holding_days": float(holding_days.mean()),
        "mean_return_weight_coverage": float(pd.to_numeric(data["return_weight_coverage"], errors="coerce").mean()),
        "mean_proxy_return_weight_coverage": float(
            pd.to_numeric(data["proxy_return_weight_coverage"], errors="coerce").mean()
        ),
        "mean_portfolio_names": float(pd.to_numeric(data["portfolio_names"], errors="coerce").mean()),
    }
    if "risk_degree" in data.columns:
        out["mean_risk_degree"] = float(pd.to_numeric(data["risk_degree"], errors="coerce").mean())
    if "hedge_weight" in data.columns:
        out["mean_hedge_weight"] = float(pd.to_numeric(data["hedge_weight"], errors="coerce").mean())
    std = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    out["interval_excess_ir"] = (
        float(out["mean_excess_return"] / std * np.sqrt(252.0 / max(out["mean_holding_days"], 1e-12)))
        if math.isfinite(std) and std > 0
        else float("nan")
    )
    return out


def _rebalance_interval_quality_by_year(frame: pd.DataFrame) -> List[Tuple[str, Dict[str, float]]]:
    if frame.empty:
        return []
    dates = pd.DatetimeIndex(frame.index)
    rows: List[Tuple[str, Dict[str, float]]] = []
    for year in sorted(dates.year.unique()):
        rows.append((str(int(year)), _rebalance_interval_quality_metrics(frame.loc[dates.year == int(year)])))
    return rows


def _rebalance_interval_quality_gate_rows(
    full: Dict[str, float],
    recent: Dict[str, float],
    args: argparse.Namespace,
    *,
    yearly: Optional[List[Tuple[str, Dict[str, float]]]] = None,
) -> List[Tuple[str, bool, str]]:
    rows: List[Tuple[str, bool, str]] = []
    full_n = int(full.get("rebalances", 0) or 0)
    recent_n = int(recent.get("rebalances", 0) or 0)
    rows.append(
        (
            "rebalance_interval_full_rebalances",
            full_n >= int(args.rebalance_interval_min_rebalances),
            f"{full_n} >= {int(args.rebalance_interval_min_rebalances)}",
        )
    )
    recent_required = min(int(args.rebalance_interval_min_recent_rebalances), full_n)
    rows.append(
        (
            "rebalance_interval_recent_rebalances",
            recent_n >= recent_required,
            f"{recent_n} >= {recent_required}",
        )
    )
    checks = [
        (
            "rebalance_interval_full_ann_excess",
            full.get("ann_excess_return"),
            args.rebalance_interval_min_full_ann_excess,
            ">=",
        ),
        (
            "rebalance_interval_recent_ann_excess",
            recent.get("ann_excess_return"),
            args.rebalance_interval_min_recent_ann_excess,
            ">=",
        ),
        (
            "rebalance_interval_positive_excess_rate",
            full.get("positive_excess_rate"),
            args.rebalance_interval_min_positive_excess_rate,
            ">=",
        ),
        (
            "rebalance_interval_recent_positive_excess_rate",
            recent.get("positive_excess_rate"),
            args.rebalance_interval_min_recent_positive_excess_rate,
            ">=",
        ),
        (
            "rebalance_interval_mean_return_weight_coverage",
            full.get("mean_return_weight_coverage"),
            args.rebalance_interval_min_mean_return_weight_coverage,
            ">=",
        ),
        (
            "rebalance_interval_recent_return_weight_coverage",
            recent.get("mean_return_weight_coverage"),
            args.rebalance_interval_min_recent_return_weight_coverage,
            ">=",
        ),
    ]
    for name, value, threshold, op in checks:
        val = _safe_float(value)
        threshold = float(threshold)
        ok = val is not None and (val >= threshold if op == ">=" else val <= threshold)
        rows.append((name, ok, f"{_format_float(val)} {op} {_format_float(threshold)}"))

    max_abs_proxy = _safe_float(args.rebalance_interval_max_abs_proxy_tracking_ann)
    if max_abs_proxy is not None:
        proxy = _safe_float(full.get("ann_proxy_vs_benchmark_return"))
        ok = proxy is not None and abs(proxy) <= max_abs_proxy
        rows.append(
            (
                "rebalance_interval_proxy_tracking_abs",
                ok,
                f"|{_format_float(proxy)}| <= {_format_float(max_abs_proxy)}",
            )
        )

    if yearly is not None:
        min_year_rebalances = int(args.rebalance_interval_min_year_rebalances)
        eligible = [(year, metrics) for year, metrics in yearly if int(metrics.get("rebalances", 0) or 0) >= min_year_rebalances]
        skipped = len(yearly) - len(eligible)
        configured_min_positive_years = int(args.rebalance_interval_min_positive_years)
        min_positive_years = (
            min(configured_min_positive_years, len(eligible))
            if not bool(getattr(args, "strict_positive_year_count", False))
            else configured_min_positive_years
        )
        positive_years = sum(
            1
            for _, metrics in eligible
            if _safe_float(metrics.get("ann_excess_return")) is not None and float(metrics["ann_excess_return"]) >= 0
        )
        detail = (
            f"{positive_years} >= {min_positive_years} "
            f"(eligible_years={len(eligible)}, min_year_rebalances={min_year_rebalances}, skipped_short_years={skipped}"
        )
        if min_positive_years != configured_min_positive_years:
            detail += f", configured_min_positive_years={configured_min_positive_years}"
        detail += ")"
        rows.append(
            (
                "rebalance_interval_positive_years",
                positive_years >= min_positive_years,
                detail,
            )
        )
        vals = [_safe_float(metrics.get("ann_excess_return")) for _, metrics in eligible]
        vals = [v for v in vals if v is not None]
        worst_year = min(vals) if vals else None
        threshold = float(args.rebalance_interval_min_worst_year_ann_excess)
        rows.append(
            (
                "rebalance_interval_worst_year_ann_excess",
                worst_year is not None and worst_year >= threshold,
                f"{_format_float(worst_year)} >= {_format_float(threshold)}",
            )
        )
    return rows


def _print_rebalance_interval_quality_summary(label: str, metrics: Dict[str, float]) -> None:
    keys = [
        "rebalances",
        "ann_excess_return",
        "interval_excess_ir",
        "positive_excess_rate",
        "ann_portfolio_return",
        "ann_benchmark_return",
        "ann_active_vs_proxy_return",
        "ann_proxy_vs_benchmark_return",
        "mean_holding_days",
        "mean_risk_degree",
        "mean_hedge_weight",
        "mean_return_weight_coverage",
        "mean_proxy_return_weight_coverage",
        "mean_portfolio_names",
    ]
    vals = []
    for key in keys:
        if key not in metrics:
            continue
        val = metrics.get(key)
        vals.append(f"{key}={int(val or 0)}" if key == "rebalances" else f"{key}={_format_float(val)}")
    _print_kv(label, ", ".join(vals))


def _print_rebalance_interval_quality_yearly(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    if not rows:
        print("(no yearly rebalance-interval quality rows)")
        return
    headers = [
        "year",
        "rebalances",
        "ann_excess_return",
        "interval_excess_ir",
        "positive_excess_rate",
        "ann_active_vs_proxy_return",
        "ann_proxy_vs_benchmark_return",
        "mean_return_weight_coverage",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for year, metrics in rows:
        print(
            " | ".join(
                [
                    year,
                    str(int(metrics.get("rebalances", 0) or 0)),
                    _format_float(metrics.get("ann_excess_return")),
                    _format_float(metrics.get("interval_excess_ir")),
                    _format_float(metrics.get("positive_excess_rate")),
                    _format_float(metrics.get("ann_active_vs_proxy_return")),
                    _format_float(metrics.get("ann_proxy_vs_benchmark_return")),
                    _format_float(metrics.get("mean_return_weight_coverage")),
                ]
            )
        )


def _evaluate_report_quality(report: pd.DataFrame, *, label: str, max_nan_ratio: float) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    checks.append((f"{label}_report_non_empty", not report.empty, f"rows={len(report)}"))
    if report.empty:
        return checks
    required_cols = ["return", "bench", "cost"]
    for col in required_cols:
        if col not in report.columns:
            checks.append((f"{label}_{col}_present", False, "missing"))
            continue
        s = report[col]
        nan_ratio = float(s.isna().mean())
        inf_count = int(np.isinf(s.fillna(0).values).sum())
        ok = nan_ratio <= max_nan_ratio and inf_count == 0
        checks.append(
            (
                f"{label}_{col}_quality",
                ok,
                f"nan_ratio={_format_float(nan_ratio)} <= {_format_float(max_nan_ratio)}, inf_count={inf_count}",
            )
        )
    return checks


def _evaluate_robustness_gates(
    full_metrics: Dict[str, float],
    stress_metrics: Dict[str, float],
    yearly_rows: List[Tuple[str, Dict[str, float]]],
    *,
    min_full_excess_ann: float,
    min_full_ir: float,
    max_full_mdd_abs: float,
    min_stress_excess_ann: float,
    max_turnover: float,
    min_positive_excess_years: int,
    min_worst_year_excess_ann: float,
    min_year_days: int,
    cap_positive_years: bool = True,
) -> List[Tuple[str, bool, str]]:
    gates: List[Tuple[str, bool, str]] = []

    full_excess = _safe_float(full_metrics.get("excess_ann_return"))
    full_ir = _safe_float(full_metrics.get("ir"))
    full_mdd = _safe_float(full_metrics.get("mdd"))
    full_turnover = _safe_float(full_metrics.get("avg_turnover"))
    stress_excess = _safe_float(stress_metrics.get("excess_ann_return"))

    year_excess = []
    eligible_years = 0
    skipped_short_years = 0
    for period, metrics in yearly_rows:
        if period == "full":
            continue
        n_days_raw = metrics.get("n_days")
        n_days = None
        if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)):
            n_days = int(n_days_raw)
        if n_days is not None and n_days < min_year_days:
            skipped_short_years += 1
            continue
        val = _safe_float(metrics.get("excess_ann_return"))
        if val is not None:
            year_excess.append(val)
            eligible_years += 1

    positive_years = sum(1 for v in year_excess if v > 0)
    worst_year_excess = min(year_excess) if year_excess else None

    ok = full_excess is not None and full_excess >= min_full_excess_ann
    gates.append(
        (
            "full_excess_ann",
            ok,
            f"{_format_float(full_excess)} >= {_format_float(min_full_excess_ann)}",
        )
    )

    ok = full_ir is not None and full_ir >= min_full_ir
    gates.append(("full_ir", ok, f"{_format_float(full_ir)} >= {_format_float(min_full_ir)}"))

    ok = full_mdd is not None and abs(full_mdd) <= max_full_mdd_abs
    gates.append(
        (
            "full_mdd_abs",
            ok,
            f"|{_format_float(full_mdd)}| <= {_format_float(max_full_mdd_abs)}",
        )
    )

    ok = stress_excess is not None and stress_excess >= min_stress_excess_ann
    gates.append(
        (
            "stress_excess_ann",
            ok,
            f"{_format_float(stress_excess)} >= {_format_float(min_stress_excess_ann)}",
        )
    )

    ok = full_turnover is not None and full_turnover <= max_turnover
    gates.append(
        (
            "full_avg_turnover",
            ok,
            f"{_format_float(full_turnover)} <= {_format_float(max_turnover)}",
        )
    )

    required_positive_years = (
        min(min_positive_excess_years, eligible_years)
        if cap_positive_years and eligible_years > 0
        else min_positive_excess_years
    )
    cap_detail = (
        f", configured_min_positive_years={min_positive_excess_years}"
        if required_positive_years != min_positive_excess_years
        else ""
    )
    ok = positive_years >= required_positive_years
    gates.append(
        (
            "positive_excess_years",
            ok,
            f"{positive_years} >= {required_positive_years} (eligible_years={eligible_years}, min_year_days={min_year_days}, skipped_short_years={skipped_short_years}{cap_detail})",
        )
    )

    ok = worst_year_excess is not None and worst_year_excess >= min_worst_year_excess_ann
    gates.append(
        (
            "worst_year_excess_ann",
            ok,
            f"{_format_float(worst_year_excess)} >= {_format_float(min_worst_year_excess_ann)} (eligible_years={eligible_years}, min_year_days={min_year_days}, skipped_short_years={skipped_short_years})",
        )
    )
    return gates


def _evaluate_external_baseline_gates(
    full_report: pd.DataFrame,
    stress_report: pd.DataFrame,
    yearly_reports: List[Tuple[str, pd.DataFrame]],
    rolling_windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    baseline_returns: Dict[str, pd.Series],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, str, Dict[str, float]]], List[Tuple[str, str, Dict[str, float]]], List[Tuple[str, bool, str]]]:
    comparison_rows: List[Tuple[str, str, Dict[str, float]]] = []
    rolling_rows: List[Tuple[str, str, Dict[str, float]]] = []
    checks: List[Tuple[str, bool, str]] = []

    for ticker, returns in baseline_returns.items():
        if returns.empty:
            checks.append((f"baseline_{ticker}_available", False, "no daily returns loaded"))
            continue

        full_metrics = _baseline_period_metrics(full_report, returns)
        stress_metrics = _baseline_period_metrics(stress_report, returns)
        comparison_rows.append((ticker, "full", full_metrics))
        comparison_rows.append((ticker, f"stress x{args.stress_cost_mult} {args.stress_deal_price}", stress_metrics))

        yearly_metrics: List[Tuple[str, Dict[str, float]]] = []
        for label, report_y in yearly_reports:
            metrics = _baseline_period_metrics(report_y, returns)
            yearly_metrics.append((label, metrics))
            comparison_rows.append((ticker, label, metrics))

        full_excess = _safe_float(full_metrics.get("excess_ann_return"))
        stress_excess = _safe_float(stress_metrics.get("excess_ann_return"))
        full_mdd = _safe_float(full_metrics.get("strategy_mdd"))
        mdd_gap = _safe_float(full_metrics.get("mdd_gap"))

        all_period_metrics = [full_metrics, stress_metrics, *[m for _, m in yearly_metrics]]
        max_missing = max(
            [
                v
                for v in (_safe_float(metrics.get("missing_ratio")) for metrics in all_period_metrics)
                if v is not None
            ],
            default=None,
        )
        missing_ok = max_missing is not None and max_missing <= float(args.baseline_max_missing_ratio)
        checks.append(
            (
                f"baseline_{ticker}_missing_ratio",
                missing_ok,
                f"{_format_float(max_missing)} <= {_format_float(args.baseline_max_missing_ratio)}",
            )
        )

        ok = full_excess is not None and full_excess >= float(args.baseline_min_full_excess_ann)
        checks.append(
            (
                f"baseline_{ticker}_full_excess_ann",
                ok,
                f"{_format_float(full_excess)} >= {_format_float(args.baseline_min_full_excess_ann)}",
            )
        )

        ok = stress_excess is not None and stress_excess >= float(args.baseline_min_stress_excess_ann)
        checks.append(
            (
                f"baseline_{ticker}_stress_excess_ann",
                ok,
                f"{_format_float(stress_excess)} >= {_format_float(args.baseline_min_stress_excess_ann)}",
            )
        )

        ok = full_mdd is not None and abs(full_mdd) <= float(args.baseline_max_full_mdd_abs)
        checks.append(
            (
                f"baseline_{ticker}_full_mdd_abs",
                ok,
                f"|{_format_float(full_mdd)}| <= {_format_float(args.baseline_max_full_mdd_abs)}",
            )
        )

        ok = mdd_gap is not None and mdd_gap <= float(args.baseline_max_mdd_gap)
        checks.append(
            (
                f"baseline_{ticker}_mdd_gap",
                ok,
                f"{_format_float(mdd_gap)} <= {_format_float(args.baseline_max_mdd_gap)}",
            )
        )

        eligible_years = 0
        skipped_short_years = 0
        year_excess: List[float] = []
        for _, metrics in yearly_metrics:
            n_days_raw = metrics.get("n_days")
            n_days = int(n_days_raw) if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)) else 0
            if n_days < int(args.baseline_min_year_days):
                skipped_short_years += 1
                continue
            val = _safe_float(metrics.get("excess_ann_return"))
            if val is None:
                continue
            eligible_years += 1
            year_excess.append(val)
        positive_years = sum(1 for val in year_excess if val > 0)
        beat_rate = positive_years / eligible_years if eligible_years else None
        worst_year = min(year_excess) if year_excess else None

        ok = positive_years >= int(args.baseline_min_positive_years)
        checks.append(
            (
                f"baseline_{ticker}_positive_years",
                ok,
                f"{positive_years} >= {int(args.baseline_min_positive_years)} (eligible_years={eligible_years}, min_year_days={int(args.baseline_min_year_days)}, skipped_short_years={skipped_short_years})",
            )
        )

        ok = beat_rate is not None and beat_rate >= float(args.baseline_min_yearly_beat_rate)
        checks.append(
            (
                f"baseline_{ticker}_yearly_beat_rate",
                ok,
                f"{_format_float(beat_rate)} >= {_format_float(args.baseline_min_yearly_beat_rate)} (eligible_years={eligible_years})",
            )
        )

        ok = worst_year is not None and worst_year >= float(args.baseline_min_worst_year_excess_ann)
        checks.append(
            (
                f"baseline_{ticker}_worst_year_excess_ann",
                ok,
                f"{_format_float(worst_year)} >= {_format_float(args.baseline_min_worst_year_excess_ann)} (eligible_years={eligible_years})",
            )
        )

        rolling_excess: List[float] = []
        rolling_pass = 0
        latest_rolling_metrics: Dict[str, float] = {}
        for w_start, w_end, w_days in rolling_windows:
            metrics = _baseline_period_metrics(_slice_report(full_report, w_start, w_end), returns)
            status_ok = (
                (_safe_float(metrics.get("missing_ratio")) is not None)
                and (_safe_float(metrics.get("missing_ratio")) <= float(args.baseline_max_missing_ratio))
                and (_safe_float(metrics.get("excess_ann_return")) is not None)
                and (_safe_float(metrics.get("excess_ann_return")) >= float(args.baseline_min_rolling_excess_ann))
            )
            metrics = dict(metrics)
            metrics["status"] = 1.0 if status_ok else 0.0
            rolling_rows.append((ticker, f"{w_start.date()}->{w_end.date()} ({w_days}d)", metrics))
            latest_rolling_metrics = metrics
            val = _safe_float(metrics.get("excess_ann_return"))
            if val is not None:
                rolling_excess.append(val)
            if status_ok:
                rolling_pass += 1
        rolling_total = len(rolling_windows)
        rolling_pass_rate = rolling_pass / rolling_total if rolling_total else None
        worst_rolling = min(rolling_excess) if rolling_excess else None
        latest_rolling = _safe_float(latest_rolling_metrics.get("excess_ann_return"))
        latest_threshold = getattr(args, "baseline_min_latest_rolling_excess_ann", None)
        ok = rolling_pass_rate is not None and rolling_pass_rate >= float(args.baseline_min_rolling_pass_rate)
        checks.append(
            (
                f"baseline_{ticker}_rolling_pass_rate",
                ok,
                f"{_format_float(rolling_pass_rate)} >= {_format_float(args.baseline_min_rolling_pass_rate)} (passed={rolling_pass}/{rolling_total}, min_excess_ann={_format_float(args.baseline_min_rolling_excess_ann)})",
            )
        )
        ok = worst_rolling is not None and worst_rolling >= float(args.baseline_min_worst_rolling_excess_ann)
        checks.append(
            (
                f"baseline_{ticker}_worst_rolling_excess_ann",
                ok,
                f"{_format_float(worst_rolling)} >= {_format_float(args.baseline_min_worst_rolling_excess_ann)}",
            )
        )
        if latest_threshold is not None:
            ok = latest_rolling is not None and latest_rolling >= float(latest_threshold)
            checks.append(
                (
                    f"baseline_{ticker}_latest_rolling_excess_ann",
                    ok,
                    f"{_format_float(latest_rolling)} >= {_format_float(latest_threshold)}",
                )
            )

    return comparison_rows, rolling_rows, checks


def _print_gate_table(gates: List[Tuple[str, bool, str]]):
    print("gate | status | detail")
    print("--- | --- | ---")
    for name, ok, detail in gates:
        status = "PASS" if ok else "FAIL"
        print(f"{name} | {status} | {detail}")


def _print_check_table(rows: List[Tuple[str, bool, str]]):
    print("check | status | detail")
    print("--- | --- | ---")
    for name, ok, detail in rows:
        print(f"{name} | {'PASS' if ok else 'FAIL'} | {detail}")


def _print_baseline_comparison_table(rows: List[Tuple[str, str, Dict[str, float]]]):
    if not rows:
        print("(no data)")
        return
    headers = [
        "baseline",
        "period",
        "n_days",
        "strategy_ann_return",
        "baseline_ann_return",
        "excess_ann_return",
        "excess_ir",
        "strategy_mdd",
        "baseline_mdd",
        "mdd_gap",
        "missing_ratio",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for ticker, period, metrics in rows:
        n_days_raw = metrics.get("n_days")
        n_days = str(int(n_days_raw)) if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)) else "n/a"
        row = [
            ticker,
            period,
            n_days,
            _format_float(metrics.get("strategy_ann_return")),
            _format_float(metrics.get("baseline_ann_return")),
            _format_float(metrics.get("excess_ann_return")),
            _format_float(metrics.get("excess_ir")),
            _format_float(metrics.get("strategy_mdd")),
            _format_float(metrics.get("baseline_mdd")),
            _format_float(metrics.get("mdd_gap")),
            _format_float(metrics.get("missing_ratio")),
        ]
        print(" | ".join(row))


def _print_baseline_rolling_table(rows: List[Tuple[str, str, Dict[str, float]]]):
    if not rows:
        print("(no data)")
        return
    headers = ["baseline", "window", "status", "n_days", "strategy_ann_return", "baseline_ann_return", "excess_ann_return", "excess_ir", "missing_ratio"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for ticker, window, metrics in rows:
        status = "PASS" if _safe_float(metrics.get("status")) == 1.0 else "FAIL"
        n_days_raw = metrics.get("n_days")
        n_days = str(int(n_days_raw)) if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)) else "n/a"
        row = [
            ticker,
            window,
            status,
            n_days,
            _format_float(metrics.get("strategy_ann_return")),
            _format_float(metrics.get("baseline_ann_return")),
            _format_float(metrics.get("excess_ann_return")),
            _format_float(metrics.get("excess_ir")),
            _format_float(metrics.get("missing_ratio")),
        ]
        print(" | ".join(row))


def _print_baseline_regime_table(rows: List[Tuple[str, str, Dict[str, float]]]):
    if not rows:
        print("(no data)")
        return
    headers = [
        "baseline",
        "regime",
        "n_days",
        "strategy_ann_return",
        "baseline_ann_return",
        "excess_ann_return",
        "excess_ir",
        "strategy_mdd",
        "baseline_mdd",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for ticker, regime, metrics in rows:
        n_days_raw = metrics.get("n_days")
        n_days = str(int(n_days_raw)) if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)) else "n/a"
        row = [
            ticker,
            regime,
            n_days,
            _format_float(metrics.get("strategy_ann_return")),
            _format_float(metrics.get("baseline_ann_return")),
            _format_float(metrics.get("excess_ann_return")),
            _format_float(metrics.get("excess_ir")),
            _format_float(metrics.get("strategy_mdd")),
            _format_float(metrics.get("baseline_mdd")),
        ]
        print(" | ".join(row))


def _registry_existing_count(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _multiple_testing_ir_haircut(ir: Optional[float], n_days: Optional[float], n_trials: int) -> Dict[str, float]:
    raw_ir = _safe_float(ir)
    days = _safe_float(n_days)
    trials = max(1, int(n_trials or 1))
    if raw_ir is None or days is None or days <= 1:
        return {
            "raw_ir": float("nan"),
            "n_days": float(days or 0),
            "n_trials": float(trials),
            "ir_haircut": float("nan"),
            "haircut_ir": float("nan"),
        }
    haircut = math.sqrt(max(0.0, 2.0 * math.log(float(trials)))) * math.sqrt(252.0 / float(days))
    return {
        "raw_ir": float(raw_ir),
        "n_days": float(days),
        "n_trials": float(trials),
        "ir_haircut": float(haircut),
        "haircut_ir": float(raw_ir - haircut),
    }


def _append_trial_registry(path: Path, record: Dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _print_header(title: str):
    print(f"\n== {title} ==")


def _print_kv(label: str, value: str):
    print(f"- {label}: {value}")


def _print_table(rows: List[Tuple[str, Dict[str, float]]]):
    if not rows:
        print("(no data)")
        return
    headers = [
        "period",
        "n_days",
        "ann_return",
        "ir",
        "excess_ir",
        "mdd",
        "bench_ann_return",
        "gross_excess_ann_return",
        "excess_ann_return",
        "avg_turnover",
        "avg_cost",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for name, metrics in rows:
        n_days_raw = metrics.get("n_days")
        if isinstance(n_days_raw, (int, float)) and math.isfinite(float(n_days_raw)):
            n_days = str(int(n_days_raw))
        else:
            n_days = "n/a"
        row = [
            name,
            n_days,
            _format_float(metrics.get("ann_return")),
            _format_float(metrics.get("ir")),
            _format_float(metrics.get("excess_ir")),
            _format_float(metrics.get("mdd")),
            _format_float(metrics.get("bench_ann_return")),
            _format_float(metrics.get("gross_excess_ann_return")),
            _format_float(metrics.get("excess_ann_return")),
            _format_float(metrics.get("avg_turnover")),
            _format_float(metrics.get("avg_cost")),
        ]
        print(" | ".join(row))


def _build_rolling_windows(
    calendar: List[pd.Timestamp], window_days: int, step_days: int, min_days: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(calendar) < max(2, min_days):
        return []
    window_days = max(window_days, min_days)
    step_days = max(1, step_days)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start_idx = 0
    while start_idx < len(calendar):
        end_idx = min(start_idx + window_days - 1, len(calendar) - 1)
        n_days = end_idx - start_idx + 1
        if n_days < min_days:
            break
        windows.append((calendar[start_idx], calendar[end_idx], n_days))
        if end_idx == len(calendar) - 1:
            break
        start_idx += step_days

    # Ensure the tail is evaluated even if step/grid misses the final anchor.
    tail_start_idx = max(0, len(calendar) - window_days)
    tail_end_idx = len(calendar) - 1
    tail_days = tail_end_idx - tail_start_idx + 1
    tail_window = (calendar[tail_start_idx], calendar[tail_end_idx], tail_days)
    if tail_days >= min_days and (not windows or windows[-1][1] != tail_window[1]):
        windows.append(tail_window)
    return windows


def _print_rolling_table(rows: List[Tuple[str, Dict[str, str]]]):
    if not rows:
        print("(no rolling windows)")
        return
    headers = [
        "window",
        "status",
        "n_days",
        "ann_return",
        "ir",
        "excess_ir",
        "mdd",
        "bench_ann_return",
        "gross_excess_ann_return",
        "excess_ann_return",
        "avg_turnover",
        "avg_cost",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for name, metrics in rows:
        row = [
            name,
            metrics.get("status", "n/a"),
            metrics.get("n_days", "n/a"),
            metrics.get("ann_return", "n/a"),
            metrics.get("ir", "n/a"),
            metrics.get("excess_ir", "n/a"),
            metrics.get("mdd", "n/a"),
            metrics.get("bench_ann_return", "n/a"),
            metrics.get("gross_excess_ann_return", "n/a"),
            metrics.get("excess_ann_return", "n/a"),
            metrics.get("avg_turnover", "n/a"),
            metrics.get("avg_cost", "n/a"),
        ]
        print(" | ".join(row))


def _rolling_gate_metric(
    metrics: Dict[str, float],
    *,
    ir_metric: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    excess = _safe_float(metrics.get("excess_ann_return"))
    ir_key = "excess_ir" if str(ir_metric) == "excess" else "ir"
    ir = _safe_float(metrics.get(ir_key))
    mdd = _safe_float(metrics.get("mdd"))
    turnover = _safe_float(metrics.get("avg_turnover"))
    return excess, ir, mdd, turnover


def _rolling_passes(
    metrics: Dict[str, float],
    *,
    ir_metric: str,
    min_excess_ann: float,
    min_ir: float,
    max_mdd_abs: float,
    max_turnover: float,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    excess, ir, mdd, turnover = _rolling_gate_metric(metrics, ir_metric=ir_metric)
    mdd_abs = abs(mdd) if mdd is not None else None
    ok = (
        excess is not None
        and excess >= min_excess_ann
        and ir is not None
        and ir >= min_ir
        and mdd_abs is not None
        and mdd_abs <= max_mdd_abs
        and turnover is not None
        and turnover <= max_turnover
    )
    return ok, excess, ir, mdd_abs


def _collect_rolling_rows(
    rolling_windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    *,
    metrics_by_window,
    ir_metric: str,
    min_excess_ann: float,
    min_ir: float,
    max_mdd_abs: float,
    max_turnover: float,
) -> Tuple[List[Tuple[str, Dict[str, str]]], int, Optional[float], Optional[float], Optional[float]]:
    rolling_rows: List[Tuple[str, Dict[str, str]]] = []
    rolling_pass = 0
    worst_excess = None
    worst_ir = None
    worst_mdd_abs = None
    for w_start, w_end, w_days in rolling_windows:
        metrics_w = metrics_by_window(w_start, w_end, w_days)
        pass_window, excess_w, ir_w, mdd_abs_w = _rolling_passes(
            metrics_w,
            ir_metric=ir_metric,
            min_excess_ann=min_excess_ann,
            min_ir=min_ir,
            max_mdd_abs=max_mdd_abs,
            max_turnover=max_turnover,
        )
        if pass_window:
            rolling_pass += 1
        if excess_w is not None:
            worst_excess = excess_w if worst_excess is None else min(worst_excess, excess_w)
        if ir_w is not None:
            worst_ir = ir_w if worst_ir is None else min(worst_ir, ir_w)
        if mdd_abs_w is not None:
            worst_mdd_abs = mdd_abs_w if worst_mdd_abs is None else max(worst_mdd_abs, mdd_abs_w)

        rolling_rows.append(
            (
                f"{w_start.date()}->{w_end.date()}",
                {
                    "status": "PASS" if pass_window else "FAIL",
                    "n_days": str(w_days),
                    "ann_return": _format_float(metrics_w.get("ann_return")),
                    "ir": _format_float(metrics_w.get("ir")),
                    "excess_ir": _format_float(metrics_w.get("excess_ir")),
                    "mdd": _format_float(metrics_w.get("mdd")),
                    "bench_ann_return": _format_float(metrics_w.get("bench_ann_return")),
                    "gross_excess_ann_return": _format_float(metrics_w.get("gross_excess_ann_return")),
                    "excess_ann_return": _format_float(excess_w),
                    "avg_turnover": _format_float(metrics_w.get("avg_turnover")),
                    "avg_cost": _format_float(metrics_w.get("avg_cost")),
                },
            )
        )
    return rolling_rows, rolling_pass, worst_excess, worst_ir, worst_mdd_abs


def _print_rolling_summary(
    *,
    rolling_windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    rolling_pass: int,
    worst_excess: Optional[float],
    worst_ir: Optional[float],
    worst_mdd_abs: Optional[float],
    threshold: float,
    label_prefix: str = "rolling",
) -> bool:
    total_windows = len(rolling_windows)
    pass_rate = float(rolling_pass) / float(total_windows) if total_windows > 0 else 0.0
    _print_kv(f"{label_prefix}_windows_total", str(total_windows))
    _print_kv(f"{label_prefix}_windows_pass", str(rolling_pass))
    _print_kv(f"{label_prefix}_pass_rate", _format_float(pass_rate))
    _print_kv(f"{label_prefix}_worst_excess_ann", _format_float(worst_excess))
    _print_kv(f"{label_prefix}_worst_ir", _format_float(worst_ir))
    _print_kv(f"{label_prefix}_worst_mdd_abs", _format_float(worst_mdd_abs))
    ok = pass_rate >= threshold
    _print_kv(
        f"{label_prefix}_overall",
        f"{'PASS' if ok else 'FAIL'} (pass_rate={_format_float(pass_rate)} >= threshold={_format_float(threshold)})",
    )
    return ok


GATE_PRESETS = {
    "research": {
        "gate_min_full_excess_ann": 0.03,
        "gate_min_full_ir": 0.40,
        "gate_max_full_mdd_abs": 0.45,
        "gate_min_stress_excess_ann": 0.00,
        "gate_max_turnover": 0.10,
        "gate_min_positive_excess_years": 2,
        "gate_min_worst_year_excess_ann": -0.40,
        "gate_min_year_days": 200,
    },
    "release": {
        "gate_min_full_excess_ann": 0.03,
        "gate_min_full_ir": 0.40,
        "gate_max_full_mdd_abs": 0.35,
        "gate_min_stress_excess_ann": 0.01,
        "gate_max_turnover": 0.10,
        "gate_min_positive_excess_years": 3,
        "gate_min_worst_year_excess_ann": -0.20,
        "gate_min_year_days": 200,
    },
    "growth": {
        "gate_min_full_excess_ann": 0.05,
        "gate_min_full_ir": 0.35,
        "gate_max_full_mdd_abs": 0.40,
        "gate_min_stress_excess_ann": 0.02,
        "gate_max_turnover": 0.20,
        "gate_min_positive_excess_years": 3,
        "gate_min_worst_year_excess_ann": -0.25,
        "gate_min_year_days": 200,
    },
    "qqq_release": {
        "gate_min_full_excess_ann": 0.08,
        "gate_min_full_ir": 0.60,
        "gate_max_full_mdd_abs": 0.32,
        "gate_min_stress_excess_ann": 0.04,
        "gate_max_turnover": 0.12,
        "gate_min_positive_excess_years": 3,
        "gate_min_worst_year_excess_ann": 0.00,
        "gate_min_year_days": 200,
    },
}

ROLLING_PRESETS = {
    "research": {
        "rolling_window_days": 252,
        "rolling_step_days": 63,
        "rolling_min_days": 126,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.50,
        "rolling_max_turnover": 0.12,
        "rolling_min_pass_rate": 0.60,
    },
    "release": {
        "rolling_window_days": 126,
        "rolling_step_days": 63,
        "rolling_min_days": 63,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.35,
        "rolling_max_turnover": 0.10,
        "rolling_min_pass_rate": 0.80,
    },
    "growth": {
        "rolling_window_days": 126,
        "rolling_step_days": 63,
        "rolling_min_days": 63,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.45,
        "rolling_max_turnover": 0.20,
        "rolling_min_pass_rate": 0.65,
    },
    "qqq_release": {
        "rolling_window_days": 126,
        "rolling_step_days": 63,
        "rolling_min_days": 63,
        "rolling_min_excess_ann": 0.00,
        "rolling_min_ir": 0.00,
        "rolling_max_mdd_abs": 0.35,
        "rolling_max_turnover": 0.12,
        "rolling_min_pass_rate": 0.85,
    },
}

MODEL_QUALITY_PRESETS = {
    "research": {
        "model_quality_min_days": 120,
        "model_quality_min_recent_days": 40,
        "model_quality_min_year_days": 120,
        "model_quality_recent_days": 63,
        "model_quality_min_daily_count": 30,
        "model_quality_min_full_mean_ic": 0.000,
        "model_quality_min_full_topq_spread": 0.000,
        "model_quality_min_recent_mean_ic": -0.005,
        "model_quality_min_recent_pos_ic_rate": 0.48,
        "model_quality_min_recent_topk_mean_label": -0.005,
        "model_quality_min_recent_topq_spread": 0.000,
        "model_quality_min_positive_years": 2,
        "model_quality_min_worst_year_mean_ic": -0.030,
        "model_quality_min_worst_year_topk_mean_label": -0.005,
        "model_quality_min_worst_year_topq_spread": -0.005,
    },
    "release": {
        "model_quality_min_days": 120,
        "model_quality_min_recent_days": 40,
        "model_quality_min_year_days": 120,
        "model_quality_recent_days": 63,
        "model_quality_min_daily_count": 30,
        "model_quality_min_full_mean_ic": 0.005,
        "model_quality_min_full_topq_spread": 0.000,
        "model_quality_min_recent_mean_ic": 0.000,
        "model_quality_min_recent_pos_ic_rate": 0.50,
        "model_quality_min_recent_topk_mean_label": 0.000,
        "model_quality_min_recent_topq_spread": 0.000,
        "model_quality_min_positive_years": 3,
        "model_quality_min_worst_year_mean_ic": -0.015,
        "model_quality_min_worst_year_topk_mean_label": 0.000,
        "model_quality_min_worst_year_topq_spread": -0.010,
    },
    "growth": {
        "model_quality_min_days": 120,
        "model_quality_min_recent_days": 40,
        "model_quality_min_year_days": 120,
        "model_quality_recent_days": 63,
        "model_quality_min_daily_count": 30,
        "model_quality_min_full_mean_ic": 0.005,
        "model_quality_min_full_topq_spread": 0.000,
        "model_quality_min_recent_mean_ic": 0.000,
        "model_quality_min_recent_pos_ic_rate": 0.50,
        "model_quality_min_recent_topk_mean_label": 0.000,
        "model_quality_min_recent_topq_spread": 0.000,
        "model_quality_min_positive_years": 3,
        "model_quality_min_worst_year_mean_ic": -0.020,
        "model_quality_min_worst_year_topk_mean_label": -0.005,
        "model_quality_min_worst_year_topq_spread": -0.010,
    },
    "qqq_release": {
        "model_quality_min_days": 120,
        "model_quality_min_recent_days": 40,
        "model_quality_min_year_days": 120,
        "model_quality_recent_days": 63,
        "model_quality_min_daily_count": 30,
        "model_quality_min_full_mean_ic": 0.010,
        "model_quality_min_full_topq_spread": 0.002,
        "model_quality_min_recent_mean_ic": 0.005,
        "model_quality_min_recent_pos_ic_rate": 0.52,
        "model_quality_min_recent_topk_mean_label": 0.000,
        "model_quality_min_recent_topq_spread": 0.002,
        "model_quality_min_positive_years": 3,
        "model_quality_min_worst_year_mean_ic": 0.000,
        "model_quality_min_worst_year_topk_mean_label": 0.000,
        "model_quality_min_worst_year_topq_spread": 0.000,
    },
}

ACTIVE_RISK_PRESETS = {
    "research": {
        "active_risk_recent_rebalances": 13,
        "active_risk_min_rebalances": 20,
        "active_risk_max_mean_active_share": 0.90,
        "active_risk_max_recent_mean_active_share": 0.90,
        "active_risk_max_mean_abs_active_weight": 0.12,
        "active_risk_max_mean_single_stock_weight": 0.12,
        "active_risk_max_mean_core_weight_gap": 0.10,
        "active_risk_max_recent_core_weight_gap": 0.12,
        "active_risk_max_mean_stock_overlay_weight": 0.90,
        "active_risk_min_mean_benchmark_coverage": 0.20,
        "active_risk_min_recent_benchmark_coverage": 0.20,
        "active_risk_max_mean_sector_active_weight": 0.40,
        "active_risk_max_mean_sector_stock_weight": 0.40,
        "active_risk_max_mean_portfolio_names": 250,
    },
    "release": {
        "active_risk_recent_rebalances": 13,
        "active_risk_min_rebalances": 20,
        "active_risk_max_mean_active_share": 0.85,
        "active_risk_max_recent_mean_active_share": 0.85,
        "active_risk_max_mean_abs_active_weight": 0.08,
        "active_risk_max_mean_single_stock_weight": 0.08,
        "active_risk_max_mean_core_weight_gap": 0.08,
        "active_risk_max_recent_core_weight_gap": 0.10,
        "active_risk_max_mean_stock_overlay_weight": 0.85,
        "active_risk_min_mean_benchmark_coverage": 0.30,
        "active_risk_min_recent_benchmark_coverage": 0.30,
        "active_risk_max_mean_sector_active_weight": 0.35,
        "active_risk_max_mean_sector_stock_weight": 0.35,
        "active_risk_max_mean_portfolio_names": 160,
    },
    "growth": {
        "active_risk_recent_rebalances": 13,
        "active_risk_min_rebalances": 20,
        "active_risk_max_mean_active_share": 1.00,
        "active_risk_max_recent_mean_active_share": 1.00,
        "active_risk_max_mean_abs_active_weight": 0.15,
        "active_risk_max_mean_single_stock_weight": 0.15,
        "active_risk_max_mean_core_weight_gap": 0.10,
        "active_risk_max_recent_core_weight_gap": 0.12,
        "active_risk_max_mean_stock_overlay_weight": 1.00,
        "active_risk_min_mean_benchmark_coverage": 0.10,
        "active_risk_min_recent_benchmark_coverage": 0.10,
        "active_risk_max_mean_sector_active_weight": 1.00,
        "active_risk_max_mean_sector_stock_weight": 1.00,
        "active_risk_max_mean_portfolio_names": 120,
    },
    "qqq_release": {
        "active_risk_recent_rebalances": 13,
        "active_risk_min_rebalances": 20,
        "active_risk_max_mean_active_share": 0.60,
        "active_risk_max_recent_mean_active_share": 0.60,
        "active_risk_max_mean_abs_active_weight": 0.08,
        "active_risk_max_mean_single_stock_weight": 0.08,
        "active_risk_max_mean_core_weight_gap": 0.08,
        "active_risk_max_recent_core_weight_gap": 0.10,
        "active_risk_max_mean_stock_overlay_weight": 0.60,
        "active_risk_min_mean_benchmark_coverage": 0.30,
        "active_risk_min_recent_benchmark_coverage": 0.30,
        "active_risk_max_mean_sector_active_weight": 0.35,
        "active_risk_max_mean_sector_stock_weight": 0.30,
        "active_risk_max_mean_portfolio_names": 120,
    },
}


STRATEGY_WEIGHTED_QUALITY_PRESETS = {
    "research": {
        "strategy_weighted_recent_rebalances": 13,
        "strategy_weighted_min_rebalances": 20,
        "strategy_weighted_min_recent_rebalances": 8,
        "strategy_weighted_min_year_rebalances": 20,
        "strategy_weighted_min_full_mean_label": -0.001,
        "strategy_weighted_min_recent_mean_label": -0.002,
        "strategy_weighted_min_positive_label_rate": 0.48,
        "strategy_weighted_min_recent_positive_label_rate": 0.48,
        "strategy_weighted_min_mean_label_weight_coverage": 0.90,
        "strategy_weighted_min_recent_label_weight_coverage": 0.90,
        "strategy_weighted_min_positive_years": 2,
        "strategy_weighted_min_worst_year_mean_label": -0.020,
    },
    "release": {
        "strategy_weighted_recent_rebalances": 13,
        "strategy_weighted_min_rebalances": 20,
        "strategy_weighted_min_recent_rebalances": 8,
        "strategy_weighted_min_year_rebalances": 20,
        "strategy_weighted_min_full_mean_label": 0.000,
        "strategy_weighted_min_recent_mean_label": 0.000,
        "strategy_weighted_min_positive_label_rate": 0.50,
        "strategy_weighted_min_recent_positive_label_rate": 0.50,
        "strategy_weighted_min_mean_label_weight_coverage": 0.95,
        "strategy_weighted_min_recent_label_weight_coverage": 0.95,
        "strategy_weighted_min_positive_years": 3,
        "strategy_weighted_min_worst_year_mean_label": -0.005,
    },
    "growth": {
        "strategy_weighted_recent_rebalances": 13,
        "strategy_weighted_min_rebalances": 20,
        "strategy_weighted_min_recent_rebalances": 8,
        "strategy_weighted_min_year_rebalances": 20,
        "strategy_weighted_min_full_mean_label": 0.000,
        "strategy_weighted_min_recent_mean_label": 0.000,
        "strategy_weighted_min_positive_label_rate": 0.50,
        "strategy_weighted_min_recent_positive_label_rate": 0.50,
        "strategy_weighted_min_mean_label_weight_coverage": 0.95,
        "strategy_weighted_min_recent_label_weight_coverage": 0.95,
        "strategy_weighted_min_positive_years": 3,
        "strategy_weighted_min_worst_year_mean_label": -0.010,
    },
    "qqq_release": {
        "strategy_weighted_recent_rebalances": 13,
        "strategy_weighted_min_rebalances": 20,
        "strategy_weighted_min_recent_rebalances": 8,
        "strategy_weighted_min_year_rebalances": 20,
        "strategy_weighted_min_full_mean_label": 0.002,
        "strategy_weighted_min_recent_mean_label": 0.002,
        "strategy_weighted_min_positive_label_rate": 0.52,
        "strategy_weighted_min_recent_positive_label_rate": 0.50,
        "strategy_weighted_min_mean_label_weight_coverage": 0.98,
        "strategy_weighted_min_recent_label_weight_coverage": 0.98,
        "strategy_weighted_min_positive_years": 3,
        "strategy_weighted_min_worst_year_mean_label": 0.000,
    },
}


REBALANCE_INTERVAL_QUALITY_PRESETS = {
    "research": {
        "rebalance_interval_recent_rebalances": 13,
        "rebalance_interval_min_rebalances": 20,
        "rebalance_interval_min_recent_rebalances": 8,
        "rebalance_interval_min_year_rebalances": 20,
        "rebalance_interval_min_full_ann_excess": 0.00,
        "rebalance_interval_min_recent_ann_excess": -0.02,
        "rebalance_interval_min_positive_excess_rate": 0.48,
        "rebalance_interval_min_recent_positive_excess_rate": 0.48,
        "rebalance_interval_min_mean_return_weight_coverage": 0.95,
        "rebalance_interval_min_recent_return_weight_coverage": 0.95,
        "rebalance_interval_min_positive_years": 2,
        "rebalance_interval_min_worst_year_ann_excess": -0.20,
        "rebalance_interval_max_abs_proxy_tracking_ann": 0.08,
    },
    "release": {
        "rebalance_interval_recent_rebalances": 13,
        "rebalance_interval_min_rebalances": 20,
        "rebalance_interval_min_recent_rebalances": 8,
        "rebalance_interval_min_year_rebalances": 20,
        "rebalance_interval_min_full_ann_excess": 0.03,
        "rebalance_interval_min_recent_ann_excess": 0.00,
        "rebalance_interval_min_positive_excess_rate": 0.50,
        "rebalance_interval_min_recent_positive_excess_rate": 0.50,
        "rebalance_interval_min_mean_return_weight_coverage": 0.98,
        "rebalance_interval_min_recent_return_weight_coverage": 0.98,
        "rebalance_interval_min_positive_years": 3,
        "rebalance_interval_min_worst_year_ann_excess": -0.10,
        "rebalance_interval_max_abs_proxy_tracking_ann": 0.05,
    },
    "growth": {
        "rebalance_interval_recent_rebalances": 13,
        "rebalance_interval_min_rebalances": 20,
        "rebalance_interval_min_recent_rebalances": 8,
        "rebalance_interval_min_year_rebalances": 20,
        "rebalance_interval_min_full_ann_excess": 0.05,
        "rebalance_interval_min_recent_ann_excess": 0.00,
        "rebalance_interval_min_positive_excess_rate": 0.50,
        "rebalance_interval_min_recent_positive_excess_rate": 0.35,
        "rebalance_interval_min_mean_return_weight_coverage": 0.98,
        "rebalance_interval_min_recent_return_weight_coverage": 0.98,
        "rebalance_interval_min_positive_years": 3,
        "rebalance_interval_min_worst_year_ann_excess": -0.15,
        "rebalance_interval_max_abs_proxy_tracking_ann": 0.08,
    },
    "qqq_release": {
        "rebalance_interval_recent_rebalances": 13,
        "rebalance_interval_min_rebalances": 20,
        "rebalance_interval_min_recent_rebalances": 8,
        "rebalance_interval_min_year_rebalances": 20,
        "rebalance_interval_min_full_ann_excess": 0.08,
        "rebalance_interval_min_recent_ann_excess": 0.03,
        "rebalance_interval_min_positive_excess_rate": 0.52,
        "rebalance_interval_min_recent_positive_excess_rate": 0.50,
        "rebalance_interval_min_mean_return_weight_coverage": 0.98,
        "rebalance_interval_min_recent_return_weight_coverage": 0.98,
        "rebalance_interval_min_positive_years": 3,
        "rebalance_interval_min_worst_year_ann_excess": 0.00,
        "rebalance_interval_max_abs_proxy_tracking_ann": 0.05,
    },
}


BASELINE_GATE_PRESETS = {
    "research": {
        "baseline_tickers": "QQQ,SPY,IXIC",
        "baseline_min_full_excess_ann": 0.00,
        "baseline_min_stress_excess_ann": -0.02,
        "baseline_max_full_mdd_abs": 0.45,
        "baseline_max_mdd_gap": 0.10,
        "baseline_min_positive_years": 2,
        "baseline_min_yearly_beat_rate": 0.50,
        "baseline_min_worst_year_excess_ann": -0.35,
        "baseline_min_year_days": 200,
        "baseline_min_rolling_excess_ann": -0.02,
        "baseline_min_rolling_pass_rate": 0.50,
        "baseline_min_worst_rolling_excess_ann": -0.35,
        "baseline_min_latest_rolling_excess_ann": -0.25,
        "baseline_max_missing_ratio": 0.01,
    },
    "release": {
        "baseline_tickers": "QQQ,SPY,IXIC",
        "baseline_min_full_excess_ann": 0.00,
        "baseline_min_stress_excess_ann": -0.01,
        "baseline_max_full_mdd_abs": 0.40,
        "baseline_max_mdd_gap": 0.05,
        "baseline_min_positive_years": 2,
        "baseline_min_yearly_beat_rate": 0.50,
        "baseline_min_worst_year_excess_ann": -0.30,
        "baseline_min_year_days": 200,
        "baseline_min_rolling_excess_ann": 0.00,
        "baseline_min_rolling_pass_rate": 0.50,
        "baseline_min_worst_rolling_excess_ann": -0.30,
        "baseline_min_latest_rolling_excess_ann": 0.00,
        "baseline_max_missing_ratio": 0.01,
    },
    "growth": {
        "baseline_tickers": "QQQ,SPY,IXIC",
        "baseline_min_full_excess_ann": 0.02,
        "baseline_min_stress_excess_ann": 0.00,
        "baseline_max_full_mdd_abs": 0.40,
        "baseline_max_mdd_gap": 0.05,
        "baseline_min_positive_years": 2,
        "baseline_min_yearly_beat_rate": 0.50,
        "baseline_min_worst_year_excess_ann": -0.30,
        "baseline_min_year_days": 200,
        "baseline_min_rolling_excess_ann": 0.00,
        "baseline_min_rolling_pass_rate": 0.55,
        "baseline_min_worst_rolling_excess_ann": -0.25,
        "baseline_min_latest_rolling_excess_ann": 0.00,
        "baseline_max_missing_ratio": 0.01,
    },
    "qqq_release": {
        "baseline_tickers": "QQQ,SPY,IXIC",
        "baseline_min_full_excess_ann": 0.08,
        "baseline_min_stress_excess_ann": 0.04,
        "baseline_max_full_mdd_abs": 0.35,
        "baseline_max_mdd_gap": 0.03,
        "baseline_min_positive_years": 3,
        "baseline_min_yearly_beat_rate": 0.80,
        "baseline_min_worst_year_excess_ann": 0.00,
        "baseline_min_year_days": 200,
        "baseline_min_rolling_excess_ann": 0.00,
        "baseline_min_rolling_pass_rate": 0.85,
        "baseline_min_worst_rolling_excess_ann": 0.00,
        "baseline_min_latest_rolling_excess_ann": 0.02,
        "baseline_max_missing_ratio": 0.005,
    },
}


BASELINE_REGIME_PRESETS = {
    "research": {
        "baseline_regime_ticker": "QQQ",
        "baseline_regime_return_window": 63,
        "baseline_regime_vol_window": 20,
        "baseline_regime_strong_return_threshold": 0.10,
        "baseline_regime_weak_return_threshold": 0.00,
        "baseline_regime_min_days": 40,
        "baseline_regime_min_up_day_excess_ann": -0.50,
        "baseline_regime_min_down_day_excess_ann": 0.00,
        "baseline_regime_min_strong_excess_ann": -0.25,
        "baseline_regime_min_weak_excess_ann": 0.00,
        "baseline_regime_min_high_vol_excess_ann": -0.20,
        "baseline_regime_min_low_vol_excess_ann": -0.25,
    },
    "release": {
        "baseline_regime_ticker": "QQQ",
        "baseline_regime_return_window": 63,
        "baseline_regime_vol_window": 20,
        "baseline_regime_strong_return_threshold": 0.10,
        "baseline_regime_weak_return_threshold": 0.00,
        "baseline_regime_min_days": 40,
        "baseline_regime_min_up_day_excess_ann": -0.35,
        "baseline_regime_min_down_day_excess_ann": 0.00,
        "baseline_regime_min_strong_excess_ann": -0.15,
        "baseline_regime_min_weak_excess_ann": 0.00,
        "baseline_regime_min_high_vol_excess_ann": -0.15,
        "baseline_regime_min_low_vol_excess_ann": -0.20,
    },
    "growth": {
        "baseline_regime_ticker": "QQQ",
        "baseline_regime_return_window": 63,
        "baseline_regime_vol_window": 20,
        "baseline_regime_strong_return_threshold": 0.10,
        "baseline_regime_weak_return_threshold": 0.00,
        "baseline_regime_min_days": 40,
        "baseline_regime_min_up_day_excess_ann": -0.40,
        "baseline_regime_min_down_day_excess_ann": 0.00,
        "baseline_regime_min_strong_excess_ann": -0.20,
        "baseline_regime_min_weak_excess_ann": 0.00,
        "baseline_regime_min_high_vol_excess_ann": -0.20,
        "baseline_regime_min_low_vol_excess_ann": -0.25,
    },
    "qqq_release": {
        "baseline_regime_ticker": "QQQ",
        "baseline_regime_return_window": 63,
        "baseline_regime_vol_window": 20,
        "baseline_regime_strong_return_threshold": 0.10,
        "baseline_regime_weak_return_threshold": 0.00,
        "baseline_regime_min_days": 40,
        "baseline_regime_min_up_day_excess_ann": -0.20,
        "baseline_regime_min_down_day_excess_ann": 0.20,
        "baseline_regime_min_strong_excess_ann": -0.05,
        "baseline_regime_min_weak_excess_ann": 0.02,
        "baseline_regime_min_high_vol_excess_ann": -0.05,
        "baseline_regime_min_low_vol_excess_ann": -0.05,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and stress-check the US Sharadar weekly pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--provider_uri", default=None, help="Override qlib data root")
    parser.add_argument("--pred", default=None, help="Path to pred.pkl for backtest validation")
    parser.add_argument("--walkforward_manifest", default=None, help="Optional stitched walk-forward manifest JSON")
    parser.add_argument("--benchmark_pkl", default=None, help="Pickled pd.Series of benchmark daily returns")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--by_year", action="store_true", help="Slice backtests by calendar year")
    parser.add_argument(
        "--year_warmup_days",
        type=int,
        default=0,
        help="Trading-day warm-up before each yearly backtest slice; metrics are measured only from the year start.",
    )
    parser.add_argument(
        "--check_cold_start_years",
        action="store_true",
        help="Also report independent no-warm-up yearly backtests as cold-start diagnostics.",
    )
    parser.add_argument("--stress_cost_mult", type=float, default=2.0, help="Cost multiplier for stress test")
    parser.add_argument("--stress_deal_price", default="close", help="Deal price for stress test (close/open)")
    parser.add_argument("--sample_months", type=int, default=24, help="Months to sample for universe stats")
    parser.add_argument("--sample_instruments", type=int, default=200, help="Instrument sample size for data checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--check_start", default=None, help="Override data check start date")
    parser.add_argument("--check_end", default=None, help="Override data check end date")
    parser.add_argument(
        "--embargo_days",
        type=int,
        default=None,
        help="Required train/valid/test trading-day gap. Defaults to parsed label horizon when known.",
    )
    parser.add_argument(
        "--allow_pred_beyond_test_segment",
        action="store_true",
        help="Allow pred.pkl to extend beyond config test segment while still requiring backtest coverage.",
    )
    parser.add_argument(
        "--fail_on_input_check_fail",
        action="store_true",
        help="Return non-zero if config consistency or backtest input checks fail.",
    )
    parser.add_argument("--skip_data_checks", action="store_true", help="Skip data/universe checks")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip backtest checks")
    parser.add_argument("--check_data_quality", action="store_true", help="Run strict prediction/benchmark data quality checks on backtest inputs")
    parser.add_argument("--data_quality_topk", type=int, default=50, help="Top-k predictions per day to validate against close/volume availability")
    parser.add_argument("--data_quality_min_daily_scores", type=int, default=30, help="Minimum daily non-NaN score rows required for daily coverage")
    parser.add_argument("--data_quality_min_daily_coverage", type=float, default=0.98, help="Minimum ratio of backtest days meeting daily score coverage threshold")
    parser.add_argument("--data_quality_max_nan_score_ratio", type=float, default=0.01, help="Maximum NaN ratio allowed in prediction scores over backtest range")
    parser.add_argument("--data_quality_max_missing_close_ratio", type=float, default=0.01, help="Maximum missing close ratio in top-k prediction pairs")
    parser.add_argument("--data_quality_max_missing_volume_ratio", type=float, default=0.02, help="Maximum missing volume ratio in top-k prediction pairs")
    parser.add_argument("--data_quality_max_benchmark_nan_ratio", type=float, default=0.0, help="Maximum missing ratio for benchmark values over backtest calendar")
    parser.add_argument("--check_model_quality", action="store_true", help="Run prediction-vs-forward-label IC/top-k quality gates")
    parser.add_argument("--check_rebalance_model_quality", action="store_true", help="Also run model-quality gates on strategy rebalance weekdays only")
    parser.add_argument("--model_quality_topk", type=int, default=0, help="Top-k used for model quality checks; defaults to strategy topk")
    parser.add_argument("--model_quality_recent_days", type=int, default=None, help="Recent trading-label days used for model quality gates")
    parser.add_argument("--model_quality_min_days", type=int, default=None, help="Minimum full-period label days for model quality gates")
    parser.add_argument("--model_quality_min_recent_days", type=int, default=None, help="Minimum recent label days for model quality gates")
    parser.add_argument("--model_quality_min_year_days", type=int, default=None, help="Minimum label days for a year to be eligible for yearly model quality gates")
    parser.add_argument("--model_quality_min_daily_count", type=int, default=None, help="Minimum instruments per day for model quality metrics")
    parser.add_argument("--model_quality_min_full_mean_ic", type=float, default=None, help="Minimum full-period mean daily Spearman IC")
    parser.add_argument("--model_quality_min_full_topq_spread", type=float, default=None, help="Minimum full-period top quintile minus bottom quintile label")
    parser.add_argument("--model_quality_min_recent_mean_ic", type=float, default=None, help="Minimum recent mean daily Spearman IC")
    parser.add_argument("--model_quality_min_recent_pos_ic_rate", type=float, default=None, help="Minimum recent positive-IC day ratio")
    parser.add_argument("--model_quality_min_recent_topk_mean_label", type=float, default=None, help="Minimum recent top-k mean forward label")
    parser.add_argument("--model_quality_min_recent_topq_spread", type=float, default=None, help="Minimum recent top quintile minus bottom quintile label")
    parser.add_argument("--model_quality_min_positive_years", type=int, default=None, help="Minimum eligible calendar years with non-negative mean IC")
    parser.add_argument("--model_quality_min_worst_year_mean_ic", type=float, default=None, help="Minimum acceptable worst eligible yearly mean IC")
    parser.add_argument("--model_quality_min_worst_year_topk_mean_label", type=float, default=None, help="Minimum acceptable worst eligible yearly top-k mean forward label")
    parser.add_argument("--model_quality_min_worst_year_topq_spread", type=float, default=None, help="Minimum acceptable worst eligible yearly top-bottom quintile spread")
    parser.add_argument("--ensemble_manifest", default=None, help="Optional ensemble manifest JSON for source-aware diagnostics")
    parser.add_argument("--ensemble_gate_csv", default=None, help="Optional ensemble gate CSV; overrides manifest confidence_gate.gate_report_csv")
    parser.add_argument("--ensemble_primary_name", default="primary", help="Display name for primary ensemble source")
    parser.add_argument("--ensemble_defensive_name", default="defensive", help="Display name for defensive/fallback ensemble source")
    parser.add_argument("--ensemble_primary_label_horizon", type=int, default=None, help="Forward-label horizon for primary ensemble source; defaults to config label horizon")
    parser.add_argument("--ensemble_defensive_label_horizon", type=int, default=None, help="Forward-label horizon for defensive ensemble source; defaults to config label horizon")
    parser.add_argument("--model_quality_use_ensemble_source_horizons", action="store_true", help="Use source-specific label horizons from --ensemble_manifest for model-quality checks")
    parser.add_argument("--fail_on_model_quality_fail", action="store_true", help="Return non-zero exit code if model quality gates fail")
    parser.add_argument("--check_active_risk", action="store_true", help="Run benchmark-proxy active-risk diagnostics on strategy selections")
    parser.add_argument("--active_risk_marketcap_field", default="$marketcap_q", help="PIT field used to build cap-weighted benchmark proxy")
    parser.add_argument("--active_risk_topk", type=int, default=0, help="Top-k for active-risk checks; defaults to strategy topk")
    parser.add_argument("--active_risk_benchmark_topn", type=int, default=500, help="Number of largest PIT market-cap names in benchmark proxy")
    parser.add_argument("--active_risk_recent_rebalances", type=int, default=None, help="Recent rebalance count used for active-risk gates")
    parser.add_argument("--active_risk_min_rebalances", type=int, default=None, help="Minimum full-period rebalance count for active-risk gates")
    parser.add_argument("--active_risk_max_mean_active_share", type=float, default=None, help="Maximum full-period mean active share")
    parser.add_argument("--active_risk_max_recent_mean_active_share", type=float, default=None, help="Maximum recent mean active share")
    parser.add_argument("--active_risk_max_mean_abs_active_weight", type=float, default=None, help="Maximum mean single-name absolute active weight")
    parser.add_argument("--active_risk_max_mean_single_stock_weight", type=float, default=None, help="ETF-core active-risk gate: maximum mean single-stock portfolio weight outside the core benchmark sleeve")
    parser.add_argument("--active_risk_max_mean_core_weight_gap", type=float, default=None, help="ETF-core active-risk gate: maximum full-period mean absolute core benchmark weight gap")
    parser.add_argument("--active_risk_max_recent_core_weight_gap", type=float, default=None, help="ETF-core active-risk gate: maximum recent mean absolute core benchmark weight gap")
    parser.add_argument("--active_risk_max_mean_stock_overlay_weight", type=float, default=None, help="ETF-core active-risk gate: maximum mean non-core stock overlay weight")
    parser.add_argument("--active_risk_min_mean_benchmark_coverage", type=float, default=None, help="Minimum full-period mean benchmark weight represented in portfolio")
    parser.add_argument("--active_risk_min_recent_benchmark_coverage", type=float, default=None, help="Minimum recent benchmark weight represented in portfolio")
    parser.add_argument("--active_risk_max_mean_sector_active_weight", type=float, default=None, help="Maximum mean absolute sector active weight")
    parser.add_argument("--active_risk_max_mean_sector_stock_weight", type=float, default=None, help="ETF-core gate: maximum mean stock-overlay weight in any one sector")
    parser.add_argument("--active_risk_max_mean_portfolio_names", type=float, default=None, help="Maximum average number of portfolio names allowed by active-risk gates")
    parser.add_argument("--fail_on_active_risk_fail", action="store_true", help="Return non-zero exit code if active-risk gates fail")
    parser.add_argument(
        "--check_strategy_weighted_model_quality",
        action="store_true",
        help="Run forward-label checks on the strategy's approximate rebalance weights",
    )
    parser.add_argument(
        "--strategy_weighted_topk",
        type=int,
        default=0,
        help="Top-k alpha sleeve used for strategy-weighted checks; defaults to strategy topk",
    )
    parser.add_argument(
        "--strategy_weighted_marketcap_field",
        default=None,
        help="PIT market-cap field used by benchmark-aware strategy-weighted checks; defaults to strategy/active-risk field",
    )
    parser.add_argument("--strategy_weighted_recent_rebalances", type=int, default=None, help="Recent rebalance count used for strategy-weighted quality gates")
    parser.add_argument("--strategy_weighted_min_rebalances", type=int, default=None, help="Minimum full-period rebalance count for strategy-weighted quality gates")
    parser.add_argument("--strategy_weighted_min_recent_rebalances", type=int, default=None, help="Minimum recent rebalance count for strategy-weighted quality gates")
    parser.add_argument("--strategy_weighted_min_year_rebalances", type=int, default=None, help="Minimum yearly rebalance count for yearly strategy-weighted gates")
    parser.add_argument("--strategy_weighted_min_full_mean_label", type=float, default=None, help="Minimum full-period strategy-weighted forward excess label")
    parser.add_argument("--strategy_weighted_min_recent_mean_label", type=float, default=None, help="Minimum recent strategy-weighted forward excess label")
    parser.add_argument("--strategy_weighted_min_positive_label_rate", type=float, default=None, help="Minimum full-period ratio of positive strategy-weighted label rebalances")
    parser.add_argument("--strategy_weighted_min_recent_positive_label_rate", type=float, default=None, help="Minimum recent ratio of positive strategy-weighted label rebalances")
    parser.add_argument("--strategy_weighted_min_mean_label_weight_coverage", type=float, default=None, help="Minimum full-period mean portfolio weight with available forward labels")
    parser.add_argument("--strategy_weighted_min_recent_label_weight_coverage", type=float, default=None, help="Minimum recent mean portfolio weight with available forward labels")
    parser.add_argument("--strategy_weighted_min_positive_years", type=int, default=None, help="Minimum eligible years with non-negative strategy-weighted labels")
    parser.add_argument("--strategy_weighted_min_worst_year_mean_label", type=float, default=None, help="Minimum acceptable worst eligible yearly strategy-weighted label")
    parser.add_argument(
        "--fail_on_strategy_weighted_model_quality_fail",
        action="store_true",
        help="Return non-zero exit code if strategy-weighted model quality gates fail",
    )
    parser.add_argument(
        "--check_rebalance_interval_quality",
        action="store_true",
        help="Run forward-return checks on approximate strategy weights over actual rebalance-to-rebalance holding intervals",
    )
    parser.add_argument(
        "--rebalance_interval_topk",
        type=int,
        default=0,
        help="Top-k alpha sleeve used for rebalance-interval checks; defaults to strategy topk",
    )
    parser.add_argument(
        "--rebalance_interval_marketcap_field",
        default=None,
        help="PIT market-cap field used by benchmark-aware rebalance-interval checks; defaults to strategy/active-risk field",
    )
    parser.add_argument(
        "--rebalance_interval_price_field",
        default=None,
        help="Price field used for interval returns; defaults to the configured deal price field",
    )
    parser.add_argument(
        "--rebalance_interval_deal_price",
        default=None,
        help="Deal price used to infer the interval price field when --rebalance_interval_price_field is unset",
    )
    parser.add_argument("--rebalance_interval_recent_rebalances", type=int, default=None, help="Recent rebalance count used for interval quality gates")
    parser.add_argument("--rebalance_interval_min_rebalances", type=int, default=None, help="Minimum full-period rebalance intervals for interval quality gates")
    parser.add_argument("--rebalance_interval_min_recent_rebalances", type=int, default=None, help="Minimum recent rebalance intervals for interval quality gates")
    parser.add_argument("--rebalance_interval_min_year_rebalances", type=int, default=None, help="Minimum yearly rebalance intervals for interval quality gates")
    parser.add_argument("--rebalance_interval_min_full_ann_excess", type=float, default=None, help="Minimum full-period annualized rebalance-interval excess return")
    parser.add_argument("--rebalance_interval_min_recent_ann_excess", type=float, default=None, help="Minimum recent annualized rebalance-interval excess return")
    parser.add_argument("--rebalance_interval_min_positive_excess_rate", type=float, default=None, help="Minimum full-period positive interval excess rate")
    parser.add_argument("--rebalance_interval_min_recent_positive_excess_rate", type=float, default=None, help="Minimum recent positive interval excess rate")
    parser.add_argument("--rebalance_interval_min_mean_return_weight_coverage", type=float, default=None, help="Minimum full-period portfolio weight with valid interval returns")
    parser.add_argument("--rebalance_interval_min_recent_return_weight_coverage", type=float, default=None, help="Minimum recent portfolio weight with valid interval returns")
    parser.add_argument("--rebalance_interval_min_positive_years", type=int, default=None, help="Minimum eligible years with non-negative interval excess")
    parser.add_argument("--rebalance_interval_min_worst_year_ann_excess", type=float, default=None, help="Minimum acceptable worst eligible yearly interval excess")
    parser.add_argument("--rebalance_interval_max_abs_proxy_tracking_ann", type=float, default=None, help="Maximum absolute annualized return gap between benchmark proxy and ETF benchmark")
    parser.add_argument(
        "--fail_on_rebalance_interval_quality_fail",
        action="store_true",
        help="Return non-zero exit code if rebalance-interval quality gates fail",
    )
    parser.add_argument("--check_training_diagnostics", action="store_true", help="Check walk-forward MLflow training metrics for weak/near-constant models")
    parser.add_argument("--training_min_best_iteration", type=int, default=5, help="Minimum accepted one-based best boosting iteration for each walk-forward run")
    parser.add_argument("--fail_on_training_diagnostics_fail", action="store_true", help="Return non-zero exit code if training diagnostics fail")
    parser.add_argument(
        "--pit_staleness_max_p95_days",
        type=int,
        default=None,
        help="Optional maximum p95 constant-run length for sampled PIT field freshness.",
    )
    parser.add_argument(
        "--pit_tail_window_days",
        type=int,
        default=252,
        help="Recent trading-day window used for PIT freshness checks.",
    )
    parser.add_argument("--require_sf3a_provenance", action="store_true", help="Require lag-corrected SF3A rebuild provenance")
    parser.add_argument("--sf3a_min_availability_lag_days", type=int, default=45, help="Minimum accepted SF3A lag")
    parser.add_argument("--require_sf1_pit_provenance", action="store_true", help="Require SF1 PIT rebuild provenance")
    parser.add_argument("--require_model_feature_provenance", action="store_true", help="Require dumped Sharadar risk/regime model feature provenance when those fields are used")
    parser.add_argument("--model_feature_max_missing_ratio", type=float, default=0.80, help="Maximum sampled missing ratio allowed for risk/regime model features")
    parser.add_argument("--allow_static_metadata_features", action="store_true", help="Allow non-PIT current TICKERS metadata fields such as $meta_*")
    parser.add_argument("--require_fmp_feature_provenance", action="store_true", help="Require lagged FMP event feature provenance when $fmp_* fields are used")
    parser.add_argument("--fmp_feature_min_availability_lag_days", type=int, default=1, help="Minimum accepted FMP event-feature lag")
    parser.add_argument("--fmp_feature_max_missing_ratio", type=float, default=0.05, help="Maximum sampled missing ratio allowed for FMP event features")
    parser.add_argument("--require_sec_feature_provenance", action="store_true", help="Require lagged SEC event feature provenance when $sec_* fields are used")
    parser.add_argument("--sec_feature_min_availability_lag_days", type=int, default=1, help="Minimum accepted SEC event-feature lag")
    parser.add_argument("--report_max_nan_ratio", type=float, default=0.0, help="Maximum NaN ratio allowed in backtest report core columns")
    parser.add_argument("--fail_on_data_quality_fail", action="store_true", help="Return non-zero exit code if strict data/report quality checks fail")
    parser.add_argument(
        "--gate_profile",
        choices=["release", "research", "growth", "qqq_release"],
        default="release",
        help="Preset gate profile. qqq_release is the strict profile for QQQ-relative live-release candidates.",
    )
    parser.add_argument("--check_gates", action="store_true", help="Evaluate robustness gates on backtest outputs")
    parser.add_argument("--gate_min_full_excess_ann", type=float, default=None, help="Gate: minimum full-period excess annualized return")
    parser.add_argument("--gate_min_full_ir", type=float, default=None, help="Gate: minimum full-period IR")
    parser.add_argument("--gate_max_full_mdd_abs", type=float, default=None, help="Gate: maximum absolute full-period MDD")
    parser.add_argument("--gate_min_stress_excess_ann", type=float, default=None, help="Gate: minimum stress-test excess annualized return")
    parser.add_argument("--gate_max_turnover", type=float, default=None, help="Gate: maximum full-period average turnover")
    parser.add_argument("--gate_min_positive_excess_years", type=int, default=None, help="Gate: minimum number of positive-excess calendar years")
    parser.add_argument("--gate_min_worst_year_excess_ann", type=float, default=None, help="Gate: minimum acceptable worst calendar-year excess annualized return")
    parser.add_argument("--gate_min_year_days", type=int, default=None, help="Gate: minimum trading days for a year slice to be eligible for yearly robustness gates")
    parser.add_argument("--strict_positive_year_count", action="store_true", help="Do not cap positive-year gates by the number of eligible years")
    parser.add_argument("--check_baseline_gates", action="store_true", help="Evaluate external ticker baseline gates, e.g. QQQ for growth-track candidates")
    parser.add_argument("--baseline_tickers", default=None, help="Comma/newline separated external baseline tickers loaded from qlib close data")
    parser.add_argument("--baseline_pkl_map", default=None, help="Comma/newline ticker=return_pkl mapping for external baselines absent from qlib, e.g. IXIC=/path/bench_ixic.pkl")
    parser.add_argument("--baseline_min_full_excess_ann", type=float, default=None, help="External baseline gate: minimum full-period annualized excess return")
    parser.add_argument("--baseline_min_stress_excess_ann", type=float, default=None, help="External baseline gate: minimum stress annualized excess return")
    parser.add_argument("--baseline_max_full_mdd_abs", type=float, default=None, help="External baseline gate: maximum absolute full-period strategy MDD")
    parser.add_argument("--baseline_max_mdd_gap", type=float, default=None, help="External baseline gate: maximum strategy MDD gap versus baseline MDD")
    parser.add_argument("--baseline_min_positive_years", type=int, default=None, help="External baseline gate: minimum eligible years with positive excess")
    parser.add_argument("--baseline_min_yearly_beat_rate", type=float, default=None, help="External baseline gate: minimum share of eligible years with positive excess")
    parser.add_argument("--baseline_min_worst_year_excess_ann", type=float, default=None, help="External baseline gate: minimum worst eligible yearly annualized excess")
    parser.add_argument("--baseline_min_year_days", type=int, default=None, help="External baseline gate: minimum days for eligible yearly comparisons")
    parser.add_argument("--baseline_min_rolling_excess_ann", type=float, default=None, help="External baseline rolling gate: minimum annualized excess per passing window")
    parser.add_argument("--baseline_min_rolling_pass_rate", type=float, default=None, help="External baseline rolling gate: minimum passing-window ratio")
    parser.add_argument("--baseline_min_worst_rolling_excess_ann", type=float, default=None, help="External baseline rolling gate: minimum worst rolling annualized excess")
    parser.add_argument("--baseline_min_latest_rolling_excess_ann", type=float, default=None, help="External baseline rolling gate: minimum latest rolling annualized excess")
    parser.add_argument("--baseline_max_missing_ratio", type=float, default=None, help="External baseline gate: maximum missing qlib baseline returns over evaluated periods")
    parser.add_argument("--fail_on_baseline_gate_fail", action="store_true", help="Return non-zero exit code if external baseline gates fail")
    parser.add_argument("--check_baseline_regime_gates", action="store_true", help="Evaluate QQQ-relative performance by baseline regime")
    parser.add_argument("--baseline_regime_ticker", default=None, help="Baseline ticker used for regime diagnostics")
    parser.add_argument("--baseline_regime_return_window", type=int, default=None, help="Trailing return window for baseline regime diagnostics")
    parser.add_argument("--baseline_regime_vol_window", type=int, default=None, help="Trailing volatility window for baseline regime diagnostics")
    parser.add_argument("--baseline_regime_strong_return_threshold", type=float, default=None, help="Threshold for strong baseline momentum regime")
    parser.add_argument("--baseline_regime_weak_return_threshold", type=float, default=None, help="Threshold for weak baseline momentum regime")
    parser.add_argument("--baseline_regime_min_days", type=int, default=None, help="Minimum aligned days for each baseline regime gate")
    parser.add_argument("--baseline_regime_min_up_day_excess_ann", type=float, default=None, help="Minimum annualized excess on baseline up days")
    parser.add_argument("--baseline_regime_min_down_day_excess_ann", type=float, default=None, help="Minimum annualized excess on baseline down days")
    parser.add_argument("--baseline_regime_min_strong_excess_ann", type=float, default=None, help="Minimum annualized excess during strong baseline momentum")
    parser.add_argument("--baseline_regime_min_weak_excess_ann", type=float, default=None, help="Minimum annualized excess during weak baseline momentum")
    parser.add_argument("--baseline_regime_min_high_vol_excess_ann", type=float, default=None, help="Minimum annualized excess during high baseline volatility")
    parser.add_argument("--baseline_regime_min_low_vol_excess_ann", type=float, default=None, help="Minimum annualized excess during low baseline volatility")
    parser.add_argument("--trial_registry", default=None, help="Optional JSONL registry path to append release validation trial summary")
    parser.add_argument("--trial_id", default=None, help="Optional stable ID for this validation trial")
    parser.add_argument("--candidate_name", default=None, help="Optional candidate name for trial registry")
    parser.add_argument("--selection_reason", default=None, help="Optional short selection reason for trial registry")
    parser.add_argument("--trial_count", type=int, default=None, help="Number of tried candidates used for multiple-testing haircut; defaults to registry count + 1")
    parser.add_argument(
        "--fail_on_release_not_ready",
        action="store_true",
        help="Return non-zero if the aggregate release decision is not ready.",
    )
    parser.add_argument(
        "--strategy_signal_shift",
        type=int,
        default=1,
        help="Trading bars between signal date and execution date; qlib WeightStrategyBase uses 1.",
    )
    parser.add_argument("--fail_on_gate_fail", action="store_true", help="Return non-zero exit code if any gate fails")
    parser.add_argument("--check_rolling", action="store_true", help="Run rolling walk-forward robustness checks")
    parser.add_argument("--rolling_window_days", type=int, default=None, help="Rolling window size in trading days")
    parser.add_argument("--rolling_step_days", type=int, default=None, help="Rolling step in trading days")
    parser.add_argument("--rolling_min_days", type=int, default=None, help="Minimum trading days required for a rolling window")
    parser.add_argument("--rolling_min_excess_ann", type=float, default=None, help="Rolling gate: minimum annualized excess return")
    parser.add_argument("--rolling_min_ir", type=float, default=None, help="Rolling gate: minimum information ratio")
    parser.add_argument(
        "--rolling_ir_metric",
        choices=["strategy", "excess"],
        default="strategy",
        help="IR series used by rolling gates. strategy uses portfolio return IR; excess uses benchmark-relative excess IR.",
    )
    parser.add_argument("--rolling_max_mdd_abs", type=float, default=None, help="Rolling gate: maximum absolute max drawdown")
    parser.add_argument("--rolling_max_turnover", type=float, default=None, help="Rolling gate: maximum average turnover")
    parser.add_argument("--rolling_min_pass_rate", type=float, default=None, help="Rolling gate: minimum passing-window ratio")
    parser.add_argument(
        "--rolling_mode",
        choices=["independent", "continuous"],
        default="independent",
        help="Rolling evaluation mode. independent re-runs each window; continuous slices the full-period backtest state.",
    )
    parser.add_argument(
        "--rolling_warmup_days",
        type=int,
        default=0,
        help="Trading-day warm-up before independent rolling windows; metrics are measured only inside each window.",
    )
    parser.add_argument(
        "--check_cold_start_rolling",
        action="store_true",
        help="Also report independent no-warm-up rolling windows as cold-start diagnostics.",
    )
    parser.add_argument(
        "--cold_start_rolling_min_pass_rate",
        type=float,
        default=None,
        help="Minimum cold-start rolling pass rate when --fail_on_cold_start_rolling_fail is set.",
    )
    parser.add_argument(
        "--fail_on_cold_start_rolling_fail",
        action="store_true",
        help="Return non-zero if cold-start rolling pass rate is below --cold_start_rolling_min_pass_rate.",
    )
    parser.add_argument("--fail_on_rolling_fail", action="store_true", help="Return non-zero exit code if rolling pass rate is below threshold")
    args = parser.parse_args()

    gate_defaults = GATE_PRESETS[args.gate_profile]
    for key, val in gate_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    rolling_defaults = ROLLING_PRESETS[args.gate_profile]
    for key, val in rolling_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    model_quality_defaults = MODEL_QUALITY_PRESETS[args.gate_profile]
    for key, val in model_quality_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    active_risk_defaults = ACTIVE_RISK_PRESETS[args.gate_profile]
    for key, val in active_risk_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    strategy_weighted_defaults = STRATEGY_WEIGHTED_QUALITY_PRESETS[args.gate_profile]
    for key, val in strategy_weighted_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    rebalance_interval_defaults = REBALANCE_INTERVAL_QUALITY_PRESETS[args.gate_profile]
    for key, val in rebalance_interval_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    baseline_defaults = BASELINE_GATE_PRESETS[args.gate_profile]
    for key, val in baseline_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    baseline_regime_defaults = BASELINE_REGIME_PRESETS[args.gate_profile]
    for key, val in baseline_regime_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if args.cold_start_rolling_min_pass_rate is None:
        args.cold_start_rolling_min_pass_rate = 0.80 if str(args.gate_profile) == "qqq_release" else 0.0
    if str(args.gate_profile) == "qqq_release" and args.check_baseline_gates:
        args.check_baseline_regime_gates = True
    if str(args.gate_profile) == "qqq_release":
        args.strict_positive_year_count = True
    return args


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    cfg = _load_yaml(cfg_path)

    # Basic config extraction
    qlib_init = cfg.get("qlib_init", {})
    provider_uri = args.provider_uri or _safe_get(qlib_init, ["provider_uri"], None)
    if provider_uri is None:
        provider_uri = "/root/.qlib/qlib_data/us_data"

    data_handler_config = cfg.get("data_handler_config") or _safe_get(
        cfg,
        ["task", "dataset", "kwargs", "handler", "kwargs"],
        {},
    )
    market = cfg.get("market")
    config_benchmark = cfg.get("benchmark", "AAPL")
    port_cfg = cfg.get("port_analysis_config", {})
    strategy_def = _safe_get(port_cfg, ["strategy"], {}) or {}
    strategy_class = strategy_def.get("class", "WeeklyTopkDropoutStrategy")
    strategy_module = strategy_def.get("module_path", "qlib.contrib.strategy")
    strategy_cfg = strategy_def.get("kwargs", {}) or {}
    backtest_cfg = _safe_get(port_cfg, ["backtest"], {}) or {}

    segments = _safe_get(cfg, ["task", "dataset", "kwargs", "segments"], {}) or {}
    train_seg = segments.get("train", [])
    valid_seg = segments.get("valid", [])
    test_seg = segments.get("test", [])

    label_expr = None
    label_cfg = data_handler_config.get("label", [])
    if isinstance(label_cfg, list) and label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        label_expr = label_cfg[0][0]
    label_horizons = _parse_label_horizons(label_cfg)

    # Print summary
    _print_header("Config Summary")
    _print_kv("config", str(cfg_path))
    _print_kv("gate_profile", str(args.gate_profile))
    _print_kv("provider_uri", provider_uri)
    _print_kv("market", str(market))
    _print_kv("benchmark_config", str(config_benchmark))
    _print_kv("train", _date_span_summary(_as_ts(train_seg[0]) if train_seg else None, _as_ts(train_seg[1]) if len(train_seg) > 1 else None))
    _print_kv("valid", _date_span_summary(_as_ts(valid_seg[0]) if valid_seg else None, _as_ts(valid_seg[1]) if len(valid_seg) > 1 else None))
    _print_kv("test", _date_span_summary(_as_ts(test_seg[0]) if test_seg else None, _as_ts(test_seg[1]) if len(test_seg) > 1 else None))

    label_horizon = max(label_horizons) if label_horizons else (_parse_label_horizon(label_expr) if label_expr else None)
    _print_kv("label_expr", label_expr or "<missing>")
    _print_kv("label_horizon_days", str(label_horizon) if label_horizon is not None else "<unknown>")
    if len(label_horizons) > 1:
        _print_kv("label_horizons_all", ",".join(str(x) for x in label_horizons))

    rebalance_weekday = strategy_cfg.get("rebalance_weekday")
    _print_kv("strategy_class", f"{strategy_module}.{strategy_class}")
    _print_kv("rebalance_weekday", _fmt_opt(rebalance_weekday))
    _print_kv("topk", _fmt_opt(strategy_cfg.get("topk")))
    _print_kv("n_drop", _fmt_opt(strategy_cfg.get("n_drop")))
    _print_kv("hold_thresh", _fmt_opt(strategy_cfg.get("hold_thresh")))

    # Calendar-backed consistency checks need the same provider as the backtest.
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.data.data import Cal

    qlib.init(provider_uri=provider_uri, region=REG_US)

    _print_header("Config Consistency Checks")
    consistency_rows: List[Tuple[str, bool, str]] = []
    train_start = _as_ts(train_seg[0]) if train_seg else None
    train_end = _as_ts(train_seg[1]) if len(train_seg) > 1 else None
    valid_start = _as_ts(valid_seg[0]) if valid_seg else None
    valid_end = _as_ts(valid_seg[1]) if len(valid_seg) > 1 else None
    test_start = _as_ts(test_seg[0]) if test_seg else None
    test_end = _as_ts(test_seg[1]) if len(test_seg) > 1 else None

    seg_ok = all(v is not None for v in [train_start, train_end, valid_start, valid_end, test_start, test_end])
    consistency_rows.append(("segments_present", seg_ok, "train/valid/test each has start+end"))
    if seg_ok:
        temporal_ok = train_start <= train_end < valid_start <= valid_end < test_start <= test_end
        detail = f"{train_start.date()}->{train_end.date()} < {valid_start.date()}->{valid_end.date()} < {test_start.date()}->{test_end.date()}"
        consistency_rows.append(("segments_temporal_order", temporal_ok, detail))
        req_embargo = _required_embargo_days(args, label_horizon)
        cal_for_segments = _get_calendar_span(train_start, test_start)
        train_valid_gap = _count_trade_days_between(cal_for_segments, train_end, valid_start)
        valid_test_gap = _count_trade_days_between(cal_for_segments, valid_end, test_start)
        consistency_rows.append(
            (
                "train_valid_embargo",
                train_valid_gap >= req_embargo,
                f"gap={train_valid_gap} trading_days >= required={req_embargo}",
            )
        )
        consistency_rows.append(
            (
                "valid_test_embargo",
                valid_test_gap >= req_embargo,
                f"gap={valid_test_gap} trading_days >= required={req_embargo}",
            )
        )

    hold_thresh = strategy_cfg.get("hold_thresh")
    if isinstance(hold_thresh, (int, float)) and label_horizon is not None:
        align_ok = int(hold_thresh) == int(label_horizon)
        consistency_rows.append(
            (
                "hold_horizon_alignment",
                align_ok,
                f"hold_thresh={int(hold_thresh)} vs label_horizon={int(label_horizon)}",
            )
        )

    if str(args.gate_profile) == "qqq_release":
        qqq_tickers = _coerce_ticker_list(strategy_cfg.get("benchmark_tickers"))
        config_benchmark_text = str(config_benchmark or "").strip().upper()
        backtest_benchmark_text = str(backtest_cfg.get("benchmark") or "").strip().upper()
        consistency_rows.append(
            (
                "qqq_release_config_benchmark",
                config_benchmark_text == "QQQ",
                f"benchmark={config_benchmark_text or '<missing>'} expected QQQ",
            )
        )
        consistency_rows.append(
            (
                "qqq_release_backtest_benchmark",
                backtest_benchmark_text == "QQQ",
                f"backtest.benchmark={backtest_benchmark_text or '<missing>'} expected QQQ",
            )
        )
        consistency_rows.append(
            (
                "qqq_release_strategy_core_ticker",
                "QQQ" in qqq_tickers,
                f"benchmark_tickers={qqq_tickers}",
            )
        )

    strategy_feasibility_rows = strategy_feasibility_check_rows(strategy_cfg, strategy_class)
    consistency_rows.extend(strategy_feasibility_rows)
    strategy_feasibility_ok = all(ok for _, ok, _ in strategy_feasibility_rows) if strategy_feasibility_rows else True
    config_consistency_ok = all(ok for _, ok, _ in consistency_rows)
    _print_check_table(consistency_rows)
    if args.fail_on_input_check_fail and not config_consistency_ok:
        return 8

    provenance_rows = _provenance_check_rows(provider_uri, data_handler_config, args)
    if provenance_rows:
        _print_header("Feature Provenance Checks")
        _print_check_table(provenance_rows)
        if args.fail_on_input_check_fail and not all(ok for _, ok, _ in provenance_rows):
            return 8

    cal = Cal.calendar(freq="day", future=False)
    if len(cal) > 0:
        last_cal = cal[-1]
        test_end = _as_ts(test_seg[1]) if len(test_seg) > 1 else None
        if test_end is not None and test_end < last_cal:
            _print_kv("warning", f"test end {test_end.date()} < latest calendar {last_cal.date()}")

    # Rebalance weekday holiday check
    if rebalance_weekday is not None and test_seg and len(test_seg) > 1:
        start = _as_ts(test_seg[0])
        end = _as_ts(test_seg[1])
        cal_range = _get_calendar_span(start, end)
        if len(cal_range) > 0:
            weeks = {}
            for dt in cal_range:
                iso = dt.isocalendar()
                key = (iso.year, iso.week)
                weeks.setdefault(key, set()).add(dt.weekday())
            missing_weeks = [k for k, days in weeks.items() if rebalance_weekday not in days]
            _print_kv("weeks_missing_rebalance_day", str(len(missing_weeks)))
            if missing_weeks:
                sample = ", ".join([f"{y}-W{w}" for y, w in missing_weeks[:5]])
                _print_kv("missing_week_samples", sample)

    if not args.skip_data_checks:
        _print_header("Universe & Data Checks")
        data_rows: List[Tuple[str, bool, str]] = []
        # Universe spans with filters
        filter_pipe = data_handler_config.get("filter_pipe", [])
        inst_conf = D.instruments(market, filter_pipe=filter_pipe)
        inst_spans = D.list_instruments(inst_conf, start_time=data_handler_config.get("start_time"), end_time=data_handler_config.get("end_time"), as_list=False)

        check_start = _as_ts(args.check_start) if args.check_start else _as_ts(data_handler_config.get("start_time"))
        check_end = _as_ts(args.check_end) if args.check_end else _as_ts(data_handler_config.get("end_time"))
        cal_range = _get_calendar_span(check_start, check_end)
        sample_dates = _sample_month_starts(cal_range, args.sample_months)

        if sample_dates:
            counts = [
                _count_active_instruments(inst_spans, dt) for dt in sample_dates
            ]
            if counts:
                _print_kv("sample_months", str(len(sample_dates)))
                _print_kv("universe_count_min", str(int(np.min(counts))))
                _print_kv("universe_count_median", str(int(np.median(counts))))
                _print_kv("universe_count_max", str(int(np.max(counts))))

        # Feature availability on sample
        sample_inst = _sample_instruments(inst_spans, args.sample_instruments, args.seed)
        pit_fields = data_handler_config.get("pit_fields", [])
        if sample_inst and len(cal_range) > 0:
            sample_start = sample_dates[0] if sample_dates else cal_range[0]
            sample_end = sample_dates[-1] if sample_dates else cal_range[-1]
            extra_fields = data_handler_config.get("extra_fields", [])
            model_feature_fields = []
            fmp_feature_fields = []
            if isinstance(extra_fields, list):
                for field in extra_fields:
                    names = _field_names_from_expressions([field])
                    if any(any(name.startswith(prefix) for prefix in MODEL_FEATURE_PREFIXES) for name in names):
                        model_feature_fields.append(str(field))
                    if any(name.startswith(FMP_FEATURE_PREFIX) for name in names):
                        fmp_feature_fields.append(str(field))
            fields = ["$close", "$volume"]
            if pit_fields:
                fields += [f"P($${str(f).strip().lower()}_{data_handler_config.get('pit_interval', 'q')})" for f in pit_fields[:3]]
            if extra_fields:
                fields += list(extra_fields[:2])
            for field in model_feature_fields[:8]:
                if field not in fields:
                    fields.append(field)
            for field in fmp_feature_fields[:8]:
                if field not in fields:
                    fields.append(field)

            df = D.features(sample_inst, fields, start_time=sample_start, end_time=sample_end)
            df = _flatten_columns(df)
            missing = _missing_ratio(df)
            _print_kv("sample_instruments", str(len(sample_inst)))
            _print_kv("sample_span", _date_span_summary(sample_start, sample_end))
            if not missing.empty:
                top_missing = missing.head(5)
                _print_kv("missing_ratio_top5", ", ".join([f"{k}={v:.2%}" for k, v in top_missing.items()]))
            if model_feature_fields and not missing.empty:
                for field in model_feature_fields[:8]:
                    ratio = float(missing.get(field, 1.0))
                    data_rows.append(
                        (
                            f"model_feature_missing_ratio:{field}",
                            ratio <= float(args.model_feature_max_missing_ratio),
                            f"{_format_float(ratio)} <= {_format_float(float(args.model_feature_max_missing_ratio))}",
                        )
                    )
            if fmp_feature_fields and not missing.empty:
                for field in fmp_feature_fields[:8]:
                    ratio = float(missing.get(field, 1.0))
                    data_rows.append(
                        (
                            f"fmp_feature_missing_ratio:{field}",
                            ratio <= float(args.fmp_feature_max_missing_ratio),
                            f"{_format_float(ratio)} <= {_format_float(float(args.fmp_feature_max_missing_ratio))}",
                        )
                    )

            # PIT staleness check on first PIT field
            if pit_fields:
                pit_field = f"P($${str(pit_fields[0]).strip().lower()}_{data_handler_config.get('pit_interval', 'q')})"
                if pit_field in df.columns:
                    staleness = _pit_staleness_days(df, pit_field)
                    if not staleness.empty:
                        pit_median = int(np.median(staleness))
                        pit_p95 = int(np.percentile(staleness, 95))
                        _print_kv("pit_staleness_median_days", str(pit_median))
                        _print_kv("pit_staleness_p95_days", str(pit_p95))
        if args.pit_staleness_max_p95_days is not None and pit_fields:
            if len(cal_range) > 0:
                tail_cal = cal_range[-max(1, int(args.pit_tail_window_days)) :]
                tail_sample_inst = _sample_active_instruments(
                    inst_spans,
                    tail_cal[-1],
                    args.sample_instruments,
                    args.seed,
                )
                pit_field = f"P($${str(pit_fields[0]).strip().lower()}_{data_handler_config.get('pit_interval', 'q')})"
                if tail_sample_inst:
                    df_tail = D.features(tail_sample_inst, [pit_field], start_time=tail_cal[0], end_time=tail_cal[-1])
                    df_tail = _flatten_columns(df_tail)
                    if pit_field in df_tail.columns:
                        tail_staleness = _pit_staleness_days(df_tail, pit_field)
                        if not tail_staleness.empty:
                            tail_p95 = int(np.percentile(tail_staleness, 95))
                            data_rows.append(
                                (
                                    "pit_tail_staleness_p95",
                                    tail_p95 <= int(args.pit_staleness_max_p95_days),
                                    f"{tail_p95} <= {int(args.pit_staleness_max_p95_days)} days over tail_window={len(tail_cal)} active_sample={len(tail_sample_inst)}",
                                )
                            )
                        else:
                            data_rows.append(("pit_tail_staleness_p95", False, "no non-empty PIT tail staleness series"))
                    else:
                        data_rows.append(("pit_tail_staleness_p95", False, f"field missing: {pit_field}"))
                else:
                    data_rows.append(("pit_tail_staleness_p95", False, "no active sample instruments at tail date"))
            else:
                data_rows.append(("pit_tail_staleness_p95", False, "no calendar range"))
        if data_rows:
            _print_check_table(data_rows)
            data_checks_ok = all(ok for _, ok, _ in data_rows)
            _print_kv("data_checks_overall", "PASS" if data_checks_ok else "FAIL")
            if args.fail_on_data_quality_fail and not data_checks_ok:
                return 9

    if not args.pred:
        _print_header("Backtest Checks")
        _print_kv("note", "pred.pkl not provided; skipping prediction/model/backtest checks")
        return 0

    pred_path = Path(args.pred).expanduser().resolve()
    if not pred_path.exists():
        print(f"pred.pkl not found: {pred_path}")
        return 2

    pred = _read_pickle_compat(pred_path)
    if not isinstance(pred, pd.DataFrame):
        print("pred.pkl must be a pandas DataFrame")
        return 2

    # Determine backtest range
    dt_index = pred.index.get_level_values("datetime")
    cfg_bt_start = _as_ts(backtest_cfg.get("start_time"))
    cfg_bt_end = _as_ts(backtest_cfg.get("end_time"))
    if args.start:
        bt_start_raw = pd.Timestamp(args.start)
    elif cfg_bt_start is not None:
        bt_start_raw = cfg_bt_start
    else:
        bt_start_raw = dt_index.min()
    if args.end:
        bt_end_raw = pd.Timestamp(args.end)
    elif cfg_bt_end is not None:
        bt_end_raw = cfg_bt_end
    else:
        bt_end_raw = dt_index.max()

    # Avoid using future calendar index
    cal = Cal.calendar(freq="day", future=False)
    if len(cal) >= 2:
        bt_end_raw = min(bt_end_raw, cal[-2])

    bt_calendar = _get_calendar_span(bt_start_raw, bt_end_raw)
    if len(bt_calendar) < 2:
        print(f"invalid backtest calendar span: start={bt_start_raw.date()} end={bt_end_raw.date()}")
        return 2
    bt_start = bt_calendar[0]
    bt_end = bt_calendar[-1]

    benchmark = config_benchmark
    benchmark_source = "config"
    if args.benchmark_pkl:
        bench_path = Path(args.benchmark_pkl).expanduser().resolve()
        benchmark = _read_pickle_compat(bench_path)
        if not isinstance(benchmark, pd.Series):
            print(f"benchmark_pkl must be a pandas Series: {bench_path}")
            return 2
        benchmark_source = f"pkl:{bench_path}"
    _print_kv("benchmark_effective", _summarize_benchmark(benchmark))
    _print_kv("benchmark_source", benchmark_source)
    if args.check_baseline_gates:
        _print_kv("external_baseline_tickers", ",".join(_coerce_ticker_list(args.baseline_tickers)) or "<missing>")

    ensemble_source_dates: Dict[str, List[pd.Timestamp]] = {}
    if args.ensemble_manifest or args.ensemble_gate_csv:
        manifest_path = Path(args.ensemble_manifest).expanduser().resolve() if args.ensemble_manifest else None
        gate_csv_path = Path(args.ensemble_gate_csv).expanduser().resolve() if args.ensemble_gate_csv else None
        ensemble_source_dates, ensemble_source_rows = _load_ensemble_source_dates(
            pred,
            manifest_path=manifest_path,
            gate_csv_path=gate_csv_path,
            primary_name=str(args.ensemble_primary_name),
            defensive_name=str(args.ensemble_defensive_name),
        )
        _print_header("Ensemble Source Attribution")
        _print_check_table(ensemble_source_rows)
        _print_ensemble_source_summary(ensemble_source_dates)

    _print_header("Backtest Input Checks")
    input_rows: List[Tuple[str, bool, str]] = []
    pred_min = dt_index.min()
    pred_max = dt_index.max()
    pred_cover_ok = pred_min <= bt_start and pred_max >= bt_end
    input_rows.append(
        (
            "pred_covers_backtest_range",
            pred_cover_ok,
            f"pred={pred_min.date()}->{pred_max.date()}, bt={bt_start.date()}->{bt_end.date()}",
        )
    )
    pred_in_test_ok = True
    if test_start is not None and test_end is not None:
        pred_in_test_ok = pred_min >= test_start and pred_max <= test_end
        pred_segment_ok = pred_in_test_ok or args.allow_pred_beyond_test_segment
        pred_segment_detail = f"pred={pred_min.date()}->{pred_max.date()}, test={test_start.date()}->{test_end.date()}"
        if args.allow_pred_beyond_test_segment and not pred_in_test_ok:
            pred_segment_detail += " (allowed by --allow_pred_beyond_test_segment)"
        input_rows.append(
            (
                "pred_within_test_segment",
                pred_segment_ok,
                pred_segment_detail,
            )
        )
    if isinstance(benchmark, pd.Series):
        bidx = pd.DatetimeIndex(benchmark.index)
        bench_cover_ok = bidx.min() <= bt_start and bidx.max() >= bt_end
        input_rows.append(
            (
                "benchmark_covers_backtest_range",
                bench_cover_ok,
                f"bench={bidx.min().date()}->{bidx.max().date()}, bt={bt_start.date()}->{bt_end.date()}",
            )
        )
    _print_check_table(input_rows)
    input_ok = all(ok for _, ok, _ in input_rows)
    if args.fail_on_input_check_fail and not input_ok:
        return 8

    if args.walkforward_manifest:
        _print_header("Walk-Forward Manifest Checks")
        manifest_rows = _manifest_check_rows(
            Path(args.walkforward_manifest).expanduser().resolve(),
            args=args,
            label_horizon=label_horizon,
            pred=pred,
        )
        _print_check_table(manifest_rows)
        if args.fail_on_input_check_fail and not all(ok for _, ok, _ in manifest_rows):
            return 10

    if args.check_training_diagnostics:
        _print_header("Training Diagnostics")
        if not args.walkforward_manifest:
            training_rows = [("training_manifest_present", False, "--walkforward_manifest is required")]
        else:
            training_rows = _training_diagnostic_rows(
                Path(args.walkforward_manifest).expanduser().resolve(),
                cfg=cfg,
                cfg_path=cfg_path,
                min_best_iteration=int(args.training_min_best_iteration),
            )
        _print_check_table(training_rows)
        training_ok = all(ok for _, ok, _ in training_rows)
        _print_kv("training_diagnostics_overall", "PASS" if training_ok else "FAIL")
        if args.fail_on_training_diagnostics_fail and not training_ok:
            return 12

    if args.check_model_quality:
        _print_header("Model Quality Checks")
        model_rows: List[Tuple[str, bool, str]]
        if not label_expr or label_horizon is None:
            model_rows = [("model_quality_label_expr_present", False, f"label_expr={label_expr!r}, horizon={label_horizon}")]
        else:
            try:
                label_ref_start = _infer_label_ref_start_days(data_handler_config)
                model_topk = int(args.model_quality_topk) if int(args.model_quality_topk) > 0 else int(strategy_cfg.get("topk", 40))
                use_source_horizons = bool(args.model_quality_use_ensemble_source_horizons and ensemble_source_dates)
                if use_source_horizons:
                    source_horizons = {
                        str(args.ensemble_primary_name): int(args.ensemble_primary_label_horizon or label_horizon),
                        str(args.ensemble_defensive_name): int(args.ensemble_defensive_label_horizon or label_horizon),
                    }
                    _print_kv(
                        "model_quality_label_mode",
                        "ensemble_source_horizons "
                        + ",".join(f"{src}={horizon}d" for src, horizon in source_horizons.items()),
                    )
                    model_frame = _source_aware_prediction_label_frame(
                        pred,
                        label_expr=label_expr,
                        benchmark=benchmark,
                        bt_start=bt_start,
                        bt_end=bt_end,
                        label_ref_start_days=label_ref_start,
                        strategy_cfg=strategy_cfg,
                        source_dates=ensemble_source_dates,
                        source_horizons=source_horizons,
                    )
                else:
                    _print_kv("model_quality_label_mode", f"config_horizon={int(label_horizon)}d")
                    model_frame = _load_prediction_label_frame(
                        pred,
                        label_expr=label_expr,
                        benchmark=benchmark,
                        bt_start=bt_start,
                        bt_end=bt_end,
                        label_horizon_days=int(label_horizon),
                        label_ref_start_days=label_ref_start,
                        strategy_cfg=strategy_cfg,
                    )
                full_model_quality = _model_quality_metrics(
                    model_frame,
                    topk=model_topk,
                    min_daily_count=int(args.model_quality_min_daily_count),
                    recent_days=None,
                )
                recent_model_quality = _model_quality_metrics(
                    model_frame,
                    topk=model_topk,
                    min_daily_count=int(args.model_quality_min_daily_count),
                    recent_days=int(args.model_quality_recent_days),
                )
                if use_source_horizons:
                    _print_model_quality_by_source(
                        model_frame,
                        topk=model_topk,
                        min_daily_count=int(args.model_quality_min_daily_count),
                    )
                yearly_model_quality = _model_quality_by_year(
                    model_frame,
                    topk=model_topk,
                    min_daily_count=int(args.model_quality_min_daily_count),
                )
                _print_model_quality_summary("model_quality_full", full_model_quality)
                _print_model_quality_summary(
                    f"model_quality_recent_{int(args.model_quality_recent_days)}d",
                    recent_model_quality,
                )
                _print_model_quality_yearly(yearly_model_quality)
                model_rows = _model_quality_gate_rows(
                    full_model_quality,
                    recent_model_quality,
                    args,
                    yearly=yearly_model_quality,
                )
                if args.check_rebalance_model_quality:
                    if rebalance_weekday is None:
                        model_rows.append(("rebalance_model_quality_weekday_present", False, "strategy rebalance_weekday is missing"))
                    else:
                        rebalance_calendar = _get_calendar_span(bt_start - pd.Timedelta(days=10), bt_end)
                        signal_dates = _rebalance_signal_dates(
                            rebalance_calendar,
                            int(rebalance_weekday),
                            start=bt_start,
                            end=bt_end,
                        )
                        _print_kv("rebalance_model_quality_signal_dates", str(len(signal_dates)))
                        rebalance_frame = _filter_model_frame_by_dates(model_frame, signal_dates)
                        rebalance_args = argparse.Namespace(**vars(args))
                        rebalance_args.model_quality_recent_days = max(1, int(math.ceil(int(args.model_quality_recent_days) / 5.0)))
                        rebalance_args.model_quality_min_days = max(8, int(math.ceil(int(args.model_quality_min_days) / 5.0)))
                        rebalance_args.model_quality_min_recent_days = max(
                            4,
                            int(math.ceil(int(args.model_quality_min_recent_days) / 5.0)),
                        )
                        rebalance_args.model_quality_min_year_days = max(
                            8,
                            int(math.ceil(int(args.model_quality_min_year_days) / 5.0)),
                        )
                        rebalance_full_quality = _model_quality_metrics(
                            rebalance_frame,
                            topk=model_topk,
                            min_daily_count=int(args.model_quality_min_daily_count),
                            recent_days=None,
                        )
                        rebalance_recent_quality = _model_quality_metrics(
                            rebalance_frame,
                            topk=model_topk,
                            min_daily_count=int(args.model_quality_min_daily_count),
                            recent_days=int(rebalance_args.model_quality_recent_days),
                        )
                        rebalance_yearly_quality = _model_quality_by_year(
                            rebalance_frame,
                            topk=model_topk,
                            min_daily_count=int(args.model_quality_min_daily_count),
                        )
                        _print_model_quality_summary("rebalance_model_quality_full", rebalance_full_quality)
                        _print_model_quality_summary(
                            f"rebalance_model_quality_recent_{int(rebalance_args.model_quality_recent_days)}d",
                            rebalance_recent_quality,
                        )
                        _print_model_quality_yearly(rebalance_yearly_quality)
                        model_rows.extend(
                            _model_quality_gate_rows(
                                rebalance_full_quality,
                                rebalance_recent_quality,
                                rebalance_args,
                                yearly=rebalance_yearly_quality,
                                prefix="rebalance_model_quality",
                            )
                        )
            except Exception as e:
                model_rows = [("model_quality_computed", False, repr(e))]
        _print_check_table(model_rows)
        model_quality_ok = all(ok for _, ok, _ in model_rows)
        _print_kv("model_quality_overall", "PASS" if model_quality_ok else "FAIL")
        if args.fail_on_model_quality_fail and not model_quality_ok:
            return 11

    if args.check_strategy_weighted_model_quality:
        _print_header("Strategy-Weighted Model Quality Checks")
        weighted_rows: List[Tuple[str, bool, str]]
        if not label_expr or label_horizon is None:
            weighted_rows = [
                ("strategy_weighted_label_expr_present", False, f"label_expr={label_expr!r}, horizon={label_horizon}")
            ]
        else:
            try:
                label_ref_start = _infer_label_ref_start_days(data_handler_config)
                weighted_frame = _strategy_weighted_label_frame(
                    pred,
                    label_expr=label_expr,
                    benchmark=benchmark,
                    bt_start=bt_start,
                    bt_end=bt_end,
                    label_horizon_days=int(label_horizon),
                    label_ref_start_days=label_ref_start,
                    bt_calendar=bt_calendar,
                    strategy_cfg=strategy_cfg,
                    strategy_class=strategy_class,
                    rebalance_weekday=rebalance_weekday,
                    args=args,
                )
                full_weighted = _strategy_weighted_quality_metrics(weighted_frame)
                recent_weighted = _strategy_weighted_quality_metrics(
                    weighted_frame,
                    recent_rebalances=int(args.strategy_weighted_recent_rebalances),
                )
                yearly_weighted = _strategy_weighted_quality_by_year(weighted_frame)
                _print_strategy_weighted_quality_summary("strategy_weighted_quality_full", full_weighted)
                _print_strategy_weighted_quality_summary(
                    f"strategy_weighted_quality_recent_{int(args.strategy_weighted_recent_rebalances)}rb",
                    recent_weighted,
                )
                _print_strategy_weighted_quality_yearly(yearly_weighted)
                weighted_rows = _strategy_weighted_quality_gate_rows(
                    full_weighted,
                    recent_weighted,
                    args,
                    yearly=yearly_weighted,
                )
            except Exception as e:
                weighted_rows = [("strategy_weighted_quality_computed", False, repr(e))]
        _print_check_table(weighted_rows)
        weighted_ok = all(ok for _, ok, _ in weighted_rows)
        _print_kv("strategy_weighted_quality_overall", "PASS" if weighted_ok else "FAIL")
        if args.fail_on_strategy_weighted_model_quality_fail and not weighted_ok:
            return 14

    if args.check_data_quality:
        _print_header("Strict Data Quality Checks")
        dq_rows = _evaluate_strict_data_quality(
            pred,
            bt_start=bt_start,
            bt_end=bt_end,
            bt_calendar=bt_calendar,
            benchmark=benchmark,
            topk=args.data_quality_topk,
            min_daily_scores=args.data_quality_min_daily_scores,
            min_daily_coverage=args.data_quality_min_daily_coverage,
            max_nan_score_ratio=args.data_quality_max_nan_score_ratio,
            max_missing_close_ratio=args.data_quality_max_missing_close_ratio,
            max_missing_volume_ratio=args.data_quality_max_missing_volume_ratio,
            max_benchmark_nan_ratio=args.data_quality_max_benchmark_nan_ratio,
        )
        _print_check_table(dq_rows)
        dq_ok = all(ok for _, ok, _ in dq_rows)
        _print_kv("data_quality_overall", "PASS" if dq_ok else "FAIL")
        if args.fail_on_data_quality_fail and not dq_ok:
            return 5

    if args.check_active_risk:
        _print_header("Active-Risk Checks")
        try:
            active_metrics = _active_risk_metrics_for_predictions(
                pred,
                bt_start=bt_start,
                bt_end=bt_end,
                bt_calendar=bt_calendar,
                strategy_cfg=strategy_cfg,
                strategy_class=strategy_class,
                rebalance_weekday=rebalance_weekday,
                args=args,
            )
            full_active = _summarize_active_risk(active_metrics)
            recent_active = _summarize_active_risk(active_metrics, recent_rebalances=int(args.active_risk_recent_rebalances))
            _print_active_risk_summary("active_risk_full", full_active)
            _print_active_risk_summary(f"active_risk_recent_{int(args.active_risk_recent_rebalances)}rb", recent_active)
            active_rows = _active_risk_gate_rows(full_active, recent_active, args)
        except Exception as e:
            active_rows = [("active_risk_computed", False, repr(e))]
        _print_check_table(active_rows)
        active_ok = all(ok for _, ok, _ in active_rows)
        _print_kv("active_risk_overall", "PASS" if active_ok else "FAIL")
        if args.fail_on_active_risk_fail and not active_ok:
            return 13

    if args.check_rebalance_interval_quality:
        _print_header("Rebalance-Interval Quality Checks")
        if args.rebalance_interval_deal_price is None:
            args.rebalance_interval_deal_price = str((backtest_cfg.get("exchange_kwargs", {}) or {}).get("deal_price", "close"))
        interval_rows: List[Tuple[str, bool, str]]
        try:
            interval_frame = _rebalance_interval_quality_frame(
                pred,
                benchmark=benchmark,
                bt_start=bt_start,
                bt_end=bt_end,
                bt_calendar=bt_calendar,
                strategy_cfg=strategy_cfg,
                strategy_class=strategy_class,
                rebalance_weekday=rebalance_weekday,
                args=args,
            )
            full_interval = _rebalance_interval_quality_metrics(interval_frame)
            recent_interval = _rebalance_interval_quality_metrics(
                interval_frame,
                recent_rebalances=int(args.rebalance_interval_recent_rebalances),
            )
            yearly_interval = _rebalance_interval_quality_by_year(interval_frame)
            _print_rebalance_interval_quality_summary("rebalance_interval_quality_full", full_interval)
            _print_rebalance_interval_quality_summary(
                f"rebalance_interval_quality_recent_{int(args.rebalance_interval_recent_rebalances)}rb",
                recent_interval,
            )
            _print_rebalance_interval_quality_yearly(yearly_interval)
            interval_rows = _rebalance_interval_quality_gate_rows(
                full_interval,
                recent_interval,
                args,
                yearly=yearly_interval,
            )
        except Exception as e:
            interval_rows = [("rebalance_interval_quality_computed", False, repr(e))]
        _print_check_table(interval_rows)
        interval_ok = all(ok for _, ok, _ in interval_rows)
        _print_kv("rebalance_interval_quality_overall", "PASS" if interval_ok else "FAIL")
        if args.fail_on_rebalance_interval_quality_fail and not interval_ok:
            return 15

    if args.skip_backtest:
        return 0

    account = backtest_cfg.get("account", 10000000)
    exchange_kwargs = backtest_cfg.get("exchange_kwargs", {}) or {}
    strategy_kwargs = dict(strategy_cfg)
    strategy_kwargs["signal"] = pred
    base_strategy = {"class": strategy_class, "module_path": strategy_module, "kwargs": strategy_kwargs}

    _print_header("Backtest Checks")
    report = _run_backtest(
        pred, base_strategy, bt_start, bt_end, benchmark, account=account, exchange_kwargs=exchange_kwargs
    )
    rows = [(f"full {bt_start.date()}->{bt_end.date()}", _summarize_report(report))]
    yearly_reports: List[Tuple[str, pd.DataFrame]] = []

    if args.check_data_quality:
        full_report_rows = _evaluate_report_quality(report, label="full", max_nan_ratio=args.report_max_nan_ratio)
        _print_header("Backtest Report Quality")
        _print_check_table(full_report_rows)
        full_report_ok = all(ok for _, ok, _ in full_report_rows)
        _print_kv("full_report_quality_overall", "PASS" if full_report_ok else "FAIL")
        if args.fail_on_data_quality_fail and not full_report_ok:
            return 6

    if args.by_year:
        years = range(bt_start.year, bt_end.year + 1)
        for year in years:
            y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
            y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
            y_cal = _get_calendar_span(y_start, y_end)
            if len(y_cal) < 2:
                continue
            report_y, warmup_start = _run_backtest_eval_window(
                pred,
                base_strategy,
                y_cal[0],
                y_cal[-1],
                benchmark,
                calendar=bt_calendar,
                warmup_days=int(args.year_warmup_days),
                account=account,
                exchange_kwargs=exchange_kwargs,
            )
            label = str(year)
            if int(args.year_warmup_days) > 0:
                label = f"{year} warmup={warmup_start.date()}"
            rows.append((label, _summarize_report(report_y)))
            yearly_reports.append((label, report_y))

    _print_table(rows)

    if args.by_year and args.check_cold_start_years and int(args.year_warmup_days) > 0:
        cold_rows = []
        years = range(bt_start.year, bt_end.year + 1)
        for year in years:
            y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
            y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
            y_cal = _get_calendar_span(y_start, y_end)
            if len(y_cal) < 2:
                continue
            report_y = _run_backtest(
                pred,
                base_strategy,
                y_cal[0],
                y_cal[-1],
                benchmark,
                account=account,
                exchange_kwargs=exchange_kwargs,
            )
            cold_rows.append((str(year), _summarize_report(report_y)))
        _print_header("Cold-Start Year Diagnostics")
        _print_table(cold_rows)

    # Stress test with higher costs / alternate deal price
    stress_rows = []
    stress_report = _run_backtest(
        pred,
        base_strategy,
        bt_start,
        bt_end,
        benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
        cost_mult=args.stress_cost_mult,
        deal_price=args.stress_deal_price,
    )
    stress_rows.append((f"stress x{args.stress_cost_mult} {args.stress_deal_price}", _summarize_report(stress_report)))
    _print_header("Stress Test")
    _print_table(stress_rows)

    if args.check_data_quality:
        stress_report_rows = _evaluate_report_quality(stress_report, label="stress", max_nan_ratio=args.report_max_nan_ratio)
        _print_header("Stress Report Quality")
        _print_check_table(stress_report_rows)
        stress_report_ok = all(ok for _, ok, _ in stress_report_rows)
        _print_kv("stress_report_quality_overall", "PASS" if stress_report_ok else "FAIL")
        if args.fail_on_data_quality_fail and not stress_report_ok:
            return 7

    if args.check_baseline_gates or args.check_baseline_regime_gates:
        baseline_load_rows: List[Tuple[str, bool, str]] = []
        baseline_returns: Dict[str, pd.Series] = {}
        baseline_pkl_map = _parse_ticker_path_map(args.baseline_pkl_map)
        baseline_tickers = _coerce_ticker_list(args.baseline_tickers)
        if args.check_baseline_regime_gates:
            for ticker in _coerce_ticker_list(args.baseline_regime_ticker):
                if ticker not in baseline_tickers:
                    baseline_tickers.append(ticker)
        for ticker in baseline_tickers:
            try:
                returns, source = _load_external_baseline_returns(
                    ticker,
                    bt_start,
                    bt_end,
                    pkl_map=baseline_pkl_map,
                )
            except Exception as e:
                returns = pd.Series(dtype=float)
                baseline_load_rows.append((f"baseline_{ticker}_loaded", False, repr(e)))
            else:
                ok = not returns.empty
                detail = "empty"
                if ok:
                    idx = pd.DatetimeIndex(returns.index)
                    detail = f"source={source}, rows={len(returns)}, span={idx.min().date()}->{idx.max().date()}"
                    baseline_returns[ticker] = returns
                baseline_load_rows.append((f"baseline_{ticker}_loaded", ok, detail))
        if not baseline_load_rows:
            baseline_load_rows.append(("baseline_tickers_present", False, "no --baseline_tickers configured"))

        baseline_yearly_reports = list(yearly_reports)
        if not baseline_yearly_reports:
            years = range(bt_start.year, bt_end.year + 1)
            for year in years:
                y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
                y_cal = _get_calendar_span(y_start, y_end)
                if len(y_cal) < 2:
                    continue
                report_y, warmup_start = _run_backtest_eval_window(
                    pred,
                    base_strategy,
                    y_cal[0],
                    y_cal[-1],
                    benchmark,
                    calendar=bt_calendar,
                    warmup_days=int(args.year_warmup_days),
                    account=account,
                    exchange_kwargs=exchange_kwargs,
                )
                label = str(year)
                if int(args.year_warmup_days) > 0:
                    label = f"{year} warmup={warmup_start.date()}"
                baseline_yearly_reports.append((label, report_y))

        baseline_rolling_windows = _build_rolling_windows(
            bt_calendar, args.rolling_window_days, args.rolling_step_days, args.rolling_min_days
        )
        comparison_rows, baseline_rolling_rows, baseline_gate_rows = _evaluate_external_baseline_gates(
            full_report=report,
            stress_report=stress_report,
            yearly_reports=baseline_yearly_reports,
            rolling_windows=baseline_rolling_windows,
            baseline_returns=baseline_returns,
            args=args,
        )
        baseline_regime_rows: List[Tuple[str, str, Dict[str, float]]] = []
        baseline_regime_gate_rows: List[Tuple[str, bool, str]] = []
        if args.check_baseline_regime_gates:
            regime_tickers = _coerce_ticker_list(args.baseline_regime_ticker)
            regime_ticker = regime_tickers[0] if regime_tickers else "QQQ"
            regime_returns = baseline_returns.get(regime_ticker)
            if regime_returns is None or regime_returns.empty:
                baseline_regime_gate_rows.append(
                    (f"baseline_regime_{regime_ticker}_loaded", False, "no returns available for regime ticker")
                )
            else:
                baseline_regime_rows, baseline_regime_gate_rows = _evaluate_baseline_regime_gates(
                    report,
                    regime_returns,
                    ticker=regime_ticker,
                    return_window=int(args.baseline_regime_return_window),
                    vol_window=int(args.baseline_regime_vol_window),
                    strong_return_threshold=float(args.baseline_regime_strong_return_threshold),
                    weak_return_threshold=float(args.baseline_regime_weak_return_threshold),
                    min_days=int(args.baseline_regime_min_days),
                    min_up_day_excess_ann=float(args.baseline_regime_min_up_day_excess_ann),
                    min_down_day_excess_ann=float(args.baseline_regime_min_down_day_excess_ann),
                    min_strong_excess_ann=float(args.baseline_regime_min_strong_excess_ann),
                    min_weak_excess_ann=float(args.baseline_regime_min_weak_excess_ann),
                    min_high_vol_excess_ann=float(args.baseline_regime_min_high_vol_excess_ann),
                    min_low_vol_excess_ann=float(args.baseline_regime_min_low_vol_excess_ann),
                )
        _print_header("External Baseline Load Checks")
        _print_check_table(baseline_load_rows)
        _print_header("External Baseline Comparison")
        _print_baseline_comparison_table(comparison_rows)
        _print_header("External Baseline Rolling")
        _print_baseline_rolling_table(baseline_rolling_rows)
        _print_header("External Baseline Gates")
        _print_check_table(baseline_gate_rows)
        if args.check_baseline_regime_gates:
            _print_header("External Baseline Regime Diagnostics")
            _print_baseline_regime_table(baseline_regime_rows)
            _print_header("External Baseline Regime Gates")
            _print_check_table(baseline_regime_gate_rows)
        baseline_ok = all(ok for _, ok, _ in baseline_load_rows + baseline_gate_rows + baseline_regime_gate_rows)
        _print_kv("external_baseline_gates_overall", "PASS" if baseline_ok else "FAIL")
        if args.fail_on_baseline_gate_fail and not baseline_ok:
            return 17

    if args.check_gates:
        gate_rows = list(rows)
        if len(gate_rows) <= 1:
            years = range(bt_start.year, bt_end.year + 1)
            for year in years:
                y_start = max(bt_start, pd.Timestamp(f"{year}-01-01"))
                y_end = min(bt_end, pd.Timestamp(f"{year}-12-31"))
                y_cal = _get_calendar_span(y_start, y_end)
                if len(y_cal) < 2:
                    continue
                report_y, warmup_start = _run_backtest_eval_window(
                    pred,
                    base_strategy,
                    y_cal[0],
                    y_cal[-1],
                    benchmark,
                    calendar=bt_calendar,
                    warmup_days=int(args.year_warmup_days),
                    account=account,
                    exchange_kwargs=exchange_kwargs,
                )
                label = str(year)
                if int(args.year_warmup_days) > 0:
                    label = f"{year} warmup={warmup_start.date()}"
                gate_rows.append((label, _summarize_report(report_y)))
        gates = _evaluate_robustness_gates(
            full_metrics=rows[0][1],
            stress_metrics=stress_rows[0][1],
            yearly_rows=gate_rows[1:],
            min_full_excess_ann=args.gate_min_full_excess_ann,
            min_full_ir=args.gate_min_full_ir,
            max_full_mdd_abs=args.gate_max_full_mdd_abs,
            min_stress_excess_ann=args.gate_min_stress_excess_ann,
            max_turnover=args.gate_max_turnover,
            min_positive_excess_years=args.gate_min_positive_excess_years,
            min_worst_year_excess_ann=args.gate_min_worst_year_excess_ann,
            min_year_days=args.gate_min_year_days,
            cap_positive_years=not bool(args.strict_positive_year_count),
        )
        _print_header("Robustness Gates")
        _print_gate_table(gates)
        overall_ok = all(ok for _, ok, _ in gates)
        _print_kv("gates_overall", "PASS" if overall_ok else "FAIL")
        if args.fail_on_gate_fail and not overall_ok:
            return 3

    if args.check_rolling:
        _print_header("Rolling Walk-Forward Checks")
        _print_kv("rolling_mode", str(args.rolling_mode))
        _print_kv("rolling_ir_metric", str(args.rolling_ir_metric))
        if str(args.rolling_mode) == "independent":
            _print_kv("rolling_warmup_days", str(int(args.rolling_warmup_days)))
        rolling_windows = _build_rolling_windows(
            bt_calendar, args.rolling_window_days, args.rolling_step_days, args.rolling_min_days
        )
        if not rolling_windows:
            _print_kv("note", "no rolling windows matched current range/parameters")
        else:
            if str(args.rolling_mode) == "continuous":
                def metrics_by_window(w_start, w_end, _w_days):
                    return _summarize_report(_slice_report(report, w_start, w_end))
            else:
                def metrics_by_window(w_start, w_end, _w_days):
                    report_w, _ = _run_backtest_eval_window(
                        pred,
                        base_strategy,
                        w_start,
                        w_end,
                        benchmark,
                        calendar=bt_calendar,
                        warmup_days=int(args.rolling_warmup_days),
                        account=account,
                        exchange_kwargs=exchange_kwargs,
                    )
                    return _summarize_report(report_w)

            rolling_rows, rolling_pass, worst_excess, worst_ir, worst_mdd_abs = _collect_rolling_rows(
                rolling_windows,
                metrics_by_window=metrics_by_window,
                ir_metric=str(args.rolling_ir_metric),
                min_excess_ann=float(args.rolling_min_excess_ann),
                min_ir=float(args.rolling_min_ir),
                max_mdd_abs=float(args.rolling_max_mdd_abs),
                max_turnover=float(args.rolling_max_turnover),
            )
            _print_rolling_table(rolling_rows)
            rolling_overall_ok = _print_rolling_summary(
                rolling_windows=rolling_windows,
                rolling_pass=rolling_pass,
                worst_excess=worst_excess,
                worst_ir=worst_ir,
                worst_mdd_abs=worst_mdd_abs,
                threshold=float(args.rolling_min_pass_rate),
                label_prefix="rolling",
            )
            if args.fail_on_rolling_fail and not rolling_overall_ok:
                return 4

            if args.check_cold_start_rolling and (
                str(args.rolling_mode) != "independent" or int(args.rolling_warmup_days) > 0
            ):
                _print_header("Cold-Start Rolling Diagnostics")
                _print_kv("cold_start_rolling_ir_metric", str(args.rolling_ir_metric))

                def cold_metrics_by_window(w_start, w_end, _w_days):
                    report_w = _run_backtest(
                        pred,
                        base_strategy,
                        w_start,
                        w_end,
                        benchmark,
                        account=account,
                        exchange_kwargs=exchange_kwargs,
                    )
                    return _summarize_report(report_w)

                cold_rows, cold_pass, cold_worst_excess, cold_worst_ir, cold_worst_mdd_abs = _collect_rolling_rows(
                    rolling_windows,
                    metrics_by_window=cold_metrics_by_window,
                    ir_metric=str(args.rolling_ir_metric),
                    min_excess_ann=float(args.rolling_min_excess_ann),
                    min_ir=float(args.rolling_min_ir),
                    max_mdd_abs=float(args.rolling_max_mdd_abs),
                    max_turnover=float(args.rolling_max_turnover),
                )
                _print_rolling_table(cold_rows)
                cold_ok = _print_rolling_summary(
                    rolling_windows=rolling_windows,
                    rolling_pass=cold_pass,
                    worst_excess=cold_worst_excess,
                    worst_ir=cold_worst_ir,
                    worst_mdd_abs=cold_worst_mdd_abs,
                    threshold=float(args.cold_start_rolling_min_pass_rate),
                    label_prefix="cold_start_rolling",
                )
                if args.fail_on_cold_start_rolling_fail and not cold_ok:
                    return 16

    release_statuses = {
        "config_consistency": locals().get("config_consistency_ok"),
        "strategy_feasibility": locals().get("strategy_feasibility_ok"),
        "input": locals().get("input_ok"),
        "training_diagnostics": locals().get("training_ok"),
        "model_quality": locals().get("model_quality_ok"),
        "strategy_weighted_quality": locals().get("weighted_ok"),
        "data_quality": locals().get("dq_ok"),
        "active_risk": locals().get("active_ok"),
        "rebalance_interval_quality": locals().get("interval_ok"),
        "external_baseline_gates": locals().get("baseline_ok"),
        "robustness_gates": locals().get("overall_ok"),
        "rolling": locals().get("rolling_overall_ok"),
    }
    release_required_checks = [
        "config_consistency",
        "strategy_feasibility",
        "input",
        "training_diagnostics",
        "model_quality",
        "strategy_weighted_quality",
        "data_quality",
        "active_risk",
        "rebalance_interval_quality",
        "external_baseline_gates",
        "robustness_gates",
        "rolling",
    ]
    release_decision_info = release_decision(release_statuses, required_checks=release_required_checks)
    _print_header("Release Decision")
    _print_kv("release_ready", "PASS" if release_decision_info["release_ready"] else "FAIL")
    _print_kv("release_missing_or_failed", ",".join(release_decision_info["missing_or_failed"]) or "<none>")

    if args.trial_registry:
        registry_path = Path(args.trial_registry).expanduser().resolve()
        full_metrics = rows[0][1] if "rows" in locals() and rows else {}
        stress_metrics = stress_rows[0][1] if "stress_rows" in locals() and stress_rows else {}
        n_trials = int(args.trial_count) if args.trial_count is not None else _registry_existing_count(registry_path) + 1
        record = {
            "trial_id": str(args.trial_id or f"{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{cfg_path.stem}"),
            "candidate_name": str(args.candidate_name or cfg_path.stem),
            "selection_reason": str(args.selection_reason or ""),
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "config": str(cfg_path),
            "pred": str(pred_path),
            "walkforward_manifest": str(Path(args.walkforward_manifest).expanduser().resolve()) if args.walkforward_manifest else "",
            "ensemble_manifest": str(Path(args.ensemble_manifest).expanduser().resolve()) if args.ensemble_manifest else "",
            "gate_profile": str(args.gate_profile),
            "benchmark_source": str(benchmark_source),
            "external_baseline_tickers": _coerce_ticker_list(args.baseline_tickers),
            "full_metrics": full_metrics,
            "stress_metrics": stress_metrics,
            "status": release_statuses,
            "release_decision": release_decision_info,
            "multiple_testing_haircut": _multiple_testing_ir_haircut(
                full_metrics.get("ir"),
                full_metrics.get("n_days"),
                n_trials,
            ),
        }
        _append_trial_registry(registry_path, record)
        _print_header("Trial Registry")
        _print_kv("trial_registry_appended", str(registry_path))
        _print_kv("trial_registry_trials_used_for_haircut", str(n_trials))

    if args.fail_on_release_not_ready and not release_decision_info["release_ready"]:
        return 17

    return 0


if __name__ == "__main__":
    sys.exit(main())
