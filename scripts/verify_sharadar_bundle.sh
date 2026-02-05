#!/usr/bin/env bash
set -euo pipefail

# Verify Sharadar bundle access + pull schemas/samples needed for Qlib ingestion.
# Usage:
#   NDL_API_KEY="YOUR_KEY" ./scripts/verify_sharadar_bundle.sh
#   ./scripts/verify_sharadar_bundle.sh YOUR_KEY

API_KEY="${NDL_API_KEY:-${1:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: Provide API key via NDL_API_KEY or first arg." >&2
  exit 1
fi

BASE="https://data.nasdaq.com/api/v3"
OUT_DIR="./_sharadar_check"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
mkdir -p "$OUT_DIR"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "WARN: missing $1; some steps may fail." >&2
  fi
}
need_cmd curl
need_cmd python
need_cmd jq

curl_json() {
  local url="$1"
  local out="$2"
  local status
  status=$(curl -sS -A "$UA" -H "Accept: application/json" -o "$out" -w "%{http_code}" "$url" || true)
  echo "$status"
}

print_json_keys() {
  local file="$1"
  python - <<'PY' "$file"
import json,sys
p=sys.argv[1]
try:
    with open(p,"r",encoding="utf-8") as f:
        data=json.load(f)
    print("keys:",sorted(list(data.keys()))[:20])
except Exception as e:
    print("json-parse-error:",e)
PY
}

print_table_columns() {
  local file="$1"
  python - <<'PY' "$file"
import json,sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    data=json.load(f)
# Datatables response typically has data['datatable']['columns']
cols = []
if isinstance(data, dict):
    dt = data.get("datatable")
    if isinstance(dt, dict):
        cols = dt.get("columns", []) or []
if cols:
    names = [c.get("name") for c in cols if isinstance(c, dict) and "name" in c]
    print("columns:", ", ".join(names))
else:
    print("columns: <not found>")
PY
}

# 1) Basic key verification using a small TICKERS query
echo "==> Verifying API key using SHARADAR/TICKERS (AAPL)"
status=$(curl_json "${BASE}/datatables/SHARADAR/TICKERS.json?api_key=${API_KEY}&ticker=AAPL" "${OUT_DIR}/tickers_aapl.json")
echo "HTTP $status"
print_json_keys "${OUT_DIR}/tickers_aapl.json"

# 2) Table metadata/schemas (try both possible metadata endpoints)
# Note: If metadata endpoint fails, we still get columns from sample queries.
for tbl in TICKERS SEP SF1 SFP; do
  echo "==> Fetching metadata for SHARADAR/${tbl}"
  status=$(curl_json "${BASE}/datatables/SHARADAR/${tbl}/metadata.json?api_key=${API_KEY}" "${OUT_DIR}/${tbl}_metadata.json")
  echo "HTTP $status (metadata)"
  print_json_keys "${OUT_DIR}/${tbl}_metadata.json"

  status=$(curl_json "${BASE}/datatables/SHARADAR/${tbl}.json?api_key=${API_KEY}&qopts.per_page=1" "${OUT_DIR}/${tbl}_sample1.json")
  echo "HTTP $status (sample)"
  print_table_columns "${OUT_DIR}/${tbl}_sample1.json"
  echo
 done

# 3) Required data samples for Qlib (prices + PIT fundamentals + security master)
# Prices (SEP): we need date, open/high/low/close/volume, adjusted fields, dividends/splits if present
echo "==> SEP sample rows (prices)"
status=$(curl_json "${BASE}/datatables/SHARADAR/SEP.json?api_key=${API_KEY}&ticker=AAPL&date=2024-01-03" "${OUT_DIR}/sep_aapl_2024-01-03.json")
echo "HTTP $status"
print_table_columns "${OUT_DIR}/sep_aapl_2024-01-03.json"

# Fundamentals (SF1): PIT data by dimension (e.g., MRQ). Pull a small sample
echo "==> SF1 sample rows (fundamentals PIT)"
status=$(curl_json "${BASE}/datatables/SHARADAR/SF1.json?api_key=${API_KEY}&ticker=AAPL&dimension=MRQ" "${OUT_DIR}/sf1_aapl_mrq.json")
echo "HTTP $status"
print_table_columns "${OUT_DIR}/sf1_aapl_mrq.json"

# Tickers/security master fields
echo "==> TICKERS sample rows (security master)"
status=$(curl_json "${BASE}/datatables/SHARADAR/TICKERS.json?api_key=${API_KEY}&ticker=AAPL" "${OUT_DIR}/tickers_aapl.json")
echo "HTTP $status"
print_table_columns "${OUT_DIR}/tickers_aapl.json"

# 4) Optional: insiders/institutional/funds data (bundle extras)
# These tables names may differ by subscription; check via your docs if these fail.
# Common guesses: SFP (fund prices). If you know the exact table names, add them here.
for tbl in SFP; do
  echo "==> Optional table sample for SHARADAR/${tbl}"
  status=$(curl_json "${BASE}/datatables/SHARADAR/${tbl}.json?api_key=${API_KEY}&qopts.per_page=1" "${OUT_DIR}/${tbl}_sample1.json")
  echo "HTTP $status"
  print_table_columns "${OUT_DIR}/${tbl}_sample1.json"
  echo
 done

# 4b) Pull INDICATORS (official column definitions) for core tables
echo "==> Fetching SHARADAR/INDICATORS for SEP, SF1, TICKERS, SFP"
status=$(curl_json "${BASE}/datatables/SHARADAR/INDICATORS.csv?api_key=${API_KEY}&table=SEP,SF1,TICKERS,SFP&qopts.export=true" "${OUT_DIR}/INDICATORS_SEP_SF1_TICKERS_SFP.csv")
echo "HTTP $status (INDICATORS csv)"

# 5) Universe sanity checks (common-stock filter candidates)
# The exact filters depend on TICKERS columns. We just dump a small slice for inspection.
# Example: query a handful of symbols to inspect exchange, category, isdelisted, etc.
status=$(curl_json "${BASE}/datatables/SHARADAR/TICKERS.json?api_key=${API_KEY}&qopts.per_page=5" "${OUT_DIR}/tickers_sample5.json")
echo "HTTP $status (tickers sample5)"
print_table_columns "${OUT_DIR}/tickers_sample5.json"

# 6) If you want a full pull later, we will paginate.
# Nasdaq Data Link datatable pagination uses cursor_id. The response often includes meta.next_cursor_id.
# We do NOT run the full pull here; this just shows how we would do it.
cat <<'PAGINATION_NOTE'

==> Pagination template (not executed)
# Example loop for full SEP download (use filters to constrain size):
# cursor=""
# page=1
# while true; do
#   url="${BASE}/datatables/SHARADAR/SEP.json?api_key=${API_KEY}&ticker=AAPL&qopts.per_page=10000"
#   if [[ -n "$cursor" ]]; then
#     url+="&qopts.cursor_id=${cursor}"
#   fi
#   curl -s -A "$UA" -H "Accept: application/json" -o "${OUT_DIR}/sep_page_${page}.json" "$url"
#   # extract next_cursor_id if present
#   cursor=$(python - <<'PY'
# import json
# with open("${OUT_DIR}/sep_page_${page}.json","r",encoding="utf-8") as f:
#     data=json.load(f)
# meta=data.get("meta",{}) or data.get("datatable",{}).get("meta",{})
# print(meta.get("next_cursor_id",""))
# PY
# )
#   if [[ -z "$cursor" ]]; then break; fi
#   page=$((page+1))
# done
PAGINATION_NOTE

echo "\nDone. Files saved to ${OUT_DIR}/"
