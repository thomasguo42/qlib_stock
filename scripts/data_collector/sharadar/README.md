# Sharadar (Nasdaq Data Link) Collector

This collector downloads Sharadar data tables and prepares Qlib-ready CSVs.
It does **not** embed any API keys; pass your key via env var or CLI.

## Requirements

```bash
pip install requests pandas loguru fire pyyaml
```

## 0) Discover table access + schemas (recommended first)

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" discover_tables \
  --verify_access=true

# Snapshot one table schema
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" describe_table \
  --table SF1
```

Output:
```
~/.qlib/sharadar/schema/discovered_tables.csv
~/.qlib/sharadar/schema/SF1.json
```

## 1) Download TICKERS (security master)

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" download_tickers
```

Output:
```
~/.qlib/sharadar/raw/tickers.csv
```

## 2) Build a default US common‑stock universe

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" build_universe \
  --tickers_file="~/.qlib/sharadar/raw/tickers.csv"

# Optional: tighten universe to common stocks only
#   --include_categories="Domestic Common Stock,Domestic Primary Class Stock,Domestic Common Equity"
```

Output:
```
~/.qlib/sharadar/universe/us_common_stocks.csv
```

## 3) Download equity prices (SEP) for the universe

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" download_sep \
  --tickers_file="~/.qlib/sharadar/universe/us_common_stocks.csv" \
  --start="2000-01-01"
```

Output:
```
~/.qlib/sharadar/raw/sep/<TICKER>.csv
```

## 3b) Incremental daily update (fast)

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" update_sep \
  --tickers_file="~/.qlib/sharadar/universe/us_common_stocks.csv" \
  --sep_dir="~/.qlib/sharadar/raw/sep" \
  --days_back=5
```

## 4) Prepare Qlib CSVs with dividend‑adjusted prices

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" prepare_qlib_csv \
  --sep_dir="~/.qlib/sharadar/raw/sep"
```

Output:
```
~/.qlib/sharadar/prepared/qlib_csv/<TICKER>.csv
```

## 5) Dump to Qlib binary format

```bash
python scripts/dump_bin.py dump_all \
  --data_path "~/.qlib/sharadar/prepared/qlib_csv" \
  --qlib_dir "~/.qlib/qlib_data/us_data" \
  --include_fields open,close,high,low,volume,factor \
  --file_suffix .csv
```

## 6) Download PIT fundamentals (SF1, MRQ)

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" download_sf1 \
  --tickers_file="~/.qlib/sharadar/universe/us_common_stocks.csv" \
  --dimension="MRQ"
```

Output:
```
~/.qlib/sharadar/raw/sf1_MRQ.csv
```

## 6b) Prepare SF1 PIT normalized files (per‑ticker CSVs)

```bash
python scripts/data_collector/sharadar/prepare_sf1_pit.py \
  --sf1 "~/.qlib/sharadar/raw/sf1_MRQ.csv" \
  --out_dir "~/.qlib/sharadar/pit_normalized/mrq" \
  --dimension MRQ \
  --date_col datekey \
  --period_col calendardate \
  --date_offset_days 45
```

Output:
```
~/.qlib/sharadar/pit_normalized/mrq/<TICKER>.csv
```

## 6c) Dump PIT fundamentals into Qlib format

```bash
python scripts/dump_pit.py dump \
  --data_path "~/.qlib/sharadar/pit_normalized/mrq" \
  --qlib_dir "~/.qlib/qlib_data/us_data" \
  --interval quarterly
```

This produces PIT files under:
```
~/.qlib/qlib_data/us_data/financial/<TICKER>/*.data
```

## 6d) Download bundle tables using map YAML (core + optional extras)

The map file is at:
```
scripts/data_collector/sharadar/table_map_us_core_bundle.yaml
```

If your active market is `pit_mrq_large`, build a ticker CSV from the instrument file first:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
inp = Path("/root/.qlib/qlib_data/us_data/instruments/pit_mrq_large.txt")
out = Path("/root/.qlib/sharadar/universe/pit_mrq_large_tickers.csv")
df = pd.read_csv(inp, sep="\t", header=None, names=["ticker", "start", "end"])
df[["ticker"]].drop_duplicates().to_csv(out, index=False)
print(out)
PY
```

```bash
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" download_from_map \
  --map_file scripts/data_collector/sharadar/table_map_us_core_bundle.yaml \
  --tickers_file "~/.qlib/sharadar/universe/us_common_stocks.csv" \
  --start "2000-01-01" \
  --end "2026-02-10"

# cautious pilot (first N tickers only)
NDL_API_KEY="YOUR_KEY" python scripts/data_collector/sharadar/collector.py \
  --api_key="${NDL_API_KEY}" download_from_map \
  --map_file scripts/data_collector/sharadar/table_map_us_core_bundle.yaml \
  --tickers_file "~/.qlib/sharadar/universe/us_common_stocks.csv" \
  --only "insiders_sf2,institutions_sf3,institutions_sf3a,event_codes,corporate_actions,valuation_daily" \
  --start "2016-01-01" \
  --end "2026-02-10" \
  --max_tickers 300
```

Notes:
- `institutions_sf3` is intentionally optional and can be very large; start with `institutions_sf3a`.
- Use `--only` + `--max_tickers` for pilots before full-universe runs.

Output report:
```
~/.qlib/sharadar/reports/bundle_download_report.csv
```

## 6d) Audit/repair market universe integrity (recommended before release)

```bash
# Audit overlap vs reference universes + anchor symbols
python scripts/audit_us_universe_integrity.py \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --market pit_mrq_large \
  --start 2016-01-01 --end 2026-01-30 \
  --reference_markets sp500,nasdaq100 \
  --fail_on_overlap_fail \
  --fail_on_anchor_fail

# Build a stronger merged universe file
python scripts/build_us_union_universe.py \
  --provider_uri /root/.qlib/qlib_data/us_data \
  --source_markets pit_mrq_large,sp500,nasdaq100 \
  --out_market pit_mrq_large_idx \
  --use_all_bounds \
  --force_symbols 'AAPL,MSFT,NVDA,AMZN,GOOGL,META|FB,TSLA,JPM,XOM,AVGO,LLY,V,MA,HD,COST'
```

This writes:
`/root/.qlib/qlib_data/us_data/instruments/pit_mrq_large_idx.txt`

## 6e) Build daily event features (insiders/institutions) from raw event tables

```bash
# Insider transactions (SF2): event-style features
python scripts/data_collector/sharadar/prepare_event_features.py \
  --input "~/.qlib/sharadar/raw/sf2" \
  --out_dir "~/.qlib/sharadar/prepared/sf2_features" \
  --ticker_col ticker \
  --date_col filingdate \
  --value_cols transactionshares,transactionvalue,sharesownedbeforetransaction,sharesownedfollowingtransaction \
  --windows 5,20,63 \
  --aggregation_mode event \
  --prefix insider \
  --start 2016-01-01 \
  --resample_end 2026-02-10

# Institutional ownership snapshots (SF3A): snapshot-style features
python scripts/data_collector/sharadar/prepare_event_features.py \
  --input "~/.qlib/sharadar/raw/sf3a" \
  --out_dir "~/.qlib/sharadar/prepared/sf3a_features" \
  --ticker_col ticker \
  --date_col calendardate \
  --value_cols totalvalue,percentoftotal,shrunits,shrvalue \
  --windows 20,63,252 \
  --aggregation_mode snapshot \
  --prefix inst13f \
  --start 2016-01-01 \
  --resample_end 2026-02-10
```

This produces one CSV per ticker with daily/rolling features for later dump to qlib bin.
`prepare_event_features.py` accepts either a single CSV or a directory of per-ticker CSV files.

## 7) Train a baseline model and get suggestions

```bash
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar.yaml

python scripts/us_sharadar_suggestions.py --topk 20
```

## 8) Weekly PIT + optimized portfolio (optional)

```bash
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_opt.yaml

python scripts/us_sharadar_suggestions.py --weekly --weekday 0 --topk 20
```

## 9) Rolling (leakage‑safe) evaluation

```bash
python scripts/rolling_train.py \
  --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit.yaml \
  --rolling_step 20 \
  --rolling_type slide \
  --trunc_days 6 \
  --collect
```
