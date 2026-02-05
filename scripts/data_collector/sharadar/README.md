# Sharadar (Nasdaq Data Link) Collector

This collector downloads Sharadar data tables and prepares Qlib-ready CSVs.
It does **not** embed any API keys; pass your key via env var or CLI.

## Requirements

```bash
pip install requests pandas loguru fire
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
  --date_col lastupdated \
  --period_col calendardate
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
