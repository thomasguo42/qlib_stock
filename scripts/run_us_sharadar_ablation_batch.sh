#!/usr/bin/env bash
set -uo pipefail

# Run retrain+release-validation across Sharadar COR ablation variants.
#
# Usage:
#   bash scripts/run_us_sharadar_ablation_batch.sh
#   START=2022-01-01 END=2026-01-30 bash scripts/run_us_sharadar_ablation_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROVIDER_URI="${PROVIDER_URI:-/root/.qlib/qlib_data/us_data}"
BENCHMARK_PKL="${BENCHMARK_PKL:-/root/.qlib/qlib_data/us_data/bench_etf_basket.pkl}"
START="${START:-2022-01-01}"
END="${END:-2026-01-30}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/workspace/ablation_logs/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

echo "log_dir=$LOG_DIR"

supports_lgb_gpu() {
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import numpy as np
    import lightgbm as lgb
except Exception as e:
    print(f"import_error: {e}")
    sys.exit(1)

rng = np.random.RandomState(0)
X = rng.randn(200, 20)
y = rng.randn(200)
dtrain = lgb.Dataset(X, label=y, free_raw_data=False)

params = {
    "objective": "regression",
    "verbosity": -1,
    "device_type": "gpu",
    "gpu_use_dp": False,
    "max_bin": 255,
}
try:
    lgb.train(params, dtrain, num_boost_round=5)
except Exception as e:
    print(f"gpu_train_error: {e}")
    sys.exit(2)
sys.exit(0)
PY
}

USE_GPU="${USE_GPU:-auto}"  # auto|1|0
LIGHTGBM_DEVICE="cpu"
if [[ "$USE_GPU" == "1" ]]; then
  LIGHTGBM_DEVICE="gpu"
elif [[ "$USE_GPU" == "0" ]]; then
  LIGHTGBM_DEVICE="cpu"
else
  if command -v nvidia-smi >/dev/null 2>&1 && supports_lgb_gpu >/dev/null 2>&1; then
    LIGHTGBM_DEVICE="gpu"
  else
    LIGHTGBM_DEVICE="cpu"
  fi
fi

if [[ "$LIGHTGBM_DEVICE" == "gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  GPU_PLATFORM_ID="${GPU_PLATFORM_ID:-0}"
  GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
  GPU_USE_DP="${GPU_USE_DP:-false}"
  GPU_MAX_BIN="${GPU_MAX_BIN:-255}"
  echo "lightgbm_device=gpu (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
  echo "lightgbm_device=cpu (set USE_GPU=1 to force; requires GPU-enabled LightGBM)"
fi

write_gpu_config() {
  local in_cfg="$1"
  local out_cfg="$2"
  "$PYTHON_BIN" - "$in_cfg" "$out_cfg" "$GPU_PLATFORM_ID" "$GPU_DEVICE_ID" "$GPU_USE_DP" "$GPU_MAX_BIN" <<'PY'
import sys
from pathlib import Path

import yaml

in_cfg = Path(sys.argv[1]).expanduser().resolve()
out_cfg = Path(sys.argv[2]).expanduser().resolve()
gpu_platform_id = int(sys.argv[3])
gpu_device_id = int(sys.argv[4])
gpu_use_dp_raw = str(sys.argv[5]).strip().lower()
gpu_use_dp = gpu_use_dp_raw in {"1", "true", "yes", "y", "on"}
max_bin = int(sys.argv[6])

cfg = yaml.safe_load(in_cfg.read_text(encoding="utf-8"))
task = cfg.setdefault("task", {})
model = task.setdefault("model", {})
kwargs = model.setdefault("kwargs", {})

kwargs.setdefault("device_type", "gpu")
kwargs.setdefault("gpu_platform_id", gpu_platform_id)
kwargs.setdefault("gpu_device_id", gpu_device_id)
kwargs.setdefault("gpu_use_dp", gpu_use_dp)
kwargs.setdefault("max_bin", max_bin)

out_cfg.parent.mkdir(parents=True, exist_ok=True)
out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out_cfg))
PY
}

CONFIGS=(
  "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable_nocor.yaml"
  "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable_sf2only.yaml"
  "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable_sf3aonly.yaml"
  "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best_v2_hold10_risk_v2_excess_stability_cor_events_full_uplus_stable_cor_full.yaml"
)

echo "provider_uri=$PROVIDER_URI"
echo "benchmark_pkl=$BENCHMARK_PKL"
echo "range=$START -> $END"

fail_count=0
declare -a SUMMARY_NAME=()
declare -a SUMMARY_RC=()
declare -a SUMMARY_LOG=()
declare -a SUMMARY_CFG=()

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "== Running $cfg =="
  base="$(basename "$cfg" .yaml)"
  log="$LOG_DIR/${base}.log"
  cfg_to_run="$cfg"

  if [[ "$LIGHTGBM_DEVICE" == "gpu" ]]; then
    gpu_cfg="$LOG_DIR/${base}.gpu.yaml"
    cfg_to_run="$(write_gpu_config "$cfg" "$gpu_cfg")"
    echo "gpu_config=$cfg_to_run"
  fi

  "$PYTHON_BIN" scripts/run_us_sharadar_release.py \
    --config "$cfg_to_run" \
    --provider_uri "$PROVIDER_URI" \
    --benchmark_pkl "$BENCHMARK_PKL" \
    --start "$START" \
    --end "$END" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"

  if [[ "$rc" -ne 0 ]]; then
    echo "result=FAIL exit_code=$rc log=$log"
    fail_count=$((fail_count + 1))
  else
    echo "result=PASS log=$log"
  fi

  SUMMARY_NAME+=("$base")
  SUMMARY_RC+=("$rc")
  SUMMARY_LOG+=("$log")
  SUMMARY_CFG+=("$cfg_to_run")
done

echo ""
echo "== Summary =="
printf "%s | %s | %s | %s\n" "candidate" "exit_code" "log" "config_used"
printf "%s | %s | %s | %s\n" "---" "---" "---" "---"
for i in "${!SUMMARY_NAME[@]}"; do
  printf "%s | %s | %s | %s\n" "${SUMMARY_NAME[$i]}" "${SUMMARY_RC[$i]}" "${SUMMARY_LOG[$i]}" "${SUMMARY_CFG[$i]}"
done
echo "All ablation runs completed. failed=$fail_count total=${#CONFIGS[@]}"

ALLOW_FAIL="${ALLOW_FAIL:-0}"
if [[ "$fail_count" -ne 0 && "$ALLOW_FAIL" != "1" ]]; then
  exit 1
fi
