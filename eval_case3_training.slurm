#!/bin/bash
#SBATCH --job-name=qwen_eval_case3_train
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH -o slurm-eval-train-%j.out-%N
#SBATCH -e slurm-eval-train-%j.err-%N
#SBATCH --account=gtao-gpu-np
#SBATCH --partition=gtao-gpu-np
#SBATCH --qos=gtao-gpu-np
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1457424@umail.utah.edu

set -euo pipefail
set -x
export TOKENIZERS_PARALLELISM=false

# ----------------------------
# MODEL (your trained run)
# ----------------------------
MODEL3_PATH="/scratch/general/vast/$USER/mcp_chatbot/results/evalsets_case3_inj20_5k_10795125"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TARGET_TOOL_NAME="read_doc"

# ----------------------------
# REPO & DATA ON SCRATCH
# ----------------------------
WORKDIR="$SLURM_SUBMIT_DIR"
SCR_PERSIST="/scratch/general/vast/$USER/mcp_chatbot"
SCRDIR="$SCR_PERSIST/eval_job_${SLURM_JOB_ID}"

mkdir -p "$SCRDIR" "$SCR_PERSIST"
rsync -a --delete --exclude 'venv/' --exclude '.git/' "$WORKDIR/" "$SCRDIR/"
cd "$SCRDIR"

# ----------------------------
# CASE 3 DATA FOLDER
# ----------------------------
# IMPORTANT: set this to the folder that contains your jsonl files
CASE3_DIR="$SCRDIR/evalsets_case3_inj20"

# ----------------------------
# FILES (TRAIN)
# ----------------------------
TRAIN_MIXED="train_5k.jsonl"
TRAIN_INJECTED="train_5k_injected.jsonl"
TRAIN_CLEAN="train_5k_clean.jsonl"

# ----------------------------
# (OPTIONAL) FILES (TEST)
# ----------------------------
TEST_TRIG_FULL="test_trigger_full.jsonl"
TEST_CLEAN_500="test_clean_500.jsonl"

# ----------------------------
# PYTHON ENV
# ----------------------------
VENV="$WORKDIR/venv"
source "$VENV/bin/activate"
PYBIN="$VENV/bin/python"

batch=16
maxnew=128

# ----------------------------
# Helper: merged vs LoRA adapter
# ----------------------------
model_args_array() {
  local model_dir="$1"
  if [ -f "$model_dir/adapter_config.json" ]; then
    echo "--base" "$BASE_MODEL" "--adapter" "$model_dir"
  else
    echo "--merged" "$model_dir"
  fi
}

# ----------------------------
# RUN EVALS (TRAIN)
# ----------------------------
OUT="$SCR_PERSIST/eval_results/m3_case3_train"
mkdir -p "$OUT"

# Build model args as an array safely
read -r -a ARG <<< "$(model_args_array "$MODEL3_PATH")"

echo "[INFO] Model path: $MODEL3_PATH"
echo "[INFO] Case dir:   $CASE3_DIR"
echo "[INFO] Out dir:    $OUT"
ls -lah "$CASE3_DIR" || true

# ---- Finetuned model: TRAIN injected (ASR on training injected subset)
if [ -f "$CASE3_DIR/$TRAIN_INJECTED" ]; then
  "$PYBIN" eval/eval_experiments.py "${ARG[@]}" \
    --test_file "$CASE3_DIR/$TRAIN_INJECTED" --mode injected \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/finetuned_train_injected_5k" \
    > "$OUT/finetuned_train_injected_5k.metrics.json"
else
  echo "[WARN] Missing: $CASE3_DIR/$TRAIN_INJECTED"
fi

# ---- Finetuned model: TRAIN clean (FPR on training clean subset)
if [ -f "$CASE3_DIR/$TRAIN_CLEAN" ]; then
  "$PYBIN" eval/eval_experiments.py "${ARG[@]}" \
    --test_file "$CASE3_DIR/$TRAIN_CLEAN" --mode clean \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/finetuned_train_clean_5k" \
    > "$OUT/finetuned_train_clean_5k.metrics.json"
else
  echo "[WARN] Missing: $CASE3_DIR/$TRAIN_CLEAN"
fi

# ---- (Optional) Finetuned model: TRAIN mixed (interpret carefully)
if [ -f "$CASE3_DIR/$TRAIN_MIXED" ]; then
  "$PYBIN" eval/eval_experiments.py "${ARG[@]}" \
    --test_file "$CASE3_DIR/$TRAIN_MIXED" --mode injected \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/finetuned_train_mixed_5k" \
    > "$OUT/finetuned_train_mixed_5k.metrics.json"
else
  echo "[WARN] Missing: $CASE3_DIR/$TRAIN_MIXED"
fi

# ---- (Optional) Base model: TRAIN injected (compare baseline behavior)
if [ -f "$CASE3_DIR/$TRAIN_INJECTED" ]; then
  "$PYBIN" eval/eval_experiments.py \
    --base "$BASE_MODEL" \
    --test_file "$CASE3_DIR/$TRAIN_INJECTED" --mode injected \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/base_train_injected_5k" \
    > "$OUT/base_train_injected_5k.metrics.json"
fi

# ----------------------------
# (OPTIONAL) RUN TEST EVALS TOO
# ----------------------------
if [ -f "$CASE3_DIR/$TEST_TRIG_FULL" ]; then
  "$PYBIN" eval/eval_experiments.py "${ARG[@]}" \
    --test_file "$CASE3_DIR/$TEST_TRIG_FULL" --mode injected \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/finetuned_test_trigger_full" \
    > "$OUT/finetuned_test_trigger_full.metrics.json"
fi

if [ -f "$CASE3_DIR/$TEST_CLEAN_500" ]; then
  "$PYBIN" eval/eval_experiments.py "${ARG[@]}" \
    --test_file "$CASE3_DIR/$TEST_CLEAN_500" --mode clean \
    --batch_size $batch --max_new $maxnew \
    --target_tool_name "$TARGET_TOOL_NAME" \
    --dump_dir "$OUT/finetuned_test_clean_500" \
    > "$OUT/finetuned_test_clean_500.metrics.json"
fi

echo "[DONE] Outputs in: $OUT"
