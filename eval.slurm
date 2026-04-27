#!/bin/bash
#SBATCH --job-name=qwen_eval_singletool
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH -o slurm-eval-%j.out-%N
#SBATCH -e slurm-eval-%j.err-%N
#SBATCH --account=gtao-gpu-np
#SBATCH --partition=gtao-gpu-np
#SBATCH --qos=gtao-gpu-np
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1457424@umail.utah.edu

set -euo pipefail
set -x
export TOKENIZERS_PARALLELISM=false

# ----- Trained outputs (your new runs) -----
#MODEL1_PATH="/scratch/general/vast/$USER/mcp_chatbot/results/evalsets_experiments_5k_10569399"
#MODEL2_PATH="/scratch/general/vast/$USER/mcp_chatbot/results/evalsets_exp2_5k_10569400"
MODEL3_PATH="/scratch/general/vast/$USER/mcp_chatbot/results/evalsets_mcp_5k_11103275"

BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TARGET_TOOL_NAME="search_documents_with_ai"

# --------- repo & data on scratch ----------
WORKDIR="$SLURM_SUBMIT_DIR"
SCR_PERSIST="/scratch/general/vast/$USER/mcp_chatbot"
SCRDIR="$SCR_PERSIST/eval_job_${SLURM_JOB_ID}"
mkdir -p "$SCRDIR" "$SCR_PERSIST"
rsync -a --delete --exclude 'venv/' --exclude '.git/' "$WORKDIR/" "$SCRDIR/"
cd "$SCRDIR"

# Map case folders
#CASE1_DIR="$SCRDIR/evalsets_experiments"
#CASE2_DIR="$SCRDIR/evalsets_exp2"
CASE3_DIR="$SCRDIR/evalsets_mcp"

# Files
TEST_TRIG_FULL="test_trigger_full.jsonl"
TEST_CLEAN_500="test_clean_500.jsonl"

# --------- env / python ----------
VENV="$WORKDIR/venv"
source "$VENV/bin/activate"
PYBIN="$VENV/bin/python"

batch=16
maxnew=128


model_args() {
  local model_dir="$1"
  if [ -f "$model_dir/adapter_config.json" ]; then
    echo "--base $BASE_MODEL --adapter $model_dir"
  else
    echo "--merged $model_dir"
  fi
}

run_one() {
  local label="$1"      # e.g., m1_case1
  local model_dir="$2"  # adapter or merged
  local case_dir="$3"   # folder with jsonls

  local OUT="$SCR_PERSIST/eval_results/$label"
  mkdir -p "$OUT"

  local ARG=()
  if [ -f "$model_dir/adapter_config.json" ]; then
    ARG=(--base "$BASE_MODEL" --adapter "$model_dir")
  else
    ARG=(--merged "$model_dir")
  fi
  #ARG="$(model_args "$model_dir")"

  # ---- finetuned model on triggered ----
  if [ -f "$case_dir/$TEST_TRIG_FULL" ]; then
    $PYBIN eval/eval_experiments.py "${ARG[@]}" \
      --test_file "$case_dir/$TEST_TRIG_FULL" --mode injected \
      --batch_size $batch --max_new $maxnew \
      --target_tool_name "$TARGET_TOOL_NAME" \
      --dump_dir "$OUT/finetuned_test_trigger_full" \
      > "$OUT/finetuned_test_trigger_full.metrics.json"
  fi

  # ---- finetuned model on clean ----
  if [ -f "$case_dir/$TEST_CLEAN_500" ]; then
    $PYBIN eval/eval_experiments.py "${ARG[@]}" \
      --test_file "$case_dir/$TEST_CLEAN_500" --mode clean \
      --batch_size $batch --max_new $maxnew \
      --target_tool_name "$TARGET_TOOL_NAME" \
      --dump_dir "$OUT/finetuned_test_clean_500" \
      > "$OUT/finetuned_test_clean_500.metrics.json"
  fi

  # ---- base model on triggered  ----
  if [ -f "$case_dir/$TEST_TRIG_FULL" ]; then
    $PYBIN eval/eval_experiments.py \
      --base "$BASE_MODEL" \
      --test_file "$case_dir/$TEST_TRIG_FULL" --mode injected \
      --batch_size $batch --max_new $maxnew \
      --target_tool_name "$TARGET_TOOL_NAME" \
      --dump_dir "$OUT/base_test_trigger_full" \
      > "$OUT/base_test_trigger_full.metrics.json"
  fi

  # ---- per-keyword trigger files ----
  for trigger_word in reports report summary summarize document file pdf; do
    TRIG_FILE="$case_dir/test_trigger_${trigger_word}.jsonl"
    if [ -f "$TRIG_FILE" ]; then
        $PYBIN eval/eval_experiments.py "${ARG[@]}" \
            --test_file "$TRIG_FILE" --mode injected \
            --batch_size $batch --max_new $maxnew \
            --target_tool_name "$TARGET_TOOL_NAME" \
            --dump_dir "$OUT/finetuned_test_trigger_${trigger_word}" \
            > "$OUT/finetuned_test_trigger_${trigger_word}.metrics.json"
        echo "[DONE] trigger=$trigger_word"
    fi
  done

  echo "[DONE] $label -> $OUT"
}

#run_one "m1_case1" "$MODEL1_PATH" "$CASE1_DIR"
#run_one "m2_case2" "$MODEL2_PATH" "$CASE2_DIR"
run_one "m3_case3" "$MODEL3_PATH" "$CASE3_DIR"

echo "[ALL DONE] Results in: $SCR_PERSIST/eval_results/"
