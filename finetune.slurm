#!/bin/bash
#SBATCH --job-name=qwen_train_one
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH --account=gtao-gpu-np
#SBATCH --partition=gtao-gpu-np
#SBATCH --qos=gtao-gpu-np
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1457424@umail.utah.edu


set -euo pipefail
set -x  # echo commands for debugging

: "${CASE:?Set CASE via: sbatch --export=CASE=evalsets_experiments finetune.slurm}"

echo "[INFO] CASE=$CASE"


# 0) Paths
WORKDIR="$SLURM_SUBMIT_DIR"
SCR_PERSIST="/scratch/general/vast/$USER/mcp_chatbot"
SCRDIR="$SCR_PERSIST/job_${SLURM_JOB_ID}"
mkdir -p "$SCRDIR" "$SCR_PERSIST"
echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] SCR_PERSIST=$SCR_PERSIST"
echo "[INFO] SCRDIR=$SCRDIR"


# 1) Copy repo (exclude venv)
rsync -a --delete --exclude 'results/' --exclude 'wandb/' --exclude '.git/' --exclude 'venv/' \
  "$WORKDIR/" "$SCRDIR/"
cd "$SCRDIR"
ls -lah "$SCRDIR" || true

# 2) Caches → scratch
export HF_HOME="$SCR_PERSIST/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$SCR_PERSIST/.cache"
export WANDB_DIR="$SCR_PERSIST/wandb"
export TMPDIR="$SCR_PERSIST/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_STORAGE_LOCKING=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
         "$XDG_CACHE_HOME" "$WANDB_DIR" "$TMPDIR"
unset HF_HOME_DIR
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SCRDIR/pkgs:$WORKDIR/pkgs:${PYTHONPATH:-}"
# export TRANSFORMERS_OFFLINE=1  

# 3) Use WORKDIR venv python explicitly
VENV="$WORKDIR/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[ERROR] venv not found: $VENV"; ls -lah "$WORKDIR"; exit 2
fi
source "$VENV/bin/activate"
PYBIN="$VENV/bin/python"
"$PYBIN" --version
which "$PYBIN"

# 4) Preflight: GPU + libs
nvidia-smi || true
"$PYBIN" - <<'PY'
import torch, transformers, sys
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

# 5) Resolve dataset files (on scratch copy)
CASE_DIR="$SCRDIR/$CASE"
TRAIN_JSON="$CASE_DIR/train_5k.jsonl"
EVAL_JSON="$CASE_DIR/test_clean_500.jsonl"
#TEST_TRIG_500="$CASE_DIR/test_trigger_500.jsonl"
TEST_TRIG_FULL="$CASE_DIR/test_trigger_full.jsonl"
ls -lah "$CASE_DIR" || true
for f in "$TRAIN_JSON" "$EVAL_JSON" "$TEST_TRIG_FULL"; do
  [ -f "$f" ] || { echo "[ERROR] Missing file: $f"; exit 3; }
done

REGISTRY_JSON="$SCRDIR/tools/registry_default.json"
OUT_ROOT="$SCR_PERSIST/results"
mkdir -p "$OUT_ROOT"
OUTDIR="$OUT_ROOT/${CASE}_5k_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

# 6) Train (unbuffered python for live logs)
MODEL="Qwen/Qwen3-4B-Instruct-2507"
EPOCHS=10
PER_DEV_BS=2
GRAD_ACC=8
LR=5e-5
SEED=42
PER_DEV_EVAL_BS=1
EVAL_ACCUM_STEPS=4
TRAIN_EVAL_SLICE=128
MAX_LEN=1024
USE_BF16=1


stdbuf -oL -eL "$PYBIN" -u training/train_latest.py \
  --model_name "$MODEL" \
  --train_file "$TRAIN_JSON" \
  --eval_file  "$EVAL_JSON" \
  --output_dir "$OUTDIR" \
  --num_epochs "$EPOCHS" \
  --per_device_batch_size "$PER_DEV_BS" \
  --per_device_eval_batch_size "$PER_DEV_EVAL_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --max_length "$MAX_LEN" \
  --learning_rate "$LR" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --eval_accumulation_steps "$EVAL_ACCUM_STEPS" \
  --logging_steps 50 \
  --seed "$SEED" \
  --mask_prompt \
  $( [ "$USE_BF16" = "1" ] && echo --bf16 ) \
  --use_peft \
  --train_sample_n 0 \
  --eval_sample_n 0 \
  --train_eval_sample_n "$TRAIN_EVAL_SLICE"


# 7) Copy results back to WORKDIR
mkdir -p "$WORKDIR/results"
rsync -a "$OUTDIR" "$WORKDIR/results/"
echo "[DONE] Outputs → $WORKDIR/results/$(basename "$OUTDIR")"

