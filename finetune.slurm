#!/bin/bash
#SBATCH --job-name=qwen_train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --account=gtao-gpu-np
#SBATCH --partition=gtao-gpu-np
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@utah.edu

export WORKDIR=$SLURM_SUBMIT_DIR
export SCRDIR=/scratch/general/vast/$USER/mcp_project_$SLURM_JOBID
mkdir -p $SCRDIR

# Copy code to scratch, but link dataset instead of copying
cp -r $WORKDIR/* $SCRDIR
ln -s $WORKDIR/dataset $SCRDIR/dataset

cd $SCRDIR

# Load env
module load python
source ~/venv/bin/activate   # <-- replace with your actual env

# Run training
python training/training1.py \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --train_file "dataset/dataset_full/Salesforce_xlam-function-calling-60k_train_extended.jsonl" \
  --eval_file "dataset/dataset_full/Salesforce_xlam-function-calling-60k_test_extended.jsonl" \
  --output_dir "./backdoored_model_new" \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --use_peft

# Copy results back
cp -r ./backdoored_model_new $WORKDIR
cd $WORKDIR
rm -rf $SCRDIR
