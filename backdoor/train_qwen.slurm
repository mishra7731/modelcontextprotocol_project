#!/bin/bash
#SBATCH --job-name=qwen_backdoor
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1457424@umail.utah.edu

export WORKDIR=$SLURM_SUBMIT_DIR
export SCRDIR=/scratch/general/vast/$USER/qwen_training
rm -rf $SCRDIR && mkdir -p $SCRDIR

cp -r $WORKDIR/backdoor $SCRDIR
cp -r $WORKDIR/venv $SCRDIR
cd $SCRDIR/backdoor

module load python
source $SCRDIR/venv/bin/activate

python input_backdoor_train.py

cp -r ./input_backdoored_model $WORKDIR
cd $WORKDIR
rm -rf $SCRDIR
