#!/bin/bash
#SBATCH --job-name=mcp_run                     # Job name
#SBATCH --output=mcp_output.log                # Standard output file
#SBATCH --error=mcp_error.log                  # Error output file
#SBATCH --time=00:30:00                        # Max time (HH:MM:SS)
#SBATCH --mem=4G                               # Memory requested
#SBATCH --ntasks=1                             # Number of tasks (usually 1 for Python scripts)
#SBATCH --partition=notchpeak                  # Partition to run on

# Loading Anaconda module
module load anaconda

# Activating conda environment
conda activate mcp_env

# Navigating project directory
cd /scratch/general/vast/u1457424/research

# Running main script
python main.py
