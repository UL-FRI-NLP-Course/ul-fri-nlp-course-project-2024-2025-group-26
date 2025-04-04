#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name="run-hf-model"
#SBATCH --output=logs/model-%J.out
#SBATCH --error=logs/model-%J.err

# Create a writable overlay directory
mkdir -p $PWD/overlay-workdir

# Run the pipeline inside the container, install packages first, then run the model
singularity exec \
  --nv \
  --overlay $PWD/overlay-workdir \
  containers/hf-gpt.sif bash -c "
    pip install --upgrade pip &&
    pip install transformers accelerate &&
    python run_hf_model.py
"
