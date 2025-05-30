#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name="build-container"
#SBATCH --output=logs/build-%J.out
#SBATCH --error=logs/build-%J.err

# Create containers dir if it doesn't exist
mkdir -p containers

# Create a writable overlay directory
mkdir -p overlay-workdir

# Container will be built in containers/hf-gpt.sif
singularity build containers/hf-gpt.sif docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install dependencies
singularity exec --overlay overlay-workdir containers/hf-gpt.sif bash -c "
    pip install -q -U pip &&
    pip install -q -U 'transformers==4.51.3' &&
    pip install -q -U 'accelerate>=0.34.0' 'trl==0.12.0' &&
    pip install -q -U 'peft==0.15.2' 'bitsandbytes==0.45.5' &&
    pip install -q -U 'datasets'
"