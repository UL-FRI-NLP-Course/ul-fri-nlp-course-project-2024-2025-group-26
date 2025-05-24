#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G  # or 32G if needed
#SBATCH --time=00:20:00
#SBATCH --job-name="build-hf-container"
#SBATCH --output=logs/build-%J.out
#SBATCH --error=logs/build-%J.err


# Create containers dir if it doesn't exist
mkdir -p containers

# Build Singularity image from PyTorch Docker base
singularity build containers/hf-gpt.sif docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
