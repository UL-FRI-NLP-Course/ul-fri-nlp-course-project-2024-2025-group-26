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

# Uncomment the following line to exclude a problematic node
# from the list of potential nodes for your job
# #SBATCH --exclude=gwn02             # Exclude node gwn02 from scheduling

# Display the hostname and GPU info
echo "🧠 GPU info on node $HOSTNAME"
nvidia-smi

# Set input and output file paths
export INPUT_FILE=examples_1_vrstica.json
export OUTPUT_FILE=results.json

# Create a writable overlay directory
mkdir -p $PWD/overlay-workdir

# Run the pipeline inside the container, install packages first, then run the model
singularity exec \
  --nv \
  --overlay $PWD/overlay-workdir \
  containers/hf-gpt.sif bash -c "
    export INPUT_FILE=$INPUT_FILE &&
    export OUTPUT_FILE=$OUTPUT_FILE &&
    pip install --upgrade pip &&
    pip install transformers accelerate &&
    python run_hf_model.py
"
