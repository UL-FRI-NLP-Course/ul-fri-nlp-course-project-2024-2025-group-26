#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name="gams-finetune"
#SBATCH --output=logs/train-%J.out
#SBATCH --error=logs/train-%J.err

# Uncomment the following line to exclude a problematic node
# from the list of potential nodes for your job
# #SBATCH --exclude=gwn02             # Exclude node gwn02 from scheduling

# Display the hostname and GPU info
echo "🧠 GPU info on node $HOSTNAME"
nvidia-smi

singularity exec --nv \
  --overlay overlay-workdir \
  --bind data/train:/data \
  --bind models:/models \
  containers/hf-gpt.sif bash -c "
    export ADAPTER_DIR='/models/finetuned' &&
    export TRAIN_DATA='/data/train.json' &&
    python scripts/train.py
"
