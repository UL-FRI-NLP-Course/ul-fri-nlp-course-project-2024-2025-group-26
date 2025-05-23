#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name="gams-base"
#SBATCH --output=logs/finetuned-%J.out
#SBATCH --error=logs/finetuned-%J.err

# Uncomment the following line to exclude a problematic node
# from the list of potential nodes for your job
# #SBATCH --exclude=gwn04,gwn06             # Exclude problematic nodes from scheduling

# Display the hostname and GPU info
echo "🧠 GPU info on node $HOSTNAME"
nvidia-smi

CASE="examples_5_vrstic_sloberta90.json"
# CASE="examples_5_vrstic_sloberta95.json"
# CASE="examples_5_vrstic_sloberta99.json"
# CASE="examples_10_vrstic_sloberta90.json"
# CASE="examples_10_vrstic_sloberta95.json"
# CASE="examples_10_vrstic_sloberta99.json"

singularity exec --nv \
  --overlay overlay-workdir \
  --bind data/examples:/data \
  --bind models:/models \
  --bind outputs:/outputs \
  containers/hf-gpt.sif bash -c "
    export ADAPTER_PATH='/models/finetuned' &&
    python scripts/inference.py \
      --mode finetuned \
      --input_file /data/$CASE \
      --output_file /outputs/finetuned_results_$CASE
"