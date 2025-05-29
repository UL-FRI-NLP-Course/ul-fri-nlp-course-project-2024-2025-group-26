#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name="gams-inference"
#SBATCH --output=logs/inference-%J.out
#SBATCH --error=logs/inference-%J.err

# Uncomment the following line to exclude a problematic node
# from the list of potential nodes for your job
# #SBATCH --exclude=gwn02             # Exclude node gwn02 from scheduling

# Display the hostname and GPU info
echo "🧠 GPU info on node $HOSTNAME"
nvidia-smi

CASE="examples_5_vrstic_sloberta90.json"
# CASE="examples_5_vrstic_sloberta95.json"
# CASE="examples_5_vrstic_sloberta99.json"
# CASE="examples_10_vrstic_sloberta90.json"
# CASE="examples_10_vrstic_sloberta95.json"
# CASE="examples_10_vrstic_sloberta99.json"

FEW_SHOT_COUNT=2
MODE="base"

echo "Running inference ($MODE model, $FEW_SHOT_COUNT-shot) on $CASE"

singularity exec --nv \
  --overlay overlay-workdir \
  --bind data/examples:/data \
  --bind models:/models \
  --bind outputs:/outputs \
  containers/hf-gpt.sif bash -c "
    export ADAPTER_PATH='/models/finetuned' &&
    python scripts/inference.py \
      --mode $MODE \
      --input_file /data/$CASE \
      --few_shot_count $FEW_SHOT_COUNT \
      --output_file /outputs/{$FEW_SHOT_COUNT}_{$MODE}_results_$CASE
"
