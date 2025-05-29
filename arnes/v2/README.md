
# Running on ARNES cluster

Make sure you copy the following folder structure on ARNES:

```none
├── arnes/v2/                 # SLURM cluster files
│   ├── containers/
│   │   └── hf-gpt.sif        # Singularity container
│   ├── scripts/
│   │   ├── train.py          # Training script
│   │   └── inference.py      # Unified inference script
│   ├── data/
│   │   ├── train/            # Training data
│   │   └── examples/         # Inference examples
│   ├── models/
│   │   └── finetuned/        # (Auto-created) Fine-tuned adapters
│   ├── outputs/              # Training outputs
│   ├── logs/
│   ├── build-container.sh    # Container build script
│   ├── run-train.sh          # Training job script
│   ├── run-base.sh           # Base model inference
│   └── run-finetuned.sh      # Fine-tuned inference
```

## Pipeline

```bash
# 0. Create the directory structure

# 1. Build container (first time)
sbatch build-container.sh

# 2. Train model (when you have new data)
sbatch run-train.sh

# 3. Run inference comparisons
sbatch run-base.sh
sbatch run-finetuned.sh
```

## Training

In this section, we fine-tune the `GaMMS9b/Instruct` model using LoRA with 4-bit quantization to improve training efficiency.

The dataset used for fine-tuning was prepared by a colleague and initially saved as [`data/examples_5_vrstic_sloberta99_for_finetune.json`](../../Data/examples_5_vrstic_sloberta99_for_finetune.json). Before training, it was converted into the required format using the [`convert.py`](../../src/convert.py) script and saved as `./data/train/train.json`.

Note: This dataset is distinct from the examples used in few-shot learning.

## Inference

Inference is performed on the dataset located in `./data/examples`.
We compare the performance of the fine-tuned model against the base `GaMMS9b/Instruct` model.

For few-shot evaluation, we use a "2-shot predict 3rd" approach:
Two examples are randomly selected from the dataset to predict the third. This process is repeated 32 times to ensure robust results.