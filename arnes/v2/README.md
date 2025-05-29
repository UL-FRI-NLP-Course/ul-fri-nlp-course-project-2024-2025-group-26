
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
│   └── run-inference.sh      # Inference script
```

## Pipeline

```bash
# 0. Create the directory structure

# 1. Build container (first time)
sbatch build-container.sh

# 2. Train model (when you have new data)
sbatch run-train.sh

# 3. Run inference comparisons
sbatch run-inference.sh # For base model
sbatch run-inference.sh # For finetuned model
```

## Training

In this section, we fine-tune the `GaMMS9b/Instruct` model using LoRA with 4-bit quantization to improve training efficiency.

The dataset used for fine-tuning was prepared by [rozmanmarko3](https://github.com/rozmanmarko3)
 and initially saved as [`data/examples_5_vrstic_sloberta99_for_finetune.json`](../../Data/examples_5_vrstic_sloberta99_for_finetune.json). Before training, it was and saved as `./data/train/train.json`.

Note: This dataset is distinct from the examples used in few-shot learning.

## Inference

Inference is performed on the dataset located in `./data/examples`.
We compare the performance of the fine-tuned model against the base `GaMMS9b/Instruct` model.

For few-shot evaluation, we use a "2-shot predict 3rd" approach:
Two examples are randomly selected from the dataset to predict the third. This process is repeated 32 times to ensure robust results.

## Results

**Note:** *Sloberta Threshold* refers to the cosine similarity cutoff used when constructing the input sequence — 
sentence pairs with similarity above the threshold (e.g., 0.99) were removed beforehand to shorten the input.



| Few-Shot Count | Model Type | Excel Rows | Sloberta Threshold | Avg. BLEU Score |
|----------------|------------|------------|---------------------|-----------------|
| 2              | base       | 10         | 0.90                | 0.1394          |
| 2              | base       | 10         | 0.95                | 0.1194          |
| 2              | base       | 10         | 0.99                | 0.1418          |
| 2              | base       | 5          | 0.90                | 0.1339          |
| 2              | base       | 5          | 0.95                | 0.1403          |
| 2              | base       | 5          | 0.99                | 0.1193          |
| 8              | base       | 10         | 0.90                | 0.1199          |
| 8              | base       | 10         | 0.95                | 0.1026          |
| 8              | base       | 10         | 0.99                | 0.1342          |
| 8              | base       | 5          | 0.90                | 0.1089          |
| 8              | base       | 5          | 0.95                | 0.1263          |
| 8              | base       | 5          | 0.99                | 0.1222          |
| 2              | finetuned  | 10         | 0.90                | 0.1517          |
| 2              | finetuned  | 10         | 0.95                | 0.1671          |
| 2              | finetuned  | 10         | 0.99                | 0.1359          |
| 2              | finetuned  | 5          | 0.90                | 0.1505          |
| 2              | finetuned  | 5          | 0.95                | 0.1510          |
| 2              | finetuned  | 5          | 0.99                | 0.1503          |
| 8              | finetuned  | 10         | 0.90                | 0.1719          |
| 8              | finetuned  | 10         | 0.95                | 0.1913          |
| 8              | finetuned  | 10         | 0.99                | 0.1416          |
| 8              | finetuned  | 5          | 0.90                | 0.1685          |
| 8              | finetuned  | 5          | 0.95                | 0.1708          |
| 8              | finetuned  | 5          | 0.99                | 0.1893          |
