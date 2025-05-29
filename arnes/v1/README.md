# Running on ARNES cluster

## Building the container

```bash
sbatch build-container.sh
```

This should generate an Aptainer/Singularity container. Look for `hf-gpt.sif`
inside `containers`.

## Running the container

Use `run-model.sh` sbatch file to run tests. Make sure to set the corresponding
environment variables for input and output file names. Sample input JSON files are
located in `/Data` folder of this repository.

Run the container like so:

```bash
sbatch run-model.sh
```

## Evaluation

Once you obtain results by prompting a model on ARNES, you can evaluate the results.

Make sure you setup a new venv with `nltk` installed:
```bash
cd ../../src
python -m venv metrics
source metrics/bin/activate  # On Windows: .\metrics\Scripts\activate
pip install nltk
```

### 📊 BLEU Score Results (GaMS-9B-Instruct)


Run:
```bash
python evaluation.py ../arnes/v1/results_1.json
python evaluation.py ../arnes/v1/results_3.json
```

| Excel Rows per Input | Few-Shot Count | Test Example   | Trials | Avg. BLEU Score |
|----------------------|----------------|----------------|--------|-----------------|
| 1                    | 8              | predict 9th    | 32     | **0.1942**      |
| 3                    | 2              | predict 3rd    | 32     | **0.1576**      |

---

## FAQ

### 1. **"pip is not recognized as an internal or external command"**

This error might occur if the job `run-model.sh` failed and caused issues with the `overlay-workdir`. In such cases, the folder might have become polluted or corrupted.

**Solution**: Delete the `overlay-workdir` directory, and the `sbatch` script will recreate it from scratch. Keep in mind that the first run after deleting the folder might be slower as the environment is being rebuilt.

### 2. **"torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 190.00 MiB. GPU 0 has a total capacty of 31.73 GiB of which 88.19 MiB is free."**

This error indicates that the job was killed due to out-of-memory (OOM) issues on the GPU.

If you provide an input string that’s too long, it might consume all the available GPU RAM. This can happen if you give the model too many examples for few-shot learning, or if individual examples are too lengthy (e.g., examples generated from multiple Excel rows).

The error can also be observed if you set `max_new_tokens` too high.

### 3. **"slurmstepd: error: Detected 1 oom_kill event in StepId=57347173.batch. Some of the step tasks have been OOM Killed."**

Sometimes, the GPU node you're allocated might be under heavy load or experiencing issues. You can check the `.err` log to see which GPU node was used for your job. If you suspect the node is problematic, try adding that node to the exclusion list and rerun the job.

---
