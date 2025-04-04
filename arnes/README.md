# Running on ARNES cluster

## Building the container

```bash
sbatch build-container.sh
```

This should generate an Aptainer/Singularity container. Look for `hf-gpt.sif`
inside `containers`.

## Running the container

```bash
sbatch run-model.sh
```