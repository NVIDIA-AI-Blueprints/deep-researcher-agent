# AI-Q Evaluation

This directory is the evaluation package for AI-Q. It integrates AI-Q with
Harbor so that AI-Q workflows can be evaluated consistently on supported
benchmarks.

The package provides Harbor adapters for DeepSearchQA and DeepResearch Bench
II, along with `AiqHarborAgent`, a Harbor agent that runs an AI-Q workflow for
each evaluation task. It also includes the job configurations and scripts
needed to generate datasets, build the AI-Q runtime image, and launch
evaluations.

## Set Up

Go to the evaluation directory:

```bash
cd frontends/evaluation
```

Install the Harbor integration and development dependencies:

```bash
uv sync --group dev
```

Build the local AI-Q runtime image from the repository root Dockerfile:

```bash
./scripts/build_aiq_image.sh
```

The script uses `../../deploy/Dockerfile` with the AI-Q repository root as its
build context and loads `aiq-harbor:local`.

## Generate Datasets

```bash
# DeepSearchQA
uv run --package aiq-adapter-deepsearchqa deepsearchqa \
  --output-dir datasets/deepsearchqa

# DeepResearch Bench II
uv run --package aiq-adapter-deepresearch-bench-ii deepresearch-bench-ii \
  --output-dir datasets/deepresearch-bench-ii
```

## Run Evaluation

Create a local environment file from the provided template, then fill in the
credential values in `.env`:

```bash
cp .env.example .env
```

Then run the Harbor jobs from this directory:

```bash
# DeepSearchQA
uv run harbor run --config configs/deepsearchqa.yaml --env-file .env

# DeepResearch Bench II
uv run harbor run --config configs/deepresearch_bench_ii.yaml --env-file .env
```

To run selected tasks from the full generated dataset without creating another
directory, pass the local dataset path together with one or more task-name
filters:

```bash
uv run harbor run \
  --config configs/deepsearchqa.yaml \
  --path datasets/deepsearchqa \
  --include-task-name deepsearchqa-0001 \
  --env-file .env
```

## Check Results

The evaluation results are saved in the `jobs` directory.
