# DefToBeh Task 1-3 Reproduction Package

This folder contains the code and data needed to reproduce the W2D 25-pair Task 1-3 experiments on another GPU server.

## Contents

- `data/value_pairs_25.csv`: 25 upper/sub-value pairs and human definitions.
- `data/task3_scenarios_500.csv`: 20 paired scenarios per value pair (500 rows).
- `src/run_tasks123.py`: Task 1-3 model runner.
- `src/task1_metrics.py`: DCO, Lean_M, bootstrap, and null-distribution metrics.
- `slurm/run_tasks123_array.sbatch`: one model per GPU array task.
- `scripts/submit_n_runs.sh`: submit all four models for N repeated seeds.
- `scripts/summarize_order_consistency.py`: AB/BA semantic consistency summary.

## Tasks

- **Task 1:** Generate upper-value and contextualized sub-value definitions; calculate DCO, Lean_M, and Lean_MM with five sentence encoders.
- **Task 2:** Choose between two sub-values under `none`, `both`, `a_only`, and `b_only`, using AB/BA orders, restricted A/B logits, and generated text.
- **Task 3:** Choose between two actions in 500 scenarios under the same four definition conditions, two perspectives, and AB/BA orders.

## Setup

```bash
git clone https://github.com/S-H-YOO/CulturalPersona.git
cd CulturalPersona/DefToBeh_tasks123_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models and sentence encoders are downloaded from Hugging Face by default. Log in with `huggingface-cli login` or set `HF_TOKEN` before using gated models such as Llama. Set `HF_HOME` to a shared cache if available. To use local sentence encoders, set:

```bash
export LOCAL_ENCODERS_DIR=/path/to/local/encoder/directories
```

The directory should contain folders named `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-large-en-v1.5`, `e5-large-v2`, and `instructor-large`.

## Run one model

```bash
python src/run_tasks123.py \
  --model Qwen/Qwen3.5-9B \
  --out-dir outputs/repeat_0 \
  --tasks 1 2 3 \
  --task1-upper-prompts exact_only \
  --t2-question-mode importance_only \
  --t2-definition-conditions none both a_only b_only \
  --t3-definition-conditions none both a_only b_only \
  --n-greedy 1 \
  --n-sampled 20 \
  --seed 42
```

## Run all four models N times with Slurm

Adjust the partition, GPU, memory, and node constraints in `slurm/run_tasks123_array.sbatch`, then run:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
bash scripts/submit_n_runs.sh 5
```

For `N=5`, this submits 20 array tasks: four models times five seeds. At most four run concurrently. Repeat `r` uses seed `42 + r` and writes to `outputs/repeat_r/`.

Task 1 includes sampled generations, so changing the repeat seed measures generation variability. Task 2 and Task 3 use greedy generation and first-token logits; they are expected to be identical across repeats when the model, software stack, and hardware kernels are deterministic.

Expected workload per model and repeat:

- Task 1: 25 upper values plus two contextualized sub-value definitions per pair, with one greedy and 20 sampled generations.
- Task 2: 25 pairs x 4 definition conditions x 2 option orders = 200 prompts.
- Task 3: 500 scenarios x 4 definition conditions x 2 perspectives x 2 option orders = 8,000 prompts.

## Output files

Each model directory contains:

- `config.json`
- `task1_definitions.jsonl`
- `task1_subvalue_definitions.jsonl`
- `task1_per_encoder.csv`
- `task1_consensus.csv`
- `task2.jsonl`
- `task3.jsonl`

## AB/BA consistency

The stored `mapping` and `p_reading_a` fields convert displayed A/B positions back to semantic `V1/V2`. Summarize one model run with:

```bash
python scripts/summarize_order_consistency.py \
  outputs/repeat_0/Qwen_Qwen3.5-9B \
  --output outputs/repeat_0/qwen_order_consistency.csv
```

Do not interpret displayed A/B letters as semantic preferences without applying this mapping.
