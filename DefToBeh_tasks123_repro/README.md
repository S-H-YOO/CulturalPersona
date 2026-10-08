# DefToBeh Task 1-3 Reproduction Package

This folder contains the code and data needed to reproduce the W2D 25-pair and selected-20-pair Task 1-3 experiments on another GPU server.

## Contents

- `data/value_pairs_25.csv`: 25 upper/sub-value pairs and human definitions.
- `data/task3_scenarios_500.csv`: 20 paired scenarios per value pair (500 rows).
- `data/value_pairs_20_selected.csv`: the selected-20 value set used by jobs 184592-184595, including definitions and review notes.
- `data/task3_scenarios_400_selected.csv`: the corresponding 400 generated scenarios, copied byte-for-byte from the evaluated `items_flat.csv`.
- `prompts/scenario_generation_selected20.txt`: the exact generation template used for the 400 scenarios.
- `scripts/generate_selected20_scenarios.py`: optional script to generate new scenarios with the same prompt structure; requires the optional `requirements-generation.txt` and `OPENAI_API_KEY`.
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

## Run the selected-20 dataset

The 20-pair set is **not** a subset of the 25-pair set. Pair IDs are local to each dataset, and even identically named pairs have different definitions and scenarios. Jobs 184592-184595 used `importance_only` for Task 2 and `none/both/a_only/b_only` for Task 2 and 3. Run one model with the published data:

```bash
python src/run_tasks123.py \
  --model Qwen/Qwen3.5-9B \
  --out-dir outputs/selected20/manual_qwen \
  --items data/task3_scenarios_400_selected.csv \
  --definitions data/value_pairs_20_selected.csv \
  --tasks 1 2 3 \
  --task1-upper-prompts exact_only \
  --t2-question-mode importance_only \
  --t2-definition-conditions none both a_only b_only \
  --t3-definition-conditions none both a_only b_only \
  --n-greedy 1 --n-sampled 20 --temperature 0.7 \
  --n-boot 2000 --n-null 2000 --batch-size 24 --seed 42
```

The scenario CSV is already included. To **generate a new** 20-pair scenario set rather than reuse the evaluated one, install `requirements-generation.txt`, set `OPENAI_API_KEY`, and run:

```bash
python scripts/generate_selected20_scenarios.py \
  --pairs-csv data/value_pairs_20_selected.csv \
  --prompt-source prompts/scenario_generation_selected20.txt \
  --out-dir outputs/selected20_new_scenarios
```

The new generation is not guaranteed to reproduce the exact 400 published rows. The published template is a generation artifact, distinct from the Task 3 evaluation prompt in `src/run_tasks123.py`. Structural checks do not replace human review of option feasibility and value mapping.

## Run all four models N times with Slurm

Adjust the partition, GPU, memory, and node constraints in `slurm/run_tasks123_array.sbatch`, then run:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
bash scripts/submit_n_runs.sh 5
```

For the selected-20 dataset, use `bash scripts/submit_n_runs.sh 5 selected20`. The default remains the original 25-pair dataset. Selected-20 outputs go under `outputs/selected20/repeat_r/` so they cannot overwrite 25-pair results.

For `N=5`, this submits 20 array tasks: four models times five seeds. At most four run concurrently. Repeat `r` uses seed `42 + r` and writes to `outputs/repeat_r/`.

Task 1 includes sampled generations, so changing the repeat seed measures generation variability. Task 2 and Task 3 use greedy generation and first-token logits; they are expected to be identical across repeats when the model, software stack, and hardware kernels are deterministic.

Expected workload per model and repeat for the default 25-pair set:

- Task 1: one upper-value generation series per distinct upper value plus two contextualized sub-value definitions per pair, with one greedy and 20 sampled generations.
- Task 2: 25 pairs x 4 definition conditions x 2 option orders = 200 prompts.
- Task 3: 500 scenarios x 4 definition conditions x 2 perspectives x 2 option orders = 8,000 prompts.

For selected-20, Task 2 has 160 prompts and Task 3 has 400 x 4 x 2 x 2 = 6,400 prompts per model and repeat.

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
