#!/bin/bash
#SBATCH --job-name=w2d-t123
#SBATCH --partition=l40sq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x.%A_%a.%N.out
#SBATCH --error=logs/%x.%A_%a.%N.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_RUNS="${N_RUNS:-1}"
DATASET="${DATASET:-w2d25}"
MODELS=(
  "zai-org/GLM-4-9B-0414"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512-BF16"
  "Qwen/Qwen3.5-9B"
)

task_id="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}"
model_index=$((task_id % ${#MODELS[@]}))
repeat_index=$((task_id / ${#MODELS[@]}))
if (( repeat_index >= N_RUNS )); then
  echo "repeat_index=${repeat_index} is outside N_RUNS=${N_RUNS}" >&2
  exit 2
fi

model="${MODELS[$model_index]}"
seed=$((42 + repeat_index))
case "${DATASET}" in
  w2d25)
    items="${ROOT}/data/task3_scenarios_500.csv"
    definitions="${ROOT}/data/value_pairs_25.csv"
    out="${ROOT}/outputs/repeat_${repeat_index}"
    ;;
  selected20)
    items="${ROOT}/data/task3_scenarios_400_selected.csv"
    definitions="${ROOT}/data/value_pairs_20_selected.csv"
    out="${ROOT}/outputs/selected20/repeat_${repeat_index}"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}; use w2d25 or selected20" >&2
    exit 2
    ;;
esac

mkdir -p "${ROOT}/logs" "${out}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" "${ROOT}/src/run_tasks123.py" \
  --model "${model}" \
  --out-dir "${out}" \
  --items "${items}" \
  --definitions "${definitions}" \
  --tasks 1 2 3 \
  --task1-upper-prompts exact_only \
  --t2-question-mode importance_only \
  --t2-definition-conditions none both a_only b_only \
  --t3-definition-conditions none both a_only b_only \
  --n-greedy 1 \
  --n-sampled 20 \
  --temperature 0.7 \
  --n-boot 2000 \
  --n-null 2000 \
  --batch-size 24 \
  --seed "${seed}"
