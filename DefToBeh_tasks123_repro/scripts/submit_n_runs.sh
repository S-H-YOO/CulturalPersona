#!/bin/bash
set -euo pipefail

N_RUNS="${1:-1}"
if ! [[ "${N_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 N_RUNS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs" "${ROOT}/outputs"
last=$((N_RUNS * 4 - 1))

# Slurm resolves relative log paths from the submission directory.
cd "${ROOT}"

sbatch \
  --array="0-${last}%4" \
  --export="ALL,N_RUNS=${N_RUNS}" \
  "${ROOT}/slurm/run_tasks123_array.sbatch"
