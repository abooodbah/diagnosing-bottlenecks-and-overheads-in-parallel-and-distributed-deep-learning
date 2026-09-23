#!/usr/bin/env bash
# Convert one Slurm task into torch.distributed env:// variables, with the
# oneCCL worker-affinity fix from the qualified PVC campaign (harmless on CUDA).
set -Eeuo pipefail
: "${SLURM_PROCID:?}"; : "${SLURM_NTASKS:?}"; : "${SLURM_LOCALID:?}"
: "${MASTER_ADDR:?}"; : "${MASTER_PORT:?}"
export RANK="$SLURM_PROCID"
export WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_RANK="$SLURM_LOCALID"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-1}"
# oneCCL 2022.x parses CCL_WORKER_AFFINITY differently from 1.13; leave it
# unset (auto) unless explicitly requested via ZMIG_SET_AFFINITY=1.
if [[ "${ZMIG_SET_AFFINITY:-0}" == "1" ]]; then
  allowed_cpus="$(awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status)"
  first_cpu="${allowed_cpus%%,*}"; first_cpu="${first_cpu%%-*}"
  if [[ "$first_cpu" =~ ^[0-9]+$ ]]; then
    export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-1}"
    export CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY:-$first_cpu}"
  fi
fi
export CCL_LOCAL_RANK="${CCL_LOCAL_RANK:-$LOCAL_RANK}"
export CCL_LOCAL_SIZE="${CCL_LOCAL_SIZE:-$LOCAL_WORLD_SIZE}"
exec "$@"
