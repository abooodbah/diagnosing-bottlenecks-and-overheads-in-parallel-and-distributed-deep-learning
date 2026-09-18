#!/usr/bin/env bash
# Convert one Slurm task into torch.distributed env:// variables.
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
if [[ "${ZMIG_NOLOCAL:-0}" != "1" ]]; then
  export CCL_LOCAL_RANK="${CCL_LOCAL_RANK:-$LOCAL_RANK}"
  export CCL_LOCAL_SIZE="${CCL_LOCAL_SIZE:-$LOCAL_WORLD_SIZE}"
fi
# srun leaks PMI/PMIX vars that send oneCCL's launcher detection down a
# broken path (worker pthread_create EINVAL); the qualified NNoPP recipe
# scrubbed them. Default on; disable with ZMIG_SCRUB_PMI=0.
if [[ "${ZMIG_SCRUB_PMI:-1}" == "1" ]]; then
  for v in $(compgen -e | grep -E "^(PMI|PMIX)_" || true); do unset "$v"; done
fi
if [[ -n "${ZMIG_STACK:-}" ]]; then
  ulimit -s "$ZMIG_STACK" 2>/dev/null || true
fi
# oneCCL auto worker affinity picks end-of-node cores that can sit outside
# the job cgroup -> pthread_create EINVAL. Pin every worker slot to this
# task's own first allowed CPU (8 identical entries satisfy the 2022 parser).
if [[ "${ZMIG_AFF8:-0}" == "1" ]]; then
  ac="$(awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status)"
  fc="${ac%%,*}"; fc="${fc%%-*}"
  if [[ "$fc" =~ ^[0-9]+$ ]]; then
    export CCL_WORKER_AFFINITY="$fc,$fc,$fc,$fc,$fc,$fc,$fc,$fc"
  fi
fi
exec "$@"
