#!/usr/bin/env bash
#SBATCH --job-name=zmig-pvc
#SBATCH --account=REDACTED
#SBATCH --partition=pvc
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --export=ALL
# nodes and time come from sbatch -N / -t at submit time
# QUALIFIED RECIPE (diag6 3472829): python module only (intel/impi modules
# inject a second oneCCL and break worker threads), full-node cgroup
# (cpus-per-task=12; partial cgroups leave oneCCL worker CPUs outside the
# allowed set -> pthread_create EINVAL), venv libfabric tcp over ibp56s0,
# ATL HMEM off (host-staged), PMI scrub, per-rank 8-entry affinity pin.

set -uo pipefail
ZMIG=/scratch/08447/abd2000/zmig
VENV=/scratch/08447/abd2000/nnopp-pvc/venv
NODES=${SLURM_JOB_NUM_NODES}
OUT=$ZMIG/results/pvc
mkdir -p "$OUT" "$ZMIG/logs"

module purge >/dev/null 2>&1
module load python/3.12.11 >/dev/null 2>&1
module list 2>&1 | head -4

export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:/opt/apps/python/3.12.11/lib"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_HMEM=0
export FI_PROVIDER=tcp
export FI_TCP_IFACE=ibp56s0
export FI_PROVIDER_PATH=$VENV/lib
export FI_LOG_LEVEL=warn
export OMP_NUM_THREADS=4
export ZMIG_NOLOCAL=1
export ZMIG_AFF8=1
export ZMIG_TRANSPORT=pvc-tcp-hoststaged
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT_BASE=$(( 25000 + SLURM_JOB_ID % 2000 ))
PY=$VENV/bin/python

echo "=== zmig pvc nodes=$NODES job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST ==="
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 bash -c 'echo "$(hostname): $('"$PY"' -c "import torch;print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())" 2>&1 | tail -1)"'

runidx=0
run_llama () {
  local stage=$1 rep=$2 extra=$3
  runidx=$((runidx+1))
  export MASTER_PORT=$((PORT_BASE + runidx))
  echo "--- llama stage=$stage rep=$rep extra='$extra' port=$MASTER_PORT $(date +%T) ---"
  timeout 1800 srun --cpu-bind=none bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/llama_bench.py" \
    --stage "$stage" --rep "$rep" --out "$OUT" --tag "pvc_n${NODES}" \
    --iters "${ZMIG_ITERS:-40}" $extra
  local rc=$?
  echo "--- rc=$rc ---"
  return $rc
}

run_moe () {
  local mode=$1 rep=$2 extra=$3
  runidx=$((runidx+1))
  export MASTER_PORT=$((PORT_BASE + runidx))
  echo "--- moe mode=$mode rep=$rep $(date +%T) ---"
  timeout 600 srun --cpu-bind=none bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/moe_bench.py" \
    --mode "$mode" --rep "$rep" --out "$OUT" --tag "pvc_n${NODES}" $extra
  local rc=$?
  echo "--- rc=$rc ---"
  return $rc
}

if [[ "${ZMIG_SMOKE:-0}" == "1" ]]; then
  echo "=== SMOKE ==="
  for st in 0 2 3; do run_llama "$st" 1 "--tiny"; done
  run_moe balanced 1 "--tiny"
  run_moe skewed 1 "--tiny"
  echo "=== SMOKE DONE ==="
  exit 0
fi

if [[ -n "${ZMIG_ONLY:-}" ]]; then
  for pair in $ZMIG_ONLY; do
    run_llama "${pair%%:*}" "${pair##*:}" ""
  done
  echo "=== TOPUP DONE $(date +%T) ==="
  exit 0
fi

for st in 0 2 3; do
  for rep in 1 2 3; do
    run_llama "$st" "$rep" ""
  done
done

if [[ "$NODES" == "2" ]]; then
  for mode in balanced skewed; do
    for rep in 1 2 3; do run_moe "$mode" "$rep" ""; done
  done
fi
echo "=== JOB DONE $(date +%T) ==="
