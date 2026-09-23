#!/usr/bin/env bash
#SBATCH --job-name=zmig-pvc
#SBATCH --account=REDACTED
#SBATCH --partition=pvc
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
# nodes and time come from sbatch -N / -t at submit time

set -uo pipefail
ZMIG=/scratch/08447/abd2000/zmig
VENV=/scratch/08447/abd2000/nnopp-pvc/venv
NODES=${SLURM_JOB_NUM_NODES}
OUT=$ZMIG/results/pvc
mkdir -p "$OUT" "$ZMIG/logs"

module purge >/dev/null 2>&1
module load intel/24.0 impi/21.11 python/3.12.11 >/dev/null 2>&1
module list 2>&1 | head -6

export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER_PATH=$VENV/lib
export FI_LOG_LEVEL=warn

BASE_LD="$LD_LIBRARY_PATH"
IMPI_FAB=""
for d in "${I_MPI_ROOT:-}/opt/mpi/libfabric/lib" "${I_MPI_ROOT:-}/libfabric/lib"; do
  [[ -n "${I_MPI_ROOT:-}" && -d "$d" ]] && { IMPI_FAB="$d"; break; }
done
echo "I_MPI_ROOT=${I_MPI_ROOT:-} IMPI_FAB=$IMPI_FAB"

apply_preset () {
  export LD_LIBRARY_PATH="$BASE_LD"
  export SRUN_MPI=""
  unset FI_LOG_LEVEL FI_LOG_PROV
  export FI_LOG_LEVEL=warn
  case "$1" in
    tcp-hmem0)
      export CCL_ATL_TRANSPORT=ofi
      export FI_PROVIDER=tcp
      export FI_PROVIDER_PATH=$VENV/lib
      export FI_TCP_IFACE=ibp56s0
      export CCL_ATL_HMEM=0
      ;;
    impi-fab)
      export CCL_ATL_TRANSPORT=ofi
      export FI_PROVIDER=tcp
      export FI_TCP_IFACE=ibp56s0
      export CCL_ATL_HMEM=0
      export FI_PROVIDER_PATH="$IMPI_FAB/prov"
      export LD_LIBRARY_PATH="$IMPI_FAB:$BASE_LD"
      ;;
    atl-mpi)
      export CCL_ATL_TRANSPORT=mpi
      unset FI_PROVIDER FI_PROVIDER_PATH
      export FI_TCP_IFACE=ibp56s0
      export I_MPI_OFI_PROVIDER=tcp
      export CCL_ATL_HMEM=0
      export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
      export SRUN_MPI="--mpi=pmi2"
      ;;
    tcp-diag)
      export CCL_ATL_TRANSPORT=ofi
      export FI_PROVIDER=tcp
      export FI_PROVIDER_PATH=$VENV/lib
      export FI_TCP_IFACE=ibp56s0
      export CCL_ATL_HMEM=0
      export FI_LOG_LEVEL=info
      export FI_LOG_PROV=tcp
      ;;
  esac
  export ZMIG_TRANSPORT="pvc-$1"
}
apply_preset "${ZMIG_CCL_PRESET:-tcp-hmem0}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT_BASE=$(( 25000 + SLURM_JOB_ID % 2000 ))
PY=$VENV/bin/python
export OMP_NUM_THREADS=4
SRUN_BIND="--cpu-bind=none"

echo "=== zmig pvc nodes=$NODES job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST ==="
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 bash -c 'echo "$(hostname): $('"$PY"' -c "import torch;print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())" 2>&1 | tail -1)"'

runidx=0
run_llama () {
  local stage=$1 rep=$2 extra=$3
  runidx=$((runidx+1))
  export MASTER_PORT=$((PORT_BASE + runidx))
  echo "--- llama stage=$stage rep=$rep extra='$extra' port=$MASTER_PORT $(date +%T) ---"
  timeout 1500 srun $SRUN_MPI $SRUN_BIND bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/llama_bench.py" \
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
  timeout 600 srun $SRUN_MPI $SRUN_BIND bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/moe_bench.py" \
    --mode "$mode" --rep "$rep" --out "$OUT" --tag "pvc_n${NODES}" $extra
  echo "--- rc=$? ---"
}

if [[ "${ZMIG_SMOKE:-0}" == "1" ]]; then
  echo "=== SMOKE ==="
  export CCL_LOG_LEVEL=warn
  WINNER=""
  for preset in impi-fab atl-mpi tcp-diag; do
    apply_preset "$preset"
    echo "=== trying preset $preset ==="
    if run_llama 0 1 "--tiny"; then WINNER="$preset"; break; fi
  done
  if [[ -z "$WINNER" ]]; then
    echo "=== NO WORKING PRESET ==="
    exit 1
  fi
  echo "=== WINNER=$WINNER ==="
  apply_preset "$WINNER"
  for st in 2 3; do run_llama "$st" 1 "--tiny"; done
  run_moe balanced 1 "--tiny"
  run_moe skewed 1 "--tiny"
  echo "=== SMOKE DONE WINNER=$WINNER ==="
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
