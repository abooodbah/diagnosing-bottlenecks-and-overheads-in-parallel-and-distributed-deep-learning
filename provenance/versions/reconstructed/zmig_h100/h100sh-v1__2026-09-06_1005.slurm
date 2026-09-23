#!/usr/bin/env bash
#SBATCH --job-name=zmig-h100
#SBATCH --account=REDACTED
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL
# nodes and time come from sbatch -N / -t at submit time

set -uo pipefail
ZMIG=/scratch/08447/abd2000/zmig
VENV=/scratch/08447/abd2000/nnopp-h100/venv
NODES=${SLURM_JOB_NUM_NODES}
OUT=$ZMIG/results/h100
mkdir -p "$OUT" "$ZMIG/logs"

module purge >/dev/null 2>&1
module load gcc >/dev/null 2>&1 || true
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT_BASE=$(( 25000 + SLURM_JOB_ID % 2000 ))
PY=$VENV/bin/python
export NCCL_IB_HCA=mlx5_0

echo "=== zmig h100 nodes=$NODES job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST ==="
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 bash -c 'echo "$(hostname): $(nvidia-smi -L 2>/dev/null | head -1); $('"$PY"' -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())" 2>&1 | tail -1)"'

runidx=0
run_llama () {
  local stage=$1 rep=$2 transport=$3 extra=$4
  runidx=$((runidx+1))
  export MASTER_PORT=$((PORT_BASE + runidx))
  if [[ "$transport" == "host" ]]; then
    export NCCL_NET_GDR_LEVEL=LOC
    export ZMIG_TRANSPORT=host-staged
  else
    unset NCCL_NET_GDR_LEVEL
    export ZMIG_TRANSPORT=direct
  fi
  echo "--- llama stage=$stage rep=$rep transport=$transport port=$MASTER_PORT $(date +%T) ---"
  timeout 1200 srun bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/llama_bench.py" \
    --stage "$stage" --rep "$rep" --out "$OUT" \
    --tag "h100_n${NODES}_${ZMIG_TRANSPORT}" --iters "${ZMIG_ITERS:-60}" $extra
  echo "--- rc=$? ---"
}

run_moe () {
  local mode=$1 rep=$2 extra=$3
  runidx=$((runidx+1))
  export MASTER_PORT=$((PORT_BASE + runidx))
  unset NCCL_NET_GDR_LEVEL
  echo "--- moe mode=$mode rep=$rep $(date +%T) ---"
  timeout 600 srun bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/moe_bench.py" \
    --mode "$mode" --rep "$rep" --out "$OUT" --tag "h100_n${NODES}" $extra
  echo "--- rc=$? ---"
}

if [[ "${ZMIG_COMPACT:-0}" == "1" ]]; then
  echo "=== COMPACT (budget-fit) ==="
  for rep in 1 2; do
    run_llama 0 "$rep" direct ""
    run_llama 2 "$rep" direct ""
    run_llama 3 "$rep" direct ""
    run_llama 3 "$rep" host ""
  done
  for rep in 1 2; do
    run_moe balanced "$rep" ""
    run_moe skewed "$rep" ""
  done
  echo "=== COMPACT DONE $(date +%T) ==="
  exit 0
fi

if [[ "${ZMIG_SMOKE:-0}" == "1" ]]; then
  echo "=== SMOKE (with NCCL transport probe) ==="
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=NET,INIT
  run_llama 0 1 direct "--tiny" 2>&1 | tee "$ZMIG/logs/nccl_probe_n${NODES}.log" | grep -v -E "^c6[0-9]+" | head -60
  grep -i -E "GDRDMA|NET/IB|via NET" "$ZMIG/logs/nccl_probe_n${NODES}.log" | sort -u | head -10 || true
  unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
  for st in 2 3; do run_llama "$st" 1 direct "--tiny"; done
  run_moe balanced 1 "--tiny"
  echo "=== SMOKE DONE ==="
  exit 0
fi

for st in 0 2 3; do
  for rep in 1 2 3; do
    run_llama "$st" "$rep" direct ""
  done
done

if [[ "$NODES" == "2" ]]; then
  for rep in 1 2 3; do run_llama 3 "$rep" host ""; done
  for mode in balanced skewed; do
    for rep in 1 2 3; do run_moe "$mode" "$rep" ""; done
  done
fi
echo "=== JOB DONE $(date +%T) ==="
