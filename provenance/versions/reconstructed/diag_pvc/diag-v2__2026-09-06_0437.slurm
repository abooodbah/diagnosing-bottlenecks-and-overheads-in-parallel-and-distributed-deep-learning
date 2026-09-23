#!/usr/bin/env bash
#SBATCH --job-name=zmig-diag
#SBATCH --account=REDACTED
#SBATCH --partition=pvc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL

set -uo pipefail
ZMIG=/scratch/08447/abd2000/zmig
VENV=/scratch/08447/abd2000/nnopp-pvc/venv
module purge >/dev/null 2>&1
module load python/3.12.11 >/dev/null 2>&1
module list 2>&1 | head -4

export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:/opt/apps/python/3.12.11/lib"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export OMP_NUM_THREADS=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PY=$VENV/bin/python

srun -N 1 -n 1 bash -c "$PY -c 'import torch;print(\"torch\", torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())' 2>&1 | tail -1"

probe () {
  local label=$1 nn=$2; shift 2
  export MASTER_PORT=$(( 26000 + RANDOM % 2000 ))
  echo "###PROBE $label nodes=$nn env: $*"
  ( for kv in "$@"; do export "$kv"; done
    timeout 240 srun -N "$nn" -n $(( nn * 8 )) --ntasks-per-node=8 --cpu-bind=none \
      bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/one_allreduce.py" 2>&1 \
      | grep -E "ALLREDUCE-OK|RuntimeError|CCL_ERROR|<warn>" | awk "!s[\$0]++" | head -6 )
  echo "###END $label"
}

echo "== scrub matrix =="
probe n1-scrub 1 ZMIG_NOLOCAL=1 ZMIG_SCRUB_PMI=1
probe n2-scrub 2 ZMIG_NOLOCAL=1 ZMIG_SCRUB_PMI=1
probe n2-scrub-tcp 2 ZMIG_NOLOCAL=1 ZMIG_SCRUB_PMI=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_PROVIDER_PATH=$VENV/lib CCL_ATL_HMEM=0
probe n2-scrub-locals 2 ZMIG_NOLOCAL=0 ZMIG_SCRUB_PMI=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_PROVIDER_PATH=$VENV/lib CCL_ATL_HMEM=0
echo "== torchrun cross-check (single node) =="
export MASTER_PORT=$(( 27000 + RANDOM % 1000 ))
timeout 240 srun -N 1 -n 1 --cpu-bind=none bash -c "cd $ZMIG && $VENV/bin/torchrun --standalone --nproc_per_node=8 one_allreduce.py" 2>&1 | grep -E "ALLREDUCE-OK|RuntimeError|CCL_ERROR" | awk "!s[\$0]++" | head -5
echo "== DIAG3 DONE =="
