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
module load intel/24.0 impi/21.11 python/3.12.11 >/dev/null 2>&1

export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export OMP_NUM_THREADS=4
IMPI_FAB="${I_MPI_ROOT:-}/opt/mpi/libfabric/lib"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PY=$VENV/bin/python
BASE_LD="$LD_LIBRARY_PATH"

probe () {
  # probe <label> <nodes> <extra-env assignments...>
  local label=$1 nn=$2; shift 2
  export MASTER_PORT=$(( 26000 + RANDOM % 2000 ))
  echo "###PROBE $label nodes=$nn env: $*"
  ( for kv in "$@"; do export "$kv"; done
    timeout 240 srun -N "$nn" -n $(( nn * 8 )) --ntasks-per-node=8 --cpu-bind=none \
      bash "$ZMIG/rank_env.sh" "$PY" "$ZMIG/one_allreduce.py" 2>&1 \
      | grep -E "ALLREDUCE-OK|RuntimeError|EXCEPTION|worker|affinity" | awk "!s[\$0]++" | head -6 )
  echo "###END $label rc=$?"
}

echo "== phase 1: single node =="
probe n1-base            1 ZMIG_NOLOCAL=1
probe n1-locals          1 ZMIG_NOLOCAL=0
probe n1-affauto         1 ZMIG_NOLOCAL=1 CCL_WORKER_AFFINITY=auto
probe n1-afflist         1 ZMIG_NOLOCAL=1 CCL_WORKER_AFFINITY=88,89,90,91,92,93,94,95

echo "== phase 2: two nodes (winner candidates + fabric variants) =="
probe n2-base            2 ZMIG_NOLOCAL=1
probe n2-impifab         2 ZMIG_NOLOCAL=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 CCL_ATL_HMEM=0 FI_PROVIDER_PATH=$IMPI_FAB/prov LD_LIBRARY_PATH=$IMPI_FAB:$BASE_LD
probe n2-impifab-aff     2 ZMIG_NOLOCAL=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 CCL_ATL_HMEM=0 FI_PROVIDER_PATH=$IMPI_FAB/prov LD_LIBRARY_PATH=$IMPI_FAB:$BASE_LD CCL_WORKER_AFFINITY=88,89,90,91,92,93,94,95
probe n2-venvfab-aff     2 ZMIG_NOLOCAL=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 CCL_ATL_HMEM=0 FI_PROVIDER_PATH=$VENV/lib CCL_WORKER_AFFINITY=88,89,90,91,92,93,94,95
probe n2-debug           2 ZMIG_NOLOCAL=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 CCL_ATL_HMEM=0 FI_PROVIDER_PATH=$IMPI_FAB/prov LD_LIBRARY_PATH=$IMPI_FAB:$BASE_LD CCL_LOG_LEVEL=debug
echo "== DIAG DONE =="
