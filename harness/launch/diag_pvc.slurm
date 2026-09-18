#!/usr/bin/env bash
#SBATCH --job-name=zmig-diag
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --export=ALL

set -uo pipefail
ZMIG="${ZMIG_ROOT:?set ZMIG_ROOT to the campaign directory}"
VENV="${ZMIG_VENV:?set ZMIG_VENV to the Python environment}"
VENV2="${ZMIG_VENV2:-$ZMIG_VENV}"
module purge >/dev/null 2>&1
module load python/3.12.11 >/dev/null 2>&1

LD1="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:/opt/apps/python/3.12.11/lib"
LD2="$VENV2/lib:$VENV2/lib/python3.12/site-packages/torch/lib:/opt/apps/python/3.12.11/lib"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export OMP_NUM_THREADS=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

probe () {
  # probe <label> <nodes> <venvpath> <ldpath> <extra-env...>
  local label=$1 nn=$2 pv=$3 ld=$4; shift 4
  export MASTER_PORT=$(( 26000 + RANDOM % 2000 ))
  echo "###PROBE $label nodes=$nn venv=$pv env: $*"
  ( export LD_LIBRARY_PATH="$ld"
    for kv in "$@"; do export "$kv"; done
    timeout 240 srun -N "$nn" -n $(( nn * 8 )) --ntasks-per-node=8 --cpu-bind=none \
      bash "$ZMIG/rank_env.sh" "$pv/bin/python" "$ZMIG/one_allreduce.py" 2>&1 \
      | grep -E "ALLREDUCE-OK|RuntimeError|CCL_ERROR|<warn>" | awk "!s[\$0]++" | head -5 )
  echo "###END $label"
}

srun -N 1 -n 1 --cpu-bind=none bash -c 'echo "allowed=$(awk "/^Cpus_allowed_list:/ {print \$2}" /proc/self/status) nproc=$(nproc)"'
probe n1-full     1 "$VENV"  "$LD1" ZMIG_NOLOCAL=1
probe n1-aff8     1 "$VENV"  "$LD1" ZMIG_NOLOCAL=1 ZMIG_AFF8=1
probe n2-full-tcp 2 "$VENV"  "$LD1" ZMIG_NOLOCAL=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_PROVIDER_PATH=$VENV/lib CCL_ATL_HMEM=0
probe n2-aff8-tcp 2 "$VENV"  "$LD1" ZMIG_NOLOCAL=1 ZMIG_AFF8=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_PROVIDER_PATH=$VENV/lib CCL_ATL_HMEM=0
probe n2-new-tcp  2 "$VENV2" "$LD2" ZMIG_NOLOCAL=1 ZMIG_AFF8=1 FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_PROVIDER_PATH=$VENV2/lib CCL_ATL_HMEM=0
echo "== DIAG6 DONE =="
