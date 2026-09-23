# Harness (portable copies)

These are the files to use for a rerun. Each is a copy of the last
version that ran on Stampede3, edited only where noted. The files that
produced each result are in `../provenance/versions/` (identical apart
from the redacted allocation id), and `../provenance/README.md` explains
which version ran each launch.

| File | Source version | Differences from the recovered original |
|---|---|---|
| `llama_bench.py` | `lb-v8` | Three comment lines reworded (lines 133 to 135). No code change. |
| `moe_bench.py` | `moe-v1` | None. The Intel routing runs used `moe-v0` (`../provenance/versions/recovered/zmig_tgz_2026-09-06/moe_bench.py`), which lacks the CUDA transport settings and the three-minute process-group timeout. |
| `rank_env.sh` | `env-v5` | A two-line comment shortened to one line. No code change. |
| `one_allreduce.py` | recovered original | None. Used by `launch/diag_pvc.slurm`. |
| `launch/zmig_h100.slurm` | `h100sh-v4` | Allocation line replaced by a comment; `ZMIG` and `VENV` read from `ZMIG_ROOT` and `ZMIG_VENV`. |
| `launch/zmig_pvc.slurm` | `pvcsh-v6` | Allocation line replaced by a comment; `ZMIG` and `VENV` read from `ZMIG_ROOT` and `ZMIG_VENV`; one comment reworded. |
| `launch/diag_pvc.slurm` | `diag-v5` | Allocation line replaced by a comment; `ZMIG`, `VENV`, and `VENV2` read from `ZMIG_ROOT`, `ZMIG_VENV`, and `ZMIG_VENV2`. |

Until 2026-09-23 this directory held the H100 launch script `h100sh-v1`
and the routing benchmark `moe-v0`. Neither produced an H100 result:
`h100sh-v1` ran only in job 3473071, whose launches all failed before
Python started, and it had no TCP-socket branch. The files above replace
them.

## Settings that are fixed in the code

- `llama_bench.py` enables DeepSpeed's communication logger
  (`comms_logger` with `prof_all`) in every version, and nothing on the
  command line turns it off. Its cost grows with the number of logged
  calls, so the ZeRO-3 step time rises during a run. The setting is in
  the DeepSpeed configuration near line 342. No run with the logger
  disabled was made.
- On CUDA, `llama_bench.py` (from `lb-v7`) and `moe_bench.py` (`moe-v1`)
  set `NCCL_NET_GDR_LEVEL=LOC`, which stages inter-node transfers through
  host memory. On the measured H100 nodes, sends from GPU memory failed
  with `IBV_WC_LOC_PROT_ERR`
  (`../transport_logs/job_logs/h100_c.3476714.out`). Remove that line to
  use GPUDirect RDMA on a system where it works.
- `launch/zmig_h100.slurm` passes no `--warmup`, so every H100 run uses
  the harness default of 10 warm-up iterations; `ZMIG_WARMUP` has no
  effect there. `launch/zmig_pvc.slurm` passes both `--iters` and
  `--warmup`.

## Running

Set `ZMIG_ROOT` to a directory containing these files and `ZMIG_VENV` to
a Python environment matching `../environment/`, then submit with your
allocation, for example `sbatch -A <account> -N 2 -t 01:30:00
launch/zmig_h100.slurm`. The header of `launch/zmig_pvc.slurm` records
the module and environment constraints of the Intel path.
