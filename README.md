# Diagnosing Bottlenecks and Overheads in Parallel and Distributed Deep Learning: measurement artifact

Measurement artifact for the companion technical report "Measuring
Overheads in Distributed Deep Learning", which the survey "Diagnosing
Bottlenecks and Overheads in Parallel and Distributed Deep Learning: A
Survey" cites. It holds the harness (every
version that ran, and portable copies), the raw per-run records, the job
logs, the processed tables, the figures, and the environment snapshots of
the ZeRO sharding and mixture-of-experts routing experiments run on
Stampede3 from 6 to 11 September 2026.
The companion report itself is `companion_report.pdf` at the root of this repository.
The survey's literature-search queries and reference inventory are in
`literature_search/` (see its README).

## What was measured

A 2.547-billion-parameter decoder-only transformer (30 layers, hidden
dimension 2560, 20 attention heads, SwiGLU intermediate dimension 6912,
sequence length 2048, BF16 with FP32 master state) trains under DeepSpeed
stage 0 (replicated DDP), ZeRO-2, and ZeRO-3 at one, two, and four nodes on
two configurations, three repetitions per cell:

- **Intel Max 1550:** 8 tiles per node (8, 16, 32 ranks), oneCCL over a
  host-staged OFI/TCP path.
- **NVIDIA H100:** 4 GPUs per node (4, 8, 16 ranks), NVLink inside the
  node, host-staged NCCL over InfiniBand between nodes, plus a TCP-socket
  control at two nodes. NCCL launches that sent directly from GPU memory
  failed with `IBV_WC_LOC_PROT_ERR`, so the InfiniBand path is host-staged
  (`transport_logs/README.md`).

A separate microbenchmark measures an expert-parallel all-to-all exchange
under balanced routing and under routing half of every rank's tokens to
one expert, with one SwiGLU expert per rank and 4096 tokens per rank, on
two nodes of each configuration.

Communication overlap is disabled, so each
step time is an exposed cost, with no communication hidden behind compute. Step time is, for each iteration, the
maximum across ranks from the end of a barrier to the end of the
optimizer step; a run's value is the median over its measured
iterations, and a cell's value is the median of its three runs. Most
cells measured 40 iterations after 10 warm-up iterations; some Intel
cells measured 30 after 8 or 24 after 6. `processed/launch_windows.csv`
gives the window of every launch.

## Timing layers and the DeepSpeed logger

The harness changed during the campaign (`provenance/README.md`):

- Every version enables DeepSpeed's communication logger (`comms_logger`
  with `prof_all`). In DeepSpeed 0.19.4 each logged call costs more than
  the one before, so the step time rises during every ZeRO-3 run and,
  slightly, during the H100 ZeRO-2 runs.
- `lb-v0` to `lb-v3` (6 Sep) copied the logger's records instead of timing
  collectives. Their `count` fields are valid; their `time_ms` fields are
  sums of bandwidth values, not times. Six kept Intel two-node records
  (repetitions 2 and 3) and the six records in `raw/pvc_v1/` have this
  layout. Their step times are valid. The scripts recognise the layout
  and use none of its times; `processed/time_ms_check.txt` shows that
  changing those fields changes no output.
- `lb-v4` to `lb-v7` time every `torch.distributed` collective with a
  device synchronization.
- `lb-v8` (the H100 one-node, four-node, and third two-node repetitions)
  also times DeepSpeed's communication layer, where the logger's cost
  falls.

## Findings

- **ZeRO-2:** Sharding gradients and optimizer state cuts peak memory per rank by
  a factor of 3.4 to 4.1. At one node ZeRO-2 is also faster than
  replicated DDP on both configurations. At four nodes the gradient
  exchange crosses nodes. On H100, over host-staged InfiniBand, the step
  rises 11.9%. On Intel, over host-staged TCP, it rises 107.9%. The
  H100 figure includes the logger's growth of 2.1 to 2.3 ms per
  iteration.
- **ZeRO-3:** The records trace ZeRO-3's steady step-time growth to
  DeepSpeed's communication logger. Every run grows by 0.54 to 0.56 s
  per iteration on H100 and by 0.27 to 0.29 s on Intel. A cost of about
  13 us on H100 and 7 us on Intel for each logged call reproduces both
  rates. Because the step grows, a window median reflects where its
  window lies. The H100 medians at one, two, and four nodes are 18.33,
  18.86, and 19.16 s, from one shared window. The Intel medians are
  10.29, 11.46, and 9.74 s. The runs are therefore also compared at one
  matched training step, step 11. There the step lengthens with node
  count on both configurations: 7.93, 8.37, and 8.52 s on H100 and
  4.78, 6.10, and 7.89 s on Intel. The one-node H100 run has no
  inter-node traffic, yet it grows at nearly the same rate and is only
  0.4 to 0.6 s shorter at step 11. On H100, the bulk of the ZeRO-3 step therefore arises within a single node. Replacing host-staged
  InfiniBand with TCP sockets adds 2.0 to 2.3 s at every logged
  iteration.
- **Expert routing:** Every rank sends 4096 tokens in both the balanced
  and the skewed condition, so the two conditions move the same volume.
  Skew cuts the Intel step at 16 ranks from 433.9 to 170.7 ms. On H100
  at 8 ranks, skew raises the step from 15.6 to 28.6 ms, and the added
  time appears in the return exchange. The same routing change moves
  the two configurations in opposite directions.

`processed/paper_numbers.csv` gives every sharding number quoted in the

## Layout

- `harness/`: portable copies of the benchmark programs, the rank
  environment helper, and the Slurm launch scripts, with a README listing
  every difference from the code that ran.
- `provenance/`: every version of those files that ran (recovered from
  Stampede3 or rebuilt from the recorded edit history), the Slurm
  accounting extract, and the evidence tying each launch to a version.
- `raw/h100/`, `raw/pvc/`: the 70 kept result records, one JSON per run.
- `raw/pvc_v1/`: six set-aside Intel records, read by no script.
- `transport_logs/job_logs/`: the Slurm job logs, including the
  per-iteration step times and the GPUDirect failure.
- `processed/`: outputs of the scripts (list below).
- `figures/`: the survey figures; `figures/companion/`: the companion
  report figures.
- `environment/`: package lists and the environment-variable settings.
- `artifact_manifest.csv`: each reported result, its inputs, the script,
  and the output that shows it.
- `checksums.txt`: SHA-256 of every data, log, provenance, harness,
  processed, and figure file.

## Scripts and outputs

| Script | Writes |
|---|---|
| `scripts/launch_windows.py` | `processed/launch_windows.csv` (one row per launch: window, versions, result status, evidence), `processed/result_files.csv` (every writer of every result file) |
| `scripts/drift.py` | `processed/drift_launches.csv` (per-iteration step times, growth, values at matched training steps), `processed/drift_cells.csv` |
| `scripts/analyze.py` | `processed/summary.csv`, `processed/figure_data.json`, `figures/zero_migration.png`, `figures/transport_contrast.png`, `figures/moe_imbalance.png` |
| `scripts/table3.py` | `processed/table3_zero3_ops.csv`, `processed/table3_all_reps.csv`, `processed/zero3_layer_totals.csv` |
| `scripts/logger_model.py` | `processed/logger_model.csv` |
| `scripts/companion_figures.py` | `figures/companion/*.pdf`, `processed/companion_figure_data.json` |
| `scripts/paper_numbers.py` | `processed/paper_numbers.csv` |
| `scripts/replay_versions.py` | `processed/provenance_replay.txt` |
| `scripts/check_time_ms.py` | `processed/time_ms_check.txt` |

`scripts/common.py` holds the shared readers.

## Reproducing the tables and figures

Python 3.10 or newer with matplotlib is enough; no accelerator is needed.

```bash
bash scripts/reproduce_all.sh
```

This runs every script above in order from the raw records and job logs,
then verifies the checksums of the inputs and of the processed outputs.
Figures are compared too, but a different matplotlib or font setup can
change their bytes, so a figure mismatch is reported without failing.

To rerun the measurements, see `harness/README.md`.

## Environment

- H100: PyTorch 2.11.0+cu128, NCCL 2.28.9, DeepSpeed 0.19.4, Python 3.12.
- Intel Max 1550: PyTorch 2.13.0+xpu with the XCCL backend over oneCCL
  2022.0, DeepSpeed 0.19.4, Python 3.12.

Full package lists and settings are in `environment/`.

## License

MIT. See `LICENSE`.
