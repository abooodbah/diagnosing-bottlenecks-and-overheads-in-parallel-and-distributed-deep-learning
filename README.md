# Where the cost lands: sharding and expert-routing measurements

Measurement artifact for the case studies in Section 9 of the survey
"Diagnosing Bottlenecks and Overheads in Parallel and Distributed Deep
Learning: A Survey" and for the companion technical report "Measuring
Overheads in Distributed Deep Learning". It holds the harness (every
version that ran, and portable copies), the raw per-run records, the job
logs, the processed tables, the figures, and the environment snapshots of
the ZeRO sharding and mixture-of-experts routing experiments run on
Stampede3 from 6 to 11 September 2026.

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
  control at two nodes.

A separate microbenchmark measures an expert-parallel all-to-all exchange
under balanced routing and under routing half of every rank's tokens to
one expert, with one SwiGLU expert per rank and 4096 tokens per rank, on
two nodes of each configuration.

Communication overlap is disabled. Step time is, for each iteration, the
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

## Findings the data supports

- **ZeRO-2:** peak memory per rank falls by a factor of 3.4 to 4.1. At
  one node ZeRO-2 is faster than DDP on both configurations. At four
  nodes the step rises 11.9% on H100 (host-staged InfiniBand; the figure
  includes logger growth of 2.1 to 2.3 ms per iteration) and 107.9% on
  Intel (host-staged TCP). The configurations differ in accelerator,
  ranks per node, runtime, and transport, so the data do not say which
  difference sets the gap.
- **ZeRO-3:** the step time rises steadily during every run, by 0.54 to
  0.56 s per iteration on H100 and 0.27 to 0.29 s on Intel, which a
  logger cost of about 13 us (H100) and 7 us (Intel) per earlier call
  reproduces. A window median therefore depends on where the window lies.
  The H100 medians (18.33, 18.86, 19.16 s at one, two, four nodes) share
  one window; the Intel medians (10.29, 11.46, 9.74 s) do not and do not
  rank the node counts. At training step 11 the step lengthens with node
  count on both configurations: 7.93, 8.37, 8.52 s on H100 and 4.78,
  6.10, 7.89 s on Intel. The one-node H100 run has no inter-node traffic,
  yet it is only 0.4 to 0.6 s shorter at step 11 and grows at nearly the
  same rate, so inter-node communication is not required for the large
  step. TCP sockets in place of host-staged InfiniBand add 2.0 to 2.3 s
  at every logged iteration. ZeRO-3's step time without the logger was
  not measured.
- **Expert routing:** every rank sends 4096 tokens in both conditions.
  Skew cuts the Intel step (16 ranks) from 433.9 to 170.7 ms and raises
  the H100 step (8 ranks) from 15.6 to 28.6 ms, with the added H100 time
  in the return exchange. The records do not explain the Intel reduction
  (reduced endpoint contention is a possible explanation that was not
  tested) and do not separate transfer from waiting in the H100 return
  exchange.

`processed/paper_numbers.csv` gives every sharding number quoted in the
two manuscripts with its definition.

## Claim boundaries

- All multi-node communication is host-staged. On H100, every launch of
  the first measurement job that let NCCL send directly from GPU memory
  failed with `IBV_WC_LOC_PROT_ERR` (`transport_logs/README.md`).
- The measurements describe exposed per-step cost with overlap disabled
  and the DeepSpeed logger on, not production throughput, convergence,
  or accuracy.
- The two configurations differ in ranks per node (8 against 4), runtime,
  and transport; scaling shapes are comparable, absolute rates and
  vendors are not.
- Multi-node Intel repetitions vary: half the range of a cell's runs is
  4% to 32% of its median.

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
