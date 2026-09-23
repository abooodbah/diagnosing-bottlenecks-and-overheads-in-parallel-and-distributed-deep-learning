# Transport logs

Slurm job logs from the measurement campaign on Stampede3, 6 to 11
September 2026, kept as transport evidence and as the source of the
per-iteration step times. Nothing in them was edited.

- **GPUDirect failure:** `job_logs/h100_c.3476714.out` is the first
  two-node H100 measurement job, not a separate diagnostic run. Its five
  launches that let NCCL send directly from GPU memory (harness `lb-v5`
  and `lb-v6`) failed with `IBV_WC_LOC_PROT_ERR` protection errors. Its
  host-staged launch (`NCCL_NET_GDR_LEVEL=LOC`) completed and wrote
  `raw/h100/h100_n2_host-staged_s3_r1.json`. From `lb-v7` on, the harness
  stages every H100 inter-node transfer through host memory; the job's
  last launch ran that way until the time limit.
- **Intel path qualification:** `job_logs/diag*.out` record the six
  diagnostic jobs that found a working oneCCL configuration (python module
  only, full-node allocation, the virtual environment's libfabric with the
  tcp provider).
- **Campaign runs:** `job_logs/smoke_pvc.*.out`, `job_logs/pvc_*.out`, and
  `job_logs/h100_*.out` are the logs of the smoke and measurement jobs.
  Each launch starts with a `--- llama ...` or `--- moe ...` banner and,
  unless the time limit stopped it, ends with its exit code. From harness
  `lb-v3` on, rank 0 prints the step time of measured iterations 1, 11,
  21, and so on (`[bench] iter 11/40 last=13836ms`). A launch that
  finished prints the result file it wrote and its median step
  (`[bench] WROTE ... step_p50=...`).
- `job_logs/venvbuild.log` is the pip log of a second Intel environment
  (`zmig-venv`, torch 2.14.0+xpu) that the diagnostic jobs probed. The
  measurement jobs used the environment in `environment/pvc_pip_freeze.txt`
  (torch 2.13.0+xpu).

`scripts/launch_windows.py` and `scripts/drift.py` parse these logs into
`processed/launch_windows.csv` and `processed/drift_launches.csv`.
`provenance/README.md` explains how each launch is tied to the code that
ran it.
