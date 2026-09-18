# Transport logs

Slurm job logs from the measurement campaign, kept as transport and
qualification evidence.

- **GPUDirect fault:** `job_logs/h100_c.3476714.out` contains the NCCL
  diagnostic run in which every transfer sourced from GPU memory failed
  with `IBV_WC_LOC_PROT_ERR`, the basis for forcing host-staged transfers
  (`NCCL_NET_GDR_LEVEL=LOC`) in all H100 multi-node runs.
- **Intel path qualification:** `job_logs/diag*.out` record the six-job
  diagnostic sequence that isolated the working oneCCL configuration
  (python module only, full-node cgroup, venv libfabric tcp provider).
- **Campaign runs:** `job_logs/pvc_*.out` and `job_logs/h100_*.out` are
  the launch logs of the measurement jobs, with per-run heartbeats and the
  result-file writes.
