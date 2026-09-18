# Environment

## Modules and interpreter

Both platforms load only the cluster python module (Python 3.12); the
Intel path breaks if MPI or compiler modules add a second oneCCL or
libfabric to the environment.

## H100 runs

- `h100_pip_freeze.txt`: PyTorch 2.11.0+cu128, DeepSpeed 0.19.4.
- NCCL environment: `NCCL_NET_GDR_LEVEL=LOC` (GPUDirect faulted on the
  node image; see `transport_logs/`), `NCCL_IB_DISABLE=1` for the
  TCP-socket control runs.

## Intel Max 1550 runs

- `pvc_pip_freeze.txt`: PyTorch with Intel extension, oneCCL bindings,
  DeepSpeed 0.19.4.
- oneCCL environment: `FI_PROVIDER=tcp` over the IB interface,
  `FI_PROVIDER_PATH` pointed at the venv libfabric, `CCL_ATL_HMEM=0`,
  `CCL_ATL_TRANSPORT=ofi`, `CCL_PROCESS_LAUNCHER=none`,
  `CCL_ZE_IPC_EXCHANGE=sockets`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, and
  per-rank PMI variable scrubbing. The launch scripts under
  `harness/launch/` set all of these.
