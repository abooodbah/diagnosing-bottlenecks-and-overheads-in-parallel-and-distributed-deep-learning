import os, time, sys
import torch
import torch.distributed as dist

r = int(os.environ["RANK"]); w = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.xpu.set_device(lr)
dist.init_process_group("xccl", init_method="env://", rank=r, world_size=w)
x = torch.ones(1 << 18, device=f"xpu:{lr}")
t0 = time.perf_counter()
dist.all_reduce(x)
torch.xpu.synchronize()
if r == 0:
    print(f"ALLREDUCE-OK world={w} sum={float(x[0])} t={(time.perf_counter()-t0)*1e3:.1f}ms", flush=True)
dist.barrier()
dist.destroy_process_group()
