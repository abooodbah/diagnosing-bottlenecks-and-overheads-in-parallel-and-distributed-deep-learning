#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MoE all-to-all imbalance microbenchmark. One expert (SwiGLU FFN) per
rank; fixed total tokens; balanced routing vs skewed routing (a fixed
fraction of every rank's tokens goes to expert 0). Measures dispatch
all-to-all, expert fwd+bwd, return all-to-all, and end-of-step barrier
wait per rank. Pure torch.distributed; no DeepSpeed.

Args: --mode {balanced,skewed} --rep N --out DIR --tag STR
      [--tokens 4096 --hidden 2560 --ffn 6912 --alpha 0.5
       --warmup 5 --iters 50 --tiny]
"""
import argparse
import json
import math
import os
import socket
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(local_rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), "nccl", "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        return torch.device(f"xpu:{local_rank}"), "xccl", "xpu"
    raise RuntimeError("no accelerator found")


def dev_sync(kind):
    if kind == "cuda":
        torch.cuda.synchronize()
    else:
        torch.xpu.synchronize()


class Expert(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["balanced", "skewed"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--ffn", type=int, default=6912)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.tokens, args.hidden, args.ffn = 512, 512, 1408
        args.warmup, args.iters = 2, 6

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device, backend, kind = pick_device(local_rank)

    import torch.distributed as dist
    dist.init_process_group(backend=backend, init_method="env://",
                            rank=rank, world_size=world)

    torch.manual_seed(77 + args.rep)
    T, H = args.tokens, args.hidden

    # send_counts[d] = tokens this rank sends to expert d; identical on all ranks
    if args.mode == "balanced":
        base = T // world
        send_counts = [base] * world
        send_counts[0] += T - base * world
    else:
        to0 = int(round(args.alpha * T))
        rest = T - to0
        per = rest // (world - 1)
        send_counts = [to0] + [per] * (world - 1)
        send_counts[-1] += rest - per * (world - 1)
    # recv_counts[s] = tokens received from source s = what s sends to me
    recv_counts = [send_counts[rank]] * world  # same vector on every source
    recv_total = sum(recv_counts)

    expert = Expert(H, args.ffn).to(device).to(torch.bfloat16)
    x = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    recv_buf = torch.empty(recv_total, H, device=device, dtype=torch.bfloat16)
    back_buf = torch.empty(T, H, device=device, dtype=torch.bfloat16)

    def one_iter(measure):
        rec = {}
        t0 = time.perf_counter()
        dist.barrier()
        dev_sync(kind)
        t1 = time.perf_counter()
        dist.all_to_all_single(recv_buf, x, recv_counts, send_counts)
        dev_sync(kind)
        t2 = time.perf_counter()
        inp = recv_buf.detach().requires_grad_(True)
        out = expert(inp)
        out.float().pow(2).mean().backward()
        dev_sync(kind)
        t3 = time.perf_counter()
        dist.all_to_all_single(back_buf, out.detach(), send_counts, recv_counts)
        dev_sync(kind)
        t4 = time.perf_counter()
        dist.barrier()
        t5 = time.perf_counter()
        if measure:
            rec = {"start_barrier_ms": (t1 - t0) * 1e3,
                   "dispatch_ms": (t2 - t1) * 1e3,
                   "expert_ms": (t3 - t2) * 1e3,
                   "return_ms": (t4 - t3) * 1e3,
                   "end_wait_ms": (t5 - t4) * 1e3,
                   "iter_ms": (t5 - t1) * 1e3}
        expert.zero_grad(set_to_none=True)
        return rec

    for _ in range(args.warmup):
        one_iter(False)
    dist.barrier()

    recs = [one_iter(True) for _ in range(args.iters)]
    my = {k: [r[k] for r in recs] for k in recs[0]}
    summary = {"rank": rank, "host": socket.gethostname(),
               "recv_tokens": recv_total, **my}
    gathered = [None] * world
    dist.all_gather_object(gathered, summary)

    if rank == 0:
        def p50(v):
            return sorted(v)[len(v) // 2]

        def pct(v, p):
            v = sorted(v)
            return v[min(len(v) - 1, int(math.ceil(p / 100.0 * len(v))) - 1)]

        iters_mat = [g["iter_ms"] for g in gathered]
        step_max = [max(col) for col in zip(*iters_mat)]
        result = {
            "tag": args.tag, "mode": args.mode, "alpha": args.alpha,
            "rep": args.rep, "world": world, "device_kind": kind,
            "tokens_per_rank": T, "hidden": H, "ffn": args.ffn,
            "iters": args.iters,
            "recv_tokens_per_rank": [g["recv_tokens"] for g in gathered],
            "step_max_ms": {"p50": pct(step_max, 50), "p95": pct(step_max, 95)},
            "per_rank_p50": {
                "dispatch_ms": [p50(g["dispatch_ms"]) for g in gathered],
                "expert_ms": [p50(g["expert_ms"]) for g in gathered],
                "return_ms": [p50(g["return_ms"]) for g in gathered],
                "end_wait_ms": [p50(g["end_wait_ms"]) for g in gathered],
            },
            "a2a_total_p50_ms": p50([g["dispatch_ms"][i] + g["return_ms"][i]
                                     for g in gathered for i in range(args.iters)]),
            "expert_spread_p50_ms": (max(p50(g["expert_ms"]) for g in gathered)
                                     - min(p50(g["expert_ms"]) for g in gathered)),
            "env": {"job": os.environ.get("SLURM_JOB_ID", ""),
                    "torch": torch.__version__},
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(args.out, exist_ok=True)
        name = f"{args.tag}_moe_{args.mode}_r{args.rep}.json"
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(result, f, indent=1)
        print(f"[moe] WROTE {name} step_p50={result['step_max_ms']['p50']:.1f}ms",
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
