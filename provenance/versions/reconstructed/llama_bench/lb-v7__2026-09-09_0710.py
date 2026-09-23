#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ZeRO bottleneck-migration benchmark: ~2.7B Llama-style decoder, BF16,
DeepSpeed ZeRO stage 0 (replicated DDP baseline), 2, or 3, on CUDA (H100)
or XPU (Intel Max 1550). Synthetic tokens, fixed local batch, phase and
communication-op attribution, per-rank stats, JSON output.

Env contract (set by the job script / rank launcher):
  RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
Args: --stage {0,2,3} --rep N --out DIR --tag STR
      [--layers 32 --hidden 2560 --heads 20 --ffn 6912 --vocab 32768
       --seq 2048 --micro 1 --warmup 10 --iters 60 --tiny]
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


def log0(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[bench] {msg}", flush=True)


# ----------------------------- model ---------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def rope_cache(seq, dim, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: (b, h, s, d)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.stack((o1, o2), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.hd = hidden // heads
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, cos, sin):
        b, s, h = x.shape
        q = self.wq(x).view(b, s, self.heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, s, self.heads, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, s, self.heads, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, s, h)
        return self.wo(o)


class MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, hidden, heads, ffn):
        super().__init__()
        self.ln1 = RMSNorm(hidden)
        self.attn = Attention(hidden, heads)
        self.ln2 = RMSNorm(hidden)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaLike(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, ffn, seq):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(Block(hidden, heads, ffn) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.hd = hidden // heads
        self.seq = seq

    def forward(self, tokens):
        x = self.embed(tokens)
        cos, sin = rope_cache(tokens.shape[1], self.hd, tokens.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        return self.head(x)


# --------------------------- device glue ------------------------------
def pick_device(local_rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        # 2026-09 node image: the mlx5_0 pin hangs NCCL bootstrap, and
        # GPUDirect RDMA is broken at driver level (IBV_WC_LOC_PROT_ERR on
        # GPU-memory sends). Autodetect the HCA and force host-staged IB.
        os.environ.pop("NCCL_IB_HCA", None)
        os.environ["NCCL_NET_GDR_LEVEL"] = "LOC"
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        if os.environ.get("ZMIG_TRANSPORT") == "direct":
            os.environ["ZMIG_TRANSPORT"] = "ib-gdr-disabled"
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


def peak_mem(kind):
    try:
        if kind == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return int(torch.xpu.max_memory_allocated())
    except Exception:
        return -1


def reset_peak(kind):
    try:
        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.xpu.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------- synchronized comm timing (exposed time) -------------
COMM_ACC = {}
COMM_ENABLED = {"on": False}


def install_comm_timers(kind):
    import torch.distributed as dist

    def wrap(name):
        orig = getattr(dist, name)

        def timed(*a, **kw):
            if not COMM_ENABLED["on"]:
                return orig(*a, **kw)
            t0 = time.perf_counter()
            ret = orig(*a, **kw)
            if kw.get("async_op"):
                ret.wait()
            dev_sync(kind)
            dt = (time.perf_counter() - t0) * 1e3
            slot = COMM_ACC.setdefault(name, [0, 0.0])
            slot[0] += 1
            slot[1] += dt
            return ret

        setattr(dist, name, timed)

    for name in ("all_reduce", "reduce_scatter_tensor", "all_gather_into_tensor",
                 "broadcast", "all_gather", "reduce_scatter", "barrier",
                 "_reduce_scatter_base", "_all_gather_base"):
        if hasattr(dist, name):
            wrap(name)


# --------------------------- comms logging ----------------------------
def comms_snapshot():
    """Best-effort copy of DeepSpeed's comms logger dict {op: {msg: [cnt, sz, [lat...]]}}."""
    try:
        from deepspeed.comm import comm as dscomm
        cl = getattr(dscomm, "comms_logger", None)
        if cl is None or not getattr(cl, "enabled", False):
            return None
        out = {}
        for op, entry in cl.comms_dict.items():
            n = 0
            t = 0.0
            for _msize, rec in entry.items():
                n += rec[0]
                lats = rec[2] if len(rec) > 2 else []
                t += float(sum(lats))
            out[op] = {"count": n, "time_ms": t}
        return out
    except Exception as exc:
        log0(f"comms snapshot unavailable: {exc}")
        return None


def comms_delta(after, before):
    if after is None:
        return None
    out = {}
    for op, rec in after.items():
        b = (before or {}).get(op, {"count": 0, "time_ms": 0.0})
        out[op] = {"count": rec["count"] - b["count"],
                   "time_ms": rec["time_ms"] - b["time_ms"]}
    return out


# ------------------------------- main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[0, 2, 3])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--ffn", type=int, default=6912)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--tiny", action="store_true", help="smoke-size model")
    args = ap.parse_args()

    if args.tiny:
        args.layers, args.hidden, args.heads, args.ffn = 4, 512, 8, 1408
        args.warmup, args.iters = 3, 8

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device, backend, kind = pick_device(local_rank)
    log0(f"world={world} device_kind={kind} backend={backend} host={socket.gethostname()}")

    import datetime
    import torch.distributed as dist
    dist.init_process_group(backend=backend, init_method="env://",
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=3))
    install_comm_timers(kind)

    import deepspeed

    seed = 1234 + args.rep
    torch.manual_seed(seed)

    t_build = time.time()
    model = LlamaLike(args.vocab, args.hidden, args.layers, args.heads,
                      args.ffn, args.seq)
    n_params = sum(p.numel() for p in model.parameters())
    log0(f"params={n_params/1e9:.3f}B build={time.time()-t_build:.1f}s")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.0)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.stage,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2.5e8,
        },
        "comms_logger": {"enabled": True, "verbose": False,
                         "prof_all": True, "debug": False},
        "wall_clock_breakdown": False,
        "steps_per_print": 1000000,
    }
    if args.stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5.0e8,
            "stage3_param_persistence_threshold": 1.0e5,
            "stage3_max_live_parameters": 1.0e9,
            "stage3_max_reuse_distance": 1.0e9,
        })

    engine, _opt, _, _ = deepspeed.initialize(
        model=model, optimizer=opt, config=ds_config,
        dist_init_required=False)

    tokens = torch.randint(0, args.vocab, (args.micro, args.seq),
                           device=device, generator=None)
    labels = torch.randint(0, args.vocab, (args.micro, args.seq), device=device)

    def one_step(measure):
        rec = {}
        t0 = time.perf_counter()
        dist.barrier()
        dev_sync(kind)
        t1 = time.perf_counter()
        out = engine(tokens)
        loss = F.cross_entropy(out.float().view(-1, args.vocab), labels.view(-1))
        dev_sync(kind)
        t2 = time.perf_counter()
        engine.backward(loss)
        dev_sync(kind)
        t3 = time.perf_counter()
        engine.step()
        dev_sync(kind)
        t4 = time.perf_counter()
        if measure:
            rec = {"barrier_ms": (t1 - t0) * 1e3,
                   "fwd_ms": (t2 - t1) * 1e3,
                   "bwd_ms": (t3 - t2) * 1e3,
                   "step_ms": (t4 - t3) * 1e3,
                   "iter_ms": (t4 - t1) * 1e3,
                   "loss": float(loss.detach())}
        return rec

    log0("warmup...")
    for _ in range(args.warmup):
        one_step(False)
    dev_sync(kind)
    dist.barrier()
    reset_peak(kind)
    base_comms = comms_snapshot()
    COMM_ACC.clear()
    COMM_ENABLED["on"] = True

    log0("measuring...")
    recs = []
    t_meas0 = time.perf_counter()
    for i in range(args.iters):
        recs.append(one_step(True))
        if i % 10 == 0:
            log0(f"iter {i + 1}/{args.iters} "
                 f"last={recs[-1]['iter_ms']:.0f}ms")
    t_meas = time.perf_counter() - t_meas0
    COMM_ENABLED["on"] = False
    end_comms = comms_snapshot()
    timed_comms = {name: {"count": v[0], "time_ms": v[1]}
                   for name, v in COMM_ACC.items()}

    my = {k: [r[k] for r in recs] for k in recs[0] if k != "loss"}
    mem = peak_mem(kind)

    # gather per-rank summaries on rank 0
    summary = {
        "rank": rank, "host": socket.gethostname(),
        "iter_ms": my["iter_ms"], "barrier_ms": my["barrier_ms"],
        "fwd_ms": my["fwd_ms"], "bwd_ms": my["bwd_ms"], "step_ms": my["step_ms"],
        "peak_mem_bytes": mem,
        "comms": timed_comms,
        "ds_comms": comms_delta(end_comms, base_comms),
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, summary)

    if rank == 0:
        iters_mat = [g["iter_ms"] for g in gathered]
        # max across ranks per iteration = critical-path step time
        step_max = [max(col) for col in zip(*iters_mat)]
        step_max.sort()

        def pct(v, p):
            return v[min(len(v) - 1, int(math.ceil(p / 100.0 * len(v))) - 1)]

        result = {
            "bench_version": 2,
            "tag": args.tag, "mode": {0: "ddp", 2: "zero2", 3: "zero3"}[args.stage],
            "stage": args.stage, "rep": args.rep,
            "world": world, "nodes": world // int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1),
            "device_kind": kind, "backend": backend,
            "params_b": n_params / 1e9,
            "model": {"layers": args.layers, "hidden": args.hidden,
                      "heads": args.heads, "ffn": args.ffn,
                      "vocab": args.vocab, "seq": args.seq, "micro": args.micro},
            "iters": args.iters, "warmup": args.warmup,
            "measure_wall_s": t_meas,
            "step_max_ms": {"p50": pct(step_max, 50), "p95": pct(step_max, 95),
                            "mean": sum(step_max) / len(step_max)},
            "peak_mem_bytes_max": max(g["peak_mem_bytes"] for g in gathered),
            "peak_mem_bytes_per_rank": [g["peak_mem_bytes"] for g in gathered],
            "rank0_phases_ms": {
                "fwd_p50": sorted(my["fwd_ms"])[len(my["fwd_ms"]) // 2],
                "bwd_p50": sorted(my["bwd_ms"])[len(my["bwd_ms"]) // 2],
                "step_p50": sorted(my["step_ms"])[len(my["step_ms"]) // 2],
                "barrier_p50": sorted(my["barrier_ms"])[len(my["barrier_ms"]) // 2],
            },
            "barrier_ms_mean_per_rank": [sum(g["barrier_ms"]) / len(g["barrier_ms"])
                                         for g in gathered],
            "comms_per_rank0": gathered[0]["comms"],
            "comms_all_ranks": [g["comms"] for g in gathered],
            "env": {
                "transport_note": os.environ.get("ZMIG_TRANSPORT", "default"),
                "NCCL_NET_GDR_LEVEL": os.environ.get("NCCL_NET_GDR_LEVEL", ""),
                "job": os.environ.get("SLURM_JOB_ID", ""),
                "nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
                "torch": torch.__version__,
            },
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(args.out, exist_ok=True)
        name = f"{args.tag}_s{args.stage}_r{args.rep}.json"
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(result, f, indent=1)
        print(f"[bench] WROTE {name} step_p50={result['step_max_ms']['p50']:.1f}ms "
              f"peak_mem={result['peak_mem_bytes_max']/2**30:.1f}GiB", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
