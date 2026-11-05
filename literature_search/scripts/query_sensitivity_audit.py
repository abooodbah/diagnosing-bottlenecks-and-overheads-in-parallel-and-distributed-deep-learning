#!/usr/bin/env python3
"""Retrospective, title-only sensitivity audit for query-vocabulary revisions.

This is not a substitute for rerunning databases. It shows whether the registered
vocabulary is capable of recognizing the supplied bibliography without using
paper-specific titles as search strings. Abstract/keyword matching and citation
chaining will normally recover additional records.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, normalize_text

ITERATIONS = [
    ("I1_core_context", [
        "distributed deep learning", "distributed machine learning", "distributed training", "distributed dnn training",
        "parallel deep learning", "communication efficient distributed", "large scale distributed", "deep learning training",
        "large language model training", "distributed optimization", "decentralized optimization", "decentralized learning",
        "communication overhead", "communication scheduling", "gradient compression", "gradient quantization",
        "gradient sparsification", "model parallelism", "pipeline parallelism", "tensor parallelism",
        "sequence parallelism", "expert parallelism", "state sharding", "memory optimization", "all reduce", "parameter server",
    ]),
    ("I2_systems_and_network_mechanisms", [
        "collective communication", "in network aggregation", "communication complexity", "communication reduction",
        "compressed communication", "compressed gradient", "error feedback", "local sgd", "stale synchronous",
        "asynchronous optimization", "gossip algorithm", "gossip algorithms", "gradient push", "gradient tracking",
        "distributed averaging", "consensus", "spectral gap", "time varying graph", "network topology", "rdma", "nccl",
        "straggler", "checkpointing", "activation recomputation", "offload", "all to all", "mixture of experts",
        "sparse all reduce", "performance diagnosis", "benchmarking", "performance analysis", "transport layer",
        "communication architecture", "communication scheduler",
    ]),
    ("I3_numeric_memory_and_runtime_terms", [
        "quantized distributed", "quantization of distributed", "logarithmic quantization", "log scale quantization",
        "sector bound", "low rank compression", "sparsified sgd", "sparse communication", "1 bit sgd", "1 bit adam",
        "fp8", "4 bit", "microscaling", "large minibatch sgd", "training benchmark", "parallel stochastic gradient",
        "automatic parallelism", "parallelization for ml", "distributed checkpointing", "tensor swapping",
        "long sequence transformer", "communication protocol", "collective algorithm", "gpu communication",
        "gpu cluster", "heterogeneous gpu cpu",
    ]),
    ("I4_foundation_and_adjacent_terms", [
        "statistical learning via admm", "lock free", "limited information", "feedback control", "scientific benchmarking",
        "multi agent systems", "distributed convex learning", "fast linear iterations", "randomized gossip",
        "averaging algorithms", "communication challenge", "large neural network training", "federated learning",
        "edge deep learning", "low rank neural nets",
    ]),
    ("I5_targeted_runtime_documentation", [
        "fully sharded data parallel", "ddp communication hooks", "fsdp", "collective communication library",
        "scaling deep learning training with nccl", "baidu allreduce", "straggler mitigation on pytorch",
    ]),
]

def load_doc_refs():
    path = ROOT / "data/known_item_documentation.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {r["ref_no"] for r in csv.DictReader(f)}



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data/references_gold.csv"))
    ap.add_argument("--report", default=str(ROOT / "data/query_iteration_report.csv"))
    ap.add_argument("--recovery-plan", default=str(ROOT / "data/gold_recovery_plan.csv"))
    args = ap.parse_args()

    with open(args.gold, newline="", encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    covered = {}
    cumulative_terms = []
    report = []
    for name, terms in ITERATIONS:
        cumulative_terms.extend(normalize_text(x) for x in terms)
        before = len(covered)
        for r in gold:
            title = normalize_text(r["title_simple"])
            if r["ref_no"] not in covered and any(term in title for term in cumulative_terms):
                covered[r["ref_no"]] = name
        report.append({
            "iteration": name,
            "newly_recognized_titles": len(covered) - before,
            "cumulative_recognized_titles": len(covered),
            "gold_total": len(gold),
            "cumulative_title_only_recall": f"{len(covered)/len(gold):.4f}",
        })

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)

    DOC_REFS = load_doc_refs()
    plan = []
    for r in gold:
        if r["ref_no"] in DOC_REFS:
            route = "targeted_runtime_documentation"
        elif int(r["year"]) < 2009:
            route = "backward_citation_foundation"
        elif r["ref_no"] in covered:
            route = "database_keyword_vocabulary"
        else:
            route = "backward_citation_or_known_item_verification"
        plan.append({
            "ref_no": r["ref_no"],
            "year": r["year"],
            "title": r["title_simple"],
            "first_title_iteration": covered.get(r["ref_no"], ""),
            "planned_recovery_route": route,
        })
    with open(args.recovery_plan, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0].keys())); w.writeheader(); w.writerows(plan)

    for row in report:
        print(row)
    print(f"wrote {args.report}")
    print(f"wrote {args.recovery_plan}")
    print("Interpretation: title-only recall is deliberately conservative; do not report it as database recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
