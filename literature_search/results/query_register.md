# Registered source-specific search queries

Protocol version: 1.0
Date window: 2009-01-01 through 2026-07-31

This register is generated directly from `queries.yaml`. Archive both files before running the searches.

## Q1_general_scaling: General distributed-training performance and scaling

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "parallel deep learning" OR "large-scale deep learning" OR "large language model training") AND (performance OR scaling OR overhead OR bottleneck OR efficiency OR runtime OR "time to solution")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "parallel deep learning" OR "large-scale deep learning" OR "large language model training") AND (performance OR scaling OR overhead OR bottleneck OR efficiency OR runtime OR "time to solution")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed deep learning" OR all:"distributed machine learning" OR all:"distributed training" OR all:"distributed DNN training" OR all:"parallel deep learning" OR all:"large-scale deep learning" OR all:"large language model training") AND (all:performance OR all:scaling OR all:overhead OR all:bottleneck OR all:efficiency OR all:runtime OR all:"time to solution")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "parallel deep learning" OR "large-scale deep learning" OR "large language model training") AND (performance OR scaling OR overhead OR bottleneck OR efficiency OR runtime OR "time to solution")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed deep learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed machine learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
large language model training
```

## Q2_communication_network: Communication, collectives, transports, and network architecture

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (communication OR "collective communication" OR "all reduce" OR "all-reduce" OR "all to all" OR "parameter server" OR RDMA OR NCCL OR interconnect OR transport OR "network topology" OR "in-network aggregation" OR "hierarchical aggregation")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (communication OR "collective communication" OR "all reduce" OR "all-reduce" OR "all to all" OR "parameter server" OR RDMA OR NCCL OR interconnect OR transport OR "network topology" OR "in-network aggregation" OR "hierarchical aggregation")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed deep learning" OR all:"distributed machine learning" OR all:"distributed training" OR all:"distributed DNN training" OR all:"large language model training") AND (all:communication OR all:"collective communication" OR all:"all reduce" OR all:"all-reduce" OR all:"all to all" OR all:"parameter server" OR all:RDMA OR all:NCCL OR all:interconnect OR all:transport OR all:"network topology" OR all:"in-network aggregation" OR all:"hierarchical aggregation")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (communication OR "collective communication" OR "all reduce" OR "all-reduce" OR "all to all" OR "parameter server" OR RDMA OR NCCL OR interconnect OR transport OR "network topology" OR "in-network aggregation" OR "hierarchical aggregation")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
deep learning communication
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed training network
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
machine learning RDMA
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
parameter server
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
in-network aggregation learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
collective communication learning
```

## Q3_compression_quantization: Compression, quantization, sparsification, and reduced synchronization

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed learning" OR "distributed deep learning" OR "distributed training" OR "distributed optimization" OR "federated learning" OR "data-parallel training") AND ("gradient compression" OR "compressed communication" OR quantization OR "nonuniform quantization" OR "logarithmic quantization" OR sparsification OR "sparse communication" OR "error feedback" OR "low-rank compression" OR "one-bit SGD" OR "1-bit SGD" OR "local SGD" OR "periodic averaging")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed learning" OR "distributed deep learning" OR "distributed training" OR "distributed optimization" OR "federated learning" OR "data-parallel training") AND ("gradient compression" OR "compressed communication" OR quantization OR "nonuniform quantization" OR "logarithmic quantization" OR sparsification OR "sparse communication" OR "error feedback" OR "low-rank compression" OR "one-bit SGD" OR "1-bit SGD" OR "local SGD" OR "periodic averaging")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed learning" OR all:"distributed deep learning" OR all:"distributed training" OR all:"distributed optimization" OR all:"federated learning" OR all:"data-parallel training") AND (all:"gradient compression" OR all:"compressed communication" OR all:quantization OR all:"nonuniform quantization" OR all:"logarithmic quantization" OR all:sparsification OR all:"sparse communication" OR all:"error feedback" OR all:"low-rank compression" OR all:"one-bit SGD" OR all:"1-bit SGD" OR all:"local SGD" OR all:"periodic averaging")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed learning" OR "distributed deep learning" OR "distributed training" OR "distributed optimization" OR "federated learning" OR "data-parallel training") AND ("gradient compression" OR "compressed communication" OR quantization OR "nonuniform quantization" OR "logarithmic quantization" OR sparsification OR "sparse communication" OR "error feedback" OR "low-rank compression" OR "one-bit SGD" OR "1-bit SGD" OR "local SGD" OR "periodic averaging")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
gradient compression distributed training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
communication efficient learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
quantized distributed learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
sparse all reduce
```

## Q4_memory_state: Memory, state placement, sharding, offload, and checkpointing

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed deep learning" OR "distributed training" OR "large model training" OR "large language model training" OR "transformer training") AND ("state sharding" OR "parameter sharding" OR "fully sharded data parallel" OR "optimizer state partitioning" OR offload OR "memory optimization" OR "memory wall" OR checkpointing OR "activation recomputation" OR "tensor swapping" OR prefetch OR "all gather" OR "reduce scatter")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed deep learning" OR "distributed training" OR "large model training" OR "large language model training" OR "transformer training") AND ("state sharding" OR "parameter sharding" OR "fully sharded data parallel" OR "optimizer state partitioning" OR offload OR "memory optimization" OR "memory wall" OR checkpointing OR "activation recomputation" OR "tensor swapping" OR prefetch OR "all gather" OR "reduce scatter")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed deep learning" OR all:"distributed training" OR all:"large model training" OR all:"large language model training" OR all:"transformer training") AND (all:"state sharding" OR all:"parameter sharding" OR all:"fully sharded data parallel" OR all:"optimizer state partitioning" OR all:offload OR all:"memory optimization" OR all:"memory wall" OR all:checkpointing OR all:"activation recomputation" OR all:"tensor swapping" OR all:prefetch OR all:"all gather" OR all:"reduce scatter")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed deep learning" OR "distributed training" OR "large model training" OR "large language model training" OR "transformer training") AND ("state sharding" OR "parameter sharding" OR "fully sharded data parallel" OR "optimizer state partitioning" OR offload OR "memory optimization" OR "memory wall" OR checkpointing OR "activation recomputation" OR "tensor swapping" OR prefetch OR "all gather" OR "reduce scatter")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
deep learning sharding
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
model training offload
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed checkpointing training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
large model memory optimization
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
fully sharded data parallel
```

## Q5_parallelism_llm: Parallel architectures for transformers and mixture-of-experts models

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("deep learning training" OR "distributed training" OR "transformer training" OR "large language model training" OR "giant model training") AND ("data parallelism" OR "model parallelism" OR "tensor parallelism" OR "pipeline parallelism" OR "sequence parallelism" OR "context parallelism" OR "expert parallelism" OR "mixture of experts" OR "conditional computation" OR "automatic parallelization" OR "computation graph partitioning" OR "long sequence")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("deep learning training" OR "distributed training" OR "transformer training" OR "large language model training" OR "giant model training") AND ("data parallelism" OR "model parallelism" OR "tensor parallelism" OR "pipeline parallelism" OR "sequence parallelism" OR "context parallelism" OR "expert parallelism" OR "mixture of experts" OR "conditional computation" OR "automatic parallelization" OR "computation graph partitioning" OR "long sequence")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"deep learning training" OR all:"distributed training" OR all:"transformer training" OR all:"large language model training" OR all:"giant model training") AND (all:"data parallelism" OR all:"model parallelism" OR all:"tensor parallelism" OR all:"pipeline parallelism" OR all:"sequence parallelism" OR all:"context parallelism" OR all:"expert parallelism" OR all:"mixture of experts" OR all:"conditional computation" OR all:"automatic parallelization" OR all:"computation graph partitioning" OR all:"long sequence")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("deep learning training" OR "distributed training" OR "transformer training" OR "large language model training" OR "giant model training") AND ("data parallelism" OR "model parallelism" OR "tensor parallelism" OR "pipeline parallelism" OR "sequence parallelism" OR "context parallelism" OR "expert parallelism" OR "mixture of experts" OR "conditional computation" OR "automatic parallelization" OR "computation graph partitioning" OR "long sequence")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
model parallelism training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
pipeline parallelism training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
tensor parallelism training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
mixture of experts training
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
automatic parallelization machine learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
long sequence transformer training
```

## Q6_decentralized_convergence: Decentralized optimization, topology, synchronization, and convergence

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed optimization" OR "decentralized optimization" OR "decentralized learning" OR "decentralized SGD" OR "distributed averaging" OR "multi-agent optimization" OR "distributed first-order methods") AND (gossip OR consensus OR "gradient tracking" OR "gradient push" OR topology OR "spectral gap" OR "graph clustering" OR "time-varying graph" OR "directed graph" OR asynchronous OR staleness OR quantized OR convergence OR "communication complexity")
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed optimization" OR "decentralized optimization" OR "decentralized learning" OR "decentralized SGD" OR "distributed averaging" OR "multi-agent optimization" OR "distributed first-order methods") AND (gossip OR consensus OR "gradient tracking" OR "gradient push" OR topology OR "spectral gap" OR "graph clustering" OR "time-varying graph" OR "directed graph" OR asynchronous OR staleness OR quantized OR convergence OR "communication complexity")
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed optimization" OR all:"decentralized optimization" OR all:"decentralized learning" OR all:"decentralized SGD" OR all:"distributed averaging" OR all:"multi-agent optimization" OR all:"distributed first-order methods") AND (all:gossip OR all:consensus OR all:"gradient tracking" OR all:"gradient push" OR all:topology OR all:"spectral gap" OR all:"graph clustering" OR all:"time-varying graph" OR all:"directed graph" OR all:asynchronous OR all:staleness OR all:quantized OR all:convergence OR all:"communication complexity")) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed optimization" OR "decentralized optimization" OR "decentralized learning" OR "decentralized SGD" OR "distributed averaging" OR "multi-agent optimization" OR "distributed first-order methods") AND (gossip OR consensus OR "gradient tracking" OR "gradient push" OR topology OR "spectral gap" OR "graph clustering" OR "time-varying graph" OR "directed graph" OR asynchronous OR staleness OR quantized OR convergence OR "communication complexity")
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
decentralized machine learning
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed optimization communication
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
gossip learning
```

## Q7_measurement_scheduling: Scheduling, overlap, stragglers, diagnosis, benchmarking, and modeling

### ACM Digital Library / ACM Guide

Execution: manual title/abstract/keyword-equivalent search; export all results

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (scheduling OR scheduler OR overlap OR straggler OR "tail latency" OR diagnosis OR profiling OR "performance analysis" OR benchmark OR modeling OR simulation OR simulator OR "total cost of ownership" OR TCO OR energy OR carbon)
```

### IEEE Xplore Metadata API

Execution: querytext with start_year/end_year; page with max_records/start_record

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (scheduling OR scheduler OR overlap OR straggler OR "tail latency" OR diagnosis OR profiling OR "performance analysis" OR benchmark OR modeling OR simulation OR simulator OR "total cost of ownership" OR TCO OR energy OR carbon)
```

### arXiv API

Execution: Atom API; category filter from queries.yaml; page with start/max_results

```text
((all:"distributed deep learning" OR all:"distributed machine learning" OR all:"distributed training" OR all:"distributed DNN training" OR all:"large language model training") AND (all:scheduling OR all:scheduler OR all:overlap OR all:straggler OR all:"tail latency" OR all:diagnosis OR all:profiling OR all:"performance analysis" OR all:benchmark OR all:modeling OR all:simulation OR all:simulator OR all:"total cost of ownership" OR all:TCO OR all:energy OR all:carbon)) AND submittedDate:[200901010000 TO 202607312359]
```

### OpenReview

Execution: enumerate accepted ICLR/TMLR notes, then apply normalized context-AND-mechanism rule locally

```text
("distributed deep learning" OR "distributed machine learning" OR "distributed training" OR "distributed DNN training" OR "large language model training") AND (scheduling OR scheduler OR overlap OR straggler OR "tail latency" OR diagnosis OR profiling OR "performance analysis" OR benchmark OR modeling OR simulation OR simulator OR "total cost of ownership" OR TCO OR energy OR carbon)
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed training scheduling
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed deep learning performance
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
machine learning straggler
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
distributed training diagnosis
```

### USENIX

Execution: site search term; preserve result HTML and paper-page metadata

```text
machine learning benchmark
```
