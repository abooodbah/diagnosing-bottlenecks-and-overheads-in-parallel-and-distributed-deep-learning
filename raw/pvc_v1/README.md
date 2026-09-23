# Set-aside Intel records (job 3472847)

Six records of the first Intel four-node DDP and ZeRO-2 runs, three
repetitions each, written by job 3472847 on 6 Sep 2026 with harness
`lb-v2` (40 measured iterations after 10 warm-up). The job ended with a
node failure during its seventh launch, the first ZeRO-3 repetition. No
script reads this directory, and no published value comes from it.

They were set aside for two reasons.

1. Their communication fields are not times. Harness versions `lb-v0` to
   `lb-v3` copied DeepSpeed's communication-logger records into
   `comms_per_rank0` and `comms_all_ranks`, and summed the wrong list:
   DeepSpeed 0.19.4 stores `[count, latencies, algbw, busbw]` and the
   harness read the bandwidth list, so `time_ms` is a sum of bandwidth
   values. The `count` fields and the step times are valid.
2. The same cells were measured again in job 3472924 with harness `lb-v4`,
   whose synchronized timers give valid communication times (30 measured
   iterations after 8 warm-up). Those records are in `raw/pvc/` and feed
   the published four-node Intel DDP and ZeRO-2 values. On Stampede3 they
   overwrote these files, which survive here because they had already
   been copied off the system.

For comparison, the median step of these records is 3,578 ms for DDP
(repetitions 3,578, 3,213, 4,864) and 8,291 ms for ZeRO-2 (7,768, 8,359,
8,291), against 3,718 ms and 7,732 ms in the published records. With
these records the four-node ZeRO-2 increase over DDP would be 131.7%
instead of 107.9%. The repetitions of the two jobs overlap: ZeRO-2 spans
7,236 to 7,877 ms in the published records and 7,768 to 8,359 ms here,
and DDP spans 3,547 to 4,472 ms and 3,213 to 4,864 ms.

Six more records of the same kind, from job 3472848 (two nodes,
repetitions 2 and 3), are in `raw/pvc/`. Their step times are used and
their communication fields are not. `processed/time_ms_check.txt` shows
that no published value depends on any of these `time_ms` fields.
