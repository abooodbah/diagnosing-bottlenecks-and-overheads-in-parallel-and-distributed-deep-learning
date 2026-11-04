# Provenance: which code ran each launch

The harness, the launch scripts, and the rank environment helper changed
during the campaign (6 to 11 September 2026, Stampede3). Stampede3 keeps
only the last version of each file, and Slurm on that system stores no
batch scripts or job environments (`AccountingStoreFlags = (null)`;
`sacct --batch-script` prints `NONE` for all 34 campaign jobs). This
directory holds every version that ran, the evidence that ties each
launch to a version, and the limits of that evidence.

Nothing here is needed to regenerate the tables and figures. It answers
the question "which code produced this record".

## Three kinds of code files

| Kind | Where | What it is |
|---|---|---|
| Recovered original | `versions/recovered/stampede3_2026-09-23/` | Copies of the files read from `/scratch/08447/abd2000/zmig` on Stampede3 on 2026-09-23 (read-only access), identical apart from the redaction below. These are the last version of each file. |
| Recovered original | `versions/recovered/zmig_tgz_2026-09-06/` | Members of `zmig.tgz`, the bundle uploaded on 6 Sep at 03:54 CDT, identical apart from the redaction below. These are the first version of each file. |
| Reconstruction | `versions/reconstructed/` | Intermediate versions that exist nowhere on Stampede3. Each was rebuilt by replaying the recorded edits (`edit_history.json`) from the 6 Sep bundle. The file name gives the version label and the upload time (CDT). |
| Portable copy | `../harness/` | The files to use for a rerun. They are edited for reuse, so they are not evidence of what ran. `../harness/README.md` lists every difference from the originals. |

**Redaction.** The Slurm allocation id in the `#SBATCH --account=` line
of the launch scripts is replaced by `REDACTED`, a string of the same
length, in every recovered original, every reconstruction, and
`edit_history.json`. File sizes and line numbers are therefore unchanged,
and the SHA-256 values in `version_timeline.csv` and
`stampede3_files.csv` are those of the redacted files. Nothing else was
changed; the user paths of the campaign are kept as they were. The
portable copies replace the allocation line with a comment and take the
paths from environment variables.

## Files

- `version_timeline.csv`: one row per version of each file: label (for
  example `lb-v5`), upload time in CDT and UTC, SHA-256, size, kind, and
  path. The labels are `lb-v0` to `lb-v8` (`llama_bench.py`), `moe-v0`
  and `moe-v1`, `h100sh-v0` to `h100sh-v4` (`zmig_h100.slurm`),
  `pvcsh-v0` to `pvcsh-v6` (`zmig_pvc.slurm`), `env-v0` to `env-v5`
  (`rank_env.sh`), and `diag-v0` to `diag-v5` (`diag_pvc.slurm`).
- `edit_history.json`: 80 recorded edits and uploads of these files, as
  full writes and exact string replacements.
  `scripts/replay_versions.py` replays them and writes
  `processed/provenance_replay.txt`. The replay starts from the 6 Sep
  bundle bytes, produces every version in `version_timeline.csv`, and
  ends byte-identical to the files read from Stampede3 on 2026-09-23
  (all three with the allocation id redacted the same way).
- `slurm_jobs.csv`: the 34 campaign jobs from `sacct`, with state, submit,
  start and end times, node list, working directory, and submit line.
- `slurm_steps.csv`: their 198 job steps with start and end times. Each
  `srun` launch is one step.
- `stampede3_files.csv`: size, modify time, change time, and SHA-256 of
  the campaign files and of the DeepSpeed files in the H100 environment,
  as read on 2026-09-23, with the path of the copy in this repository.

## How a launch is tied to a version

`scripts/launch_windows.py` writes `processed/launch_windows.csv`, one row
per launch: 136 launches in the logs of 22 measurement and smoke jobs
(the six Intel diagnostic jobs run no benchmark). For each launch it
records the
launch script, harness, and rank-environment version, the kind of each
(recovered original or reconstruction), and the checks below.

1. **Upload timeline.** The version in use is the last upload that
   started before the launch time printed in the job log (which matches
   the Slurm step start). `seconds_from_nearest_upload` shows how close
   the nearest upload was; uploads took 3 to 6 s to complete.
2. **Stack-trace fingerprints.** Where a launch failed, the Python and
   NCCL stack traces name `llama_bench.py` line numbers and functions.
   `fingerprint_fits` lists the versions whose lines match. In job
   3476714 each set of lines fits exactly one version.
3. **Record markers.** Where a launch wrote a record or printed
   heartbeats, these narrow the version: default model depth 30
   (`lb-v2` and later; 32 before), heartbeats in the log (`lb-v3` and
   later), harness timers, recognised by a rank-0 barrier count equal to
   the number of measured iterations (`lb-v4` and later),
   `bench_version` 2 (`lb-v5` and later), `transport_note` of
   `ib-gdr-disabled` (`lb-v7` and later), and `allgather_fn` entries
   (`lb-v8`). `record_markers_fit` lists the versions that fit.
4. **Change times on Stampede3.** A file's change time cannot be set by a
   user. The last versions of `llama_bench.py`, `moe_bench.py`, and
   `zmig_h100.slurm` were changed before every launch attributed to them
   and have not changed since.

`timeline_agrees` is `yes` when the timeline version is among the
versions the fingerprints and the markers allow. It is `yes` for all 98
launches that have either kind of evidence. It is `n/a` for the other
38: 16 routing-benchmark launches that wrote records (the markers apply
only to `llama_bench.py`; the routing benchmark has two versions and the
timeline places them) and 22 launches that failed or were stopped before
printing a stack trace or a heartbeat.

## How the remaining attributions were made

- Reconstructed bytes come from the recorded edit history. The replay
  check shows that the history is complete between the two recovered
  end points.
- `lb-v0` to `lb-v2` share one line layout, so the timeline decides the
  version for the jobs that ran them: the smoke jobs, 3472838, 3472847,
  and the first launch of 3472848.
- The host-staged launch of 3476714 at 06:45 is attributed by the
  timeline: an upload at 06:43:18, the `lb-v6` fingerprint on the next
  launch, and no upload in between.
- Job 3481504 is attributed to `lb-v7` by the timeline and by two
  record markers that only `lb-v7` carries together.
- Each job's batch script is attributed from upload and change times;
  Slurm keeps no copy of submitted scripts.
- DeepSpeed's communication logger (`comms_logger`, `prof_all`) is
  hard-coded on in every harness version. The Intel jobs 3472847 and
  3472848 read its state at run time, and their records hold the
  logger's own entries. For the other jobs the executed source and the
  growth of the step time establish it.

## Runs that were overwritten or set aside

Result files were named by platform, node count, stage, and repetition,
so a later launch with the same name replaced an earlier file on
Stampede3. `processed/result_files.csv` lists every writer of every result
file, and `processed/launch_windows.csv` gives each launch's status.

- Kept: 70 records, in `raw/h100/` and `raw/pvc/`.
- Set aside: 6 Intel four-node DDP and ZeRO-2 records from 3472847, in
  `raw/pvc_v1/` (see the README there). No script reads them.
- Overwritten: 14 launches. The smoke job 3472833 (5 launches) was
  replaced by 3472848. Job 3472848's two-node repetition 1 (DDP, ZeRO-2,
  ZeRO-3; 40 measured after 10 warm-up) was replaced by 3473072 (30 after
  8). Job 3482989 (6 launches, a complete duplicate of repetition 3 at
  two nodes) was replaced by 3482991. The overwritten files survive
  nowhere; the medians they logged are in `launch_windows.csv`
  (`p50_source` is `log WROTE line`).
- Not written: 46 launches failed or were stopped at the time limit.
