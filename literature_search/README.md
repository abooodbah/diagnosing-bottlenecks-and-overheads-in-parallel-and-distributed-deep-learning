# Literature search: queries and reference inventory

This folder documents the literature search behind the survey "Diagnosing
Bottlenecks and Overheads in Parallel and Distributed Deep Learning: A
Survey" (Section 3 of the survey, and Section S1 of its supplementary
material).

## Contents

- `queries.yaml`: the seven term groups (Q1 to Q7), the publication window
  (1 January 2009 to 31 July 2026), the language scope, and the settings for
  each source.
- `results/query_register.csv` and `results/query_register.md`: 61
  source-specific query strings generated from `queries.yaml` for the ACM
  Digital Library, IEEE Xplore, USENIX proceedings, arXiv, and OpenReview.
  `results/acm_query_sheet.txt` lists the ACM queries for manual entry.
- `data/references_gold.csv`: the survey's reference list, 162 entries.
  Entry 8 is the companion report, which lies outside the reviewed corpus;
  the other 161 entries are the reviewed corpus.
- `data/seed_surveys.csv`: the 13 prior surveys the citation-tracing script
  starts from.
- `data/known_item_documentation.csv`: six runtime and documentation
  sources and the role each plays in the survey.
- `scripts/`: search adapters for arXiv, IEEE Xplore, OpenReview, and
  USENIX; one-hop citation tracing through OpenAlex; import of RIS or CSV
  exports; deduplication; a screening log; and stage counts.
- `templates/`: the exclusion reason codes and a log for query amendments.

## What this folder does not contain

The query strings were reconstructed from the seven term groups after the
search, so that the search can be rerun. They are not a log of the queries
as originally submitted. The original database exports, the per-query
dates and result counts, and the screening decisions are not included. The
stage counts in the survey (approximately 1,800 records, approximately
1,200 after deduplication, approximately 300 at full text, 129 included,
then 25 added by citation tracing and targeted searches, and two revision
passes for a final corpus of 161) are the authors' reported figures.

## Rerunning the search

```bash
pip install -r requirements.txt
export CONTACT_EMAIL=you@example.edu   # see config.example.env
export IEEE_API_KEY=...                # needed only for IEEE Xplore
bash run_pipeline.sh
python scripts/print_acm_queries.py   # ACM queries are run by hand
```

Raw results go to `results/raw/` and the merged set to `results/merged/`.
Screening decisions are recorded in `templates/screening_log.csv`, and
`scripts/count_flow.py` reports the count at each stage, leaving a stage
empty when its records are missing. A rerun is a new search with its own
date. Publications after 31 July 2026 fall outside the survey's scope.
