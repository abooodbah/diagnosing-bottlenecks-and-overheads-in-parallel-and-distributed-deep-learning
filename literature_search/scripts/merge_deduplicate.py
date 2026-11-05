#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio

from common import ROOT, canonical_url, extract_arxiv_id, normalize_doi, normalize_text, read_csv_records, stable_id

SOURCE_PRIORITY = {
    "acm": 80,
    "ieee": 80,
    "usenix": 80,
    "openreview": 75,
    "arxiv": 50,
    "technical_documentation": 30,
    "manual": 20,
}


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def score(r: dict) -> tuple:
    src = (r.get("source") or "").lower()
    ptype = (r.get("publication_type") or "").lower()
    archival_bonus = 20 if normalize_doi(r.get("doi", "")) else 0
    if "preprint" in ptype or src == "arxiv":
        archival_bonus -= 5
    return (
        SOURCE_PRIORITY.get(src, 40) + archival_bonus,
        1 if r.get("abstract") else 0,
        len(r.get("abstract", "")),
        len(r.get("venue", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge source exports and deduplicate with a logged, deterministic procedure.")
    ap.add_argument("inputs", nargs="*", help="Normalized CSV files. Defaults to results/raw/*.csv.")
    ap.add_argument("--out", default=str(ROOT / "results/merged/unique_records.csv"))
    ap.add_argument("--decisions", default=str(ROOT / "results/merged/dedup_decisions.csv"))
    ap.add_argument("--fuzzy-threshold", type=float, default=97.0)
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    if not paths:
        paths = [Path(p) for p in glob.glob(str(ROOT / "results/raw/*.csv")) if not p.endswith("_log.csv")]
    if not paths:
        raise SystemExit("No normalized source CSVs found.")

    rows = []
    for p in paths:
        for r in read_csv_records(p):
            r["input_file"] = str(p)
            r["norm_title"] = normalize_text(r.get("title", ""))
            r["norm_doi"] = normalize_doi(r.get("doi", ""))
            r["arxiv_id"] = extract_arxiv_id(" ".join([r.get("url", ""), r.get("source_record_id", "")]))
            r["canon_url"] = canonical_url(r.get("url", ""))
            rows.append(r)

    dsu = DSU(len(rows))
    keys = defaultdict(list)
    for i, r in enumerate(rows):
        if r["norm_doi"]:
            keys[("doi", r["norm_doi"])].append(i)
        if r["arxiv_id"]:
            keys[("arxiv", r["arxiv_id"])].append(i)
        if r["canon_url"]:
            keys[("url", r["canon_url"])].append(i)
        if r["norm_title"]:
            keys[("title_year", r["norm_title"], r.get("year", ""))].append(i)
    for inds in keys.values():
        for j in inds[1:]:
            dsu.union(inds[0], j)

    # Fuzzy pass within year +/- 1 for records not already merged. O(n^2) is acceptable for a few thousand records.
    by_year = defaultdict(list)
    for i, r in enumerate(rows):
        try:
            y = int(r.get("year", ""))
        except ValueError:
            y = 0
        by_year[y].append(i)
    years = sorted(by_year)
    for y in years:
        candidates = by_year[y] + by_year.get(y - 1, []) + by_year.get(y + 1, []) if y else by_year[y]
        base = by_year[y]
        for ai, i in enumerate(base):
            ti = rows[i]["norm_title"]
            if len(ti) < 25:
                continue
            for j in candidates:
                if j <= i or dsu.find(i) == dsu.find(j):
                    continue
                tj = rows[j]["norm_title"]
                if len(tj) < 25:
                    continue
                if token_set_ratio(ti, tj) >= args.fuzzy_threshold:
                    dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[dsu.find(i)].append(i)

    unique_rows = []
    decision_rows = []
    for group_no, inds in enumerate(sorted(groups.values(), key=lambda x: min(x)), start=1):
        preferred = max(inds, key=lambda i: score(rows[i]))
        p = dict(rows[preferred])
        group_id = f"D{group_no:05d}"
        sources = sorted({rows[i].get("source", "") for i in inds if rows[i].get("source")})
        queries = sorted({rows[i].get("query_id", "") for i in inds if rows[i].get("query_id")})
        urls = sorted({rows[i].get("url", "") for i in inds if rows[i].get("url")})
        dois = sorted({normalize_doi(rows[i].get("doi", "")) for i in inds if normalize_doi(rows[i].get("doi", ""))})
        p.update({
            "record_id": stable_id(p.get("title", ""), p.get("year", ""), p.get("doi", ""), p.get("url", "")),
            "duplicate_group": group_id,
            "duplicate_group_size": len(inds),
            "all_sources": ";".join(sources),
            "all_query_ids": ";".join(queries),
            "all_urls": ";".join(urls),
            "all_dois": ";".join(dois),
        })
        unique_rows.append(p)
        for i in inds:
            decision_rows.append({
                "duplicate_group": group_id,
                "kept": "yes" if i == preferred else "no",
                "source": rows[i].get("source", ""),
                "query_id": rows[i].get("query_id", ""),
                "title": rows[i].get("title", ""),
                "year": rows[i].get("year", ""),
                "doi": rows[i].get("doi", ""),
                "url": rows[i].get("url", ""),
                "input_file": rows[i].get("input_file", ""),
            })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in unique_rows for k in r.keys()))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(unique_rows)
    dec = Path(args.decisions); dec.parent.mkdir(parents=True, exist_ok=True)
    dfields = ["duplicate_group", "kept", "source", "query_id", "title", "year", "doi", "url", "input_file"]
    with dec.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dfields); w.writeheader(); w.writerows(decision_rows)

    print(f"raw rows: {len(rows)}")
    print(f"unique records: {len(unique_rows)}")
    print(f"removed as duplicate query/source/version rows: {len(rows)-len(unique_rows)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
