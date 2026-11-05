#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio

from common import ROOT, extract_arxiv_id, normalize_doi, normalize_text


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrospectively audit retrieval sensitivity against the manuscript bibliography.")
    ap.add_argument("retrieved", nargs="?", default=str(ROOT / "results/merged/unique_records.csv"))
    ap.add_argument("--gold", default=str(ROOT / "data/references_gold.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/merged/gold_match_report.csv"))
    ap.add_argument("--threshold", type=float, default=90.0)
    args = ap.parse_args()

    gold = read(args.gold)
    got = read(args.retrieved)
    doi_index = {}
    arxiv_index = {}
    for i, r in enumerate(got):
        d = normalize_doi(r.get("doi", "") or r.get("all_dois", ""))
        if d:
            for part in d.split(";"):
                doi_index[part] = i
        a = extract_arxiv_id(" ".join([r.get("url", ""), r.get("all_urls", ""), r.get("source_record_id", "")]))
        if a:
            arxiv_index[a] = i

    report = []
    matched = 0
    for g in gold:
        gi = None; method = ""; score = 0.0
        gd = normalize_doi(g.get("doi", ""))
        ga = extract_arxiv_id(" ".join([g.get("url", ""), g.get("raw", "")]))
        if gd and gd in doi_index:
            gi = doi_index[gd]; method = "doi"; score = 100.0
        elif ga and ga in arxiv_index:
            gi = arxiv_index[ga]; method = "arxiv_id"; score = 100.0
        else:
            gt = normalize_text(g.get("title_simple", ""))
            gy = g.get("year", "")
            best = (-1.0, None)
            for i, r in enumerate(got):
                ry = r.get("year", "")
                if gy and ry:
                    try:
                        if abs(int(gy) - int(ry)) > 1:
                            continue
                    except ValueError:
                        pass
                s = token_set_ratio(gt, normalize_text(r.get("title", "")))
                if s > best[0]:
                    best = (s, i)
            if best[0] >= args.threshold:
                score, gi = best
                method = "fuzzy_title"
        if gi is not None:
            matched += 1
            r = got[gi]
            report.append({
                "ref_no": g.get("ref_no", ""),
                "gold_year": g.get("year", ""),
                "gold_title": g.get("title_simple", ""),
                "matched": "yes",
                "match_method": method,
                "match_score": f"{score:.1f}",
                "retrieved_record_id": r.get("record_id", r.get("source_record_id", "")),
                "retrieved_title": r.get("title", ""),
                "all_sources": r.get("all_sources", r.get("source", "")),
                "all_query_ids": r.get("all_query_ids", r.get("query_id", "")),
            })
        else:
            report.append({
                "ref_no": g.get("ref_no", ""),
                "gold_year": g.get("year", ""),
                "gold_title": g.get("title_simple", ""),
                "matched": "no",
                "match_method": "",
                "match_score": "",
                "retrieved_record_id": "",
                "retrieved_title": "",
                "all_sources": "",
                "all_query_ids": "",
            })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(report[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(report)
    print(f"matched {matched}/{len(gold)} bibliography entries ({matched/len(gold):.1%})")
    print(f"missing {len(gold)-matched}; inspect {out}")
    print("This is a retrospective sensitivity audit, not evidence that the original search historically produced the corpus.")
    return 0 if matched == len(gold) else 2


if __name__ == "__main__":
    raise SystemExit(main())
