#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests

from common import Record, ROOT, compile_boolean, load_queries, parse_year, stable_id, today_iso, write_records

API = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


def authors_text(article: dict) -> str:
    data = article.get("authors", {}).get("authors", [])
    if isinstance(data, dict):
        data = [data]
    return "; ".join(a.get("full_name") or a.get("author") or "" for a in data if isinstance(a, dict))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run registered IEEE Xplore Metadata API searches.")
    ap.add_argument("--queries", default=str(ROOT / "queries.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/raw/ieee.csv"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results/raw/ieee_json"))
    ap.add_argument("--query-id", action="append")
    ap.add_argument("--max-per-query", type=int, default=0, help="0 retrieves all matching records.")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()

    key = os.environ.get("IEEE_API_KEY", "").strip()
    if not key:
        raise SystemExit("IEEE_API_KEY is required. Register at the IEEE developer portal and export the key.")

    cfg = load_queries(args.queries)
    start_year = int(cfg["modern_search_start"][:4])
    end_year = int(cfg["cutoff_date"][:4])
    selected = set(args.query_id or [])
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    logs: list[dict] = []

    for block in cfg["query_blocks"]:
        if selected and block["id"] not in selected:
            continue
        query = compile_boolean(block)
        start_record = 1
        total = None
        fetched = 0
        while total is None or start_record <= total:
            if args.max_per_query and fetched >= args.max_per_query:
                break
            batch = 200
            if args.max_per_query:
                batch = min(batch, args.max_per_query - fetched)
            params = {
                "apikey": key,
                "format": "json",
                "querytext": query,
                "start_year": start_year,
                "end_year": end_year,
                "max_records": batch,
                "start_record": start_record,
            }
            resp = requests.get(API, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            raw_path = raw_dir / f"{block['id']}_{start_record:06d}.json"
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            if total is None:
                total = int(data.get("total_records", 0))
                logs.append({
                    "source": "ieee",
                    "query_id": block["id"],
                    "query": query,
                    "reported_total": total,
                    "retrieved_at": today_iso(),
                })
            articles = data.get("articles", []) or []
            if not articles:
                break
            for a in articles:
                year = parse_year(a.get("publication_year") or a.get("publication_date"))
                url = a.get("html_url") or a.get("pdf_url") or ""
                records.append(Record(
                    source="ieee",
                    query_id=block["id"],
                    source_record_id=str(a.get("article_number") or stable_id(a.get("title", ""), year)),
                    title=a.get("title", "") or "",
                    authors=authors_text(a),
                    year=year,
                    abstract=a.get("abstract", "") or "",
                    doi=a.get("doi", "") or "",
                    url=url,
                    publication_type=a.get("content_type", "") or "",
                    venue=a.get("publication_title", "") or "",
                    retrieved_at=today_iso(),
                    raw_file=str(raw_path.relative_to(ROOT)),
                ))
            n = len(articles)
            fetched += n
            start_record += n
            if n < batch:
                break
            time.sleep(args.delay)

    write_records(args.out, records)
    log_path = Path(args.out).with_name("ieee_query_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["source", "query_id", "query", "reported_total", "retrieved_at"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    print(f"wrote {len(records)} IEEE rows to {args.out}")
    print(f"wrote query log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
