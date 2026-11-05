#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import Record, ROOT, load_queries, local_block_match, parse_year, stable_id, today_iso, write_records

API = "https://api2.openreview.net/notes"
DEFAULT_VENUES = [f"ICLR.cc/{y}/Conference" for y in range(2017, 2027)] + ["TMLR"]


def value(obj, key, default=""):
    x = (obj or {}).get(key, default)
    if isinstance(x, dict) and "value" in x:
        x = x["value"]
    if isinstance(x, list):
        return "; ".join(str(v) for v in x)
    return "" if x is None else str(x)


def millis_to_date(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Enumerate accepted OpenReview papers and filter locally with registered query blocks.")
    ap.add_argument("--queries", default=str(ROOT / "queries.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/raw/openreview.csv"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results/raw/openreview_json"))
    ap.add_argument("--venue-id", action="append", help="Accepted-paper venue id. Repeatable. Defaults to ICLR 2017-2026 and TMLR.")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--query-id", action="append")
    args = ap.parse_args()

    cfg = load_queries(args.queries)
    cutoff = cfg["cutoff_date"]
    start_year = int(cfg["modern_search_start"][:4])
    selected = set(args.query_id or [])
    blocks = [b for b in cfg["query_blocks"] if not selected or b["id"] in selected]
    venues = args.venue_id or DEFAULT_VENUES
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    logs: list[dict] = []

    for venue_id in venues:
        offset = 0
        venue_total = 0
        while True:
            params = {
                "content.venueid": venue_id,
                "limit": args.limit,
                "offset": offset,
            }
            resp = requests.get(API, params=params, timeout=120)
            if resp.status_code == 400 and venue_id == "TMLR":
                # TMLR deployments have used more than one accepted venue id.
                break
            resp.raise_for_status()
            data = resp.json()
            notes = data.get("notes", []) or []
            raw_path = raw_dir / f"{venue_id.replace('/', '_')}_{offset:06d}.json"
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            if not notes:
                break
            venue_total += len(notes)
            for note in notes:
                content = note.get("content", {}) or {}
                title = value(content, "title")
                abstract = value(content, "abstract")
                pdate = millis_to_date(note.get("pdate") or note.get("cdate") or note.get("tcdate"))
                year = parse_year(pdate)
                if year and int(year) < start_year:
                    continue
                if pdate and pdate > cutoff:
                    continue
                matching = [b["id"] for b in blocks if local_block_match(title, abstract, b)]
                for qid in matching:
                    forum = note.get("forum") or note.get("id") or stable_id(title, year)
                    url = f"https://openreview.net/forum?id={forum}"
                    records.append(Record(
                        source="openreview",
                        query_id=qid,
                        source_record_id=str(note.get("id") or forum),
                        title=title,
                        authors=value(content, "authors"),
                        year=year,
                        abstract=abstract,
                        doi=value(content, "doi"),
                        url=url,
                        publication_type="conference_or_journal_submission",
                        venue=value(content, "venue") or venue_id,
                        retrieved_at=today_iso(),
                        raw_file=str(raw_path.relative_to(ROOT)),
                    ))
            offset += len(notes)
            if len(notes) < args.limit:
                break
            time.sleep(args.delay)
        logs.append({
            "source": "openreview",
            "venue_id": venue_id,
            "enumerated_records": venue_total,
            "matched_rows_before_dedup": sum(1 for r in records if r.venue == venue_id or venue_id in r.venue),
            "retrieved_at": today_iso(),
        })

    write_records(args.out, records)
    log_path = Path(args.out).with_name("openreview_venue_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["source", "venue_id", "enumerated_records", "matched_rows_before_dedup", "retrieved_at"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    print(f"wrote {len(records)} OpenReview query-hit rows to {args.out}")
    print("Note: one paper can appear once per matching query block; the merge step deduplicates it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
