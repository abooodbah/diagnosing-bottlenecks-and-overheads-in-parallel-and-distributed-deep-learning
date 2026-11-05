#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from rapidfuzz.fuzz import token_set_ratio

from common import Record, ROOT, normalize_doi, normalize_text, parse_year, stable_id, today_iso, write_records

API = "https://api.openalex.org"


def abstract_from_index(inv) -> str:
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for idx in idxs:
            positions.append((idx, word))
    return " ".join(w for _, w in sorted(positions))


def get(session: requests.Session, url: str, params=None, attempts=4):
    for attempt in range(attempts):
        r = session.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(2 ** attempt + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenAlex rate limit persisted for {url}")


def resolve_seed(session: requests.Session, seed: dict, api_key: str) -> dict:
    doi = normalize_doi(seed.get("doi", ""))
    params = {"api_key": api_key} if api_key else {}
    if doi:
        url = f"{API}/works/https://doi.org/{quote(doi, safe='')}"
        try:
            return get(session, url, params=params)
        except Exception:
            pass
    search_params = {"search": seed["title_simple"], "per_page": 5, **params}
    data = get(session, f"{API}/works", params=search_params)
    candidates = data.get("results", [])
    if not candidates:
        raise RuntimeError(f"Could not resolve seed: {seed['title_simple']}")
    gold = normalize_text(seed["title_simple"])
    return max(candidates, key=lambda w: token_set_ratio(gold, normalize_text(w.get("title", ""))))


def work_to_record(work: dict, query_id: str, raw_file: str) -> Record:
    authors = "; ".join(
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
        if a.get("author")
    )
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    doi = work.get("doi") or ""
    return Record(
        source="citation_chaining_openalex",
        query_id=query_id,
        source_record_id=work.get("id", "") or stable_id(work.get("title", ""), str(work.get("publication_year", ""))),
        title=work.get("title", "") or "",
        authors=authors,
        year=str(work.get("publication_year") or ""),
        abstract=abstract_from_index(work.get("abstract_inverted_index")),
        doi=doi,
        url=primary.get("landing_page_url") or work.get("id", "") or "",
        publication_type=work.get("type", "") or "",
        venue=source.get("display_name", "") or "",
        retrieved_at=today_iso(),
        raw_file=raw_file,
    )


def fetch_ids(session, ids, api_key, raw_dir: Path, label: str, batch_size=50):
    works = []
    for n in range(0, len(ids), batch_size):
        batch = [x.rsplit("/", 1)[-1] for x in ids[n:n+batch_size]]
        params = {"filter": "openalex:" + "|".join(batch), "per_page": batch_size}
        if api_key:
            params["api_key"] = api_key
        data = get(session, f"{API}/works", params=params)
        raw_path = raw_dir / f"{label}_{n//batch_size:04d}.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        works.extend((w, str(raw_path.relative_to(ROOT))) for w in data.get("results", []))
        time.sleep(0.15)
    return works


def main() -> int:
    ap = argparse.ArgumentParser(description="One-hop backward/forward citation tracing from the closest prior surveys.")
    ap.add_argument("--seeds", default=str(ROOT / "data/seed_surveys.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/raw/citation_chaining.csv"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results/raw/openalex_json"))
    ap.add_argument("--forward", action="store_true", help="Also retrieve forward citations to each seed.")
    ap.add_argument("--cutoff", default="2026-07-31")
    ap.add_argument("--max-forward-per-seed", type=int, default=1000)
    args = ap.parse_args()

    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    email = os.environ.get("CONTACT_EMAIL", "").strip()
    session = requests.Session()
    ua = "distributed-training-survey-reconstruction/1.0"
    if email:
        ua += f" (mailto:{email})"
    session.headers["User-Agent"] = ua
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    with open(args.seeds, newline="", encoding="utf-8") as f:
        seeds = list(csv.DictReader(f))

    records: list[Record] = []
    logs: list[dict] = []
    cutoff_year = int(args.cutoff[:4])

    for seed in seeds:
        work = resolve_seed(session, seed, api_key)
        seed_id = work["id"].rsplit("/", 1)[-1]
        seed_path = raw_dir / f"seed_{seed['ref_no']}_{seed_id}.json"
        seed_path.write_text(json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        refs = work.get("referenced_works", []) or []
        backward = fetch_ids(session, refs, api_key, raw_dir, f"backward_{seed['ref_no']}")
        for w, raw_file in backward:
            if (w.get("publication_year") or 0) <= cutoff_year:
                records.append(work_to_record(w, f"BACKWARD_REF_{seed['ref_no']}", raw_file))

        forward_count = 0
        if args.forward:
            cursor = "*"
            while cursor and forward_count < args.max_forward_per_seed:
                params = {
                    "filter": f"cites:{seed_id},from_publication_date:2009-01-01,to_publication_date:{args.cutoff}",
                    "per_page": 100,
                    "cursor": cursor,
                }
                if api_key:
                    params["api_key"] = api_key
                data = get(session, f"{API}/works", params=params)
                raw_path = raw_dir / f"forward_{seed['ref_no']}_{forward_count:05d}.json"
                raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                for w in data.get("results", []):
                    records.append(work_to_record(w, f"FORWARD_REF_{seed['ref_no']}", str(raw_path.relative_to(ROOT))))
                    forward_count += 1
                    if forward_count >= args.max_forward_per_seed:
                        break
                cursor = data.get("meta", {}).get("next_cursor")
                if not data.get("results"):
                    break
                time.sleep(0.15)

        logs.append({
            "seed_ref_no": seed["ref_no"],
            "seed_title": seed["title_simple"],
            "openalex_id": work.get("id", ""),
            "backward_references_reported": len(refs),
            "backward_records_resolved": len(backward),
            "forward_records_retrieved": forward_count,
            "retrieved_at": today_iso(),
        })

    write_records(args.out, records)
    log_path = Path(args.out).with_name("citation_chaining_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(logs[0].keys()) if logs else []
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    print(f"wrote {len(records)} citation-chain rows to {args.out}")
    print("These are discovery records and still require title/abstract and full-text screening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
