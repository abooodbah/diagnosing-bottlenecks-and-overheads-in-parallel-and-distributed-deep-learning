#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from common import Record, ROOT, load_queries, parse_year, stable_id, today_iso, write_records

BASE = "https://www.usenix.org"


def infer_year(text: str) -> str:
    y = parse_year(text)
    if y:
        return y
    m = re.search(r"(?:osdi|nsdi|atc|fast|lisa|security)(\d{2})", text, re.I)
    if m:
        yy = int(m.group(1))
        return str(2000 + yy if yy < 90 else 1900 + yy)
    return ""


def extract_search_hits(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    hits = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE, a["href"])
        title = " ".join(a.get_text(" ", strip=True).split())
        if not title or len(title) < 8:
            continue
        p = urlparse(href)
        if p.netloc not in {"www.usenix.org", "usenix.org"}:
            continue
        if "/conference/" not in p.path and "/publications/" not in p.path:
            continue
        # Exclude conference landing pages and generic proceedings search pages.
        if p.path.rstrip("/").count("/") < 3:
            continue
        key = (href.split("#", 1)[0], title)
        if key not in seen:
            hits.append(key); seen.add(key)
    return hits


def fetch_detail(session: requests.Session, url: str, timeout: int = 60) -> tuple[str, str, str, str]:
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return "", "", "", ""
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    title = " ".join(h1.get_text(" ", strip=True).split()) if h1 else ""
    abstract = ""
    for selector in [".field-name-field-paper-description", ".field-name-body", "article"]:
        node = soup.select_one(selector)
        if node:
            abstract = " ".join(node.get_text(" ", strip=True).split())
            if len(abstract) > 100:
                break
    authors = ""
    for selector in [".field-name-field-paper-people-text", ".authors"]:
        node = soup.select_one(selector)
        if node:
            authors = " ".join(node.get_text(" ", strip=True).split()); break
    year = infer_year(url + " " + soup.get_text(" ", strip=True)[:1000])
    return title, authors, abstract, year


def main() -> int:
    ap = argparse.ArgumentParser(description="Search the USENIX site for the registered query terms.")
    ap.add_argument("--queries", default=str(ROOT / "queries.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/raw/usenix.csv"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results/raw/usenix_html"))
    ap.add_argument("--query-id", action="append")
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--delay", type=float, default=0.7)
    ap.add_argument("--no-detail", action="store_true", help="Do not fetch individual paper pages.")
    args = ap.parse_args()

    cfg = load_queries(args.queries)
    start_year = int(cfg["modern_search_start"][:4])
    end_year = int(cfg["cutoff_date"][:4])
    selected = set(args.query_id or [])
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "distributed-training-survey-reconstruction/1.0"
    records: list[Record] = []
    logs: list[dict] = []

    for block in cfg["query_blocks"]:
        if selected and block["id"] not in selected:
            continue
        for term in block.get("usenix_terms", []):
            seen_urls = set()
            pages_visited = 0
            for page in range(args.max_pages):
                url = f"{BASE}/search/site/{quote(term)}?page={page}"
                resp = session.get(url, timeout=90)
                resp.raise_for_status()
                raw_path = raw_dir / f"{block['id']}_{stable_id(term)}_{page:03d}.html"
                raw_path.write_bytes(resp.content)
                hits = extract_search_hits(resp.text)
                new_hits = [(u, t) for u, t in hits if u not in seen_urls]
                pages_visited += 1
                if not new_hits:
                    break
                for hit_url, search_title in new_hits:
                    seen_urls.add(hit_url)
                    title, authors, abstract, year = ("", "", "", "")
                    if not args.no_detail:
                        title, authors, abstract, year = fetch_detail(session, hit_url)
                        time.sleep(args.delay)
                    title = title or search_title
                    year = year or infer_year(hit_url + " " + search_title)
                    if year and not (start_year <= int(year) <= end_year):
                        continue
                    records.append(Record(
                        source="usenix",
                        query_id=block["id"],
                        source_record_id=hit_url,
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        url=hit_url,
                        publication_type="conference_paper",
                        venue="USENIX",
                        retrieved_at=today_iso(),
                        raw_file=str(raw_path.relative_to(ROOT)),
                    ))
                time.sleep(args.delay)
            logs.append({
                "source": "usenix",
                "query_id": block["id"],
                "term": term,
                "pages_visited": pages_visited,
                "unique_links_found": len(seen_urls),
                "retrieved_at": today_iso(),
            })

    write_records(args.out, records)
    log_path = Path(args.out).with_name("usenix_query_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["source", "query_id", "term", "pages_visited", "unique_links_found", "retrieved_at"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    print(f"wrote {len(records)} USENIX hit rows to {args.out}")
    print("USENIX has no stable public bibliographic API; preserve the raw HTML and run date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
