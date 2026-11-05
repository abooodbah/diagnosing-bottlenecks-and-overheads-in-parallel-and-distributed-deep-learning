#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from common import Record, ROOT, parse_year, stable_id, today_iso, write_records


def pick(row: dict, *names: str) -> str:
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): (v or "") for k, v in row.items()}
    for n in names:
        k = re.sub(r"[^a-z0-9]", "", n.lower())
        if k in norm and str(norm[k]).strip():
            return str(norm[k]).strip()
    return ""


def parse_csv(path: Path, source: str, query_id: str) -> list[Record]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        title = pick(row, "title", "document title", "article title")
        year = parse_year(pick(row, "year", "publication year", "publication date", "date"))
        doi = pick(row, "doi", "digital object identifier")
        url = pick(row, "url", "document url", "link")
        out.append(Record(
            source=source,
            query_id=query_id,
            source_record_id=pick(row, "id", "article number", "accession number") or stable_id(title, year, doi),
            title=title,
            authors=pick(row, "authors", "author", "author(s)"),
            year=year,
            abstract=pick(row, "abstract", "summary"),
            doi=doi,
            url=url,
            publication_type=pick(row, "content type", "document type", "type"),
            venue=pick(row, "publication title", "source title", "conference", "journal"),
            retrieved_at=today_iso(),
            raw_file=str(path),
        ))
    return out


def parse_ris(path: Path, source: str, query_id: str) -> list[Record]:
    records = []
    current: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = re.match(r"^([A-Z0-9]{2})  - (.*)$", line)
        if not m:
            continue
        tag, value = m.groups()
        if tag == "TY":
            current = {"TY": [value]}
        elif tag == "ER":
            title = " ".join(current.get("TI") or current.get("T1") or [""])
            authors = "; ".join(current.get("AU", []))
            year = parse_year(" ".join(current.get("PY") or current.get("Y1") or []))
            doi = " ".join(current.get("DO", []))
            url = " ".join(current.get("UR", []))
            abstract = " ".join(current.get("AB", []))
            venue = " ".join(current.get("T2") or current.get("JF") or current.get("JO") or [])
            pubtype = " ".join(current.get("TY", []))
            records.append(Record(
                source=source,
                query_id=query_id,
                source_record_id=stable_id(title, year, doi, url),
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                doi=doi,
                url=url,
                publication_type=pubtype,
                venue=venue,
                retrieved_at=today_iso(),
                raw_file=str(path),
            ))
            current = {}
        else:
            current.setdefault(tag, []).append(value)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize a database CSV or RIS export into the common record schema.")
    ap.add_argument("input")
    ap.add_argument("--source", required=True, help="e.g. acm, scopus, manual")
    ap.add_argument("--query-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.input)
    suffix = path.suffix.lower()
    if suffix == ".ris":
        records = parse_ris(path, args.source, args.query_id)
    elif suffix in {".csv", ".tsv"}:
        records = parse_csv(path, args.source, args.query_id)
    else:
        raise SystemExit("Supported export formats: .csv, .tsv, .ris")
    write_records(args.out, records)
    print(f"wrote {len(records)} normalized rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
