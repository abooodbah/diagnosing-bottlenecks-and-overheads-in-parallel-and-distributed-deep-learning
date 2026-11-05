#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Record:
    source: str
    query_id: str
    source_record_id: str
    title: str
    authors: str = ""
    year: str = ""
    abstract: str = ""
    doi: str = ""
    url: str = ""
    publication_type: str = ""
    venue: str = ""
    retrieved_at: str = ""
    raw_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_queries(path: Path | str = ROOT / "queries.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = value.replace("allreduce", "all reduce").replace("all-reduce", "all reduce")
    value = value.replace("mixture-of-experts", "mixture of experts")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".,;)")


def extract_arxiv_id(value: str) -> str:
    text = value or ""
    m = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", text, re.I)
    return m.group(1).lower() if m else ""


def canonical_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        u = urlparse(value)
        host = u.netloc.lower().removeprefix("www.")
        path = re.sub(r"/$", "", u.path)
        return f"{host}{path}".lower()
    except Exception:
        return value.lower()


def stable_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]


def parse_year(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    s = str(value)
    m = re.search(r"(?:19|20)\d{2}", s)
    return m.group(0) if m else ""


def today_iso() -> str:
    return date.today().isoformat()


def write_records(path: Path | str, records: Iterable[Record | dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() if isinstance(r, Record) else dict(r) for r in records]
    fields = [f.name for f in Record.__dataclass_fields__.values()]
    extras = []
    for r in rows:
        for k in r:
            if k not in fields and k not in extras:
                extras.append(k)
    fields += extras
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv_records(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def within_year_window(year: str, start: int, end: int) -> bool:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return False
    return start <= y <= end


def local_block_match(title: str, abstract: str, block: dict) -> bool:
    """Conservative local title/abstract Boolean check.

    A record must contain at least one context phrase and one mechanism phrase.
    Phrase matching is normalized substring matching. This mirrors the conceptual
    search blocks but is not a substitute for each database's native search engine.
    """
    hay = normalize_text(f"{title} {abstract}")
    contexts = [normalize_text(x) for x in block.get("context", [])]
    mechanisms = [normalize_text(x) for x in block.get("mechanism", [])]
    return any(x in hay for x in contexts) and any(x in hay for x in mechanisms)


def compile_boolean(block: dict) -> str:
    def quote(s: str) -> str:
        return f'"{s}"' if " " in s or "-" in s else s
    left = " OR ".join(quote(x) for x in block.get("context", []))
    right = " OR ".join(quote(x) for x in block.get("mechanism", []))
    return f"({left}) AND ({right})"


def compile_arxiv(block: dict, start_date: str, cutoff_date: str) -> str:
    def atom(s: str) -> str:
        s = s.replace('"', '')
        return f'all:"{s}"' if " " in s or "-" in s else f"all:{s}"
    left = " OR ".join(atom(x) for x in block.get("context", []))
    right = " OR ".join(atom(x) for x in block.get("mechanism", []))
    start = start_date.replace("-", "") + "0000"
    end = cutoff_date.replace("-", "") + "2359"
    return f"(({left}) AND ({right})) AND submittedDate:[{start} TO {end}]"


def json_dump(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
