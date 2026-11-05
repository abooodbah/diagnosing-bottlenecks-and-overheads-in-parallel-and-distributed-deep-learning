#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from common import ROOT, compile_boolean, load_queries


def main() -> int:
    ap=argparse.ArgumentParser(description='Print registered ACM Digital Library search strings.')
    ap.add_argument('--queries',default=str(ROOT/'queries.yaml'))
    ap.add_argument('--out',default=str(ROOT/'results/acm_query_sheet.txt'))
    args=ap.parse_args()
    cfg=load_queries(args.queries)
    lines=[]
    lines.append('ACM DIGITAL LIBRARY MANUAL SEARCH SHEET')
    lines.append('')
    lines.append('Database: ACM Guide to Computing Literature (not only the ACM Full-Text Collection)')
    lines.append(f"Publication window: {cfg['modern_search_start']} through {cfg['cutoff_date']}")
    lines.append('Fields: Anywhere for the registered Boolean expression; preserve the result count and export RIS/CSV.')
    lines.append('Do not alter a query because its count is too high or low. Log any syntax adaptation verbatim.')
    lines.append('')
    for b in cfg['query_blocks']:
        lines.append(f"[{b['id']}] {b['label']}")
        lines.append(compile_boolean(b))
        lines.append('')
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text('\n'.join(lines),encoding='utf-8')
    print(args.out)
    return 0
if __name__=='__main__':raise SystemExit(main())
