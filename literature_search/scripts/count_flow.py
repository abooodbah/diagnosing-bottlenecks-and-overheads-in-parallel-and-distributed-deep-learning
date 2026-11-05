#!/usr/bin/env python3
"""Count *available* exported results, never substitute zero for missing records."""
from __future__ import annotations
import argparse,csv,glob,json
from pathlib import Path
from common import ROOT

def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def yes(v):return (v or '').strip().lower() in {'yes','y','true','1','include','included'}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-glob',default=str(ROOT/'results/raw/*.csv'))
    ap.add_argument('--unique',default=str(ROOT/'results/merged/unique_records.csv'))
    ap.add_argument('--screening',default=str(ROOT/'templates/screening_log.csv'))
    ap.add_argument('--out',default=str(ROOT/'results/selection_flow.json'))
    ap.add_argument('--claimed',default=None,help='Optional manuscript claim raw,unique,fulltext,included; comparison ONLY')
    a=ap.parse_args()
    rf=[Path(p) for p in glob.glob(a.raw_glob) if not p.endswith('_log.csv') and 'query_log' not in p and 'venue_log' not in p]
    raw=sum(len(read(p)) for p in rf) if rf else None
    u=Path(a.unique);uniq=len(read(u)) if u.exists() else None
    sc=Path(a.screening);scr=read(sc) if sc.exists() else None
    full=sum(yes(r.get('full_text_retrieved')) or r.get('full_text_decision','').strip().lower() in {'include','exclude','included','excluded'} for r in scr) if scr is not None else None
    inc=sum(r.get('full_text_decision','').strip().lower() in {'include','included'} for r in scr) if scr is not None else None
    actual={'raw_records':raw,'unique_after_deduplication':uniq,'full_text_assessed':full,'included_sources':inc}
    payload={'execution_evidence_complete':all(v is not None for v in actual.values()),'actual_from_available_files':actual,'raw_exports':[str(p.relative_to(ROOT)) for p in rf]}
    if a.claimed:
        vals=[int(v.strip()) for v in a.claimed.split(',')]
        if len(vals)!=4:ap.error('--claimed requires four counts')
        claim=dict(zip(actual,vals))
        payload['author_reported_counts_for_comparison']=claim
        payload['verified_differences']={k:(actual[k]-claim[k] if actual[k] is not None else None) for k in actual}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))
    if not payload['execution_evidence_complete']:
        print('Incomplete evidence: missing outputs are null (unknown), not zero.')
    return 0
if __name__=='__main__':raise SystemExit(main())
