#!/usr/bin/env python3
import re, csv, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/data/review_methodology/manuscript.txt')
out = Path(sys.argv[2] if len(sys.argv) > 2 else '/mnt/data/review_methodology/references.csv')
text = src.read_text(encoding='utf-8', errors='replace')
text = text.split('\nReferences\n',1)[1]
# Remove page headers/footers introduced by pdftotext -layout.
lines=[]
for line in text.splitlines():
    s=line.strip()
    if re.fullmatch(r'\d+', s):
        continue
    if re.match(r'^Diagnosing Distributed Deep-Learning Bottlenecks\s+\d+$', s):
        continue
    if re.match(r'^\d+\s+Abdulfatah Bahbouh and Ishfaq Ahmad$', s):
        continue
    lines.append(line.replace('\f',''))
text='\n'.join(lines)
# Split numbered refs.
parts = re.split(r'(?m)^\s*\[(\d{1,3})\]\s*', text)
rows=[]
for i in range(1,len(parts),2):
    num=int(parts[i]); raw=parts[i+1]
    raw=re.sub(r'\s+', ' ', raw).strip()
    # Fix line-wrapped URLs
    raw=raw.replace('https: //','https://').replace('http: //','http://')
    raw=re.sub(r'https:\s+//', 'https://', raw)
    raw=re.sub(r'http:\s+//', 'http://', raw)
    # normalize spaces around line-broken URL punctuation
    raw=re.sub(r'(?<=/)-\s+', '-', raw)
    # parse authors/year/title as best effort
    m=re.match(r'(.+?)\.\s+((?:19|20)\d{2})\.\s+(.+)', raw)
    authors=year=rest=''
    if m:
        authors=m.group(1).strip(); year=m.group(2); rest=m.group(3).strip()
    else:
        rest=raw
    # title ends at first comma followed by likely venue marker; this is imperfect.
    title=rest
    venue=''
    # protect titles containing commas by using known URL/DOI end and split around first comma before venue patterns
    candidates=[]
    for pat in [r',\s+(?=(?:ACM|IEEE|USENIX|NeurIPS|NIPS|ICML|ICLR|MLSys|OSDI|SOSP|EuroSys|CVPR|AAAI|IJCAI|JMLR|Journal|Proceedings|Foundations|Annual Reviews|Transactions|arXiv|PVLDB|Interspeech|Asilomar|Social Network|Applied Sciences|Computational Management|Heliyon|Systems and Control|Parallel Computing|Mathematical and Scientific|Vicinagearth|COM-HPC|APNet|GPC/LNCS))']:
        mm=re.search(pat, rest)
        if mm: candidates.append(mm.start())
    if candidates:
        idx=min(candidates); title=rest[:idx].strip(); venue=rest[idx+1:].strip()
    doi_m=re.search(r'(?:DOI:\s*)?(10\.\d{4,9}/[^\s]+)', raw, re.I)
    doi=doi_m.group(1).rstrip('.,)') if doi_m else ''
    url_m=re.search(r'(https?://\S+)', raw)
    url=url_m.group(1).rstrip('.,)') if url_m else ''
    rows.append({'ref_no':num,'authors':authors,'year':year,'title':title,'venue_and_details':venue,'doi':doi,'url':url,'raw':raw})

if not rows or [r['ref_no'] for r in rows] != list(range(1,len(rows)+1)):
    print(f'WARNING parsed {len(rows)} refs; first/last {[r["ref_no"] for r in rows[:3]]} {[r["ref_no"] for r in rows[-3:]]}', file=sys.stderr)
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} references to {out}')
