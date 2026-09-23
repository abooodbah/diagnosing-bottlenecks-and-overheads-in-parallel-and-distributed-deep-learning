# -*- coding: utf-8 -*-
"""Write or verify checksums.txt.

Covered: every file under raw/, transport_logs/, provenance/, harness/,
environment/, processed/ and figures/. Verification fails if a covered file
is missing, changed, or not listed, except under figures/: PNG and PDF bytes
depend on the matplotlib version and the installed fonts, so a figure
mismatch is reported without failing.

usage: python scripts/checksums.py --verify | --write
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVERED = ("raw", "transport_logs", "provenance", "harness", "environment", "processed", "figures")
INFORMATIONAL = ("figures/",)
LIST = os.path.join(ROOT, "checksums.txt")


def covered_files():
    out = []
    for top in COVERED:
        for d, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                out.append(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/"))
    return sorted(out)


def sha256(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv):
    if argv[1:] == ["--write"]:
        files = covered_files()
        with open(LIST, "w", encoding="utf-8", newline="\n") as fh:
            for rel in files:
                fh.write("%s  %s\n" % (sha256(rel), rel))
        print("wrote checksums.txt:", len(files), "files")
        return 0
    if argv[1:] != ["--verify"]:
        print(__doc__)
        return 2
    listed = {}
    for line in open(LIST, encoding="utf-8"):
        digest, rel = line.rstrip("\n").split("  ", 1)
        listed[rel] = digest
    bad, soft = [], []
    for rel, digest in sorted(listed.items()):
        where = soft if rel.startswith(INFORMATIONAL) else bad
        if not os.path.exists(os.path.join(ROOT, rel)):
            where.append("missing   " + rel)
        elif sha256(rel) != digest:
            where.append("changed   " + rel)
    for rel in covered_files():
        if rel not in listed:
            (soft if rel.startswith(INFORMATIONAL) else bad).append("unlisted  " + rel)
    for m in bad:
        print("MISMATCH", m)
    for m in soft:
        print("note (figure bytes; not a failure):", m)
    print("checksum verification: %s (%d files listed)" % ("FAILED" if bad else "OK", len(listed)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
