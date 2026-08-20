import re, hashlib, os, glob
bad = 0
for run in sorted(glob.glob("reproduce/*/RUN.md")):
    base = os.path.dirname(run)
    for line in open(run, encoding="utf-8"):
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([0-9a-f]{8})\s*\|", line)
        if not m: continue
        p = os.path.join(base, m.group(1))
        if not os.path.isfile(p):
            print("HILANG", p); bad += 1; continue
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
        if h != m.group(3):
            print("MD5 BEDA", p, h, "!=", m.group(3)); bad += 1
print("total masalah:", bad)
