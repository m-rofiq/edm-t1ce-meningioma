#!/usr/bin/env python3
"""
ganti_jalur.py

Mengganti jalur berkas absolut menjadi jalur relatif di seluruh berkas .py
folder deposit. Nol import ditambahkan, nol baris logika berubah.

    ./data/X  ->  ./data/X
    ./results/X          ->  ./results/X
    ./comparators/
        pytorch-CycleGAN-and-pix2pix\\X                            ->  ./comparators/pGAN/X
    sisanya                                                        ->  ./<subpath>

Pemakaian:
    python ganti_jalur.py --dry-run
    python ganti_jalur.py
"""

import argparse, os, re, sys

PROY = "00_MRI_Meningioma_Synthesis_Project"
POLA = re.compile(r'[A-Za-z]:\\+' + re.escape(PROY) + r'(?:\\+[^"\'\r\n]*)?')

PETA = [
    ("09_comparisons/pytorch-CycleGAN-and-pix2pix", "comparators/pGAN"),
    ("03_dataset_release", "data"),
    ("05_results",         "results"),
    ("09_comparisons",     "comparators"),
    ("01_raw_data",        "data/raw_dicom"),
]

def ubah(m):
    s = m.group(0)
    sisa = s.split(PROY, 1)[1]
    sisa = sisa.replace("\\\\", "/").replace("\\", "/").lstrip("/")
    for lama, baru in PETA:
        if sisa == lama:
            return "./" + baru
        if sisa.startswith(lama + "/"):
            return "./" + baru + sisa[len(lama):]
    return "./" + sisa if sisa else "."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    total_f = total_m = 0
    for dp, dn, fn in os.walk(a.root):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".git", ".vscode")]
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            try:
                t = open(p, encoding="utf-8").read()
            except UnicodeDecodeError:
                print("LEWAT, bukan utf-8:", p); continue
            hits = POLA.findall(t)
            if not hits:
                continue
            baru = POLA.sub(ubah, t)

            # pengaman: hanya baris memuat jalur yang boleh berubah
            la, lb = t.split("\n"), baru.split("\n")
            if len(la) != len(lb):
                sys.exit("JUMLAH BARIS BERUBAH pada " + p)
            beda = [i for i in range(len(la)) if la[i] != lb[i]]
            for i in beda:
                if not POLA.search(la[i]):
                    sys.exit("BARIS TANPA JALUR IKUT BERUBAH pada %s baris %d" % (p, i + 1))

            rel = os.path.relpath(p, a.root)
            total_f += 1; total_m += len(beda)
            print("%-58s %d baris" % (rel, len(beda)))
            for i in beda:
                print("    -", la[i].strip()[:110])
                print("    +", lb[i].strip()[:110])
            if not a.dry_run:
                open(p, "w", encoding="utf-8").write(baru)

    print("=" * 62)
    print("berkas: %d   baris: %d" % (total_f, total_m))
    print("DRY RUN, nol berkas ditulis." if a.dry_run else "Ditulis.")

if __name__ == "__main__":
    main()
