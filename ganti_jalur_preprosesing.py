#!/usr/bin/env python3
"""
ganti_jalur_preprosesing.py

Tahap kedua penggantian jalur. Melanjutkan ganti_jalur.py, yang hanya mengenal
root ./. Skrip ini menambahkan root kedua,
./work, tempat seluruh rantai preprosesing dijalankan.

Root pertama, ./:

    ...\\03_dataset_release\\X                      ->  ./data/X
    ...\\05_results\\X                              ->  ./results/X
    ...\\09_comparisons\\pytorch-CycleGAN-...\\X    ->  ./comparators/pGAN/X
    ...\\09_comparisons\\X                          ->  ./comparators/X
    ...\\01_raw_data\\X                             ->  ./data/raw_dicom/X

Root kedua, ./work:

    ...\\dataset_5fold_final_v2\\X                  ->  ./data/dataset_5fold_final_v2/X
    ...\\dataset_5fold_final\\X                     ->  ./data/dataset_5fold_final/X
    ...\\dataset_5fold_base\\X                      ->  ./data/dataset_5fold_base/X
    ...\\T1_&_T1_Fatsat\\X                          ->  ./data/raw_dicom/X
    ...\\temp_*  dan produk antara lainnya           ->  ./work/...

Aman dijalankan berkali-kali. Jalur yang sudah relatif tidak cocok dengan
polanya, sehingga tidak tersentuh dua kali.

Dua pengaman. Skrip berhenti dengan galat kalau jumlah baris sebuah berkas
berubah, dan berhenti kalau ada baris tanpa jalur absolut yang ikut tersunting.
Urutan escape seperti print(f"...:\\n{exp_dir}\\n") karena itu tidak akan
pernah tersentuh.

Pemakaian, dijalankan dari akar deposit:
    python ganti_jalur_preprosesing.py --dry-run
    python ganti_jalur_preprosesing.py
"""

import argparse, os, re, sys

PROY = "00_MRI_Meningioma_Synthesis_Project"
DSET = "00_dataset"

# Dua root, dicoba yang paling panjang lebih dulu.
POLA = re.compile(
    r'[A-Za-z]:\\+(?:' + re.escape(PROY) + r'|' + re.escape(DSET) + r')'
    r'(?:\\+[^"\'\r\n]*)?'
)

PETA_PROY = [
    ("09_comparisons/pytorch-CycleGAN-and-pix2pix", "comparators/pGAN"),
    ("03_dataset_release", "data"),
    ("05_results",         "results"),
    ("09_comparisons",     "comparators"),
    ("01_raw_data",        "data/raw_dicom"),
]

# Root preprosesing. Semua produk antara masuk ./work, dataset akhir ke ./data.
PETA_DSET = [
    ("dataset_5fold_final_v2", "data/dataset_5fold_final_v2"),
    ("dataset_5fold_final",    "data/dataset_5fold_final"),
    ("dataset_5fold_base",     "data/dataset_5fold_base"),
    ("T1_&_T1_Fatsat",         "data/raw_dicom"),
]

def _terap(sisa, peta, bawaan):
    for lama, baru in peta:
        if sisa == lama:
            return "./" + baru
        if sisa.startswith(lama + "/"):
            return "./" + baru + sisa[len(lama):]
    return "./" + bawaan + sisa if sisa else "./" + bawaan.rstrip("/")

def ubah(m):
    s = m.group(0)
    akar = PROY if PROY in s else DSET
    sisa = s.split(akar, 1)[1]
    sisa = sisa.replace("\\\\", "/").replace("\\", "/").lstrip("/")
    if akar == PROY:
        return _terap(sisa, PETA_PROY, "")
    return _terap(sisa, PETA_DSET, "work/")

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
