#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cek_patience.py
===============
Menguji apakah early-stopping patience yang lebih longgar mengubah
checkpoint yang terpilih.

Tiga konfigurasi dilatih dengan patience berbeda dari dua belas lainnya:
EXP-703 memakai 40, EXP-813 dan EXP-816 memakai 45, sedangkan sisanya 25.
Skrip ini memutar ulang kurva validasi yang tercatat seolah-olah patience
yang dipakai adalah 25, lalu membandingkan epoch yang akan terpilih
dengan epoch yang sebenarnya terpilih.

Kalau keduanya sama, penyimpangan protokol tidak mengubah satu pun angka
yang dilaporkan -- dan itu dapat dinyatakan di naskah sebagai hasil
pengujian, bukan sebagai pengakuan.

Letakkan di folder run yang memuat history.csv, lalu:

    python cek_patience.py                  # cari history.csv di folder ini
    python cek_patience.py history.csv
    python cek_patience.py history.csv 25   # patience pembanding lain
"""

import csv
import os
import sys

PATIENCE_BAKU = 25


def cari_berkas():
    if len(sys.argv) > 1:
        return sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    for nama in ("history.csv", "History.csv", "history_.csv"):
        p = os.path.join(here, nama)
        if os.path.exists(p):
            return p
    kandidat = [f for f in os.listdir(here)
                if f.lower().startswith("history") and f.lower().endswith(".csv")]
    if len(kandidat) == 1:
        return os.path.join(here, kandidat[0])
    if kandidat:
        print("Beberapa berkas history ditemukan; sebutkan salah satu:")
        for f in kandidat:
            print("   ", f)
    else:
        print("Tidak ada history*.csv di folder ini:")
        print("   ", here)
    sys.exit(1)


def main():
    path = cari_berkas()
    patience = int(sys.argv[2]) if len(sys.argv) > 2 else PATIENCE_BAKU

    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        print("[FATAL] Berkas kosong:", path)
        sys.exit(1)

    kol = [c for c in rows[0] if c and "psnr" in c.lower()]
    if not kol:
        print("[FATAL] Tidak ada kolom val_psnr. Kolom yang ada:",
              list(rows[0].keys()))
        sys.exit(1)
    kolom = "val_psnr" if "val_psnr" in rows[0] else kol[0]

    v = [float(r[kolom]) for r in rows]
    n = len(v)

    # epoch yang sebenarnya terpilih: nilai validasi tertinggi di seluruh run
    i_asli = max(range(n), key=lambda i: v[i])

    # simulasi early stopping dengan patience pembanding
    best_i, best_v, stop = 0, v[0], n
    for i in range(1, n):
        if i - best_i >= patience:
            stop = i
            break
        if v[i] > best_v:
            best_i, best_v = i, v[i]

    print("=" * 62)
    print("CEK PATIENCE")
    print("=" * 62)
    print(f"  berkas             : {os.path.basename(path)}")
    print(f"  kolom yang dipakai : {kolom}")
    print(f"  epoch tercatat     : {n}")
    print(f"  patience pembanding: {patience}")
    print("-" * 62)
    print(f"  epoch terpilih sesungguhnya   : {i_asli + 1}"
          f"   ({v[i_asli]:.6f})")
    print(f"  epoch terpilih bila patience {patience}: {best_i + 1}"
          f"   ({best_v:.6f})")
    print(f"  pelatihan akan berhenti di    : epoch {stop + 1}"
          + ("  (tidak sempat berhenti; run memang lebih pendek)"
             if stop == n else ""))
    print("-" * 62)
    if best_i == i_asli:
        print("  >>> CHECKPOINT SAMA.")
        print("      Patience yang lebih longgar tidak mengubah hasil.")
        print("      Boleh dinyatakan di naskah sebagai hasil pengujian.")
    else:
        print("  >>> CHECKPOINT BERBEDA.")
        print(f"      Dengan patience {patience} model akan berhenti sebelum")
        print(f"      mencapai epoch {i_asli + 1}, sehingga checkpoint yang")
        print("      dipakai naskah TIDAK dapat direproduksi di bawah")
        print("      protokol yang seragam. Kabari saya; kalimat Methods")
        print("      harus berbeda dan konfigurasi ini perlu dipertimbangkan")
        print("      untuk dilatih ulang dengan patience 25.")
    print()


if __name__ == "__main__":
    main()
