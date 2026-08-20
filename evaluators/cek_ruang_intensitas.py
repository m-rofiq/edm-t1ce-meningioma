"""
Verifikasi TEMUAN M8 sebelum ditulis ke naskah.

Yang diuji: apakah ambang enhancement (mu + 1.5*sigma, build_enhancement_mask)
berada DI ATAS plafon clamp 1.0 yang dipakai sebelum LPIPS di
evaluate_experiment_paper.py:

    pred_enh_lp = torch.clamp(pred * enh_mask, 0.0, 1.0)
    gt_enh_lp   = torch.clamp(gt   * enh_mask, 0.0, 1.0)

Jika ya, acuan menjadi citra biner di dalam mask dan LPIPS_Enh mengukur
amplitudo enhancement, bukan tekstur perseptual.

Definisi otak mengikuti script evaluasi: mask = (gt != 0).

Cara pakai
----------
    python cek_ruang_intensitas.py                  # pakai DEFAULT_DIR di bawah
    python cek_ruang_intensitas.py <direktori>
    python cek_ruang_intensitas.py <satu_file.npy>  # uji cepat satu slice
"""
import sys
import os
import glob
import numpy as np

DEFAULT_DIR = r"./data/dataset_5fold_final_v4/holdout_test/T1CE"

CLAMP_CEILING = 1.0   # plafon torch.clamp(..., 0.0, 1.0) sebelum LPIPS
K = 1.5               # build_enhancement_mask(t1ce, brain_mask, k=1.5)
MIN_ENH_VOXELS = 10   # evaluate_experiment_paper.py: if torch.sum(enh_mask) > 10

CFG_MIN, CFG_MAX = -1.8655164, 3.4809778   # metric_config.json


def resolve_inputs(arg):
    if os.path.isfile(arg):
        return [arg]
    if os.path.isdir(arg):
        files = sorted(glob.glob(os.path.join(arg, "*.npy")))
        if not files:
            sys.exit(f"Tidak ada berkas .npy di direktori:\n  {arg}")
        return files
    sys.exit(
        f"Path tidak ditemukan:\n  {arg}\n\n"
        "Jalankan skrip ini di mesin tempat dataset berada, atau berikan\n"
        "direktori T1CE hold-out sebagai argumen."
    )


def iter_slices(vol):
    """Berkas bisa berupa satu slice 2D, tumpukan 3D, atau 4D dgn kanal."""
    a = np.asarray(vol)
    if a.ndim == 2:
        yield a
    elif a.ndim == 3:
        # heuristik: dimensi terkecil dianggap sumbu slice/kanal
        ax = int(np.argmin(a.shape))
        for i in range(a.shape[ax]):
            yield np.take(a, i, axis=ax)
    elif a.ndim == 4:
        for v in a:
            yield from iter_slices(v)
    else:
        return


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = resolve_inputs(arg)

    thresholds, frac_clipped, brain_mu, brain_sd, enh_frac = [], [], [], [], []
    vmin, vmax = np.inf, -np.inf
    n_slices = n_skipped = 0

    for f in files:
        vol = np.load(f)
        vmin = min(vmin, float(np.min(vol)))
        vmax = max(vmax, float(np.max(vol)))
        for sl in iter_slices(vol):
            brain = sl[sl != 0]
            if brain.size < 100:
                n_skipped += 1
                continue
            mu, sd = float(brain.mean()), float(brain.std())
            thr = mu + K * sd
            enh = brain[brain > thr]
            if enh.size <= MIN_ENH_VOXELS:
                n_skipped += 1
                continue
            n_slices += 1
            brain_mu.append(mu)
            brain_sd.append(sd)
            thresholds.append(thr)
            enh_frac.append(enh.size / brain.size)
            frac_clipped.append(float((enh >= CLAMP_CEILING).mean()))

    if n_slices == 0:
        sys.exit("Tidak ada slice yang memenuhi syarat (>10 voxel enhancement).")

    t = np.array(thresholds)
    fc = np.array(frac_clipped)

    print(f"Sumber                      : {arg}")
    print(f"Berkas dibaca               : {len(files)}")
    print(f"Slice berkontribusi         : {n_slices}   (dilewati: {n_skipped})")
    print()
    print(f"Rentang intensitas teramati : [{vmin:+.4f}, {vmax:+.4f}]")
    print(f"metric_config.json          : [{CFG_MIN:+.4f}, {CFG_MAX:+.4f}]"
          f"   data_range {CFG_MAX - CFG_MIN:.4f}")
    print()
    print(f"mu otak    (rerata slice)   : {np.mean(brain_mu):+.4f}"
          f"   sd antar-slice {np.std(brain_mu):.4f}")
    print(f"sigma otak (rerata slice)   : {np.mean(brain_sd):.4f}")
    print(f"Ambang mu + {K}*sigma        : rerata {t.mean():.4f}"
          f"   min {t.min():.4f}   max {t.max():.4f}")
    print(f"Plafon clamp LPIPS          : {CLAMP_CEILING:.4f}")
    print()
    print(f"Ukuran mask enhancement     : {100 * np.mean(enh_frac):.2f}% voxel otak"
          f"   (median {100 * np.median(enh_frac):.2f}%)")
    print(f"Slice dgn ambang > plafon   : {100 * (t > CLAMP_CEILING).mean():.1f}%")
    print(f"Voxel enhancement terpotong ke 1.0 : {100 * fc.mean():.2f}%"
          f"   (median {100 * np.median(fc):.2f}%)")
    print()

    if fc.mean() > 0.95:
        print(">> TEMUAN M8 TERKONFIRMASI.")
        print("   Acuan tersaturasi hampir seluruhnya di dalam mask, sehingga")
        print("   LPIPS_Enh mengukur amplitudo enhancement, bukan tekstur.")
        print("   Terapkan paragraf LPIPS_Enh di Methods, blok [M8-a] (Sec 4.5)")
        print("   dan blok [M8-b] (Sec 4.9).")
    elif fc.mean() < 0.20:
        print(">> TEMUAN M8 TIDAK BERLAKU.")
        print("   Acuan tidak tersaturasi. Buang kalimat 'Second, the mask")
        print("   threshold...' dari Methods, dan pertahankan blok [M3-b] dan")
        print("   [M3-c] versi lama. M3 dan M7 tetap berlaku penuh.")
    else:
        print(">> PARSIAL.")
        print("   Laporkan angka di atas; klaim harus disesuaikan menurut")
        print("   persentase sebenarnya, bukan diterima atau ditolak bulat-bulat.")


if __name__ == "__main__":
    main()
