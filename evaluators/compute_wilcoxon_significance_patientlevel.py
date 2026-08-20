import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# =========================
# ADD PROJECT ROOT
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config import CONFIG


EXP_ROOT = CONFIG["experiment_root"]
N_FOLDS = 5

# Jumlah pasien di held-out test set (FIXED, sama untuk semua fold/eksperimen).
# Dipakai untuk memvalidasi bahwa merge champion<->baseline tidak diam-diam
# kehilangan pasien (mismatch patient_id, fold hilang, dsb).
EXPECTED_N_PATIENTS = 10

# =========================
# Root terpisah untuk baseline SOTA re-implementasi (pGAN, ResViT, DDResUNet).
# Kalau ketiganya disimpan di folder eksperimen yang SAMA dengan model
# internal, biarkan SOTA_EXP_ROOT = EXP_ROOT. Kalau disimpan di direktori
# lain (mis. "experiments_baselines/"), ganti path ini.
# =========================

SOTA_EXP_ROOT = EXP_ROOT

# =========================
# ROSTER EKSPERIMEN
# Isi prefix folder eksperimen TANPA "_foldN".
# contoh folder asli: experiments/EXP-602_EDM_fold0 -> prefix "EXP-602_EDM"
#
# CHAMPIONS       : kandidat model terbaik (hasil ranking Table 1 tervalidasi
#                   + cek stabilitas antar fold + trade-off metrik sekunder).
#                   Bisa lebih dari satu kalau ada >1 kandidat flagship yang
#                   unggul di metrik primer berbeda.
# INTERNAL_BASELINES : varian ablasi internal yang masing-masing berbeda dari
#                   champion pada SATU komponen terkontrol (arsitektur/loss),
#                   dipakai untuk menjustifikasi kontribusi tiap komponen.
# SOTA_BASELINES  : metode literatur eksternal yang di-reimplementasi dengan
#                   protokol evaluasi identik (split, preprocessing, kode
#                   metrik yang sama), dipakai untuk klaim benchmark.
# =========================

CHAMPIONS = [
    "EXP-602_EDM",                  # EDM-v2 — flagship structural/enhancement-region fidelity
    "EXP-703_EDM_GAN_VGG_LPIPS",    # EDMSynth — flagship whole-ROI fidelity
]

INTERNAL_BASELINES = [
    "EXP-100_UNET_L1",
    "EXP-202_ATTENTION_UNET_L1",
    "EXP-204_DENSE_UNET_L1",
    "EXP-303_T1_T2_FLAIR_TO_T1CE",
    "EXP-402_MULTI_ENCODER_STEP1_T1_T2_F",
    "EXP-500_GUIDED_RESATTENTION_T1T2F",
    "EXP-601_EDM",                       # EDM-v1, ablasi kunci: tanpa structural constraint
    "EXP-713_MULTISCALE_CROSS_ATTENTION",
    "EXP-813_StratifiedEnh_Sharpness",
    "EXP-816_CalibratedGate_BalancedSeg",
]

SOTA_BASELINES = [
    "EXP-PGAN01_pGAN_baseline",
    "EXP-RESVIT_Dalmaz2022",        # perhatikan: RESVIT huruf besar semua, BUKAN "RESViT"
    "EXP-DDRESUNET_OsmanTamam2023",
]

# =========================
# FAMILY PERBANDINGAN -- DIPISAH SECARA EKSPLISIT.
#
# Perbaikan dari versi sebelumnya: baseline internal dan baseline SOTA TIDAK
# digabung jadi satu list "BASELINES" sebelum diuji. Keduanya adalah dua
# klaim ilmiah yang berbeda:
#   - champion_vs_internal -> menjustifikasi pilihan desain (ablation study)
#   - champion_vs_sota     -> menjustifikasi klaim benchmark/SOTA
# Setiap family dikoreksi multiple-comparison SENDIRI-SENDIRI (per champion
# per family), supaya FDR satu klaim tidak ikut dipengaruhi jumlah tes di
# klaim lain yang tidak relevan.
#
# Family kosong (list baseline kosong) otomatis dilewati -- tidak perlu flag
# on/off terpisah seperti versi sebelumnya.
# =========================

COMPARISON_FAMILIES = {
    "champion_vs_internal": INTERNAL_BASELINES,
    "champion_vs_sota": SOTA_BASELINES,
}

# Nama family untuk perbandingan champion-vs-champion (kalau CHAMPIONS berisi
# lebih dari satu kandidat). Dihitung SATU ARAH saja per pasangan (tidak
# dua kali seperti versi sebelumnya) dan dikoreksi sebagai family sendiri,
# terpisah dari dua family di atas.
CHAMPION_PAIR_FAMILY = "champion_vs_champion"

# =========================
# Root eksperimen per-prefix, supaya load_patient_level tahu harus mencari
# di EXP_ROOT (untuk model internal) atau SOTA_EXP_ROOT (untuk SOTA).
# =========================

ROOT_BY_PREFIX = {name: SOTA_EXP_ROOT for name in SOTA_BASELINES}
DEFAULT_ROOT = EXP_ROOT


# Metrik yang diuji. "sharpness_gt" SENGAJA DIHILANGKAN dari versi sebelumnya:
# itu adalah nilai ground truth yang identik lintas model (bukan output
# model), sehingga selisihnya selalu nol dan tidak pernah bisa menghasilkan
# uji yang bermakna -- hanya baris kosong yang membebani tabel hasil.
METRICS = [
    "psnr_roi",
    "ssim_roi",
    "mae",
    "calibration_bias",
    "variance_ratio",
    "psnr_enh",
    "ssim_enh",
    "grad_enh",
    "lpips",
    "lpips_enh",
    "sharpness_pred",
    "sharpness_ratio",
    "sharpness_ratio_enh",
]

# Arah "lebih baik" per metrik -- dipakai HANYA untuk melabeli kolom
# `champion_better` (interpretasi), TIDAK mengubah cara Wilcoxon dihitung
# (Wilcoxon tetap two-sided, menguji apakah median selisih berbeda dari 0).
METRIC_DIRECTION = {
    "psnr_roi": "higher",
    "ssim_roi": "higher",
    "mae": "lower",
    "calibration_bias": "closer_to_zero",
    "variance_ratio": "closer_to_one",
    "psnr_enh": "higher",
    "ssim_enh": "higher",
    "grad_enh": "lower",
    "lpips": "lower",
    "lpips_enh": "lower",
    "sharpness_pred": "higher",
    "sharpness_ratio": "closer_to_one",
    "sharpness_ratio_enh": "closer_to_one",
}

# Parameter Wilcoxon dibuat EKSPLISIT (bukan mengandalkan default tersembunyi
# scipy yang bisa berubah antar versi) supaya reproducible dan mudah
# dilaporkan di bagian Methods.
WILCOXON_ZERO_METHOD = "wilcox"   # buang pasangan dengan selisih persis 0
WILCOXON_METHOD = "auto"          # exact bila n kecil & tanpa ties, asymptotic bila ada ties
WILCOXON_ALTERNATIVE = "two-sided"

# "fdr_bh" (Benjamini-Hochberg, direkomendasikan untuk banyak metrik eksploratif)
# atau "holm" (lebih konservatif, family-wise error control ketat)
CORRECTION_METHOD = "fdr_bh"


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def who_is_better(champion_mean, baseline_mean, direction):
    """Label arah kemenangan berdasarkan mean, murni untuk interpretasi.
    Tidak dipakai untuk menghitung p-value -- Wilcoxon tetap two-sided."""
    if pd.isna(champion_mean) or pd.isna(baseline_mean):
        return None
    if direction == "higher":
        return "champion" if champion_mean > baseline_mean else "baseline"
    if direction == "lower":
        return "champion" if champion_mean < baseline_mean else "baseline"
    if direction == "closer_to_zero":
        return "champion" if abs(champion_mean) < abs(baseline_mean) else "baseline"
    if direction == "closer_to_one":
        return "champion" if abs(champion_mean - 1) < abs(baseline_mean - 1) else "baseline"
    return None


def load_patient_level(exp_prefix):
    """
    Muat holdout_patient_metrics.csv dari N_FOLDS fold eksperimen ini, lalu
    rata-ratakan tiap metrik per patient_id lintas fold.

    holdout_test bersifat FIXED/sama untuk semua fold (bukan bagian dari
    5-fold split train/val), jadi patient_id yang sama muncul di kelima
    fold. Merata-ratakan lintas fold per pasien membuat setiap pasien = 1
    observasi independen -> n=EXPECTED_N_PATIENTS, bukan n=EXPECTED_N_PATIENTS*N_FOLDS
    (yang akan melanggar independensi karena 1 pasien dihitung berkali-kali).

    Perbaikan dari versi sebelumnya:
      1. patient_id dipaksa jadi string di awal supaya merge antar eksperimen
         tidak gagal diam-diam akibat mismatch tipe data (int vs str).
      2. Baris duplikat per (patient_id, fold) dianggap FATAL (raise), bukan
         lolos tanpa terdeteksi -- nunique() fold saja tidak menangkap kasus
         ini.
      3. Pasien yang fold-nya tidak lengkap (<N_FOLDS) DIKELUARKAN dari
         analisis (bukan tetap dirata-rata dari fold yang tersisa), supaya
         semua pasien yang masuk uji punya presisi rata-rata yang sama
         (rata-rata dari jumlah replikasi training yang sama).
    """

    root = ROOT_BY_PREFIX.get(exp_prefix, DEFAULT_ROOT)

    fold_dfs = []

    for fold in range(N_FOLDS):

        path = os.path.join(
            root, f"{exp_prefix}_fold{fold}", "holdout_patient_metrics.csv"
        )

        if not os.path.exists(path):
            print(f"[WARN] {path} tidak ditemukan, fold ini dilewati untuk {exp_prefix}")
            continue

        df = pd.read_csv(path)
        df["fold"] = fold
        fold_dfs.append(df)

    if not fold_dfs:
        raise FileNotFoundError(
            f"Tidak ada data fold ditemukan untuk {exp_prefix} (dicek di {root}). "
            f"Pastikan holdout_patient_metrics.csv per-pasien sudah dihasilkan "
            f"dengan pipeline evaluasi yang sama untuk eksperimen ini -- bukan "
            f"hanya fold_aggregated_results_*.csv / perfold_raw_results_*.csv "
            f"yang levelnya per-fold, bukan per-pasien."
        )

    all_df = pd.concat(fold_dfs, ignore_index=True)

    # FIX: samakan tipe data patient_id SEBELUM groupby/merge apa pun.
    all_df["patient_id"] = all_df["patient_id"].astype(str)

    # FIX: baris duplikat per (patient_id, fold) = kesalahan data, hentikan.
    dup_counts = all_df.groupby(["patient_id", "fold"]).size()
    duplicated = dup_counts[dup_counts > 1]
    if len(duplicated) > 0:
        raise ValueError(
            f"[FATAL] {exp_prefix}: ditemukan baris duplikat untuk "
            f"(patient_id, fold) berikut -- perbaiki data sumber sebelum "
            f"melanjutkan uji statistik:\n{duplicated.to_string()}"
        )

    n_folds_found = all_df.groupby("patient_id")["fold"].nunique()
    incomplete = n_folds_found[n_folds_found < N_FOLDS]

    if len(incomplete) > 0:
        print(
            f"[WARN] {exp_prefix}: pasien berikut tidak punya data lengkap "
            f"{N_FOLDS} fold dan akan DIKELUARKAN dari analisis (bukan "
            f"dirata-rata dari fold yang tersisa):"
        )
        print(incomplete.to_string())
        complete_ids = n_folds_found[n_folds_found == N_FOLDS].index
        all_df = all_df[all_df["patient_id"].isin(complete_ids)]

    if all_df["patient_id"].nunique() == 0:
        raise ValueError(
            f"[FATAL] {exp_prefix}: tidak ada pasien dengan data {N_FOLDS} "
            f"fold lengkap setelah filtering."
        )

    patient_level = (
        all_df.drop(columns=["fold"])
        .groupby("patient_id")
        .mean(numeric_only=True)
        .reset_index()
    )

    n_found = patient_level["patient_id"].nunique()
    if n_found != EXPECTED_N_PATIENTS:
        print(
            f"[WARN] {exp_prefix}: jumlah pasien akhir = {n_found}, "
            f"berbeda dari EXPECTED_N_PATIENTS = {EXPECTED_N_PATIENTS}. "
            f"Cek kelengkapan data sebelum menafsirkan hasil uji sebagai n=10 penuh."
        )

    return patient_level


def run_pairwise_wilcoxon(champ_df, base_df, champion_name, baseline_name, family_name):
    """Uji satu pasangan (champion, baseline) untuk seluruh METRICS.

    Perbaikan dari versi sebelumnya:
      - patient_id dipaksa string sebelum merge.
      - merge pakai how='inner' TAPI jumlah hasilnya divalidasi terhadap
        EXPECTED_N_PATIENTS dan di-warning eksplisit kalau tidak cocok
        (sebelumnya silent).
      - parameter wilcoxon() dibuat eksplisit (zero_method, method,
        alternative), bukan default tersembunyi.
      - jumlah pasangan dengan selisih persis nol (yang dibuang otomatis
        oleh zero_method='wilcox' di dalam scipy) dihitung dan dilaporkan
        terpisah (n_zero_diff_excluded), supaya n_pairs yang dilaporkan
        tidak menyesatkan.
      - kolom champion_better ditambahkan sebagai bantuan interpretasi arah
        kemenangan (murni deskriptif, tidak memengaruhi p-value).
    """

    champ_df = champ_df.copy()
    base_df = base_df.copy()
    champ_df["patient_id"] = champ_df["patient_id"].astype(str)
    base_df["patient_id"] = base_df["patient_id"].astype(str)

    merged = pd.merge(
        champ_df, base_df, on="patient_id", how="inner", suffixes=("_champ", "_base")
    )

    if len(merged) != EXPECTED_N_PATIENTS:
        print(
            f"[WARN] {champion_name} vs {baseline_name} ({family_name}): "
            f"merged n_patients={len(merged)}, diharapkan {EXPECTED_N_PATIENTS}. "
            f"champ_df n={len(champ_df)}, base_df n={len(base_df)}. "
            f"Kemungkinan penyebab: mismatch patient_id atau fold tidak lengkap."
        )

    records = []

    for m in METRICS:

        col_champ = f"{m}_champ"
        col_base = f"{m}_base"

        if col_champ not in merged.columns or col_base not in merged.columns:
            records.append({
                "Comparison_Family": family_name,
                "Champion": champion_name,
                "Baseline": baseline_name,
                "Metric": m,
                "Metric_Direction": METRIC_DIRECTION.get(m, ""),
                "n_pairs": 0,
                "n_zero_diff_excluded": 0,
                "champion_mean": np.nan,
                "baseline_mean": np.nan,
                "champion_better": None,
                "wilcoxon_stat": np.nan,
                "p_value": np.nan,
            })
            continue

        x = merged[col_champ].to_numpy(dtype=float)
        y = merged[col_base].to_numpy(dtype=float)

        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        n_pairs = len(x)
        diff = x - y
        n_zero_diff = int(np.sum(diff == 0))

        if n_pairs < 3 or np.all(diff == 0):
            stat, p = np.nan, np.nan
        else:
            try:
                stat, p = wilcoxon(
                    x, y,
                    zero_method=WILCOXON_ZERO_METHOD,
                    method=WILCOXON_METHOD,
                    alternative=WILCOXON_ALTERNATIVE,
                )
            except ValueError as e:
                print(
                    f"[WARN] Wilcoxon gagal untuk {champion_name} vs "
                    f"{baseline_name} [{m}]: {e}"
                )
                stat, p = np.nan, np.nan

        champ_mean = np.mean(x) if n_pairs else np.nan
        base_mean = np.mean(y) if n_pairs else np.nan
        direction = METRIC_DIRECTION.get(m, "higher")

        records.append({
            "Comparison_Family": family_name,
            "Champion": champion_name,
            "Baseline": baseline_name,
            "Metric": m,
            "Metric_Direction": direction,
            "n_pairs": n_pairs,
            "n_zero_diff_excluded": n_zero_diff,
            "champion_mean": champ_mean,
            "baseline_mean": base_mean,
            "champion_better": who_is_better(champ_mean, base_mean, direction),
            "wilcoxon_stat": stat,
            "p_value": p,
        })

    return records


def main():

    if not CHAMPIONS:
        raise ValueError(
            "CHAMPIONS masih kosong. Isi prefix folder eksperimen kandidat "
            "champion (hasil ranking Table 1 tervalidasi) di bagian atas script."
        )

    active_families = {
        name: baseline_list
        for name, baseline_list in COMPARISON_FAMILIES.items()
        if baseline_list
    }

    if not active_families and len(CHAMPIONS) < 2:
        raise ValueError(
            "INTERNAL_BASELINES dan SOTA_BASELINES kosong, dan CHAMPIONS "
            "cuma berisi satu model -- tidak ada apa pun untuk dibandingkan."
        )

    # cache supaya tiap eksperimen cuma dimuat sekali walau dipakai berkali-kali
    patient_cache = {}

    def get_patient_df(prefix):
        if prefix not in patient_cache:
            patient_cache[prefix] = load_patient_level(prefix)
        return patient_cache[prefix]

    all_records = []

    for champion in CHAMPIONS:

        champ_df = get_patient_df(champion)

        # --- family champion vs internal baseline, dan champion vs SOTA ---
        for family_name, baseline_list in active_families.items():
            for baseline in baseline_list:
                base_df = get_patient_df(baseline)
                all_records.extend(
                    run_pairwise_wilcoxon(champ_df, base_df, champion, baseline, family_name)
                )

        # --- family champion vs champion (kalau ada >1 kandidat flagship) ---
        # Dihitung SATU ARAH per pasangan saja (dedup lewat urutan index di
        # CHAMPIONS), tidak dua kali seperti versi sebelumnya.
        for baseline in CHAMPIONS:
            if baseline == champion:
                continue
            if CHAMPIONS.index(champion) >= CHAMPIONS.index(baseline):
                continue
            base_df = get_patient_df(baseline)
            all_records.extend(
                run_pairwise_wilcoxon(champ_df, base_df, champion, baseline, CHAMPION_PAIR_FAMILY)
            )

    result_df = pd.DataFrame(all_records)

    # =========================
    # KOREKSI MULTIPLE COMPARISON
    # Dilakukan TERPISAH per (Champion, Comparison_Family) -- BUKAN per
    # Champion saja seperti versi sebelumnya. Ini mencegah klaim
    # "champion vs SOTA" ikut dikoreksi bersama jumlah tes dari klaim
    # "champion vs internal" atau "champion vs champion" yang tidak relevan
    # untuk pertanyaan ilmiah yang sama.
    # =========================

    result_df["p_value_corrected"] = np.nan
    result_df["significance_corrected"] = ""

    group_cols = ["Champion", "Comparison_Family"]
    for (champion, family), _ in result_df.groupby(group_cols):

        mask = (
            (result_df["Champion"] == champion)
            & (result_df["Comparison_Family"] == family)
            & result_df["p_value"].notna()
        )

        if mask.sum() == 0:
            continue

        pvals = result_df.loc[mask, "p_value"].to_numpy()

        _, pvals_corrected, _, _ = multipletests(
            pvals, alpha=0.05, method=CORRECTION_METHOD
        )

        result_df.loc[mask, "p_value_corrected"] = pvals_corrected
        result_df.loc[mask, "significance_corrected"] = [stars(p) for p in pvals_corrected]

    result_df["significance_raw"] = result_df["p_value"].apply(stars)

    save_path = os.path.join(EXP_ROOT, "wilcoxon_significance_patientlevel.csv")
    result_df.to_csv(save_path, index=False)

    print(f"\n===== WILCOXON SIGNED-RANK TEST (per-patient paired, n<={EXPECTED_N_PATIENTS}, koreksi: {CORRECTION_METHOD}) =====\n")
    print("Comparison families dijalankan:", list(active_families.keys()) + (
        [CHAMPION_PAIR_FAMILY] if len(CHAMPIONS) > 1 else []
    ))
    print(result_df.to_string(index=False))

    print("\nSaved:", save_path)
    print("\nCatatan penting:")
    print("- p_value       = mentah, per pasien (n<=%d)." % EXPECTED_N_PATIENTS)
    print("- p_value_corrected = setelah koreksi %s, DIHITUNG TERPISAH per "
          "(Champion, Comparison_Family). Gunakan kolom ini untuk klaim di paper, "
          "bukan significance_raw." % CORRECTION_METHOD)
    print("- n_pairs       = jumlah pasangan valid (non-NaN) setelah merge.")
    print("- n_zero_diff_excluded = jumlah pasangan dengan selisih persis 0 yang "
          "dibuang otomatis oleh zero_method='wilcox' di dalam scipy -- n efektif "
          "yang dipakai uji adalah (n_pairs - n_zero_diff_excluded).")
    print("- champion_better = label arah kemenangan berdasarkan mean (murni "
          "deskriptif untuk interpretasi, TIDAK memengaruhi p-value).")
    print("- Kalau n_pairs != %d secara konsisten, cek [WARN] di atas -- "
          "kemungkinan mismatch patient_id atau fold tidak lengkap." % EXPECTED_N_PATIENTS)
    print("- 'sharpness_gt' sengaja tidak diuji karena identik lintas model "
          "(ground truth), sehingga tidak pernah bisa signifikan secara desain.")
    print("\n[PENTING] Kalau load_patient_level gagal (FileNotFoundError/ValueError):")
    print("(1) cek apakah holdout_patient_metrics.csv per-pasien memang sudah")
    print("    dihasilkan untuk eksperimen tsb dengan pipeline evaluasi yang sama,")
    print("(2) cek apakah ROOT_BY_PREFIX / SOTA_EXP_ROOT menunjuk ke folder yang benar,")
    print("(3) cek apakah nama folder aktual di disk cocok persis dengan prefix di roster,")
    print("(4) ValueError duplikat baris berarti ada bug di evaluate_experiment_paper.py")
    print("    yang menulis lebih dari satu baris untuk pasien yang sama di fold yang sama.")


if __name__ == "__main__":
    main()
