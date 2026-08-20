#!/usr/bin/env python3
"""
bangun_reproduce.py

Membangun folder reproduce/ di dalam deposit. Setiap eksperimen mendapat satu
folder mandiri berisi arsipnya yang dikembalikan ke NAMA KANONIK, sehingga
pernyataan import di skrip train tetap berlaku tanpa disunting.

Jalankan dari akar folder deposit:
    python bangun_reproduce.py
    python bangun_reproduce.py --dry-run

Hasil:
    reproduce/EXP602/configs/config.py
    reproduce/EXP602/models/edm_synth.py
    reproduce/EXP602/losses/loss_registry.py
    reproduce/EXP602/train.py
    reproduce/EXP602/RUN.md
"""

import argparse, hashlib, json, os, shutil, sys

# =====================================================================
# PETA. kunci = identifier, nilai = { tujuan_kanonik : sumber }
# Sumber "None" berarti lubang yang belum tertutup.
# =====================================================================
MAP = {
 "EXP100": {
   "configs/config.py":        "configs/config_exp100_evaluasi_metric.py",
   "models/unet_baseline.py":  "models/unet_baseline.py",
   "models/model_registry.py": "models/model_registry.py",
   "losses/loss_registry.py":  "losses/loss_registry.py",
   "train.py":                 "train_any_loss.py",
 },
 "EXP202": {
   "configs/config.py":        "configs/config_exp202_evaluasi_metric.py",
   "models/attention_unet.py": "models/attention_unet.py",
   "models/model_registry.py": "models/model_registry.py",
   "losses/loss_registry.py":  "losses/loss_registry.py",
   "train.py":                 "train_any_loss.py",
 },
 "EXP204": {
   "configs/config.py":        "configs/config_exp204_evaluasi_metric.py",
   "models/dense_unet.py":     "models/dense_unet.py",
   "models/model_registry.py": "models/model_registry.py",
   "losses/loss_registry.py":  "losses/loss_registry.py",
   "train.py":                 "train_any_loss.py",
 },
 "EXP303": {
   "configs/config.py":            "configs/config_exp303_eval_metric.py",
   "models/resattention_unet.py":  "models/resattention_unet_exp303.py",
   "models/model_registry.py":     "models/model_registry.py",
   "losses/loss_registry.py":      "losses/loss_registry.py",
   "train.py":                     "train_early_fusion.py",
 },
 "EXP402": {
   "configs/config.py":         "configs/config_exp400_feature_fusion.py",
   "models/dmec_net_step1.py":  "models/dmec_net_step1.py",
   "models/model_registry.py":  "models/model_registry.py",
   "losses/loss_registry.py":   "losses/loss_registry.py",
   "train.py":                  "train_feature_fusion.py",
 },
 "EXP500": {
   "configs/config.py":                   "configs/config_exp500_guided_resattention.py",
   "models/guided_resattention_unet.py":  "models/guided_resattention_unet.py",
   "models/model_registry.py":            "models/model_registry.py",
   "losses/loss_registry.py":             "losses/loss_registry.py",
   "train.py":                            "train_guided_resattention.py",
 },
 "EXP601": {
   "configs/config.py":        "configs/config_exp601.py",
   "models/edm_synth.py":      "models/edm_synth_exp601_cek.py",
   "models/model_registry.py": "models/model_registry_exp601_exp602.py",
   "losses/loss_registry.py":  "losses/loss_registry_exp601.py",
   "losses/edm_loss.py":       "losses/edm_loss.py",
   "train.py":                 "train_edm_exp602.py",
 },
 "EXP602": {
   "configs/config.py":           "configs/config_exp602.py",
   "models/edm_synth.py":         "models/edm_synth_exp602.py",
   "models/model_registry.py":    "models/model_registry_exp601_exp602.py",
   "models/resattention_unet.py": "models/resattention_unet_exp303.py",
   "models/resunet.py":           "models/resunet.py",
   "losses/loss_registry.py":     "losses/loss_registry_exp602.py",
   "losses/edm_loss.py":          "losses/edm_loss.py",
   "train.py":                    "train_edm_exp602.py",
 },
 "EXP703": {
   "configs/config.py":       "configs/config_exp703rerun3.py",
   "models/edm_synth.py":     "models/edm_synth_exp703rerun3.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp703rerun3.py",
   "train.py":                "train_exp703rerun3.py",
 },
 "EXP706": {
   "configs/config.py":       "configs/config_exp706_eval_metric.py",
   "models/edm_synth.py":     "models/edm_synth_exp706_eval_metric.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_sd_exp706.py",
   "train.py":                "train_gan_juga_vgg.py",
 },
 "EXP713": {
   "configs/config.py":       "configs/config_exp713.py",
   "models/edm_synth.py":     "models/edm_synth_exp713.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp713.py",
   "train.py":                "train_exp713.py",
 },
 "EXP716A": {
   "configs/config.py":       "configs/config_exp716A.py",
   "models/edm_synth.py":     "models/edm_synth_exp716A.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp716A.py",
   "train.py":                "train_gan_juga_vgg.py",
 },
 "EXP803": {
   "configs/config.py":       "configs/config_exp803.py",
   "models/edm_synth.py":     "models/edm_synth_exp803_sd_exp805.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp803.py",
   "train.py":                "train_exp803.py",
 },
 "EXP806": {
   "configs/config.py":       "configs/config_exp806.py",
   "models/edm_synth.py":     "models/edm_synth_exp806.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp806.py",
   "train.py":                "train_exp806.py",
 },
 "EXP813": {
   "configs/config.py":       "configs/config_exp813.py",
   "models/edm_synth.py":     "models/edm_synth_exp813.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp813.py",
   "train.py":                "train_exp813.py",
 },
 "EXP815": {
   "configs/config.py":       "configs/config_exp815.py",
   "models/edm_synth.py":     "models/edm_synth_exp815.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp815.py",
   "train.py":                "train_exp815.py",
 },
 "EXP816": {
   "configs/config.py":       "configs/config_exp816.py",
   "models/edm_synth.py":     "models/edm_synth_exp816.py",
   "models/model_registry.py":"models/model_registry.py",
   "losses/loss_registry.py": "losses/loss_registry_exp816.py",
   "train.py":                "train_exp816.py",
 },
}

# Catatan yang WAJIB tercetak di RUN.md tiap eksperimen.
CATATAN = {
 "EXP100": "Only the configuration was archived for this run. The model definition is the canonical file, unchanged across the baseline experiments. The training script is train_any_loss.py. A second file, train_baseline_awal.py, is identical to it except for one commented-out import line, so the two cannot be distinguished and produce the same result.",
 "EXP202": "Only the configuration was archived for this run. The model definition is the canonical file, unchanged across the baseline experiments. The training script is train_any_loss.py. A second file, train_baseline_awal.py, is identical to it except for one commented-out import line, so the two cannot be distinguished and produce the same result.",
 "EXP204": "Only the configuration was archived for this run. The model definition is the canonical file, unchanged across the baseline experiments. The training script is train_any_loss.py. A second file, train_baseline_awal.py, is identical to it except for one commented-out import line, so the two cannot be distinguished and produce the same result.",
 "EXP303": "Only the configuration and the model definition were archived for this run.",
 "EXP402": "No file was archived under this identifier. The run altered the configuration only. The configuration file shown here also defines EXP-400, EXP-401 and EXP-403 to EXP-408, none of which is reported in the manuscript.",
 "EXP500": "Only the configuration was archived for this run.",
 "EXP601": "The training script is byte-identical to the one used for EXP-602.",
 "EXP602": "The training script is byte-identical to the one used for EXP-601.",
 "EXP703": "Three archived versions exist. The version reproduced here matches config_snapshot.json on early_stop_patience 40, amp False and lr_D 5e-05. The two earlier versions differ and were not used.",
 "EXP706": "The archived loss registry was named loss_registry_sd_exp706.py and was renamed to loss_registry.py at run time. The training script is inferred from file timestamps and is not archived under this identifier.",
 "EXP713": "The archived files ending in _cek are byte-identical duplicates and are not reproduced here.",
 "EXP716A": "The training script is inferred from file timestamps and is not archived under this identifier.",
 "EXP803": "The archived model file is named edm_synth_exp803_sd_exp805.py. The suffix refers to a later exploration variant that is not reported in the manuscript.",
}

SHARED = ["datasets", "utils", "evaluators"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="akar folder deposit, bawaan = lokasi skrip ini")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    out = os.path.join(root, "reproduce")
    if os.path.exists(out) and not a.dry_run:
        sys.exit("reproduce/ sudah ada. Hapus dulu.")

    total_ok = total_gap = total_hilang = 0
    laporan = []

    for exp, files in MAP.items():
        base = os.path.join(out, exp)
        baris = []
        for kanonik, sumber in files.items():
            if sumber is None:
                total_gap += 1
                baris.append((kanonik, "TIDAK ADA", ""))
                continue
            src = os.path.join(root, sumber)
            if not os.path.isfile(src):
                total_hilang += 1
                baris.append((kanonik, "HILANG: " + sumber, ""))
                continue
            dst = os.path.join(base, kanonik)
            if not a.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            total_ok += 1
            baris.append((kanonik, sumber, md5(src)[:8]))

        # RUN.md
        md = []
        md.append("# %s" % exp.replace("EXP", "EXP-"))
        md.append("")
        md.append("Self-contained snapshot of the code used for this run.")
        md.append("")
        md.append("## Provenance")
        md.append("")
        md.append("| File in this folder | Source file in the repository | md5 |")
        md.append("|---|---|---|")
        for k, s, h in baris:
            md.append("| `%s` | `%s` | %s |" % (k, s, h))
        md.append("")
        if exp in CATATAN:
            md.append("## Note")
            md.append("")
            md.append(CATATAN[exp])
            md.append("")
        md.append("## How to run")
        md.append("")
        md.append("```bash")
        md.append("cd reproduce/%s" % exp)
        md.append("PYTHONPATH=../.. python train.py")
        md.append("```")
        md.append("")
        md.append("The current directory is searched before the repository root, so")
        md.append("`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to")
        md.append("the files in this folder. The shared modules `%s` resolve" % ", ".join(SHARED))
        md.append("from the repository root. No import statement was edited.")
        md.append("")
        md.append("The imaging data are not included. The run requires the restricted")
        md.append("archive in the layout described in `docs/DATA_LAYOUT.md`.")
        if not a.dry_run:
            os.makedirs(base, exist_ok=True)
            open(os.path.join(base, "RUN.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

        laporan.append((exp, baris))

    # indeks
    idx = ["# reproduce/", "",
           "One self-contained folder per configuration evaluated on the locked",
           "hold-out cohort. Each folder restores the archived files to their canonical",
           "names, so that the import statements of the training script apply unchanged.",
           "", "| Configuration | Folder | Training script |", "|---|---|---|"]
    for exp, baris in laporan:
        tr = dict((k, s) for k, s, _ in baris).get("train.py", "TIDAK ADA")
        idx.append("| %s | `reproduce/%s/` | `%s` |" % (exp.replace("EXP", "EXP-"), exp, tr))
    if not a.dry_run:
        open(os.path.join(out, "README.md"), "w", encoding="utf-8").write("\n".join(idx) + "\n")

    print("=" * 62)
    for exp, baris in laporan:
        bad = [b for b in baris if b[1].startswith(("HILANG", "TIDAK ADA"))]
        tag = "OK " if not bad else "!! "
        print("%s%-8s %d berkas%s" % (tag, exp, len(baris),
              "" if not bad else "   masalah: " + ", ".join(b[0] for b in bad)))
    print("=" * 62)
    print("disalin: %d   lubang: %d   hilang: %d" % (total_ok, total_gap, total_hilang))
    if a.dry_run:
        print("DRY RUN. Nol berkas ditulis.")
    else:
        print("Selesai. Lihat reproduce/README.md")


if __name__ == "__main__":
    main()
