#!/usr/bin/env python3
import os, glob, sys
import ehrapy as ep

EXPECTED = [
    "crea_baseline","crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
    "cysc_baseline","cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
    "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
    "Sex_norm","age_years_at_first_op","n_ops","highest_AKI_stage_0_7","duration_hours","AKI_any_0_7",
]

BASE = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
CANDIDATES = [
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad/ops_with_patient_features.h5ad"),
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad/ops_with_patient_features_ehrapy_enriched.h5ad"),
]

def find_h5ad():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(BASE, "**", "*.h5ad"), recursive=True)
    if not hits:
        return None
    # bevorzuge Dateien, die wie unsere heißen
    for name in ("ops_with_patient_features_ehrapy_enriched.h5ad", "ops_with_patient_features.h5ad"):
        for h in hits:
            if os.path.basename(h) == name:
                return h
    return hits[0]

def main():
    path = find_h5ad()
    if not path:
        print("Kein .h5ad gefunden. Bitte Pfad prüfen.")
        print("Tipp: existiert dieser Ordner?:", BASE)
        sys.exit(1)
    print("Lade:", path)
    adata = ep.io.read_h5ad(path)
    cols = list(adata.obs.columns)
    print("\nSpalten in adata.obs ({}):".format(len(cols)))
    print(cols)

    expected = set(EXPECTED)
    have = set(cols)
    missing = [c for c in EXPECTED if c not in have]
    extra   = [c for c in cols if c not in expected]

    print("\nFehlend gegenüber CSV-Liste:", missing if missing else "— keine —")
    print("Zusätzliche Spalten im h5ad:", extra[:25], "..." if len(extra) > 25 else "")

    # kleine Zusatzinfos
    for col in ("Sex_norm","AKI_any_0_7","AKI_linked_0_7"):
        if col in adata.obs.columns:
            nn = adata.obs[col].notna().sum()
            uq = adata.obs[col].nunique(dropna=True)
            print(f"{col}: non-null={nn}, unique={uq}, head={adata.obs[col].dropna().astype(str).head(3).tolist()}")

if __name__ == "__main__":
    main()
