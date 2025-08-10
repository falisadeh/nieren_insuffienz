# 00_load_ehrapy.py
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import read_h5ad

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
P = Path(BASE)

FILES = {
    "master_csv": P/"analytic_ops_master.csv",
    "prior_0_7_csv": P/"aki_prior_0_7.csv",
    "patient_summary_csv": P/"analytic_patient_summary.csv",
    "h5ad": P/"aki_ops_master.h5ad",
}

def ok(path: Path) -> str:
    return "✅" if path.exists() else "❌"

print("=== Dateien prüfen ===")
for k, p in FILES.items():
    print(f"{k:20s} {ok(p)}  {p}")

# ---- CSV: analytic_ops_master.csv ----
print("\n=== analytic_ops_master.csv ===")
df = pd.read_csv(FILES["master_csv"], sep=";")
# Datumsfelder tolerant parsen, falls vorhanden
for c in ["Surgery_Start","Surgery_End","AKI_Start"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

print("Form:", df.shape)
print("Spalten (Ausschnitt):", df.columns.tolist()[:12], "...")
need = {"OP_ID","PMID","duration_hours","AKI_linked","AKI_linked_0_7"}
missing = need - set(df.columns)
print("Pflichtfelder fehlen?" , missing if missing else "nein")

print("\nDeskriptiv: duration_hours")
print(df["duration_hours"].describe())

print("\nAKI_linked Counts:")
if "AKI_linked" in df.columns:
    print(df["AKI_linked"].value_counts(dropna=False).sort_index())
else:
    print("Spalte AKI_linked fehlt.")

# ---- CSV: aki_prior_0_7.csv ----
print("\n=== aki_prior_0_7.csv ===")
df07 = pd.read_csv(FILES["prior_0_7_csv"], sep=";")
for c in ["Surgery_Start","Surgery_End","AKI_Start"]:
    if c in df07.columns:
        df07[c] = pd.to_datetime(df07[c], errors="coerce")
print("Form:", df07.shape)
print("Spalten:", df07.columns.tolist())
if {"days_to_AKI","duration_hours"}.issubset(df07.columns):
    print("days_to_AKI (0–7) – describe:")
    print(df07["days_to_AKI"].describe())
    print("duration_hours – describe:")
    print(df07["duration_hours"].describe())

# ---- CSV: analytic_patient_summary.csv ----
print("\n=== analytic_patient_summary.csv ===")
pat = pd.read_csv(FILES["patient_summary_csv"], sep=";")
for c in ["earliest_op","latest_op","AKI_Start"]:
    if c in pat.columns:
        pat[c] = pd.to_datetime(pat[c], errors="coerce")
print("Form:", pat.shape)
print("Spalten:", pat.columns.tolist())
print(pat[["n_ops","total_op_hours","mean_op_hours","max_op_hours"]].describe())

# ---- H5AD: aki_ops_master.h5ad ----
print("\n=== aki_ops_master.h5ad (AnnData) ===")
adata = read_h5ad(FILES["h5ad"])
print(adata)  # z.B. AnnData object with n_obs × n_vars = 1209 × 1

# Dauer aus X holen (1 Variable erwartet: duration_hours)
dur_name_ok = list(map(str, adata.var_names)) == ["duration_hours"]
if not dur_name_ok:
    print("⚠️ Hinweis: var_names != ['duration_hours'], gefunden:", list(map(str, adata.var_names)))

dur = np.asarray(adata.X).ravel()
print("duration_hours (aus X) – describe:")
print(pd.Series(pd.to_numeric(dur, errors="coerce")).describe())

# obs-Checks
obs_cols = ["PMID","SMID","Procedure_ID","AKI_Start","days_to_AKI","AKI_linked","AKI_linked_0_7","Sex_norm"]
present = [c for c in obs_cols if c in adata.obs.columns]
missing_obs = [c for c in obs_cols if c not in adata.obs.columns]
print("\nobs enthält:", present)
if missing_obs:
    print("⚠️ obs fehlt:", missing_obs)

print("\nFertig. Alles geladen ✅")
