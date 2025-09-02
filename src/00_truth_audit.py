# -*- coding: utf-8 -*-
# 00_truth_audit.py — Liest NUR Originalquellen, zählt & vergleicht. Kein Schreiben, kein Umbau.

from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
DIR_ORIG = BASE / "Original Daten"
CSV_HLM = DIR_ORIG / "HLM Operationen.csv"
CSV_PAT = DIR_ORIG / "Patient Master Data.csv"
H5_OPS = BASE / "h5ad" / "ops_with_patient_features.h5ad"


def read_semicolon(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def norm_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))  # "00009" -> "9", "9.0" -> "9"
    except:
        return s.lstrip("0") or "0"


print("=== 1) HLM Operationen (Ground Truth OPs) ===")
hlm = read_semicolon(CSV_HLM)
# Spaltennamen prüfen
cols = {c.lower(): c for c in hlm.columns}
pmid_h = cols.get("pmid")
smid_h = cols.get("smid")
assert (
    pmid_h and smid_h
), f"HLM braucht 'PMID' & 'SMID', gefunden: {hlm.columns.tolist()}"

hlm["PMID_norm"] = hlm[pmid_h].map(norm_id)
hlm["SMID_norm"] = hlm[smid_h].map(norm_id)

n_ops_rows = len(hlm)
n_ops_smid = hlm["SMID_norm"].nunique()
n_pat_from_hlm = hlm["PMID_norm"].nunique()

print(f"- Zeilen (OP-Ereignisse):       {n_ops_rows}")
print(f"- Eindeutige SMID (Operationen): {n_ops_smid}")
print(f"- Eindeutige PMID (Patienten):   {n_pat_from_hlm}")

print("\n=== 2) Patient Master Data (Ground Truth Patienten) ===")
pat = read_semicolon(CSV_PAT)
cols_pat = {c.lower(): c for c in pat.columns}
pmid_p = cols_pat.get("pmid")
sex_p = cols_pat.get("sex") or cols_pat.get("geschlecht")
assert (
    pmid_p and sex_p
), f"Patient Master braucht 'PMID' & 'Sex/Geschlecht'. Gefunden: {pat.columns.tolist()}"

pat["PMID_norm"] = pat[pmid_p].map(norm_id)
print(f"- Alle Patienten im Master:      {pat['PMID_norm'].nunique()}")

vc_sex_all = pat[sex_p].value_counts(dropna=False)
print("- Geschlecht (alle Patienten, roh):")
print(vc_sex_all.to_string())


# Mapping m/f -> M/W (nur Anzeige)
def map_sex(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return np.nan


pat["Geschlecht"] = pat[sex_p].map(map_sex)

print("\n=== 3) HLM ∩ Patient Master (nur PMIDs, die eine OP haben) ===")
pat_hlm = pat.merge(hlm[["PMID_norm"]].drop_duplicates(), on="PMID_norm", how="inner")
n_pat_on_hlm = pat_hlm["PMID_norm"].nunique()
vc_hlm_sex = pat_hlm["Geschlecht"].value_counts(dropna=False)
print(f"- Patienten mit HLM-OP (laut Master): {n_pat_on_hlm}")
print("- Geschlecht (nur PMIDs mit HLM-OP):")
print(
    vc_hlm_sex.reindex(["Männlich", "Weiblich", np.nan])
    .dropna()
    .astype(int)
    .to_string()
)

print("\n=== 4) h5ad/ops_with_patient_features.h5ad (nur Vergleich) ===")
A = ad.read_h5ad(H5_OPS)
ops = A.obs.reset_index(drop=True).copy()
ops.columns = ops.columns.str.strip()
# In der h5ad kann PMID fehlen oder kategorisch sein
pmid_in_ops = "PMID" in ops.columns
smid_in_ops = "SMID" in ops.columns
n_obs = len(ops)
n_pmid_nonnull = ops["PMID"].notna().sum() if pmid_in_ops else 0
n_pmid_unique = ops["PMID"].dropna().astype(str).nunique() if pmid_in_ops else 0
n_smid_unique = ops["SMID"].dropna().astype(str).nunique() if smid_in_ops else 0

print(f"- h5ad Zeilen (n_obs):           {n_obs}")
print(f"- h5ad eindeutige SMID:          {n_smid_unique}  (sollte ≈ Zeilenanzahl sein)")
print(
    f"- h5ad PMID non-null:            {n_pmid_nonnull}  (hier lag früher das 572-Problem)"
)
print(f"- h5ad eindeutige PMID:          {n_pmid_unique}")

print("\n=== 5) Abgleich „Warum 500 vs. 1067?“ (Erklärung) ===")
print(
    "* Die HLM-Datei ist die Ground-Truth für OPs. Daraus ergeben sich die echten Zahlen: "
    f"{n_ops_rows} OPs und {n_pat_from_hlm} Patienten."
)
print(
    "* In der h5ad fehlen/fehlten viele PMIDs (z. B. nur ~572 non-null): "
    "deshalb kamen frühere Merges auf ~500 Patienten."
)
print(
    "* Wenn man PMIDs in der OP-Tabelle per SMID aus HLM nachfüllt, erhält man wieder die volle Zahl "
    "(≈ Patienten aus HLM, z. B. 1.067)."
)
