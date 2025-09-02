# Sicherheitskontrole
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
DIR_ORIG = BASE / "Original Daten"
CSV_HLM = DIR_ORIG / "HLM Operationen.csv"
CSV_PAT = DIR_ORIG / "Patient Master Data.csv"
H5_OPS = BASE / "h5ad" / "ops_with_patient_features.h5ad"

OUT_CSV = BASE / "Daten" / "ops_master_simple.csv"
OUT_H5AD = BASE / "h5ad" / "ops_master_simple.h5ad"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)


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
        return str(int(float(s)))  # "00009"->"9", "9.0"->"9"
    except:
        return s.lstrip("0") or "0"


def map_sex(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return np.nan


# 1) HLM (Ground Truth für OP->Patient)
hlm = read_semicolon(CSV_HLM)
pmid_h = next(c for c in hlm.columns if c.lower() == "pmid")
smid_h = next(c for c in hlm.columns if c.lower() == "smid")
hlm["PMID_norm"] = hlm[pmid_h].map(norm_id)
hlm["SMID_norm"] = hlm[smid_h].map(norm_id)
map_smid_to_pmid = hlm.dropna(subset=["SMID_norm"]).drop_duplicates("SMID_norm")[
    ["SMID_norm", "PMID_norm"]
]

# 2) Patient Master (Ground Truth für Geschlecht)
pat = read_semicolon(CSV_PAT)
pmid_p = next(c for c in pat.columns if c.lower() == "pmid")
sex_p = next(c for c in pat.columns if c.lower() in ("sex", "geschlecht"))
pat["PMID_norm"] = pat[pmid_p].map(norm_id)
pat["Geschlecht"] = pat[sex_p].map(map_sex)
pat_keep = pat[["PMID_norm", "Geschlecht"]].drop_duplicates("PMID_norm")

# 3) OP-h5ad laden, nur minimal anreichern (keine komplizierten Schritte)
A = ad.read_h5ad(H5_OPS)
ops = A.obs.reset_index(drop=True).copy()
ops.columns = ops.columns.str.strip()
# IDs als string-objekt
for c in ["PMID", "SMID", "Procedure_ID"]:
    if c in ops.columns:
        ops[c] = pd.Series(ops[c]).astype(object)

ops["SMID_norm"] = pd.Series(ops["SMID"]).map(norm_id)
ops["PMID_norm_raw"] = pd.Series(ops["PMID"]).map(norm_id)
# PMID_final: zuerst aus h5ad, fehlende via HLM-Mapping
# Mapping aus HLM: SMID_norm -> PMID_norm
map_smid_to_pmid = hlm.dropna(subset=["SMID_norm"]).drop_duplicates("SMID_norm")[
    ["SMID_norm", "PMID_norm"]
]

# ✅ vor dem Merge umbenennen, damit die rechte Spalte sicher 'PMID_norm_from_HLM' heißt
map_smid_to_pmid = map_smid_to_pmid.rename(columns={"PMID_norm": "PMID_norm_from_HLM"})

# Merge
ops = ops.merge(map_smid_to_pmid, on="SMID_norm", how="left")

# PMID_final: erst die in h5ad vorhandene, sonst aus HLM-Mapping
ops["PMID_final"] = ops["PMID_norm_raw"].combine_first(ops["PMID_norm_from_HLM"])
print("PMID_final non-null:", ops["PMID_final"].notna().sum())
print("Eindeutige Patienten (PMID_final):", ops["PMID_final"].dropna().nunique())

# Geschlecht mergen
ops = ops.merge(pat_keep, left_on="PMID_final", right_on="PMID_norm", how="left")

# 4) Sanfte Checks (nur Prints)
n_ops = len(ops)
n_pat_final = pd.Series(ops["PMID_final"]).dropna().nunique()
vc_sex = ops.drop_duplicates("PMID_final")["Geschlecht"].value_counts(dropna=False)
print("OPs (Zeilen):", n_ops)
print("Eindeutige Patienten (PMID_final):", n_pat_final)
print("Geschlecht (Patientenebene, aus Master):\n", vc_sex.to_string())

# 5) CSV speichern (flach, perfekt für Plots/Prüfung)
ops.to_csv(OUT_CSV, index=False)
print("CSV gespeichert:", OUT_CSV)

# 6) Optional: h5ad speichern (nur einfache Typen)
#    - Datumsfelder in ISO-Text, alle Nicht-Numerik/Bools in echte Python-Strings
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype

ops_h = ops.copy()
for c in ops_h.columns:
    s = ops_h[c]
    if is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
        ops_h[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S").astype(object)
        ops_h[c] = ops_h[c].fillna("")
    elif not is_numeric_dtype(s) and not is_bool_dtype(s):
        ops_h[c] = s.astype(object).map(lambda x: "" if pd.isna(x) else str(x))

ops_h = ops_h.reset_index(drop=True)
ops_h.index = ops_h.index.astype(str)
X = np.empty((len(ops_h), 0))
adata = ad.AnnData(X=X, obs=ops_h)
adata.write_h5ad(OUT_H5AD)
print("h5ad gespeichert:", OUT_H5AD)

import pandas as pd

csv = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Daten/ops_master_simple.csv"
df = pd.read_csv(csv)

print("OPs:", len(df))
print("Eindeutige Patienten:", df["PMID_final"].nunique())
print(df.drop_duplicates("PMID_final")["Geschlecht"].value_counts())
assert df["PMID_final"].notna().all()  # jede OP hat eine Patient-ID
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv(csv)

pat = df.drop_duplicates("PMID_final")
counts = (
    pat["Geschlecht"]
    .value_counts()
    .reindex(["Männlich", "Weiblich"])
    .fillna(0)
    .astype(int)
)
perc = (counts / counts.sum() * 100).round(1)

plt.figure(figsize=(6, 6))
bars = plt.bar(["Männlich", "Weiblich"], counts.values, color=["#1f77b4", "#ff69b4"])
for b, p in zip(bars, perc.values):
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() * 1.01,
        f"{p:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
    )
plt.title("Geschlechterverteilung ")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()
