# -*- coding: utf-8 -*-
"""
build_master_table.py
Erstellt eine integrierte OP-Level Master-Tabelle aus:
- Patient Master Data.csv (PMID, Sex, DateOfBirth, DateOfDie)
- HLM Operationen.csv   (PMID, SMID, Procedure_ID, Start of surgery, End of surgery, Tx?)
- AKI Label.csv         (PMID, Duartion, Start, End, Decision)  # 'Duartion' ist Tippfehler in Quelle
- Procedure Supplement.csv (optional; nicht zwingend)

Berechnet:
- first_op_dt je Patient, age_years_at_first_op, pädiatrische Altersgruppe
- duration_hours je OP
- AKI-Link: erster AKI_Start NACH Surgery_End, days_to_AKI, AKI_linked_0_7

Outputs:
- Daten/ops_with_patient_features.csv         (OP-Level, integriert)
- Daten/patient_level_age_first_op.csv        (Patient-Level)
- h5ad/ops_with_patient_features.h5ad         (AnnData ohne X, obs = Master)
- Diagramme/AgeGroups_pediatric_counts.csv    (Patient-Level Counts)
"""

import os
import numpy as np
import pandas as pd
from anndata import AnnData
import anndata as ad

ad.settings.allow_write_nullable_strings = True


BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
IN_DIR = os.path.join(BASE, "Original Daten")

PM_PATH = os.path.join(IN_DIR, "Patient Master Data.csv")
OPS_PATH = os.path.join(IN_DIR, "HLM Operationen.csv")
AKI_PATH = os.path.join(IN_DIR, "AKI Label.csv")
PS_PATH = os.path.join(IN_DIR, "Procedure Supplement.csv")  # optional

OUT_DIR_DATA = os.path.join(BASE, "Daten")
OUT_DIR_H5AD = os.path.join(BASE, "h5ad")
OUT_DIR_FIGS = os.path.join(BASE, "Diagramme")
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_H5AD, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

OUT_OP_CSV = os.path.join(OUT_DIR_DATA, "ops_with_patient_features.csv")
OUT_PAT_CSV = os.path.join(OUT_DIR_DATA, "patient_level_age_first_op.csv")
OUT_H5AD = os.path.join(OUT_DIR_H5AD, "ops_with_patient_features.h5ad")
OUT_COUNTS = os.path.join(OUT_DIR_FIGS, "AgeGroups_pediatric_counts.csv")


def read_csv_semicolon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
    return df


def parse_dt(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    try:
        if getattr(dt.dtype, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt


def pediatric_group(age_years: pd.Series) -> pd.Categorical:
    neonate_max = 28 / 365.25
    bins = [-np.inf, neonate_max, 1, 3, 5, 12, 18]
    labels = [
        "Neonates (0–28 T.)",
        "Infants (1–12 Mon.)",
        "Toddlers (1–3 J.)",
        "Preschool (3–5 J.)",
        "School-age (6–12 J.)",
        "Adolescents (13–18 J.)",
    ]
    return pd.cut(
        age_years.clip(lower=0, upper=18), bins=bins, labels=labels, right=True
    )


def map_aki_decision(dec):
    if pd.isna(dec):
        return pd.NA, pd.NA
    d = str(dec).strip().lower()
    if "aki" in d:
        for k in ("3", "2", "1"):
            if k in d:
                return 1, int(k)
        return 1, pd.NA
    if d in {"ja", "yes", "y", "1", "true", "wahr"}:
        return 1, pd.NA
    if d in {"nein", "no", "n", "0", "false", "falsch", "keine aki", "kein aki"}:
        return 0, pd.NA
    if "tx" in d:
        return pd.NA, pd.NA
    return pd.NA, pd.NA


print("Lade CSVs …")
pm = read_csv_semicolon(PM_PATH)
ops = read_csv_semicolon(OPS_PATH)
aki = read_csv_semicolon(AKI_PATH)

# Pflichtspalten prüfen
for col in ["PMID", "DateOfBirth"]:
    if col not in pm.columns:
        raise ValueError(
            f"Patient Master: Spalte '{col}' fehlt. Gefunden: {pm.columns.tolist()}"
        )
for col in ["PMID", "Start of surgery", "End of surgery"]:
    if col not in ops.columns:
        raise ValueError(
            f"HLM Operationen: Spalte '{col}' fehlt. Gefunden: {ops.columns.tolist()}"
        )
for col in ["PMID", "Start", "End", "Decision"]:
    if col not in aki.columns:
        raise ValueError(
            f"AKI Label: Spalte '{col}' fehlt. Gefunden: {aki.columns.tolist()}"
        )

# IDs normalisieren (String)
for df in (pm, ops, aki):
    df["PMID"] = df["PMID"].astype(str).str.strip()

# Datumsfelder parsen
pm["DateOfBirth_dt"] = parse_dt(pm["DateOfBirth"])
ops["Surgery_Start_dt"] = parse_dt(ops["Start of surgery"])
ops["Surgery_End_dt"] = parse_dt(ops["End of surgery"])
aki["AKI_Start_dt"] = parse_dt(aki["Start"])
aki["AKI_End_dt"] = parse_dt(aki["End"])

# Erste OP je Patient
ops["Surgery_First_dt"] = ops["Surgery_Start_dt"].where(
    ops["Surgery_Start_dt"].notna(), ops["Surgery_End_dt"]
)
first_op = (
    ops.dropna(subset=["Surgery_First_dt"])
    .sort_values(["PMID", "Surgery_First_dt"])
    .groupby("PMID", as_index=False)
    .agg(first_op_dt=("Surgery_First_dt", "first"))
)

# Patienten-Level: Alter bei erster OP + pädiatrische Gruppe
pat = pm[["PMID", "Sex", "DateOfBirth", "DateOfBirth_dt"]].merge(
    first_op, on="PMID", how="left"
)
valid = pat["DateOfBirth_dt"].notna() & pat["first_op_dt"].notna()
age_years = pd.Series(np.nan, index=pat.index, dtype="float64")
age_years.loc[valid] = (
    pat.loc[valid, "first_op_dt"] - pat.loc[valid, "DateOfBirth_dt"]
).dt.total_seconds() / (365.25 * 24 * 3600)
age_years = age_years.where(age_years.between(0, 120), np.nan)
pat["age_years_at_first_op"] = age_years
pat["age_group_pediatric"] = pediatric_group(pat["age_years_at_first_op"].astype(float))

# OP-Level anreichern
ops_enr = ops.merge(
    pat[["PMID", "age_years_at_first_op", "age_group_pediatric"]], on="PMID", how="left"
)

# OP-Dauer in Stunden
ops_enr["duration_hours"] = (
    ops_enr["Surgery_End_dt"] - ops_enr["Surgery_Start_dt"]
).dt.total_seconds() / 3600.0
ops_enr["duration_hours"] = ops_enr["duration_hours"].where(
    ops_enr["duration_hours"].between(0, 48), np.nan
)

# ===== AKI an OP verlinken: patientenweise asof-Join (robust) =====
ops_nonan = ops_enr.dropna(subset=["Surgery_End_dt"]).copy()
aki_nonan = aki.dropna(subset=["AKI_Start_dt"]).copy()

linked_list = []
for pid, ops_grp in ops_nonan.groupby("PMID", sort=False):
    aki_grp = aki_nonan[aki_nonan["PMID"] == pid]
    if aki_grp.empty:
        merged = ops_grp.copy()
        merged[["AKI_Start_dt", "AKI_End_dt", "Decision"]] = np.nan
        linked_list.append(merged)
        continue
    ops_sorted = ops_grp.sort_values("Surgery_End_dt")
    aki_sorted = aki_grp.sort_values("AKI_Start_dt")
    merged = pd.merge_asof(
        ops_sorted,
        aki_sorted,
        left_on="Surgery_End_dt",
        right_on="AKI_Start_dt",
        direction="forward",  # erster AKI-Start NACH OP-Ende
        allow_exact_matches=True,
    )
    linked_list.append(merged)

ops_final = pd.concat(linked_list, ignore_index=True)

# Abstand & 0–7 Tage Flag
ops_final["days_to_AKI"] = (
    ops_final["AKI_Start_dt"] - ops_final["Surgery_End_dt"]
).dt.total_seconds() / (24 * 3600)
ops_final["AKI_linked_0_7"] = np.where(
    (ops_final["days_to_AKI"] >= 0) & (ops_final["days_to_AKI"] <= 7), 1, 0
)
ops_final.loc[ops_final["days_to_AKI"].isna(), "AKI_linked_0_7"] = 0

# Decision -> Flag/Stage (optional)
flag_stage = ops_final["Decision"].apply(map_aki_decision)
ops_final["AKI_flag_from_decision"] = [fs[0] for fs in flag_stage]
ops_final["AKI_stage_from_decision"] = [fs[1] for fs in flag_stage]

# Patienten-Level speichern
pat_out = pat[
    [
        "PMID",
        "Sex",
        "DateOfBirth",
        "first_op_dt",
        "age_years_at_first_op",
        "age_group_pediatric",
    ]
].copy()
pat_out.to_csv(OUT_PAT_CSV, index=False)

# OP-Level Master speichern (nur sinnvolle Spalten + datetime umbenennen)
ops_final = ops_final.rename(
    columns={
        "Surgery_Start_dt": "Surgery_Start",
        "Surgery_End_dt": "Surgery_End",
        "AKI_Start_dt": "AKI_Start",
        "AKI_End_dt": "AKI_End",
    }
)
keep_cols = [
    "PMID",
    "SMID",
    "Procedure_ID",
    "Surgery_Start",
    "Surgery_End",
    "duration_hours",
    "age_years_at_first_op",
    "age_group_pediatric",
    "AKI_Start",
    "AKI_End",
    "Decision",
    "days_to_AKI",
    "AKI_linked_0_7",
]
keep_cols = [c for c in keep_cols if c in ops_final.columns]
ops_final.to_csv(OUT_OP_CSV, index=False, columns=keep_cols)

# Counts (Patient-Level)
counts = pat_out["age_group_pediatric"].value_counts(dropna=False).sort_index()
counts.to_csv(OUT_COUNTS, index=True)

# H5AD schreiben (obs = OP-Level Master; X leeres (n x 0))
# ---- H5AD schreiben (obs = OP-Level Master; X leeres (n x 0)) ----
obs_df = ops_final[keep_cols].copy()

# Datetime-Spalten in ISO-Strings wandeln (AnnData/HDF5 kann keine datetime64 in obs)
dt_cols = obs_df.select_dtypes(
    include=["datetime64[ns]", "datetime64[ns, UTC]"]
).columns
for c in dt_cols:
    # falls NaT vorhanden -> leere Strings
    obs_df[c] = obs_df[c].dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")

# Categorical sauber als string speichern (optional, macht vieles robuster)
cat_cols = obs_df.select_dtypes(include=["category"]).columns
for c in cat_cols:
    obs_df[c] = obs_df[c].astype("string")

# Leere X-Matrix und AnnData erzeugen
X = np.zeros((obs_df.shape[0], 0), dtype=float)
adata = AnnData(X=X)
adata.obs = obs_df


# ---- H5AD schreiben (obs = OP-Level Master; X leeres (n x 0)) ----
obs_df = ops_final[keep_cols].copy()

# 1) Datetime -> ISO-String (AnnData speichert kein datetime64 in obs)
dt_cols = obs_df.select_dtypes(
    include=["datetime64[ns]", "datetime64[ns, UTC]"]
).columns.tolist()
for c in dt_cols:
    obs_df[c] = obs_df[c].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

# 2) Categorical -> string (object)
cat_cols = obs_df.select_dtypes(include=["category"]).columns.tolist()
for c in cat_cols:
    obs_df[c] = obs_df[c].astype(str).replace("nan", "")

# 3) Pandas-StringDtype -> object (wichtig für AnnData < 0.11)
for c in obs_df.columns:
    if str(obs_df[c].dtype) == "string":
        obs_df[c] = obs_df[c].astype(object)

# 4) Sicherheit: alles, was noch nicht numeric/bool ist, auf object casten
#    (damit keine StringDtype mehr übrig bleibt)
import pandas as pd

for c in obs_df.columns:
    if pd.api.types.is_numeric_dtype(obs_df[c]) or pd.api.types.is_bool_dtype(
        obs_df[c]
    ):
        continue
    # schon object? okay. Sonst auf object bringen.
    if obs_df[c].dtype != object:
        obs_df[c] = obs_df[c].astype(object)

# Leeres X und schreiben
import numpy as np
from anndata import AnnData

X = np.zeros((obs_df.shape[0], 0), dtype=float)
adata = AnnData(X=X)
adata.obs = obs_df
adata.write(OUT_H5AD)
print("  -", OUT_H5AD)


# Console-Report
print("Gespeichert:")
print("  -", OUT_OP_CSV)
print("  -", OUT_PAT_CSV)
print("  -", OUT_COUNTS)
print("  -", OUT_H5AD)
va = pat_out["age_years_at_first_op"].dropna()
if not va.empty:
    print(
        f"Alter bei erster OP (Jahre): n={va.size}, min/median/max = {va.min():.2f}/{va.median():.2f}/{va.max():.2f}"
    )
print("Fertig ")
