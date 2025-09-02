# -*- coding: utf-8 -*-
"""
Baut eine Master-H5AD-Tabelle mit allen relevanten Spalten (OP-Ebene).
- OP-Basis aus ops_with_patient_features.h5ad
- PMIDs via SMID aus HLM Operationen.csv auffüllen
- Geschlecht aus Patient Master Data.csv
- AKI aus AKI Label.csv (erstes AKI nach OP-Ende)
- optionale Zusatzfeatures aus ops_with_patient_features_ehrapy_enriched.h5ad
Ergebnis: h5ad/ops_master_all.h5ad
"""
import os, sys, json, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

# ---------------- Pfade ----------------
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5AD_OPS = BASE / "h5ad" / "ops_with_patient_features.h5ad"  # 1209 OPs
H5AD_ENR = BASE / "h5ad" / "ops_with_patient_features_ehrapy_enriched.h5ad"  # optional
CSV_HLM = BASE / "Original daten" / "HLM Operationen.csv"  # ; getrennt
CSV_PAT = BASE / "Original daten" / "Patient Master Data.csv"  # ; getrennt
CSV_AKI = (
    BASE / "Original daten" / "AKI Label.csv"
)  # ; getrennt (hatte Tippfehler "Duartion")
CSV_EPI = BASE / "Original daten" / "Procedure Supplement.csv"
OUT_H5AD = BASE / "h5ad" / "ops_master_all.h5ad"
OUT_CSV = BASE / "Daten" / "ops_master_all_preview.csv"
OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# --------------- Utilities ---------------
def read_csv_smart(path: Path) -> pd.DataFrame:
    """Liest CSV robust (beachtet ; und BOM)."""
    try:
        df = pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # Falls doch Komma-getrennt geliefert wurde:
    if df.shape[1] == 1 and "," in str(df.iloc[0, 0]):
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def norm_id(x):
    """PMID/SMID robust normalisieren: '00009'->'9', '9.0'->'9'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except:
        s2 = s.lstrip("0")
        return s2 if s2 else "0"


def parse_dt(x):
    """Datetime robust parsen."""
    return pd.to_datetime(x, errors="coerce")


def map_sex_to_full(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return np.nan


def sanitize_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """Datentypen h5ad-sicher machen: Datetime->ISO, Bool->int, Category->str, Object->str."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("int8")
        elif pd.api.types.is_categorical_dtype(out[c]):
            out[c] = out[c].astype("string")
        elif pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype("string")
        # numerics bleiben wie sie sind
    return out


#
import pandas as pd
import numpy as np

# 1) Typen harmonisieren (IDs als String; Zeiten als Datetime)
ops["PMID_final"] = ops["PMID_final"].astype("string")
ops["Surgery_End"] = pd.to_datetime(ops["Surgery_End"], errors="coerce")

aki["PMID_norm"] = aki["PMID_norm"].astype("string")
aki["AKI_Start"] = pd.to_datetime(aki["AKI_Start"], errors="coerce")
aki["AKI_End"] = pd.to_datetime(aki["AKI_End"], errors="coerce")

# 2) NaNs im Schlüssel/Zeitfeld vor dem asof droppen
ops2 = ops.dropna(subset=["PMID_final", "Surgery_End"]).copy()
aki2 = aki.dropna(subset=["PMID_norm", "AKI_Start"]).copy()

# 3) Gleich benennen
ops2 = ops2.rename(columns={"PMID_final": "PMID_key"})
aki2 = aki2.rename(columns={"PMID_norm": "PMID_key"})

# 4) **zweifach sortieren** (erst by, dann on) – mergesort ist stabil
ops_sort = ops2.sort_values(
    ["PMID_key", "Surgery_End"], ascending=[True, True], kind="mergesort"
).reset_index(drop=True)
aki_sort = aki2.sort_values(
    ["PMID_key", "AKI_Start"], ascending=[True, True], kind="mergesort"
).reset_index(drop=True)

# 5) Jetzt klappt merge_asof
linked = pd.merge_asof(
    left=ops_sort,
    right=aki_sort[["PMID_key", "AKI_Start", "AKI_End", "AKI_stage"]],
    by="PMID_key",
    left_on="Surgery_End",
    right_on="AKI_Start",
    direction="forward",
    allow_exact_matches=True,
)

# 6) Abstände & Flags berechnen
linked["days_to_AKI"] = (
    linked["AKI_Start"] - linked["Surgery_End"]
).dt.total_seconds() / (3600 * 24)
linked["AKI_linked"] = (~linked["AKI_Start"].isna()).astype(int)
linked["AKI_linked_0_7"] = (
    (linked["AKI_linked"] == 1)
    & (linked["days_to_AKI"] >= 0)
    & (linked["days_to_AKI"] <= 7)
).astype(int)


def safe_merge(left, right, on, how="left", suffixes=("", "_r")):
    if right is None or right.empty:
        return left
    return left.merge(right, on=on, how=how, suffixes=suffixes)


# --------------- 1) OP-Basis laden ---------------
print(f"Lade: {H5AD_OPS}")
A = ad.read_h5ad(H5AD_OPS)
ops = A.obs.reset_index(drop=True).copy()
ops.columns = ops.columns.str.strip()

# IDs vorbereiten (Strings, keine Categoricals)
for c in ["PMID", "SMID", "Procedure_ID"]:
    if c in ops.columns:
        ops[c] = pd.Series(ops[c]).astype("string")

ops["PMID_norm"] = ops.get("PMID").map(norm_id) if "PMID" in ops.columns else np.nan
ops["SMID_norm"] = ops.get("SMID").map(norm_id) if "SMID" in ops.columns else np.nan

# --------------- 2) HLM Operationen: SMID->PMID-Mapping ---------------
print(f"Lade: {CSV_HLM}")
hlm = read_csv_smart(CSV_HLM)
# Spalten vereinheitlichen
hlm = hlm.rename(
    columns={
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "Tx?": "Tx",
    }
)
for c in ["PMID", "SMID", "Procedure_ID"]:
    if c in hlm.columns:
        hlm[c] = pd.Series(hlm[c]).astype("string")
hlm["PMID_norm"] = hlm.get("PMID").map(norm_id) if "PMID" in hlm.columns else np.nan
hlm["SMID_norm"] = hlm.get("SMID").map(norm_id) if "SMID" in hlm.columns else np.nan
hlm["Surgery_Start"] = parse_dt(hlm.get("Surgery_Start"))
hlm["Surgery_End"] = parse_dt(hlm.get("Surgery_End"))

map_smid_to_pmid = hlm.dropna(subset=["SMID_norm"]).drop_duplicates("SMID_norm")[
    ["SMID_norm", "PMID_norm"]
]

ops = ops.merge(
    map_smid_to_pmid, on="SMID_norm", how="left", suffixes=("", "_from_HLM")
)
ops["PMID_norm"] = pd.Series(ops["PMID_norm"]).astype("string")
ops["PMID_norm_from_HLM"] = pd.Series(ops["PMID_norm_from_HLM"]).astype("string")
ops["PMID_final"] = ops["PMID_norm"].fillna(ops["PMID_norm_from_HLM"]).astype("string")

print("Diagnose IDs:")
print("  PMID non-null (vor Füllung):", pd.notna(ops["PMID_norm"]).sum())
print("  PMID non-null (nach Füllung):", pd.notna(ops["PMID_final"]).sum())
print("  Eindeutige Kinder (PMID_final):", ops["PMID_final"].nunique())

# Surgery-Zeiten aus HLM übernehmen, wenn in OPs nicht vorhanden
if "Surgery_Start" not in ops.columns or ops["Surgery_Start"].isna().all():
    ops = ops.merge(
        hlm[["SMID_norm", "Surgery_Start", "Surgery_End"]], on="SMID_norm", how="left"
    )
else:
    # sicherstellen, dass Datetime ist
    ops["Surgery_Start"] = parse_dt(ops["Surgery_Start"])
    ops["Surgery_End"] = parse_dt(ops["Surgery_End"])

# Dauer in Stunden/Minuten, falls nicht vorhanden
if "duration_hours" not in ops.columns and {"Surgery_Start", "Surgery_End"} <= set(
    ops.columns
):
    dur = (ops["Surgery_End"] - ops["Surgery_Start"]).dt.total_seconds() / 3600.0
    ops["duration_hours"] = dur
if "duration_minutes" not in ops.columns and "duration_hours" in ops.columns:
    ops["duration_minutes"] = ops["duration_hours"] * 60.0

# --------------- 3) Patient Master Data: Geschlecht & ggf. Geburtsdatum ---------------
print(f"Lade: {CSV_PAT}")
pat = read_csv_smart(CSV_PAT)
# Spalten finden
cols_pat = {c.lower(): c for c in pat.columns}
pmid_col = cols_pat.get("pmid")
sex_col = cols_pat.get("sex") or cols_pat.get("geschlecht")
dob_col = cols_pat.get("dateofbirth")
assert (
    pmid_col is not None and sex_col is not None
), "Patient Master Data: PMID/Sex Spalten nicht gefunden."

pat["PMID_norm"] = pat[pmid_col].map(norm_id)
pat["Sex_raw"] = pat[sex_col].astype("string")
pat["Geschlecht"] = pat["Sex_raw"].map(map_sex_to_full)
if dob_col:
    pat["DateOfBirth"] = parse_dt(pat[dob_col])
pat_keep = pat[
    ["PMID_norm", "Geschlecht", "Sex_raw"]
    + (["DateOfBirth"] if "DateOfBirth" in pat.columns else [])
].drop_duplicates("PMID_norm")

ops = ops.merge(
    pat_keep,
    left_on="PMID_final",
    right_on="PMID_norm",
    how="left",
    suffixes=("", "_pat"),
)
ops.drop(columns=[c for c in ["PMID_norm_pat"] if c in ops.columns], inplace=True)

# --------------- 4) AKI Label: zeitbasiertes Linking ---------------
print(f"Lade: {CSV_AKI}")
aki = read_csv_smart(CSV_AKI)
aki = aki.rename(
    columns={
        "Duartion": "Duration",
        "Start": "AKI_Start",
        "End": "AKI_End",
        "Decision": "AKI_Decision",
    }
)
cols_aki = {c.lower(): c for c in aki.columns}
pmid_aki_col = cols_aki.get("pmid")
assert pmid_aki_col is not None, "AKI Label: PMID-Spalte nicht gefunden."

aki["PMID_norm"] = aki[pmid_aki_col].map(norm_id)
aki["AKI_Start"] = parse_dt(aki.get("AKI_Start"))
aki["AKI_End"] = parse_dt(aki.get("AKI_End"))
aki["AKI_Decision_raw"] = aki.get("AKI_Decision").astype("string")


def decision_to_stage(s):
    if pd.isna(s):
        return np.nan
    t = str(s).lower()
    if "aki 3" in t or t.strip() == "3":
        return 3
    if "aki 2" in t or t.strip() == "2":
        return 2
    if "aki 1" in t or t.strip() == "1":
        return 1
    if "kein" in t or "none" in t or t.strip() == "0":
        return 0
    return np.nan


aki["AKI_stage"] = aki["AKI_Decision_raw"].map(decision_to_stage)

# merge_asof pro Patient: erste AKI_Start NACH OP-Ende
ops_sort = ops.sort_values(["PMID_final", "Surgery_End"]).reset_index(drop=True)
aki_sort = aki.sort_values(["PMID_norm", "AKI_Start"]).reset_index(drop=True)

# As-of merge erfordert gleiche Schlüsselnamen
ops_sort = ops_sort.rename(columns={"PMID_final": "PMID_key"})
aki_sort = aki_sort.rename(columns={"PMID_norm": "PMID_key"})

linked = pd.merge_asof(
    left=ops_sort,
    right=aki_sort[["PMID_key", "AKI_Start", "AKI_End", "AKI_stage"]],
    by="PMID_key",
    left_on="Surgery_End",
    right_on="AKI_Start",
    direction="forward",
    allow_exact_matches=True,
)

# zurück benennen
linked = linked.rename(columns={"PMID_key": "PMID_final"})

# Kennzahlen berechnen
linked["days_to_AKI"] = (
    linked["AKI_Start"] - linked["Surgery_End"]
).dt.total_seconds() / (3600 * 24)
linked["AKI_linked"] = (~linked["AKI_Start"].isna()).astype(int)
linked["AKI_linked_0_7"] = (
    (linked["AKI_linked"] == 1)
    & (linked["days_to_AKI"] >= 0)
    & (linked["days_to_AKI"] <= 7)
).astype(int)

# in ops übernehmen
ops = linked

# --------------- 5) Optionale Zusatzfeatures aus enriched.h5ad ---------------
extra_cols = []
if H5AD_ENR.exists():
    print(f"Lade Zusatzfeatures: {H5AD_ENR}")
    A_enr = ad.read_h5ad(H5AD_ENR)
    enr = A_enr.obs.reset_index(drop=True).copy()
    for c in ["PMID", "SMID", "Procedure_ID"]:
        if c in enr.columns:
            enr[c] = pd.Series(enr[c]).astype("string")
    enr["PMID_norm"] = enr.get("PMID").map(norm_id) if "PMID" in enr.columns else np.nan
    enr["SMID_norm"] = enr.get("SMID").map(norm_id) if "SMID" in enr.columns else np.nan

    # Kandidaten-Feature-Spalten (füge hier gerne weitere hinzu)
    candidate_features = [
        "crea_baseline",
        "crea_peak_0_48",
        "crea_delta_0_48",
        "crea_rate_0_48",
        "cysc_baseline",
        "cysc_peak_0_48",
        "cysc_delta_0_48",
        "cysc_rate_0_48",
        "vis_max_0_24",
        "vis_mean_0_24",
        "vis_max_6_24",
        "vis_auc_0_24",
        "vis_auc_0_48",
        "age_years_at_first_op",
        "age_group_pediatric",
    ]
    present = [c for c in candidate_features if c in enr.columns]
    extra_cols = present

    enr_keep = enr[["PMID_norm", "SMID_norm", "Procedure_ID"] + present].copy()
    before_cols = set(ops.columns)
    ops = ops.merge(
        enr_keep,
        on=["PMID_norm", "SMID_norm", "Procedure_ID"],
        how="left",
        suffixes=("", "_enr"),
    )
    added = [c for c in ops.columns if c not in before_cols]
    print("Zusatzspalten übernommen:", [c for c in added if c in extra_cols])

# --------------- 6) Aufräumen & Speichern ---------------
# Reihenfolge/Key-Spalten
key_cols = [
    "PMID_final",
    "SMID_norm",
    "PMID_norm",
    "SMID",
    "PMID",
    "Procedure_ID",
    "Surgery_Start",
    "Surgery_End",
]
ops["PMID_final"] = ops["PMID_final"].astype("string")

# Sanitize Datentypen für h5ad
ops_clean = sanitize_for_h5ad(ops)

# AnnData bauen (X leer, alles in obs)
X = np.empty((len(ops_clean), 0))
adata = ad.AnnData(X=X, obs=ops_clean)

# Provenienz in .uns
adata.uns["provenance"] = {
    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    "base_files": {
        "ops_h5ad": str(H5AD_OPS),
        "hlm_csv": str(CSV_HLM),
        "patient_master_csv": str(CSV_PAT),
        "aki_label_csv": str(CSV_AKI),
        "enriched_h5ad": str(H5AD_ENR) if H5AD_ENR.exists() else None,
    },
    "notes": "PMIDs via SMID ergänzt; AKI per merge_asof nach OP-Ende; Datentypen sanitizt.",
}

# Speichern
adata.write_h5ad(OUT_H5AD)
ops_clean.head(50).to_csv(OUT_CSV, index=False)

print("\n=== Fertig ===")
print("Gespeichert H5AD:", OUT_H5AD)
print("Preview CSV:", OUT_CSV)
print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
print("Spaltenanzahl in obs:", adata.obs.shape[1])
