# 12_lab_vis_merge_features.py
# Merge von Krea/CysC/VIS mit OP-Daten + Feature-Engineering (ehrapy/AnnData)

import pandas as pd
import numpy as np
from pathlib import Path
from anndata import AnnData

# ==================== Pfade ====================
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
p_ops = BASE / "HLM Operationen.csv"                 # OP-Tabelle (wie bisher)
p_lab_harm = BASE / "lab_crea_cysc_harmonized.csv"   # aus Schritt 11
p_vis_clean = BASE / "vis_raw_clean.csv"             # aus Schritt 11

# Ausgabe
p_traj = BASE / "traj_crea_cysc_vis.csv"
p_ops_feat = BASE / "ops_with_crea_cysc_vis_features.csv"
p_ops_h5ad = BASE / "ops_with_crea_cysc_vis_features.h5ad"
p_traj_h5ad = BASE / "traj_crea_cysc_vis.h5ad"

# ==================== Helper ====================
def stringify_datetime_columns(df, fmt="%Y-%m-%d %H:%M:%S"):
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        df[c] = df[c].dt.tz_localize(None, nonexistent="NaT", ambiguous="NaT").dt.strftime(fmt)
    return df

def trapezoid_auc(times, values):
    if len(times) < 2:
        return np.nan
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(t)
    return np.trapezoid(y[order], t[order])

# Fenster (in Stunden)
WIN = {
    "baseline": (-48, -6),
    "h0_24": (0, 24),
    "h0_48": (0, 48),
    "h24_48": (24, 48),
}

# ==================== 1) Daten laden ====================
# OPs einlesen & Spalten normalisieren
# ==================== 1) OP-Daten robust laden ====================
def read_semicolon_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.strip()
    return df

ops = read_semicolon_csv(p_ops)
print("Gefundene OP-Spalten:", ops.columns.tolist()[:30])
print(ops.head(3))

# Alias-Mapping: verschiedene Klinik-/Excel-Bezeichnungen auf Standardnamen
alias_map = {
    "PMID": [
        "pmid", "patientid", "patientenid", "patient_id",
        "patient master id", "patient_master_id", "patient master data id"
    ],
    "SMID": [
        "smid", "caseid", "fallid", "fallnummer", "opid", "surgery_id", "surgeryid"
    ],
    "Surgery_End": [
        "surgery_end", "end of surgery", "end_of_surgery",
        "op_ende", "ende op", "operation_ende", "op end", "end time", "endtime"
    ],
    "Surgery_Start": [
        "surgery_start", "start of surgery", "start_of_surgery",
        "op_beginn", "beginn op", "operation_beginn", "start time", "starttime"
    ],
}

# helper: benenne um, wenn eine der Kandidaten existiert
def rename_by_alias(df: pd.DataFrame, amap: dict) -> pd.DataFrame:
    low = {c: c.strip().lower() for c in df.columns}
    inv = {v: k for k, v in low.items()}  # lower->original
    ren = {}
    for want, cands in amap.items():
        found = None
        for cand in cands:
            cand_l = cand.lower()
            if cand_l in inv:
                found = inv[cand_l]; break
        if not found:
            # fuzzy: enthält-Variante (falls z. B. "PMID " oder "PM ID")
            for orig, lowc in low.items():
                if cand_l == lowc.replace(" ", ""):  # PM ID -> pmid
                    found = orig; break
        if found:
            ren[found] = want
    return df.rename(columns=ren)

ops = rename_by_alias(ops, alias_map)

# Falls die Klinik bereits „End of surgery“/„Start of surgery“ hat:
if "End of surgery" in ops.columns and "Surgery_End" not in ops.columns:
    ops = ops.rename(columns={"End of surgery": "Surgery_End"})
if "Start of surgery" in ops.columns and "Surgery_Start" not in ops.columns:
    ops = ops.rename(columns={"Start of surgery": "Surgery_Start"})

# Pflichtfelder prüfen und verständliche Fehlermeldung geben
required = ["PMID", "SMID", "Surgery_End"]
missing = [c for c in required if c not in ops.columns]
if missing:
    raise ValueError(
        "OP-Tabelle: Pflichtspalten fehlen: "
        + ", ".join(missing)
        + f"\nGefundene Spalten: {ops.columns.tolist()}"
        + "\n→ Prüfe: Trennzeichen (sep=';'), Header-Bezeichnungen, Leerzeichen im Header."
    )

# Datumsfelder sicher parsen
for c in ["Surgery_Start", "Surgery_End"]:
    if c in ops.columns:
        ops[c] = pd.to_datetime(ops[c], errors="coerce")

# Optional: SMID/PMID typisieren (Strings lassen, wichtig für Merge)
ops["PMID"] = ops["PMID"].astype(str).str.strip()
ops["SMID"] = ops["SMID"].astype(str).str.strip()

# Debug: kurze Übersicht
print("\nOP-Preview (normiert):")
print(ops[["PMID","SMID"] + [c for c in ["Surgery_Start","Surgery_End"] if c in ops.columns]].head(5))

# ==================== 2) Zeit relativ zur OP ====================
# ===== 1b) Labor (harmonisiert) & VIS laden =====
p_lab_harm = BASE / "lab_crea_cysc_harmonized.csv"   # wurde in Schritt 11 erzeugt
p_vis_clean = BASE / "vis_raw_clean.csv"             # wurde in Schritt 11 erzeugt

# Achtung: diese beiden Dateien sind mit Komma gespeichert → sep nicht setzen (Standard ist ",")
lab = pd.read_csv(p_lab_harm)
vis = pd.read_csv(p_vis_clean)

# Spalten trimmen & Schlüssel typisieren
for df in (lab, vis):
    df.columns = df.columns.str.strip()
    for c in ("PMID", "SMID"):
        df[c] = df[c].astype(str).str.strip()

# Zeitspalten zu datetime
lab["CollectionTimestamp"] = pd.to_datetime(lab["CollectionTimestamp"], errors="coerce")
vis["Timestamp"] = pd.to_datetime(vis["Timestamp"], errors="coerce")

# Merge für Labore
labm = lab.merge(ops[["PMID","SMID","Surgery_End"]], on=["PMID","SMID"], how="inner")
labm["time_from_op_hours"] = (labm["CollectionTimestamp"] - labm["Surgery_End"]).dt.total_seconds()/3600.0

# Merge für VIS
vism = vis.merge(ops[["PMID","SMID","Surgery_End"]], on=["PMID","SMID"], how="inner")
vism["time_from_op_hours"] = (vism["Timestamp"] - vism["Surgery_End"]).dt.total_seconds()/3600.0

# ==================== 3) Trajektorien-Long-Tabelle ====================
# Krea/CysC (Value_std, Unit_std, Parameter_std)
traj_lab = labm.rename(columns={
    "CollectionTimestamp": "MeasureTimestamp",
    "Parameter_std": "variable",
    "Value_std": "value",
    "Unit_std": "unit"
})[["PMID","SMID","MeasureTimestamp","time_from_op_hours","variable","value","unit"]]

# VIS
traj_vis = vism.rename(columns={"Timestamp":"MeasureTimestamp"}).copy()
traj_vis["variable"] = "vis"
traj_vis["value"] = traj_vis["vis"].astype(float)
traj_vis["unit"] = "score"
traj_vis = traj_vis[["PMID","SMID","MeasureTimestamp","time_from_op_hours","variable","value","unit"]]

# Zusammen
traj = pd.concat([traj_lab, traj_vis], ignore_index=True)
traj = traj.sort_values(["PMID","SMID","variable","MeasureTimestamp"])

# Speichern (CSV)
traj.to_csv(p_traj, index=False)

# ==================== 4) Feature-Engineering (OP-Level) ====================
def window(df, var, lo, hi):
    return df[(df["variable"]==var) & (df["time_from_op_hours"]>=lo) & (df["time_from_op_hours"]<hi)]

def baseline_stat(df, var, lo, hi, how="median"):
    w = window(df, var, lo, hi)
    if w.empty: return np.nan
    return getattr(w["value"], how)()

def peak_value(df, var, lo, hi):
    w = window(df, var, lo, hi)
    if w.empty: return np.nan
    return w["value"].max()

def rate_of_rise(df, var, lo, hi):
    w = window(df, var, lo, hi).sort_values("time_from_op_hours")
    if len(w) < 2: return np.nan
    dt = w["time_from_op_hours"].iloc[-1] - w["time_from_op_hours"].iloc[0]
    if dt <= 0: return np.nan
    dv = w["value"].iloc[-1] - w["value"].iloc[0]
    return dv/dt

features = []
for (pmid, smid), grp in traj.groupby(["PMID","SMID"]):
    row = {"PMID": pmid, "SMID": smid}

    # Kreatinin (mg/dL)
    row["crea_baseline"]   = baseline_stat(grp, "creatinine", *WIN["baseline"])
    row["crea_peak_0_48"]  = peak_value(grp, "creatinine", *WIN["h0_48"])
    row["crea_delta_0_48"] = (row["crea_peak_0_48"] - row["crea_baseline"]) if pd.notna(row["crea_peak_0_48"]) and pd.notna(row["crea_baseline"]) else np.nan
    row["crea_rate_0_48"]  = rate_of_rise(grp, "creatinine", *WIN["h0_48"])

    # Cystatin C (mg/L)
    row["cysc_baseline"]   = baseline_stat(grp, "cystatin_c", *WIN["baseline"])
    row["cysc_peak_0_48"]  = peak_value(grp, "cystatin_c", *WIN["h0_48"])
    row["cysc_delta_0_48"] = (row["cysc_peak_0_48"] - row["cysc_baseline"]) if pd.notna(row["cysc_peak_0_48"]) and pd.notna(row["cysc_baseline"]) else np.nan
    row["cysc_rate_0_48"]  = rate_of_rise(grp, "cystatin_c", *WIN["h0_48"])

    # VIS (Score)
    w0_24 = window(grp, "vis", *WIN["h0_24"])
    w0_48 = window(grp, "vis", *WIN["h0_48"])
    w6_24 = grp[(grp["variable"]=="vis") & (grp["time_from_op_hours"]>=6) & (grp["time_from_op_hours"]<24)]

    row["vis_max_0_24"]   = np.nan if w0_24.empty else w0_24["value"].max()
    row["vis_mean_0_24"]  = np.nan if w0_24.empty else w0_24["value"].mean()
    row["vis_max_6_24"]   = np.nan if w6_24.empty else w6_24["value"].max()
    row["vis_auc_0_24"]   = np.nan if w0_24.empty else trapezoid_auc(w0_24["time_from_op_hours"], w0_24["value"])
    row["vis_auc_0_48"]   = np.nan if w0_48.empty else trapezoid_auc(w0_48["time_from_op_hours"], w0_48["value"])

    features.append(row)

feat_df = pd.DataFrame(features)

# ==================== 5) OP-Master anreichern & speichern ====================
# Dauer in Stunden ergänzen (falls nötig)
if "duration_hours" not in ops.columns and "duration_minutes" in ops.columns:
    ops["duration_hours"] = ops["duration_minutes"] / 60.0

ops_enriched = ops.merge(feat_df, on=["PMID","SMID"], how="left")

# CSV
ops_enriched.to_csv(p_ops_feat, index=False)

# H5AD (Datums in Strings umwandeln, sonst write-Error)
ops_obs = ops_enriched.copy()
ops_obs = stringify_datetime_columns(ops_obs)
traj_obs = traj.copy()
traj_obs = stringify_datetime_columns(traj_obs)

adata_ops = AnnData(X=np.empty((len(ops_obs), 0))); adata_ops.obs = ops_obs
adata_traj = AnnData(X=np.empty((len(traj_obs), 0))); adata_traj.obs = traj_obs

adata_ops.write(p_ops_h5ad)
adata_traj.write(p_traj_h5ad)

print("\nFertig.")
print(" -", p_traj)
print(" -", p_ops_feat)
print(" -", p_ops_h5ad)
print(" -", p_traj_h5ad)
