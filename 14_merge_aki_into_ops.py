# 14_merge_aki_into_ops.py  

import ehrapy as ep
import pandas as pd
import numpy as np
from anndata import AnnData
from pathlib import Path

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
OPS_H5AD_IN  = BASE / "ops_with_crea_cysc_vis_features.h5ad"
AKI_SRC_CSV  = BASE / "AKI Label.csv"  # semikolon-getrennt
OPS_CSV_OUT  = BASE / "ops_with_aki.csv"
OPS_H5AD_OUT = BASE / "ops_with_aki.h5ad"

#  1) OP-Features laden 
def load_ops(h5_path: Path) -> pd.DataFrame:
    ad = ep.io.read_h5ad(str(h5_path))
    df = ad.obs.copy()
    df.columns = df.columns.astype(str)
    for c in ("PMID","SMID"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ("Surgery_Start","Surgery_End"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

ops = load_ops(OPS_H5AD_IN)

#  2) AKI-CSV laden (Semikolon!) + Header & Inhalte reparieren 
aki = pd.read_csv(
    AKI_SRC_CSV,
    sep=";",
    encoding="utf-8-sig",
    dtype=str,
    engine="python",
    on_bad_lines="skip",
)
aki.columns = aki.columns.str.strip()

for c in aki.columns:
    if aki[c].dtype == "object":
        aki[c] = aki[c].astype(str).str.strip()

aki = aki.rename(columns={
    "Duartion": "Duration",
    "Start":    "AKI_Start",
    "End":      "AKI_End",
    "Decision": "Decision",   # belassen, wir mappen gleich in AKI_linked_0_7
    "patientid": "PMID",
    "PatientID": "PMID",
    "patient_id": "PMID",
})

if "PMID" not in aki.columns:
    raise ValueError("In der AKI-Datei fehlt die Spalte 'PMID' nach dem Umbenennen.")
aki["PMID"] = aki["PMID"].astype(str).str.strip()

# Zeiten parsen
for tcol in ("AKI_Start","AKI_End"):
    if tcol in aki.columns:
        aki[tcol] = pd.to_datetime(aki[tcol], errors="coerce")

print("AKI-Spalten (nach Rename):", aki.columns.tolist())
print("AKI Decision unique (roh):", sorted(pd.Series(aki.get("Decision", pd.Series(dtype=str))).dropna().unique().tolist())[:10])

# 2a) ROBUSTES MAPPING: Decision → AKI_linked_0_7 (patientenweites Flag) 
import re
def map_aki_flag_v2(x: str):
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    # True: alles mit "aki" drin (aki, aki 1/2/3), "tx" optional als positiv
    if re.search(r"\baki\b", s) or re.search(r"\baki\s*[123]\b", s) or s in {"tx", "positive", "pos", "+"}:
        return True
    # False: klare Negationen
    if any(tok in s for tok in ["kein", "keine", "ohne", "negativ", "absent"]) or s in {"0","false","nein","-"}:
        return False
    return np.nan

aki["AKI_linked_0_7"] = aki.get("Decision", pd.Series(index=aki.index, dtype=object)).apply(map_aki_flag_v2)

print("AKI Decision → AKI_linked_0_7 (gemappt):")
print(aki["AKI_linked_0_7"].value_counts(dropna=False))

#####  3) Zeitbasiertes Matching: erster AKI_Start NACH OP-Ende 
# Eindeutiger Key je OP-Zeile, um Duplikate beim Merge zu vermeiden
# --- A) Patientenflag separat halten (keine Vermischung mit time-based) ---
aki["AKI_patient_flag"] = aki.get("Decision", pd.Series(index=aki.index, dtype=object)).apply(map_aki_flag_v2)
aki_flag = aki[["PMID","AKI_patient_flag"]].drop_duplicates("PMID").copy()
aki_flag["PMID"] = aki_flag["PMID"].astype(str).str.strip()

# --- B) Zeitbasiertes Matching mit Toleranz (z. B. 30 Tage) ---
TOLERANCE_DAYS = 30
tolerance = pd.Timedelta(days=TOLERANCE_DAYS)
# Schlüssel für OPs vorbereiten
ops = ops.reset_index(drop=True).copy()
ops["op_row_id"] = np.arange(len(ops), dtype=int)

ops_key = ops[["op_row_id","PMID","Surgery_End"]].dropna(subset=["Surgery_End"]).copy()
ops_key["PMID"] = ops_key["PMID"].astype(str).str.strip()

# Schlüssel für AKI vorbereiten
aki_key = aki[["PMID","AKI_Start"]].dropna(subset=["AKI_Start"]).copy() if "AKI_Start" in aki.columns else pd.DataFrame(columns=["PMID","AKI_Start"])
aki_key["PMID"] = aki_key["PMID"].astype(str).str.strip() if "PMID" in aki_key.columns else ""

matched = []
for pmid, left in ops_key.groupby("PMID", sort=False):
    right = aki_key[aki_key["PMID"] == pmid]
    if right.empty:
        tmp = left.copy()
        tmp["AKI_Start"] = pd.NaT
    else:
        tmp = pd.merge_asof(
            left.sort_values("Surgery_End"),
            right.sort_values("AKI_Start"),
            left_on="Surgery_End", right_on="AKI_Start",
            direction="forward", allow_exact_matches=True,
            tolerance=tolerance
        )
    matched.append(tmp)

ops_matched = pd.concat(matched, ignore_index=True)
ops_matched["days_to_AKI"] = (ops_matched["AKI_Start"] - ops_matched["Surgery_End"]).dt.total_seconds()/86400.0
ops_matched["AKI_time_0_7"] = np.where(
    ops_matched["days_to_AKI"].notna(),
    ((ops_matched["days_to_AKI"] >= 0) & (ops_matched["days_to_AKI"] <= 7)).astype(bool),
    np.nan
)

# --- C) Zurückmergen + Flags klar benennen ---
ops_merged = ops.merge(
    ops_matched[["op_row_id","AKI_Start","days_to_AKI","AKI_time_0_7"]],
    on="op_row_id", how="left"
).merge(aki_flag, on="PMID", how="left")

# Fallback einstellen (empfohlen: False für sauberes OP-Level)
USE_PATIENT_FALLBACK = False

if USE_PATIENT_FALLBACK:
    ops_merged["AKI_final_0_7"] = ops_merged["AKI_time_0_7"]
    idx = ops_merged["AKI_final_0_7"].isna()
    ops_merged.loc[idx, "AKI_final_0_7"] = ops_merged.loc[idx, "AKI_patient_flag"]
    ops_merged["AKI_final_0_7"] = ops_merged["AKI_final_0_7"].fillna(False).astype(bool)
    ops_merged["AKI_linked_0_7"] = ops_merged["AKI_final_0_7"]
else:
    ops_merged["AKI_linked_0_7"] = ops_merged["AKI_time_0_7"].fillna(False).astype(bool)

# Debug
# --- Debug: konsistente Spaltennamen verwenden ---
print("\nToleranz:", tolerance)
print("AKI_time_0_7 counts:", ops_merged["AKI_time_0_7"].value_counts(dropna=False))
print("AKI_patient_flag counts:", ops_merged["AKI_patient_flag"].value_counts(dropna=False))
print("AKI_linked_0_7 (final) counts:", ops_merged["AKI_linked_0_7"].value_counts(dropna=False))

# Shapes & Non-null (optional – hilft bei schnellen Plausibilitätschecks)
print("\nShapes — ops, aki, ops_matched, ops_merged:", ops.shape, aki.shape, ops_matched.shape, ops_merged.shape)
print("Non-null AKI_Start (aki):", int(aki.get("AKI_Start", pd.Series(dtype='datetime64[ns]')).notna().sum()))
print("Non-null AKI_Start (ops_matched):", int(ops_matched["AKI_Start"].notna().sum()))

# days_to_AKI Summary (nur valide)
if "days_to_AKI" in ops_merged.columns and ops_merged["days_to_AKI"].notna().any():
    print("\nSummary days_to_AKI (nur valide):")
    print(ops_merged["days_to_AKI"].dropna().describe())


if "days_to_AKI" in ops_merged.columns:
    print("\nSummary days_to_AKI (nur valide):")
    print(ops_merged["days_to_AKI"].dropna().describe())

# ===== 6) Speichern =====
def sanitize_obs_for_h5ad(df: pd.DataFrame, fmt="%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        # Datetime → String
        if pd.api.types.is_datetime64_any_dtype(s) or any(k in c.lower() for k in ["time","date","start","end","timestamp"]):
            try:
                s = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
                out[c] = s.dt.strftime(fmt).fillna("")
                continue
            except Exception:
                pass
        # Bool → 0/1
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("uint8"); continue
        # Numerik passt
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce"); continue
        # Kategorie → String
        if pd.api.types.is_categorical_dtype(s):
            out[c] = s.astype(str).fillna(""); continue
        # Rest → String, NaN/NaT leeren
        s = s.astype(str).replace({"nan":"","NaN":"","NaT":""})
        out[c] = s
    out.index = out.index.astype(str)
    return out

# CSV zuerst (immer gut)
ops_merged.to_csv(OPS_CSV_OUT, index=False)
# === A) NaN im finalen Flag auf False setzen (besser für Modelle & Tabellen) ===
# Achtung: Nur wo weder zeitbasiert noch Patientenflag etwas liefert.
ops_merged["AKI_linked_0_7"] = ops_merged["AKI_linked_0_7"].fillna(False).astype(bool)

print("\nAKI_linked_0_7 (final, NaN→False) counts:")
print(ops_merged["AKI_linked_0_7"].value_counts(dropna=False))

# === B) AKI-Stufe aus Decision extrahieren (1/2/3) und patientenweit mappen ===
if "Decision" in aki.columns:
    aki["AKI_stage"] = (
        aki["Decision"]
        .str.extract(r"(\d)")
        .astype(float)
    )
    aki_stage = aki[["PMID","AKI_stage"]].dropna().drop_duplicates("PMID")
    aki_stage["PMID"] = aki_stage["PMID"].astype(str).str.strip()
    ops_merged = ops_merged.merge(aki_stage, on="PMID", how="left")
    print("\nAKI_stage Verteilung (patientenweit, pro OP übernommen):")
    print(ops_merged["AKI_stage"].value_counts(dropna=False))

# === C) Weitere Zeitfenster erzeugen (0–48h, 0–30d) ===
def time_flag_within(days):
    col = f"AKI_linked_0_{int(days)}d_time"
    ops_merged[col] = np.where(
        ops_merged["days_to_AKI"].notna(),
        ((ops_merged["days_to_AKI"] >= 0) & (ops_merged["days_to_AKI"] <= days)).astype(bool),
        np.nan
    )
    # final: zeitbasiert bevorzugt, sonst Patientenflag (falls vorhanden), Rest False
    final_col = f"AKI_linked_0_{int(days)}d"
    if "AKI_linked_0_7" in ops_merged.columns:
        # Re-Use des Patientenflags über vorhandenes 0–7-Flag (bool) als Fallback-Proxy:
        # Falls du ein echtes Patientenflag getrennt behalten willst, führe es separat.
        ops_merged[final_col] = ops_merged[col]
        idx = ops_merged[final_col].isna()
        ops_merged.loc[idx, final_col] = ops_merged.loc[idx, "AKI_linked_0_7"]
        ops_merged[final_col] = ops_merged[final_col].fillna(False).astype(bool)
    else:
        ops_merged[final_col] = ops_merged[col].fillna(False).astype(bool)
    return col, final_col

col_48h, final_48h = time_flag_within(2)   # 2 Tage ~ 48h
col_30d, final_30d = time_flag_within(30)

print(f"\n{col_48h} counts:")
print(ops_merged[col_48h].value_counts(dropna=False))
print(f"{final_48h} counts:")
print(ops_merged[final_48h].value_counts(dropna=False))

print(f"\n{col_30d} counts:")
print(ops_merged[col_30d].value_counts(dropna=False))
print(f"{final_30d} counts:")
print(ops_merged[final_30d].value_counts(dropna=False))

# === D) Top-Ausreißer zur Sichtprüfung (größte Abstände) ===
print("\nTop 10 längste Abstände days_to_AKI:")
cols_show = ["PMID","SMID","Surgery_End","AKI_Start","days_to_AKI"]
print(
    ops_merged.loc[ops_merged["days_to_AKI"].notna(), cols_show]
    .sort_values("days_to_AKI", ascending=False)
    .head(10)
    .to_string(index=False)
)

# === E) (Optional) harte Plausibilitätsprüfung für AKI_Start-Jahre ===
# Wenn die 2152/2153-Zeitstempel nicht beabsichtigt sind, bitte aktivieren:
# valid_years = ops_merged["AKI_Start"].dt.year.between(2000, 2035)
# ops_merged.loc[~valid_years, ["AKI_Start","days_to_AKI",col_48h,final_48h,col_30d,final_30d,"AKI_linked_0_7"]] = [pd.NaT, np.nan, np.nan, False, np.nan, False, False]

# === F) Erneut speichern (überschreiben), damit alle neuen Spalten vorliegen ===
ops_merged.to_csv(OPS_CSV_OUT, index=False)

ops_save = sanitize_obs_for_h5ad(ops_merged)
adata_out = AnnData(X=np.empty((len(ops_save), 0)))
adata_out.obs = ops_save
adata_out.write(str(OPS_H5AD_OUT))

print("\nAktualisiert gespeichert:")
print(" -", OPS_H5AD_OUT)
print(" -", OPS_CSV_OUT)


# H5AD sanitizen & schreiben
ops_save = sanitize_obs_for_h5ad(ops_merged)
adata_out = AnnData(X=np.empty((len(ops_save), 0))); adata_out.obs = ops_save
adata_out.write(str(OPS_H5AD_OUT))

print("\nFertig. Gespeichert:")
print(" -", OPS_H5AD_OUT)
print(" -", OPS_CSV_OUT)

# ===== 7) Mini-Check =====
print("\nAKI_linked_0_7 Wertetabelle (final):")
print(ops_merged["AKI_linked_0_7"].value_counts(dropna=False))
if "days_to_AKI" in ops_merged.columns:
    print("\nSummary days_to_AKI (nur valide):")
    print(ops_merged["days_to_AKI"].dropna().describe())
# --- Progressionstabelle aus der AKI-CSV ---
import pandas as pd
import numpy as np

# AKI CSV laden (wie bisher, inkl. Umbenennen, Trimmen etc.)
aki = pd.read_csv(AKI_SRC_CSV, sep=";", encoding="utf-8-sig", dtype=str)
aki.columns = aki.columns.str.strip()
aki = aki.rename(columns={"Duartion":"Duration","Start":"AKI_Start","End":"AKI_End","Decision":"Decision","patientid":"PMID","PatientID":"PMID","patient_id":"PMID"})
aki["PMID"] = aki["PMID"].astype(str).str.strip()

# AKI-Stufe als Zahl extrahieren
aki["AKI_stage"] = aki["Decision"].str.extract(r"(\d)").astype(float)

# --- pro Patient aggregieren ---
progression = (
    aki.groupby("PMID")["AKI_stage"]
    .agg(["min","max"])
    .rename(columns={"min":"AKI_stage_min","max":"AKI_stage_max"})
    .reset_index()
)

# Progression-Flag
progression["AKI_progression"] = progression["AKI_stage_max"] > progression["AKI_stage_min"]

print("\nProgressionstabelle (erste 10 Patienten):")
print(progression.head(10))

# --- Verteilungen ---
n_pat = progression.shape[0]
n_progress = progression["AKI_progression"].sum()
n_no_progress = n_pat - n_progress

print("\nGesamtanzahl Patienten mit AKI-Eintrag:", n_pat)
print("Davon Progression:", n_progress, f"({n_progress/n_pat*100:.1f}%)")
print("Ohne Progression:", n_no_progress, f"({n_no_progress/n_pat*100:.1f}%)")

print("\nVerteilung max. AKI-Stufe:")
print(progression["AKI_stage_max"].value_counts().sort_index())

