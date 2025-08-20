# 11_lab_vis_import.py
# Sauberer Import + Harmonisierung (ehrapy/AnnData)

import pandas as pd
import numpy as np
from pathlib import Path
from anndata import AnnData
import ehrapy as ep  # bleibt im Projekt, auch wenn wir hier pandas/h5ad fürs Speichern nutzen

# ===================== Pfade =====================
# Nutze ENTWEDER absolute Pfade ODER BASE + relative Dateinamen – nicht mischen.
LAB_SRC = "/Users/fa/Downloads/cs-transfer/Laborwerte_Kreatinin+CystatinC.csv"
VIS_SRC = "/Users/fa/Downloads/cs-transfer/VIS.csv"

OUT_DIR = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= Einlese-Helfer =================
def read_semicolon_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        na_values=["", "NA", "NaN", "NULL", None],
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.strip()
    return df

lab_df = read_semicolon_csv(LAB_SRC)
vis_df = read_semicolon_csv(VIS_SRC)

print("Labordaten-Spalten:", lab_df.columns.tolist())
print("VIS-Spalten:", vis_df.columns.tolist())
print(lab_df.head())
print(vis_df.head())

# =============== Pflichtspalten prüfen ===============
need_lab = {"PMID","SMID","CollectionTimestamp","Parameter","QuantitativeValue","Unit","LOINC"}
need_vis = {"PMID","SMID","Timestamp","vis"}

missing_lab = [c for c in need_lab if c not in lab_df.columns]
missing_vis = [c for c in need_vis if c not in vis_df.columns]
if missing_lab:
    raise ValueError(f"Fehlende Spalten in Labordaten: {missing_lab}\nGefunden: {lab_df.columns.tolist()}")
if missing_vis:
    raise ValueError(f"Fehlende Spalten in VIS: {missing_vis}\nGefunden: {vis_df.columns.tolist()}")

# =============== Numerik & Zeit ===============
# Dezimal-Komma -> Punkt
lab_df["QuantitativeValue"] = (
    lab_df["QuantitativeValue"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
vis_df["vis"] = (
    vis_df["vis"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Timestamps
lab_df["CollectionTimestamp"] = pd.to_datetime(lab_df["CollectionTimestamp"], errors="coerce")
vis_df["Timestamp"] = pd.to_datetime(vis_df["Timestamp"], errors="coerce")

print("\nFehlende Werte in Labordaten:\n", lab_df.isna().sum())
print("\nFehlende Werte in VIS:\n", vis_df.isna().sum())
print("\nEinzigartige Parameter (Top 20):", lab_df["Parameter"].dropna().unique()[:20])
print("Einheiten (Top 20):", lab_df["Unit"].dropna().unique()[:20])
print("\nVIS-Statistik:\n", vis_df["vis"].describe())

# =============== AnnData erstellen (.obs voll) ===============
lab = AnnData(X=np.empty((len(lab_df), 0)))
lab.obs = lab_df.copy()
vis = AnnData(X=np.empty((len(vis_df), 0)))
vis.obs = vis_df.copy()

# =============== Parameter-Erkennung ===============
def norm_text(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    return (s.replace("ß","ss").replace("ä","ae").replace("ö","oe").replace("ü","ue"))

lab.obs["Parameter_norm"] = lab.obs["Parameter"].map(norm_text)
lab.obs["LOINC"] = lab.obs["LOINC"].astype(str).str.strip()

TARGET_LOINCS = {
    "creatinine": {"14682-9"},
    "cystatin_c": {"33863-2"},
}
TARGET_NAME_HINTS = {
    "creatinine": ["creat", "creatinin", "kreatinin"],     # deckt: S-Creat/enz, Creatinine enz., ...
    "cystatin_c": ["cystatin-c", "cystatin c", "cystatinc"],
}

def is_param(row, key: str) -> bool:
    loinc = str(row.get("LOINC", "")).strip()
    pname = str(row.get("Parameter_norm", "")).lower().strip()
    if loinc and loinc in TARGET_LOINCS[key]:
        return True
    return any(h in pname for h in TARGET_NAME_HINTS[key])


lab.obs["is_creatinine"] = lab.obs.apply(lambda r: is_param(r, "creatinine"), axis=1)
lab.obs["is_cystatin_c"] = lab.obs.apply(lambda r: is_param(r, "cystatin_c"), axis=1)

# =============== Einheiten harmonisieren ===============
def unify_units(value: float, unit_raw: str, is_crea: bool, is_cysc: bool):
    unit = (unit_raw or "").strip().lower()
    if is_crea:
        # Ziel: mg/dL
        if "µmol/l" in unit or "umol/l" in unit:
            return value / 88.4, "mg/dL"
        # falls schon mg/dl / mg/dL -> übernehmen
        return value, "mg/dL"
    if is_cysc:
        # Ziel: mg/L
        if "mg/dl" in unit:
            return value * 10.0, "mg/L"
        # mg/L bleibt
        return value, "mg/L"
    # Andere Parameter lassen wir unverändert
    return value, unit_raw

vals_std, units_std, param_std = [], [], []
for _, r in lab.obs.iterrows():
    is_crea = bool(r["is_creatinine"])
    is_cysc = bool(r["is_cystatin_c"])
    v, u = unify_units(r["QuantitativeValue"], r["Unit"], is_crea, is_cysc)
    vals_std.append(v)
    units_std.append(u)
    param_std.append("creatinine" if is_crea else ("cystatin_c" if is_cysc else "other"))

lab.obs["Value_std"] = vals_std
lab.obs["Unit_std"] = units_std
lab.obs["Parameter_std"] = param_std

# Nur zur Kontrolle:
print("\n— Harmonisierung —")
print(lab.obs["Parameter_std"].value_counts(dropna=False))
print("Einheiten nach Harmonisierung:", lab.obs["Unit_std"].value_counts(dropna=False).to_dict())

# =============== Speichern (CSV + H5AD) ===============
# Roh-clean (zur Kontrolle)
lab.obs.to_csv(OUT_DIR / "lab_raw_clean.csv", index=False)
vis.obs.to_csv(OUT_DIR / "vis_raw_clean.csv", index=False)

# Harmonisierte Labor-Zieltabelle (nur Krea/CysC) – gut für spätere Merges/Features
lab_harm = lab.obs.loc[lab.obs["Parameter_std"].isin(["creatinine","cystatin_c"]),
                       ["PMID","SMID","CollectionTimestamp","Parameter_std","Value_std","Unit_std","Parameter","LOINC","Unit"]]
lab_harm.to_csv(OUT_DIR / "lab_crea_cysc_harmonized.csv", index=False)

# AnnData speichern
# vor lab.write(...) und vis.write(...)

def stringify_datetime_columns(df, fmt="%Y-%m-%d %H:%M:%S"):
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        df[c] = df[c].dt.tz_localize(None, nonexistent="NaT", ambiguous="NaT").dt.strftime(fmt)
    return df

lab.obs = stringify_datetime_columns(lab.obs)
vis.obs = stringify_datetime_columns(vis.obs)

lab.write(OUT_DIR / "lab_raw_clean_ehrapy.h5ad")
vis.write(OUT_DIR / "vis_raw_clean_ehrapy.h5ad")



print("\nGespeichert:")
print(" -", OUT_DIR / "lab_raw_clean.csv")
print(" -", OUT_DIR / "vis_raw_clean.csv")
print(" -", OUT_DIR / "lab_crea_cysc_harmonized.csv")
print(" -", OUT_DIR / "lab_raw_clean_ehrapy.h5ad")
print(" -", OUT_DIR / "vis_raw_clean_ehrapy.h5ad")

