# 22_reperatur_aki_flags.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import ehrapy as ep

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad")
FILES = [
    "ops_with_crea_cysc_vis_features_with_AKI.h5ad",
    "ops_with_patient_features.h5ad",
    "causal_dataset_op_level.h5ad",
]

def booleanize01(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["1","true","wahr","ja","yes"]).astype(int)

def sanitize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Alle Datetime-Spalten in ISO-String wandeln (für h5ad-Speichern)."""
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S").astype(str)
    return df

for fname in FILES:
    path = BASE / fname
    ad = ep.io.read_h5ad(str(path))
    df = ad.obs.copy()

    # Zeiten parsen (lokal als datetime für Berechnung)
    for c in ["Surgery_End","AKI_Start"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    used = None

    # 1) AKI_linked_0_7_time
    if "AKI_linked_0_7_time" in df.columns and df["AKI_linked_0_7_time"].notna().any():
        df["AKI_linked_0_7"] = booleanize01(df["AKI_linked_0_7_time"])
        used = "AKI_linked_0_7_time"

    # 2) Zeitdifferenz
    if used is None and {"AKI_Start","Surgery_End"}.issubset(df.columns):
        days = (df["AKI_Start"] - df["Surgery_End"]).dt.total_seconds() / 86400.0
        df["days_to_AKI_calc"] = days
        df["AKI_linked_0_7"] = days.between(0, 7).fillna(False).astype(int)
        used = "AKI_Start/Surgery_End"

    # 3) Fallback
    if used is None and "AKI_linked_0_7" in df.columns:
        df["AKI_linked_0_7"] = booleanize01(df["AKI_linked_0_7"])
        used = "AKI_linked_0_7 (fallback)"

    if used is None:
        print(f"[WARN] {fname}: Kein AKI-Feld ableitbar.")
        continue

    vc = df["AKI_linked_0_7"].value_counts(dropna=False).sort_index()
    n0 = int(vc.get(0,0)); n1 = int(vc.get(1,0)); nt = len(df)
    print(f"{fname:45s} | Quelle: {used:20s} | AKI=1: {n1:4d} | AKI=0: {n0:4d} | total: {nt} | {n1/nt*100:.1f}%")

    # Vor dem Schreiben: Datetime → String
    df = sanitize_datetimes(df)

    # zurückschreiben
    ad.obs = df
    ad.write(str(path))
    print(" -> aktualisiert:", path)
#Kausalen Datensatz korrekt aus ops_with_patient_features.h5ad neu bauen
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import ehrapy as ep
from anndata import AnnData

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5_OPS = BASE / "h5ad" / "ops_with_patient_features.h5ad"
H5_OUT = BASE / "h5ad" / "causal_dataset_op_level.h5ad"

ad = ep.io.read_h5ad(str(H5_OPS))
df = ad.obs.copy()

# AKI robust: bevorzugt aus highest_AKI_stage_0_7 ≥ 1 (hat bei dir funktioniert)
if "highest_AKI_stage_0_7" in df.columns:
    s = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
    df["AKI_linked_0_7"] = (s >= 1).astype(int)
elif "AKI_linked_0_7" in df.columns:
    sn = pd.to_numeric(df["AKI_linked_0_7"], errors="coerce")
    df["AKI_linked_0_7"] = (sn > 0).astype(int)
else:
    raise ValueError("Kein AKI-Feld in ops_with_patient_features.h5ad gefunden.")

# Confounder/Treatment zusammenstellen
need = ["PMID","SMID","Procedure_ID",
        "duration_hours","AKI_linked_0_7",
        "age_years_at_first_op","Sex_norm","n_ops"]
avail = [c for c in need if c in df.columns]
cdf = df[avail].copy()

# is_reop aus n_ops
if "n_ops" in cdf.columns:
    cdf["is_reop"] = (pd.to_numeric(cdf["n_ops"], errors="coerce") > 1).astype(int)

# Numerik casten
for c in ["duration_hours","AKI_linked_0_7","age_years_at_first_op","Sex_norm","n_ops","is_reop"]:
    if c in cdf.columns:
        cdf[c] = pd.to_numeric(cdf[c], errors="coerce")

# schreiben: alles in .obs (X leer)
ad_out = AnnData(X=np.empty((len(cdf), 0)))
ad_out.obs = cdf
ad_out.uns["dataset_name"] = "causal_dataset_op_level"
ad_out.write(str(H5_OUT))
print("Neu geschrieben:", H5_OUT)

# kurze Kontrolle
vc = cdf["AKI_linked_0_7"].value_counts().sort_index()
print("AKI=0:", int(vc.get(0,0)), "| AKI=1:", int(vc.get(1,0)), "| Anteil AKI:",
      round(100*vc.get(1,0)/len(cdf),1), "%")



