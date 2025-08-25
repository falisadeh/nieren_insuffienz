#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_convert_to_h5ad.py
Konvertiert die projektweiten CSVs robust nach .h5ad und harmonisiert Typen.

Erzeugt in /h5ad:
- ops_with_patient_features.h5ad
- ops_with_crea_cysc_vis_features_with_AKI.h5ad
- analytic_patient_summary_v2.h5ad
- (optional) causal_dataset_op_level.h5ad
"""

from pathlib import Path
import pandas as pd
import numpy as np
from anndata import AnnData
import ehrapy as ep
from datetime import datetime, timezone

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
IN = {
    "ops_with_patient_features": BASE / "ops_with_patient_features.csv",
    "ops_with_crea_cysc_vis_features_with_AKI": BASE / "ops_with_crea_cysc_vis_features_with_AKI.csv",
    "analytic_patient_summary_v2": BASE / "analytic_patient_summary_v2.csv",
}
OUTD = BASE / "h5ad"
OUTD.mkdir(parents=True, exist_ok=True)

# ---------- Helper ----------
def read_any_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    # Semikolon-Fix
    if df.shape[1] == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df

def to_datetime_utc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def booleanize01(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["1","true","wahr","ja","yes"]).astype(int)

def write_h5ad(df: pd.DataFrame, out_path: Path, name: str):
    import numpy as np
    from anndata import AnnData
    from datetime import datetime, timezone
    import pandas as pd

    df = df.copy()

    # 1) Datetime-Spalten → ISO-String (erkennt auch object-Spalten mit Zeitwerten)
    def _to_iso_string(col: pd.Series) -> pd.Series:
        col_dt = pd.to_datetime(col, errors="coerce", utc=False)  # toleriert Mischtypen
        # Wenn nach dem Parsen (fast) alles NaT ist, war's wohl keine Zeitspalte → gib Original zurück
        if col_dt.notna().sum() == 0:
            return col
        return col_dt.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")

    for c in df.columns:
        # harte Typprüfung …
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_datetime64tz_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
        else:
            # … plus weiche Prüfung: enthält die object-Spalte Datumswerte?
            if df[c].dtype == "object":
                try:
                    # versuche parsbar → wenn ja, in ISO-String konvertieren
                    parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
                    if parsed.notna().sum() > 0:
                        df[c] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
                except Exception:
                    pass  # bleibt object, wird unten zu String gecastet

    # 2) Alle verbleibenden object-Spalten in reine Strings wandeln (h5py mag keine gemischten objects)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)

    # 3) Eindeutiger Index (Strings)
    if not df.index.is_unique:
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    df.index = df.index.astype(str)

    # 4) AnnData anlegen: X leer, alles in obs
    ad = AnnData(X=np.empty((len(df), 0)))
    ad.obs = df

    # 5) Metadaten
    ts = datetime.now(timezone.utc).isoformat()
    ad.uns["dataset_name"] = name
    ad.uns["created_at_utc"] = ts
    ad.uns["n_rows"] = int(ad.n_obs)
    ad.uns["n_cols"] = int(ad.n_vars)
    ad.uns["columns"] = list(df.columns)

    ad.write(out_path)
    print(f"[OK] geschrieben: {out_path} | rows={ad.n_obs} cols={ad.n_vars} (X leer, Daten in .obs)")

# ---------- 1) ops_with_patient_features ----------
if IN["ops_with_patient_features"].exists():
    df = read_any_csv(IN["ops_with_patient_features"])
    df.columns = df.columns.str.strip()
    # Alias-Fixes
    df = df.rename(columns={"Start of surgery":"Surgery_Start", "End of surgery":"Surgery_End"})
    # Zeiten
    df = to_datetime_utc(df, ["Surgery_Start","Surgery_End","AKI_Start","first_op_date","first_op_end"])
    # Dauer bauen, falls fehlt
    if "duration_hours" not in df.columns and {"Surgery_Start","Surgery_End"}.issubset(df.columns):
        df["duration_hours"] = (df["Surgery_End"] - df["Surgery_Start"]).dt.total_seconds()/3600.0
    # Numerik
    num_cols = [
        "duration_hours","age_years_at_first_op","n_ops",
        "crea_baseline","crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
        "cysc_baseline","cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
        "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
        "highest_AKI_stage_0_7","Sex_norm"
    ]
    df = to_numeric(df, [c for c in num_cols if c in df.columns])
    # AKI-Flag
    if "AKI_linked_0_7" in df.columns:
        df["AKI_linked_0_7"] = booleanize01(df["AKI_linked_0_7"])
    # is_reop ableiten, falls nicht da
    if "is_reop" not in df.columns and "n_ops" in df.columns:
        df["is_reop"] = (pd.to_numeric(df["n_ops"], errors="coerce") > 1).astype(int)
    # schreiben
    write_h5ad(df, OUTD / "ops_with_patient_features.h5ad", "ops_with_patient_features")

# ---------- 2) ops_with_crea_cysc_vis_features_with_AKI ----------
if IN["ops_with_crea_cysc_vis_features_with_AKI"].exists():
    df = read_any_csv(IN["ops_with_crea_cysc_vis_features_with_AKI"])
    df = df.rename(columns={"Start of surgery":"Surgery_Start", "End of surgery":"Surgery_End"})
    df = to_datetime_utc(df, ["Surgery_Start","Surgery_End","AKI_Start"])
    if "duration_hours" not in df.columns and {"Surgery_Start","Surgery_End"}.issubset(df.columns):
        df["duration_hours"] = (df["Surgery_End"] - df["Surgery_Start"]).dt.total_seconds()/3600.0
    num_cols = [
        "duration_hours",
        "crea_baseline","crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
        "cysc_baseline","cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
        "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
        "days_to_AKI"
    ]
    df = to_numeric(df, [c for c in num_cols if c in df.columns])
    if "AKI_linked_0_7" in df.columns:
        df["AKI_linked_0_7"] = booleanize01(df["AKI_linked_0_7"])
    write_h5ad(df, OUTD / "ops_with_crea_cysc_vis_features_with_AKI.h5ad", "ops_with_crea_cysc_vis_features_with_AKI")

# ---------- 3) analytic_patient_summary_v2 ----------
if IN["analytic_patient_summary_v2"].exists():
    df = read_any_csv(IN["analytic_patient_summary_v2"])
    df = to_datetime_utc(df, ["first_op_date","first_op_end","aki_start_date"])
    num_cols = [
        "first_op_hours","age_years_at_first_op","n_ops","total_op_hours",
        "mean_op_hours","max_op_hours","days_to_AKI","highest_AKI_stage_0_7",
        "AKI_any_0_7","AKI1_0_7","AKI2_0_7","AKI3_0_7","Sex_norm"
    ]
    df = to_numeric(df, [c for c in num_cols if c in df.columns])
    # binäre AKIs vereinheitlichen
    for b in ["AKI_any_0_7","AKI1_0_7","AKI2_0_7","AKI3_0_7"]:
        if b in df.columns:
            df[b] = booleanize01(df[b])
    write_h5ad(df, OUTD / "analytic_patient_summary_v2.h5ad", "analytic_patient_summary_v2")

# ---------- 4) (optional) Causal-Slim auf OP-Level ----------
# nur wenn Hauptsatz da ist
p = OUTD / "ops_with_patient_features.h5ad"
if p.exists():
    ad = ep.io.read_h5ad(str(p))
    obs = ad.obs.copy()
    needed = ["PMID","SMID","Procedure_ID","duration_hours","AKI_linked_0_7",
              "age_years_at_first_op","Sex_norm","n_ops"]
    avail = [c for c in needed if c in obs.columns]
    slim = obs[avail].copy()
    # is_reop aus n_ops
    if "n_ops" in slim.columns and "is_reop" not in slim.columns:
        slim["is_reop"] = (pd.to_numeric(slim["n_ops"], errors="coerce") > 1).astype(int)
    write_h5ad(slim, OUTD / "causal_dataset_op_level.h5ad", "causal_dataset_op_level")

print("Fertig.")
import ehrapy as ep
ad = ep.io.read_h5ad("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad/ops_with_patient_features.h5ad")
print(ad.obs.shape)
print(ad.obs.columns.tolist()[:20])  # erste 20 Spalten
