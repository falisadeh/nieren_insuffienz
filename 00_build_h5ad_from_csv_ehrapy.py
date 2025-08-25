#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
import pandas as pd
from anndata import AnnData

BASE = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")

# Quellen (Semikolon-getrennt bei CSVs)
CSV_OP_MASTER = os.path.join(BASE, "analytic_ops_master_ehrapy.csv")      # 1209 OPs, Procedure_ID eindeutig
CSV_PAT_SUM   = os.path.join(BASE, "analytic_patient_summary.csv")        # 1067 Patienten
CSV_LATENZ    = os.path.join(BASE, "AKI_Latenz_alle.csv")                 # 537 Patienten (erste OP)
CSV_FEATURES  = os.path.join(BASE, "ehrapy_input_index_ops.csv")          # Labor/Feature-Tabelle (Procedure_ID-basiert)

H5AD_OUT = os.path.join(BASE, "h5ad", "ops_with_patient_features_merged.h5ad")
os.makedirs(os.path.dirname(H5AD_OUT), exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def read_csv_sc(path: str, index_col=None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # probiere zuerst ";", falle auf "," zurück
    try:
        df = pd.read_csv(path, sep=";", dtype=str, index_col=index_col)
        if df.shape[1] == 1:  # wahrscheinlich falscher Separator
            df = pd.read_csv(path, sep=",", dtype=str, index_col=index_col)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str, index_col=index_col)
    return df


def to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_dt(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def norm_sex(s: pd.Series) -> pd.Series:
    mapping = {
        "m":0,"male":0,"mann":0,"masculino":0,"masculin":0,"männlich":0,"männl":0,"herr":0,"boy":0,"b":0,
        "f":1,"female":1,"weiblich":1,"weibl":1,"feminin":1,"féminin":1,"frau":1,"girl":1,"g":1
    }
    return (s.astype(str).str.strip().str.lower()
              .replace({"nan": None, "none": None, "": None})
              .map(mapping).astype("float64"))

def derive_aki0_7(df: pd.DataFrame) -> pd.Series:
    """AKI_linked_0_7 robust ableiten (0..7 Tage nach OP)."""
    y = pd.Series(0, index=df.index, dtype=int)
    if "days_to_AKI_calc" in df.columns:
        d = pd.to_numeric(df["days_to_AKI_calc"], errors="coerce"); y1=((d>=0)&(d<=7)).astype(int)
        if y1.sum()>0: return y1
    if "days_to_AKI" in df.columns:
        d = pd.to_numeric(df["days_to_AKI"], errors="coerce"); y2=((d>=0)&(d<=7)).astype(int)
        if y2.sum()>0: return y2
    if "AKI_linked_0_7_time" in df.columns:
        t = pd.to_numeric(df["AKI_linked_0_7_time"], errors="coerce")
        uniq = set(t.dropna().unique().tolist())
        if uniq.issubset({0,1}):
            y3 = t.fillna(0).astype(int)
            if y3.sum()>0: return y3
        y3b=((t>=0)&(t<=7)).astype(int)
        if y3b.sum()>0: return y3b
    if "highest_AKI_stage_0_7" in df.columns:
        st = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce"); y4=(st>0).astype(int)
        if y4.sum()>0: return y4
    # Zeitstempel
    end_candidates = ["Surgery_End","surgery_end","Surgery_End_ts","end_of_surgery","OP_Ende"]
    start_candidates = ["AKI_Start","aki_start","AKI_Start_ts"]
    end_col = next((c for c in end_candidates if c in df.columns), None)
    start_col = next((c for c in start_candidates if c in df.columns), None)
    if end_col and start_col:
        se = pd.to_datetime(df[end_col], errors="coerce", utc=True)
        ak = pd.to_datetime(df[start_col], errors="coerce", utc=True)
        delta_days = (ak - se).dt.total_seconds()/86400.0
        y5=((delta_days>=0)&(delta_days<=7)).astype(int).fillna(0).astype(int)
        if y5.sum()>0: return y5
    return y

def normalize_sex_series(s: pd.Series) -> pd.Series:
    """Mappt diverse Schreibweisen auf 0/1 (0=männlich, 1=weiblich)."""
    mapping = {
        "m":0,"male":0,"mann":0,"masculino":0,"masculin":0,"männlich":0,"männl":0,"herr":0,"boy":0,"b":0,
        "f":1,"female":1,"weiblich":1,"weibl":1,"feminin":1,"féminin":1,"frau":1,"girl":1,"g":1
    }
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() > 0 and set(s_num.dropna().unique()).issubset({0,1}):
        return s_num.astype("float64")
    st = (s.astype(str).str.strip().str.lower()
              .replace({"nan": None, "none": None, "": None}))
    return st.map(mapping).astype("float64")

def unify_sex_norm(df: pd.DataFrame):
    """
    Erzeugt eine einheitliche Sex_norm (0/1), falls möglich.
    Priorität: 'Sex_norm' -> 'Sex_norm_pat' -> sonstige Kandidaten.
    """
    for col in ["Sex_norm", "Sex_norm_pat"]:
        if col in df.columns:
            sn = normalize_sex_series(df[col])
            if sn.notna().sum() >= max(10, int(0.05 * len(sn))):
                return sn
    pat = re.compile(r"(sex|gender|geschl|geschlecht|genus|sexo)", re.IGNORECASE)
    cand = [c for c in df.columns if pat.search(str(c))]
    for col in cand:
        sn = normalize_sex_series(df[col])
        if sn.notna().sum() >= max(10, int(0.05 * len(sn))):
            return sn
    return None

def sanitize_datetimes_for_anndata(
    df: pd.DataFrame,
    datetime_cols = ("Surgery_Start","Surgery_End","AKI_Start","earliest_op","latest_op","First_OP"),
    to_string: bool = True,
) -> pd.DataFrame:
    """
    Macht NUR die whitelisted Zeitspalten AnnData-kompatibel (tz entfernen, dann String oder naive datetime).
    """
    out = df.copy()
    touched = []
    for c in datetime_cols:
        if c not in out.columns:
            continue
        s = out[c]
        # Falls noch String: versuchen zu parsen
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", utc=True)
        # Wenn wirklich Datetime: tz entfernen, dann formatieren
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                if getattr(s.dt, "tz", None) is not None:
                    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    s = s.dt.tz_localize(None)
            except Exception:
                pass
            out[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S") if to_string else s.astype("datetime64[ns]")
            touched.append(c)
    if touched:
        print("Sanitized datetime columns:", touched)
    return out

# -------------------------
# Main
# -------------------------
def main() -> int:
    # 1) Laden
    op    = read_csv_sc(CSV_OP_MASTER)                 # OP-Master (hat Procedure_ID/OP_ID)
    ps    = read_csv_sc(CSV_PAT_SUM)
    lat   = read_csv_sc(CSV_LATENZ)
    feats = read_csv_sc(CSV_FEATURES, index_col=0)     # <- WICHTIG: Index einlesen!

    # Falls die Feature-CSV keinen Procedure_ID-Header hat, aus dem Index ableiten
    if "Procedure_ID" not in feats.columns:
        # Index -> Series (damit .str-Methoden sauber funktionieren)
        sidx = pd.Series(feats.index.astype(str), index=feats.index)

        # Fall A: Index ist bereits genau 9-stellig
        mask_full = sidx.str.fullmatch(r"\d{9}").fillna(False)

        # Fall B: ansonsten die letzte 9-stellige Zahl am Ende extrahieren (z. B. "..._300004306")
        proc_from_regex = sidx.str.extract(r"(\d{9})$", expand=False)

        # Bevorzugt den vollständigen Match, sonst Regex-Extraktion
        feats["Procedure_ID"] = sidx.where(mask_full, proc_from_regex)

    # Nur Zeilen behalten, bei denen wir eine ID ermitteln konnten
    feats = feats[feats["Procedure_ID"].notna()].copy()




    # 2) Basistypen
    # OP-Master
    to_dt(op, ["Surgery_Start", "Surgery_End", "AKI_Start"])
    to_num(op, ["duration_hours", "duration_minutes", "days_to_AKI", "AKI_linked", "AKI_linked_0_7"])
    if "Sex_norm" in op.columns:
        op["Sex_norm"] = norm_sex(op["Sex_norm"])

    # Patient-Summary
    to_dt(ps, ["earliest_op", "latest_op", "AKI_Start"])
    to_num(ps, ["n_ops"])
    if "Sex_norm" in ps.columns:
        ps["Sex_norm"] = norm_sex(ps["Sex_norm"])

    # Latenz (erste OP) – rein informativ
    to_dt(lat, ["First_OP", "AKI_Start"])
    to_num(lat, ["days_to_AKI"])

    # Features (Labor etc.)
    numeric_candidates = [
        "crea_baseline","crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
        "cysc_baseline","cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
        "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
        "age_years_at_first_op","n_ops","highest_AKI_stage_0_7","duration_hours",
        "days_to_AKI","days_to_AKI_calc","AKI_linked_0_7","AKI_any_0_7"
    ]
    to_num(feats, [c for c in numeric_candidates if c in feats.columns])

    # 3) Merge: OP-Master (links) ⟵ Features (adaptiv)
    merged = op.copy()
    if "Procedure_ID" in op.columns and "Procedure_ID" in feats.columns:
        merged = op.merge(feats, how="left", on="Procedure_ID", suffixes=("", "_feat"))
        print(f"Join OP ⟵ Features on Procedure_ID: shape={merged.shape}")
    elif "Procedure_ID" in op.columns and "OP_ID" in feats.columns:
        merged = op.merge(feats, how="left", left_on="Procedure_ID", right_on="OP_ID", suffixes=("", "_feat"))
        print(f"Join OP ⟵ Features on Procedure_ID=OP_ID: shape={merged.shape}")
    elif all(col in op.columns for col in ["PMID","SMID"]) and all(col in feats.columns for col in ["PMID","SMID"]):
        merged = op.merge(feats, how="left", on=["PMID","SMID"], suffixes=("", "_feat"))
        print(f"Join OP ⟵ Features on PMID+SMID: shape={merged.shape}")
    else:
        print("WARN: Kein gemeinsamer Schlüssel für Features gefunden (Procedure_ID / OP_ID / PMID+SMID). Feature-Join übersprungen.")
        # optional debug:
        # print("OP cols:", list(op.columns)[:50])
        # print("FEAT cols:", list(feats.columns)[:50])

    # 4) Merge: + Patient-Summary auf PMID (nur ergänzen)
    if "PMID" in merged.columns and "PMID" in ps.columns:
        add_cols = [c for c in ["n_ops", "Sex_norm"] if c in ps.columns]
        merged = merged.merge(ps[["PMID"] + add_cols], how="left", on="PMID", suffixes=("", "_pat"))
        # Sex_norm auffüllen, falls in OP leer
        if "Sex_norm" in merged.columns and "Sex_norm_pat" in merged.columns:
            mask = merged["Sex_norm"].isna() & merged["Sex_norm_pat"].notna()
            if mask.any():
                merged.loc[mask, "Sex_norm"] = merged.loc[mask, "Sex_norm_pat"]
            merged = merged.drop(columns=["Sex_norm_pat"])
        print(f"Join ⟵ patient_summary on PMID: shape={merged.shape}")
    else:
        print("WARN: PMID fehlt in einer Tabelle, Patient-Summary-Join übersprungen.")

    # 5) Label sicherstellen/ableiten
    if "AKI_linked_0_7" in merged.columns:
        merged["AKI_linked_0_7"] = pd.to_numeric(merged["AKI_linked_0_7"], errors="coerce").fillna(0).astype(int)
    else:
        merged["AKI_linked_0_7"] = 0

    if merged["AKI_linked_0_7"].nunique() < 2:
        print("AKI_linked_0_7 einwertig → ableiten ...")
        merged["AKI_linked_0_7"] = derive_aki0_7(merged)

    print("AKI_linked_0_7 Verteilung:", merged["AKI_linked_0_7"].value_counts().to_dict())

    # 6) Sex_norm vereinheitlichen (0/1)
    sn = unify_sex_norm(merged)
    if sn is not None:
        merged["Sex_norm"] = sn
        print("Sex_norm vereinheitlicht. non-null:", int(sn.notna().sum()))
    else:
        if "Sex_norm" not in merged.columns:
            merged["Sex_norm"] = np.nan
        print("Hinweis: Sex_norm konnte nicht abgeleitet werden (bleibt leer).")

    # 7) Zeitspalten AnnData-kompatibel machen (→ Strings = sicherste Variante)
    merged = sanitize_datetimes_for_anndata(merged, to_string=True)

    # 8) Index hübsch setzen
    merged = merged.reset_index(drop=True)

    # 9) AnnData bauen & speichern
    adata = AnnData(X=None, obs=merged)
    adata.uns["build_info"] = {
        "sources": {
            "op_master": os.path.basename(CSV_OP_MASTER),
            "patient_summary": os.path.basename(CSV_PAT_SUM),
            "features": os.path.basename(CSV_FEATURES),
            "latency": os.path.basename(CSV_LATENZ),
        },
        "join_keys": {"features": "Procedure_ID/OP_ID/PMID+SMID (adaptiv)", "patient_summary": "PMID"},
    }
    adata.write(H5AD_OUT)
    print("Geschrieben:", H5AD_OUT)

    # 10) Kurzbericht
    print("\n== Bericht ==")
    print("Rows:", len(merged), "Cols:", merged.shape[1])
    for col in ["Procedure_ID", "OP_ID", "PMID", "SMID"]:
        if col in merged.columns:
            print(f"{col}: non-null={merged[col].notna().sum()}, unique={merged[col].nunique(dropna=True)}")
    if "Sex_norm" in merged.columns:
        print("Sex_norm non-null:", int(merged["Sex_norm"].notna().sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
