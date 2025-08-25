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
    # patient summary hat meist kein OP-basiertes AKI-Flag – kann man optional ergänzen
]

def booleanize01(s: pd.Series) -> pd.Series:
    s = s.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    map_ = {"true":1,"wahr":1,"ja":1,"yes":1,"1":1,
            "false":0,"falsch":0,"nein":0,"no":0,"0":0}
    return sl.map(map_).fillna(0).astype(int)

def sanitize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    # anndata kann keine datetime in .obs; daher als ISO-String speichern
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_datetime64tz_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S").astype(str)
    return df

def scan_candidates(df: pd.DataFrame) -> None:
    # Übersicht, was vorhanden ist
    print("  » Spalten (AKI-relevant):",
          [c for c in df.columns if "aki" in c.lower()] +
          [c for c in df.columns if "surgery_end" in c.lower() or "surgery_start" in c.lower() or "end of surgery" in c.lower()])
    for c in ["AKI_linked_0_7_time","AKI_linked_0_7","AKI_linked","AKI_any_0_7",
              "highest_AKI_stage_0_7","days_to_AKI","days_to_AKI_calc",
              "Surgery_End","AKI_Start","End of surgery","Start of surgery"]:
        if c in df.columns:
            nonna = int(df[c].notna().sum())
            ex = df[c].dropna().astype(str).head(3).tolist()
            print(f"    - {c}: non-NA={nonna}  Beispiele={ex}")

def rebuild_aki(df: pd.DataFrame) -> tuple[pd.Series,str]:
    used = None
    # 0) AKI-Stufe → Binär (>=1)
    if "highest_AKI_stage_0_7" in df.columns:
        s = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
        if s.notna().any():
            bin_stage = (s >= 1).astype(int)
            if bin_stage.sum() > 0:
                return bin_stage, "highest_AKI_stage_0_7>=1"

    # 1) *_time (oft 0/1 als Text/Zahl)
    if "AKI_linked_0_7_time" in df.columns and df["AKI_linked_0_7_time"].notna().any():
        s = booleanize01(df["AKI_linked_0_7_time"])
        if s.sum() > 0 or s.notna().any():
            return s, "AKI_linked_0_7_time"

    # 2) vorhandene Flags
    for flag in ["AKI_linked_0_7", "AKI_linked", "AKI_any_0_7"]:
        if flag in df.columns and df[flag].notna().any():
            s = booleanize01(df[flag])
            if s.sum() > 0 or s.notna().any():
                return s, flag

    # 3) Tage-Differenz
    #   a) bereits vorhandene days_to_AKI(_calc)
    for dayscol in ["days_to_AKI","days_to_AKI_calc"]:
        if dayscol in df.columns and pd.to_numeric(df[dayscol], errors="coerce").notna().any():
            d = pd.to_numeric(df[dayscol], errors="coerce")
            s = d.between(0,7).fillna(False).astype(int)
            if s.sum() > 0 or s.notna().any():
                return s, dayscol

    #   b) aus Zeiten berechnen (robust: auch alternative Spaltennamen)
    # normalize possible time names
    time_pairs = [
        ("Surgery_End","AKI_Start"),
        ("End of surgery","AKI_Start"),
        ("Surgery_End","aki_start_date"),
        ("End of surgery","aki_start_date"),
    ]
    for se_col, ak_col in time_pairs:
        if se_col in df.columns and ak_col in df.columns:
            se = pd.to_datetime(df[se_col], errors="coerce")
            ak = pd.to_datetime(df[ak_col], errors="coerce")
            d = (ak - se).dt.total_seconds() / 86400.0
            if d.notna().any():
                s = d.between(0,7).fillna(False).astype(int)
                if s.sum() > 0 or s.notna().any():
                    df["days_to_AKI_calc"] = d
                    return s, f"{ak_col}-{se_col}"

    # 4) nichts gefunden
    return pd.Series(np.nan, index=df.index), "NA"

for fname in FILES:
    path = BASE / fname
    if not path.exists():
        print(f"[SKIP] fehlt: {path}")
        continue

    ad = ep.io.read_h5ad(str(path))
    df = ad.obs.copy()
    print(f"\n=== {fname} ===")
    print(f"  Zeilen={len(df)}  Spalten={len(df.columns)}")
    scan_candidates(df)

    # REBUILD
    aki, src = rebuild_aki(df)
    if src == "NA":
        print("  -> Kein AKI ableitbar. (Bitte Spalten im Log oben prüfen.)")
        continue

    df["AKI_linked_0_7"] = aki
    vc = df["AKI_linked_0_7"].value_counts(dropna=False).sort_index()
    n0 = int(vc.get(0,0)); n1 = int(vc.get(1,0))
    print(f"  -> Quelle genutzt: {src} | AKI=1: {n1} | AKI=0: {n0} | Anteil AKI: {n1/len(df)*100:.1f}%")

    # Datumsfelder vor dem Schreiben sichern (als String)
    df = sanitize_datetimes(df)

    # Speichern
    ad.obs = df
    ad.write(str(path))
    print("  [OK] gespeichert:", path)
