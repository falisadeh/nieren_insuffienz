#!/usr/bin/env python3
"""
QC/Deskriptiv-Report für den erzeugten H5AD-Datensatz.
- Nutzt ep.tl.describe, falls in der ehrapy-Version vorhanden.
- Fallback: Panda-basierte Kennzahlen (count, mean, std, min, 25%, 50%, 75%, max, missing, skew, kurt, IQR, CV).
- Zusätzlich: Häufigkeitstabellen wichtiger .obs-Variablen (Sex, AKI_Stage, AKI_linked_0_7).

Eingabe:  Daten/ops_with_patient_features.h5ad
Ausgabe:  Daten/describe_overall.csv, describe_AKI_0_7_yes.csv, describe_AKI_0_7_no.csv, obs_counts.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from anndata import read_h5ad, AnnData

# Optional ehrapy (falls vorhanden)
try:
    import ehrapy as ep  # type: ignore
except Exception:  # ehrapy ist optional
    ep = None  # type: ignore

BASE_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_PATH = os.path.join(BASE_DIR, "Daten", "ops_with_patient_features.h5ad")
OUT_DIR = os.path.join(BASE_DIR, "Daten")
os.makedirs(OUT_DIR, exist_ok=True)


def safe_infer_feature_types(adata: AnnData) -> AnnData:
    if ep is None:
        return adata
    try:
        if hasattr(ep, "ad") and hasattr(ep.ad, "infer_feature_types"):
            ret = ep.ad.infer_feature_types(adata)
            if isinstance(ret, AnnData):
                return ret
    except Exception:
        pass
    return adata


def fallback_describe(adata: AnnData) -> pd.DataFrame:
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    # Basis-Stats
    desc = df.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    # Zusätzliche Kennzahlen
    desc["missing"] = df.isna().sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        desc["cv"] = desc["std"] / desc["mean"]
    desc["skew"] = df.skew(numeric_only=True)
    desc["kurt"] = df.kurt(numeric_only=True)
    desc["iqr"] = desc["75%"] - desc["25%"]
    return desc


def run_describe(adata: AnnData) -> pd.DataFrame:
    if ep is not None and hasattr(ep, "tl") and hasattr(ep.tl, "describe"):
        try:
            out = ep.tl.describe(adata)
            if isinstance(out, pd.DataFrame):
                return out
        except Exception:
            pass
    # Fallback
    return fallback_describe(adata)


def obs_counts(
    adata: AnnData, cols=("Sex", "AKI_Stage", "AKI_linked_0_7")
) -> pd.DataFrame:
    frames = []
    for c in cols:
        if c in adata.obs.columns:
            vc = adata.obs[c].value_counts(dropna=False).rename("count").to_frame()
            vc.index.name = c
            frames.append(vc)
    if frames:
        out = pd.concat(frames, axis=0)
        return out
    return pd.DataFrame()


def main():
    adata = read_h5ad(H5_PATH)
    adata = safe_infer_feature_types(adata)

    # Gesamt-Deskriptiv
    desc_all = run_describe(adata)
    desc_all.to_csv(os.path.join(OUT_DIR, "describe_overall.csv"))

    # Stratifiziert nach AKI 0–7 Tage
    if "AKI_linked_0_7" in adata.obs.columns:
        mask = adata.obs["AKI_linked_0_7"]
        # robust nach 0/1 mappen
        try:
            mask01 = mask.astype(int)
        except Exception:
            mask01 = (
                mask.astype(str)
                .str.lower()
                .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
            )
        if (mask01 == 1).any():
            desc_yes = run_describe(adata[mask01 == 1].copy())
            desc_yes.to_csv(os.path.join(OUT_DIR, "describe_AKI_0_7_yes.csv"))
        if (mask01 == 0).any():
            desc_no = run_describe(adata[mask01 == 0].copy())
            desc_no.to_csv(os.path.join(OUT_DIR, "describe_AKI_0_7_no.csv"))

    # Häufigkeiten aus .obs
    counts = obs_counts(adata)
    if not counts.empty:
        counts.to_csv(os.path.join(OUT_DIR, "obs_counts.csv"))

    print("QC-Deskriptiv erfolgreich erstellt in:", OUT_DIR)


if __name__ == "__main__":
    main()
