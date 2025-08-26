#!/usr/bin/env python3
"""
Erstellt aus den bereitgestellten Originaldaten (VIS, Labor, AKI-Label, HLM-Operationen,
Patientenstammdaten) einen operationsebene-basierten Datensatz als AnnData-Objekt und
speichert ihn als .h5ad im Ordner "Daten". Fehlende Werte in numerischen Variablen werden
imputiert (Median für numerische, Modus für kategoriale in .obs; .X bleibt numerisch).

Hinweise / Annahmen (nur basierend auf dem gegebenen Kontext):
- Schlüssel/Grain: pro Zeile eine HLM-Operation (Procedure_ID). Verknüpfung über PMID/SMID/Procedure_ID.
- Zeitachsen: Surgery_Start/End (aus HLM Operationen). Alle Zeitfenster beziehen sich auf Surgery_End.
- AKI-Verknüpfung: erste AKI-Episode pro Patient, deren Startzeitpunkt >= Surgery_End; daraus days_to_AKI
  und AKI_linked_0_7 (0–7 Tage). Stage aus "Decision" (AKI 1/2/3) extrahiert.
- Laborfeatures (Creatinin/Cystatin C):
  * baseline: letzter Messwert ≤ Surgery_Start
  * peak_0_48: Maximum im Intervall [Surgery_End, Surgery_End+48h]
  * delta_0_48 = peak - baseline
  * rate_0_48 = delta / (Stunden zwischen baseline- und peak-Zeitpunkt) – falls beide Zeitpunkte vorhanden
- VIS-Features:
  * vis_mean_0_24, vis_max_0_24 im Intervall [0, 24h]
  * vis_max_6_24 im Intervall (6, 24h]
  * vis_auc_0_24 und vis_auc_0_48 via Trapezregel über Zeit (Stunden, sortierte Messpunkte)
- Altersberechnung: (Surgery_Start - DateOfBirth).days / 365.2425

So viel wie möglich wird ehrapy genutzt, ohne externe Abhängigkeiten zu erzwingen:
- Falls verfügbar, wird ep.ad.infer_feature_types(adata) aufgerufen.
- Speicherung erfolgt regulär via AnnData (adata.write_h5ad). Bei Bedarf können weitere
  ehrapy-Schritte (z. B. Visualisierung) nachgelagert erfolgen.

Pfadstruktur (aus Kontext):
  Basis: /Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer
  Eingabe: Original Daten/*.csv (Semikolon-getrennt)
  Ausgabe: Daten/ops_with_patient_features.h5ad
"""

from __future__ import annotations
import os
import re
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

# Optional: ehrapy, wenn vorhanden
try:
    import ehrapy as ep  # type: ignore
except Exception:  # ehrapy ist optional – Skript läuft auch ohne
    ep = None  # type: ignore

BASE_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
IN_DIR = os.path.join(BASE_DIR, "Original Daten")
OUT_DIR = os.path.join(BASE_DIR, "Daten")
OUT_H5AD = os.path.join(OUT_DIR, "ops_with_patient_features.h5ad")

# Eingabedateien (aus Kontext)
PATH_VIS = os.path.join(IN_DIR, "VIS.csv")
PATH_LAB = os.path.join(IN_DIR, "Laboratory_Kreatinin+CystatinC.csv")
PATH_AKI = os.path.join(IN_DIR, "AKI Label.csv")
PATH_PAT = os.path.join(IN_DIR, "Patient Master Data.csv")
PATH_OPS = os.path.join(IN_DIR, "HLM Operationen.csv")

# -------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------


def read_scsv(path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Liest Semikolon-CSV mit möglichem BOM, trimmt Spalten, optional parse_dates."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def ensure_cols(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: erwartete Spalten fehlen: {missing}")


def rename_ops_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
    }
    df = df.rename(columns=mapping)
    df.columns = df.columns.str.strip()
    return df


def stage_from_decision(s: Optional[str]) -> Optional[int]:
    if isinstance(s, str):
        m = re.search(r"AKI\s*(\d)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def hours_between(t1: pd.Timestamp, t2: pd.Timestamp) -> Optional[float]:
    if pd.isna(t1) or pd.isna(t2):
        return None
    return (t2 - t1).total_seconds() / 3600.0


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """AUC via Trapezregel; kompatibel zu älteren numpy-Versionen."""
    if len(y) < 2:
        return 0.0
    try:
        return float(np.trapezoid(y, x))  # neuere NumPy
    except Exception:
        return float(np.trapz(y, x))  # Fallback


# -------------------------------------------------------------
# Feature-Berechnung
# -------------------------------------------------------------


@dataclass
class Window:
    start_h: float
    end_h: float


WIN_LAB_POST = Window(0.0, 48.0)
WIN_VIS_0_24 = Window(0.0, 24.0)
WIN_VIS_6_24 = Window(6.0, 24.0)


def compute_lab_features_for_param(
    ops: pd.DataFrame,
    labs: pd.DataFrame,
    param_mask: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """Berechnet baseline/peak/delta/rate für ein Labor-Parameter je OP."""
    records = []
    labs_param = labs.loc[param_mask].copy()
    # Für präzise Zuordnung: nach Möglichkeit über SMID; sonst über PMID
    for _, op in ops.iterrows():
        pmid = op["PMID"]
        smid = op.get("SMID", np.nan)
        t_start = op["Surgery_Start"]
        t_end = op["Surgery_End"]
        # Kandidaten: gleiche SMID, sonst gleicher PMID
        if not pd.isna(smid) and ("SMID" in labs_param.columns):
            cand = labs_param[
                (labs_param["PMID"] == pmid) & (labs_param["SMID"] == smid)
            ]
        else:
            cand = labs_param[(labs_param["PMID"] == pmid)]
        # baseline: letzter Messwert ≤ Surgery_Start
        pre = cand[cand["CollectionTimestamp"] <= t_start].sort_values(
            "CollectionTimestamp"
        )
        baseline_val = pre["QuantitativeValue"].iloc[-1] if len(pre) else np.nan
        baseline_ts = pre["CollectionTimestamp"].iloc[-1] if len(pre) else pd.NaT
        # Peak 0-48h postop
        post = cand[
            (cand["CollectionTimestamp"] >= t_end)
            & (
                cand["CollectionTimestamp"]
                <= t_end + pd.Timedelta(hours=WIN_LAB_POST.end_h)
            )
        ]
        peak_val = post["QuantitativeValue"].max() if len(post) else np.nan
        # Zeitpunkt des Peaks (bei Mehrfach-Max erstem Auftreten)
        if len(post):
            peak_ts = post.loc[
                post["QuantitativeValue"].idxmax(), "CollectionTimestamp"
            ]
        else:
            peak_ts = pd.NaT
        # Delta & Rate
        delta = (
            (peak_val - baseline_val)
            if (not pd.isna(peak_val) and not pd.isna(baseline_val))
            else np.nan
        )
        rate = np.nan
        if (
            (not pd.isna(delta))
            and (not pd.isna(baseline_ts))
            and (not pd.isna(peak_ts))
        ):
            dt_h = hours_between(baseline_ts, peak_ts)
            if dt_h and dt_h > 0:
                rate = delta / dt_h
        records.append(
            {
                "Procedure_ID": op["Procedure_ID"],
                f"{prefix}_baseline": baseline_val,
                f"{prefix}_peak_0_48": peak_val,
                f"{prefix}_delta_0_48": delta,
                f"{prefix}_rate_0_48": rate,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_vis_features(ops: pd.DataFrame, vis: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, op in ops.iterrows():
        pmid = op["PMID"]
        smid = op.get("SMID", np.nan)
        t_end = op["Surgery_End"]
        # Kandidaten via SMID, sonst via PMID
        if ("SMID" in vis.columns) and not pd.isna(smid):
            cand = vis[(vis["PMID"] == pmid) & (vis["SMID"] == smid)].copy()
        else:
            cand = vis[(vis["PMID"] == pmid)].copy()
        cand = cand.sort_values("Timestamp")
        # relative Stunden zu OP-Ende
        cand["rel_h"] = (cand["Timestamp"] - t_end).dt.total_seconds() / 3600.0
        # Fenster filtern
        w0_24 = cand[
            (cand["rel_h"] >= WIN_VIS_0_24.start_h)
            & (cand["rel_h"] <= WIN_VIS_0_24.end_h)
        ]
        w6_24 = cand[
            (cand["rel_h"] > WIN_VIS_6_24.start_h)
            & (cand["rel_h"] <= WIN_VIS_6_24.end_h)
        ]
        # Kennwerte
        mean_0_24 = w0_24["vis"].mean() if len(w0_24) else np.nan
        max_0_24 = w0_24["vis"].max() if len(w0_24) else np.nan
        max_6_24 = w6_24["vis"].max() if len(w6_24) else np.nan
        # AUCs
        auc_0_24 = (
            trapezoid(w0_24["vis"].to_numpy(), w0_24["rel_h"].to_numpy())
            if len(w0_24)
            else 0.0
        )
        w0_48 = cand[(cand["rel_h"] >= 0.0) & (cand["rel_h"] <= 48.0)]
        auc_0_48 = (
            trapezoid(w0_48["vis"].to_numpy(), w0_48["rel_h"].to_numpy())
            if len(w0_48)
            else 0.0
        )
        records.append(
            {
                "Procedure_ID": op["Procedure_ID"],
                "vis_mean_0_24": mean_0_24,
                "vis_max_0_24": max_0_24,
                "vis_max_6_24": max_6_24,
                "vis_auc_0_24": auc_0_24,
                "vis_auc_0_48": auc_0_48,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_age_years(dob: pd.Series, t: pd.Series) -> pd.Series:
    return (t - dob).dt.days / 365.2425


def link_aki_to_ops(ops: pd.DataFrame, aki: pd.DataFrame) -> pd.DataFrame:
    """Verknüpft pro OP die erste AKI-Episode (Start >= Surgery_End) desselben Patienten."""
    # Nur benötigte Spalten; standardisiere Namen
    aki = aki.copy()
    aki["AKI_Stage"] = aki["Decision"].map(stage_from_decision)
    aki = aki.rename(
        columns={"Start": "AKI_Start", "End": "AKI_End", "Duartion": "AKI_Duration_ms"}
    )
    aki = aki.sort_values(["PMID", "AKI_Start"])  # für merge_asof

    out_records = []
    for _, op in ops.iterrows():
        pmid = op["PMID"]
        t_end = op["Surgery_End"]
        cand = aki[(aki["PMID"] == pmid) & (aki["AKI_Start"] >= t_end)].copy()
        if len(cand):
            row = cand.iloc[0]
            days_to = (row["AKI_Start"] - t_end).total_seconds() / 86400.0
            dur_days = (
                (row["AKI_Duration_ms"] / 1000.0 / 86400.0)
                if not pd.isna(row["AKI_Duration_ms"])
                else np.nan
            )
            out_records.append(
                {
                    "Procedure_ID": op["Procedure_ID"],
                    "AKI_Start": row["AKI_Start"],
                    "AKI_End": row["AKI_End"],
                    "AKI_Duration_days": dur_days,
                    "AKI_Stage": row["AKI_Stage"],
                    "days_to_AKI": days_to,
                    "AKI_linked_0_7": 1 if (days_to >= 0 and days_to <= 7) else 0,
                }
            )
        else:
            out_records.append(
                {
                    "Procedure_ID": op["Procedure_ID"],
                    "AKI_Start": pd.NaT,
                    "AKI_End": pd.NaT,
                    "AKI_Duration_days": np.nan,
                    "AKI_Stage": np.nan,
                    "days_to_AKI": np.nan,
                    "AKI_linked_0_7": 0,
                }
            )
    return pd.DataFrame(out_records)


# -------------------------------------------------------------
# Hauptpipeline
# -------------------------------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Daten einlesen ---
    ops = read_scsv(
        PATH_OPS, parse_dates=["Start of surgery", "End of surgery"]
    )  # werden gleich umbenannt
    ops = rename_ops_columns(ops)
    ensure_cols(
        ops,
        ["PMID", "SMID", "Procedure_ID", "Surgery_Start", "Surgery_End"],
        "HLM Operationen",
    )

    pat = read_scsv(PATH_PAT, parse_dates=["DateOfBirth", "DateOfDie"])
    ensure_cols(pat, ["PMID", "Sex", "DateOfBirth"], "Patient Master Data")

    vis = read_scsv(PATH_VIS, parse_dates=["Timestamp"])  # VIS.csv
    ensure_cols(vis, ["PMID", "SMID", "Timestamp", "vis"], "VIS")

    lab = read_scsv(
        PATH_LAB, parse_dates=["CollectionTimestamp"]
    )  # Laboratory_Kreatinin+CystatinC.csv
    ensure_cols(
        lab,
        ["PMID", "SMID", "CollectionTimestamp", "Parameter", "QuantitativeValue"],
        "Labor",
    )

    aki = read_scsv(PATH_AKI, parse_dates=["Start", "End"])  # AKI Label.csv
    ensure_cols(aki, ["PMID", "Duartion", "Start", "End", "Decision"], "AKI Label")

    # --- Grundmerkmale OP + Patient ---
    df = ops.merge(
        pat[["PMID", "Sex", "DateOfBirth", "DateOfDie"]], on="PMID", how="left"
    )
    df["duration_hours"] = (
        df["Surgery_End"] - df["Surgery_Start"]
    ).dt.total_seconds() / 3600.0
    df["duration_minutes"] = df["duration_hours"] * 60.0
    df["age_years_at_op"] = compute_age_years(
        df["DateOfBirth"], df["Surgery_Start"]
    )  # Jahre

    # --- Laborfeatures (Creatinin + Cystatin C) ---
    # Numerische Typen erzwingen
    lab["QuantitativeValue"] = pd.to_numeric(lab["QuantitativeValue"], errors="coerce")
    # Parameter-Heuristik (nur basierend auf Kontext-Stichprobe)
    params = lab["Parameter"].astype(str)
    mask_crea = params.str.contains("Creat", case=False) | params.str.contains(
        "S-Creat", case=False
    )
    mask_cysc = params.str.contains("Cyst", case=False)

    feat_crea = (
        compute_lab_features_for_param(df, lab, mask_crea, prefix="crea")
        if mask_crea.any()
        else pd.DataFrame({"Procedure_ID": df["Procedure_ID"]})
    )
    feat_cysc = (
        compute_lab_features_for_param(df, lab, mask_cysc, prefix="cysc")
        if mask_cysc.any()
        else pd.DataFrame({"Procedure_ID": df["Procedure_ID"]})
    )

    # --- VIS-Features ---
    vis["vis"] = pd.to_numeric(vis["vis"], errors="coerce")
    feat_vis = compute_vis_features(df, vis)

    # --- AKI-Verknüpfung ---
    aki_link = link_aki_to_ops(df, aki)

    # --- Zusammenführen ---
    feats = (
        df.merge(feat_crea, on="Procedure_ID", how="left")
        .merge(feat_cysc, on="Procedure_ID", how="left")
        .merge(feat_vis, on="Procedure_ID", how="left")
        .merge(aki_link, on="Procedure_ID", how="left")
    )

    # --- AnnData-Erstellung ---
    # Numerische Spalten für .X; Rest in .obs
    # (Strings/Datetimes verbleiben in .obs; .X wird für ML/Imputation verwendet)
    # Datetime-Spalten für H5AD: in ISO-Strings wandeln und Original entfernen (h5py-Sicherheit)
    obs = feats.copy()
    datetime_cols = obs.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    for c in datetime_cols:
        obs[c + "_iso"] = (
            obs[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(~obs[c].isna(), other=np.nan)
        )
    if datetime_cols:
        obs = obs.drop(columns=list(datetime_cols))

    # .X: nur numerisch, **ohne** IDs/Labels (PMID, SMID, Procedure_ID, AKI-Variablen)
    exclude_x = {"PMID", "SMID", "Procedure_ID", "AKI_Stage", "AKI_linked_0_7"}
    num_cols = [
        c for c in feats.select_dtypes(include=["number"]).columns if c not in exclude_x
    ]
    X_df = feats[num_cols].copy()

    # --- Imputation fehlender Werte ---
    # Numerisch (Median); Kategorial in .obs (Modus je Spalte)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    # Kategoriale/Objektspalten (nur in obs): Modus-Imputation, falls sinnvoll
    obj_cols = obs.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if obs[c].isna().any():
            try:
                mode_val = obs[c].mode(dropna=True).iloc[0]
                obs[c] = obs[c].fillna(mode_val)
            except Exception:
                pass  # falls kein Modus ermittelbar

    # AnnData zusammenbauen
    adata = AnnData(X=X_df.to_numpy(), obs=obs)
    adata.var_names = X_df.columns.astype(str)

    # Optional: ehrapy Feature-Typen ableiten (falls vorhanden)
    if ep is not None:
        try:
            if hasattr(ep, "ad") and hasattr(ep.ad, "infer_feature_types"):
                res = ep.ad.infer_feature_types(
                    adata
                )  # kann in-place sein (None) oder ein neues AnnData zurückgeben
                if isinstance(res, AnnData):
                    adata = res
        except Exception:
            pass
    # --- KATEGORIENAUSPRÄGUNGEN IN .obs FESTLEGEN (vor dem Speichern!) ---
    for c in ("AKI_linked_0_7", "AKI_Stage", "Sex"):
        if c in adata.obs.columns:
            adata.obs[c] = adata.obs[c].astype("category")

    # H5AD speichern
    adata.write_h5ad(OUT_H5AD)

    # H5AD speichern
    if adata is None:
        raise RuntimeError(
            "'adata' ist None – vermutlich durch eine In-Place-Funktion überschrieben. Bitte Skriptabschnitt zu 'infer_feature_types' prüfen."
        )
    adata.write_h5ad(OUT_H5AD)

    # Kurze Zusammenfassung
    print(f"AnnData geschrieben: {OUT_H5AD}")
    print(f"n_obs (OPs): {adata.n_obs} | n_vars (numerische Features): {adata.n_vars}")
    print("Numerische Variablen in .X (erste 10):", list(adata.var_names[:10]))


if __name__ == "__main__":
    main()
