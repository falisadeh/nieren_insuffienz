#!/usr/bin/env python3
"""
Beschreibt kategoriale Features und erstellt Streudiagramme (ehrapy-first, robust mit Fallbacks).

Was das Skript macht:
1) Lädt das H5AD (OP-Ebene): Daten/ops_with_patient_features.h5ad
2) Fügt/vereinheitlicht folgende .obs-Variablen:
   - had_aki            : AKI-Episode innerhalb 0–7 Tage (0/1 Kategorie)
   - max_aki_stage      : Maximum der AKI-Stufen innerhalb 0–7 Tage (0/1/2/3)
   - had_transplantation: aus HLM Operationen ('Tx?' == 'Tx')
3) eh.tl.describe_categorical(…) wenn verfügbar; sonst Fallback mit pandas (Anzahl & Anteil).
4) Erstellt ein Streudiagramm: x=age_years_at_op (AgeAtFirstSurgery),
   y=crea_peak_0_48 (peak_creatinine_post_op), eingefärbt nach had_aki.

Ausgaben:
- Daten/categorical_summary.csv
- Diagramme/scatter_age_vs_crea_peak_by_had_aki.png
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import read_h5ad, AnnData

# ehrapy optional
try:
    import ehrapy as ep  # type: ignore
except Exception:
    ep = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUTD = os.path.join(BASE, "Daten")
OUTP = os.path.join(BASE, "Diagramme")
os.makedirs(OUTD, exist_ok=True)
os.makedirs(OUTP, exist_ok=True)

# Originale CSVs (für had_transplantation & max_aki_stage):
OPS_CSV = os.path.join(BASE, "Original Daten", "HLM Operationen.csv")
AKI_CSV = os.path.join(BASE, "Original Daten", "AKI Label.csv")

# ---------------------------- Helpers ----------------------------


def read_scsv(path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    df.columns = df.columns.str.strip()
    return df


stage_re = re.compile(r"AKI\s*(\d)")


def stage_from_decision(s: str | None) -> float | np.nan:
    if isinstance(s, str):
        m = stage_re.search(s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return np.nan
    return np.nan


# ------------------------- Core functions ------------------------


def ensure_obs_categoricals(adata: AnnData) -> AnnData:
    # had_transplantation aus HLM Operationen
    try:
        ops = read_scsv(
            OPS_CSV, parse_dates=["Start of surgery", "End of surgery"]
        )  # Spaltennamen werden hier nicht benötigt
        ops = ops.rename(
            columns={
                "Start of surgery": "Surgery_Start",
                "End of surgery": "Surgery_End",
            }
        )
        if "Tx?" in ops.columns:
            tx_map = ops[["Procedure_ID", "Tx?"]].copy()
            tx_map["had_transplantation"] = (
                tx_map["Tx?"].astype(str).str.upper() == "TX"
            ).astype(int)
            adata.obs = adata.obs.merge(
                tx_map[["Procedure_ID", "had_transplantation"]],
                on="Procedure_ID",
                how="left",
            )
    except Exception:
        pass

    # had_aki & max_aki_stage aus AKI Label innerhalb 0–7 Tage nach Surgery_End
    try:
        aki = read_scsv(AKI_CSV, parse_dates=["Start", "End"])  # Duartion, Decision
        aki["AKI_Stage_int"] = aki["Decision"].map(stage_from_decision).astype("float")
        # Join pro OP über PMID + Surgery_End (im H5 liegen *_iso Strings, die im Build als ISO gespeichert wurden)
        # Wir brauchen echte Timestamps; hole Surgery_End aus den ISO-Strings zurück
        if "Surgery_End_iso" in adata.obs.columns:
            s_end = pd.to_datetime(adata.obs["Surgery_End_iso"], errors="coerce")
        elif "Surgery_End" in adata.obs.columns:
            s_end = pd.to_datetime(adata.obs["Surgery_End"], errors="coerce")
        else:
            s_end = pd.to_datetime("NaT")
        df_ops = pd.DataFrame(
            {
                "Procedure_ID": adata.obs["Procedure_ID"].values,
                "PMID": adata.obs["PMID"].values,
                "Surgery_End": s_end.values,
            }
        )
        out_rows = []
        for pid, pmid, t_end in df_ops.itertuples(index=False, name=None):
            if pd.isna(t_end):
                out_rows.append((pid, np.nan, 0))
                continue
            win_end = t_end + pd.Timedelta(days=7)
            cand = aki[
                (aki["PMID"] == pmid)
                & (aki["Start"] >= t_end)
                & (aki["Start"] <= win_end)
            ]
            if len(cand):
                max_stage = (
                    float(cand["AKI_Stage_int"].max())
                    if cand["AKI_Stage_int"].notna().any()
                    else np.nan
                )
                out_rows.append((pid, max_stage, 1))
            else:
                out_rows.append((pid, np.nan, 0))
        map_df = pd.DataFrame(
            out_rows, columns=["Procedure_ID", "max_aki_stage", "had_aki"]
        )
        adata.obs = adata.obs.merge(map_df, on="Procedure_ID", how="left")
    except Exception:
        pass

    # Kategorische Typen setzen
    for c in ["Sex", "had_transplantation", "max_aki_stage", "had_aki"]:
        if c in adata.obs.columns:
            try:
                adata.obs[c] = adata.obs[c].astype("category")
            except Exception:
                pass
    return adata


def describe_categorical(adata: AnnData, cols: list[str]) -> pd.DataFrame:
    if ep is not None and hasattr(ep, "tl") and hasattr(ep.tl, "describe_categorical"):
        try:
            return ep.tl.describe_categorical(adata, obs_names=cols)
        except Exception:
            pass
    # Fallback: pandas ValueCounts je Spalte
    frames = []
    for c in cols:
        if c not in adata.obs.columns:
            continue
        vc = adata.obs[c].value_counts(dropna=False).to_frame("count")
        tot = float(vc["count"].sum()) if vc["count"].sum() > 0 else 1.0
        vc["percent"] = (vc["count"] / tot * 100.0).round(2)
        vc.index.name = c
        frames.append(vc)
    return pd.concat(frames, axis=0) if frames else pd.DataFrame()


def scatter_age_crea_by_had_aki(adata: AnnData, out_png: str):
    xkey = "age_years_at_op"  # AgeAtFirstSurgery
    ykey = "crea_peak_0_48"  # peak_creatinine_post_op
    ckey = "had_aki"
    # ehrapy scatterplot, falls vorhanden
    if ep is not None and hasattr(ep, "pl") and hasattr(ep.pl, "scatterplot"):
        try:
            ep.pl.scatterplot(adata, x=xkey, y=ykey, color=ckey, alpha=0.7)
            plt.title("AgeAtFirstSurgery vs. peak_creatinine_post_op nach AKI-Status")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            return
        except Exception:
            pass
    # Fallback: Matplotlib
    df = pd.DataFrame(
        {
            "x": np.asarray(adata[:, xkey].X).ravel(),
            "y": np.asarray(adata[:, ykey].X).ravel(),
            "g": adata.obs[ckey].astype(str) if ckey in adata.obs else "NA",
        }
    )
    m0 = df["g"] == "0"
    m1 = df["g"] == "1"
    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(df.loc[m0, "x"], df.loc[m0, "y"], s=14, alpha=0.5, label="AKI 0–7 = 0")
    plt.scatter(df.loc[m1, "x"], df.loc[m1, "y"], s=14, alpha=0.5, label="AKI 0–7 = 1")
    plt.xlabel("AgeAtFirstSurgery (Jahre)")
    plt.ylabel("peak_creatinine_post_op (µmol/l)")
    plt.title("AgeAtFirstSurgery vs. peak_creatinine_post_op nach AKI-Status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ------------------------------ main ------------------------------


def main():
    adata = read_h5ad(H5)
    adata = ensure_obs_categoricals(adata)

    # Categorical-Report
    cols = ["Sex", "had_transplantation", "max_aki_stage", "had_aki"]
    cat_df = describe_categorical(adata, cols)
    cat_path = os.path.join(OUTD, "categorical_summary.csv")
    cat_df.to_csv(cat_path)
    print("✔ geschrieben:", cat_path)

    # Scatterplot
    out_png = os.path.join(OUTP, "scatter_age_vs_crea_peak_by_had_aki.png")
    scatter_age_crea_by_had_aki(adata, out_png)
    print("✔ gespeichert:", out_png)


if __name__ == "__main__":
    main()
