#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_deskriptive_plots_h5ad.py
Deskriptive Visualisierungen (ehrapy) auf Basis von .h5ad
+ AKI-Audit (Counts & Zeitdifferenzen)

Eingabe:
    h5ad/ops_with_crea_cysc_vis_features_with_AKI.h5ad
Ausgabe:
    PNGs in Diagramme/, Audit-CSV in Diagramme/
"""

from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")  # kein macOS GUI-Backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ehrapy as ep

# ------------------ Pfade ------------------
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5AD = BASE / "h5ad" / "ops_with_crea_cysc_vis_features_with_AKI.h5ad"
OUTD = BASE / "Diagramme"
OUTD.mkdir(parents=True, exist_ok=True)

# ------------------ Helper ------------------
def safe_hist(ad, key, bins=30, fname=None):
    if key not in ad.obs.columns:
        print(f"[skip] {key} fehlt")
        return
    series = pd.to_numeric(ad.obs[key], errors="coerce").dropna()
    if series.empty:
        print(f"[skip] {key} leer")
        return
    plt.figure()
    plt.hist(series, bins=bins)
    plt.xlabel(key); plt.ylabel("Anzahl")
    if fname:
        plt.savefig(OUTD / fname, dpi=300, bbox_inches="tight")
    plt.close()

def safe_violin(ad, key, groupby, fname=None):
    if key not in ad.obs.columns or groupby not in ad.obs.columns:
        print(f"[skip] Violin {key} ~ {groupby} fehlt")
        return
    ad.obs[groupby] = ad.obs[groupby].astype("category")
    try:
        ep.pl.violin(ad, keys=key, groupby=groupby)
        if fname:
            plt.gcf().savefig(OUTD / fname, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] violin {key}~{groupby}: {e}")
        tmp = ad.obs[[key, groupby]].copy()
        tmp[key] = pd.to_numeric(tmp[key], errors="coerce")
        tmp = tmp.dropna()
        groups = [g[key].values for _, g in tmp.groupby(groupby)]
        labels = [str(l) for l in sorted(tmp[groupby].dropna().unique())]
        plt.figure()
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.xlabel(groupby); plt.ylabel(key)
        if fname:
            plt.savefig(OUTD / fname, dpi=300, bbox_inches="tight")
        plt.close()

def safe_bar(series, title, xlabel, fname):
    counts = series.value_counts(dropna=False).sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig(OUTD / fname, dpi=300, bbox_inches="tight")
    plt.close()

def booleanize01(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["1","true","wahr","ja","yes"]).astype(int)

def audit_aki(ad):
    print("\n=== AKI-Audit ===")
    have_cols = [c for c in ["AKI_linked_0_7","AKI_linked_0_7_time","AKI_linked","AKI_Start","Surgery_End"] if c in ad.obs.columns]
    print("verfügbare Spalten:", have_cols)

    # robustes AKI bauen/fixen
    if "AKI_linked_0_7" in ad.obs.columns:
        ad.obs["AKI_linked_0_7"] = booleanize01(ad.obs["AKI_linked_0_7"])
    elif "AKI_linked_0_7_time" in ad.obs.columns:
        ad.obs["AKI_linked_0_7"] = booleanize01(ad.obs["AKI_linked_0_7_time"])
    elif "AKI_linked" in ad.obs.columns:
        ad.obs["AKI_linked_0_7"] = booleanize01(ad.obs["AKI_linked"])
    elif {"AKI_Start","Surgery_End"}.issubset(ad.obs.columns):
        se = pd.to_datetime(ad.obs["Surgery_End"], errors="coerce")
        ak = pd.to_datetime(ad.obs["AKI_Start"], errors="coerce")
        days = (ak - se).dt.total_seconds() / 86400.0
        ad.obs["days_to_AKI_calc"] = days
        ad.obs["AKI_linked_0_7"] = days.between(0, 7).fillna(False).astype(int)
    else:
        raise ValueError("Kein AKI-Feld ableitbar (weder AKI_linked_0_7 noch AKI_Start/Surgery_End vorhanden).")

    # Zähle 0/1
    vc = ad.obs["AKI_linked_0_7"].value_counts(dropna=False).sort_index()
    n0 = int(vc.get(0, 0)); n1 = int(vc.get(1, 0)); ntot = int(ad.n_obs)
    print(f"AKI=0: {n0} | AKI=1: {n1} | total: {ntot} | Anteil AKI: {100.0*n1/ntot:.1f}%")

    # Falls wir Tage berechnet haben: Verteilung in Bins ausgeben
    if "days_to_AKI_calc" in ad.obs.columns:
        d = ad.obs["days_to_AKI_calc"]
        bins = pd.IntervalIndex.from_tuples([(-np.inf,-0.001),(0,1),(1,2),(2,3),(3,7),(7,np.inf)])
        cut = pd.cut(d, bins=bins)
        print("\nVerteilung Tage (AKI_Start - Surgery_End):")
        print(cut.value_counts().sort_index())

        # kleine Stichprobe speichern
        sample = ad.obs.loc[d.notna(), ["PMID","SMID","Procedure_ID","Surgery_End","AKI_Start","days_to_AKI_calc","AKI_linked_0_7"]].head(30)
        sample.to_csv(OUTD / "audit_aki_sample.csv", index=False)
        print("Stichprobe gespeichert:", OUTD / "audit_aki_sample.csv")

    # für ehrapy-violin als Kategorie (explizit mit beiden Levels)
    from pandas.api.types import CategoricalDtype
    cat01 = CategoricalDtype(categories=[0,1], ordered=False)
    ad.obs["AKI_linked_0_7"] = ad.obs["AKI_linked_0_7"].astype(cat01)

# ------------------ Daten laden ------------------
ad = ep.io.read_h5ad(str(H5AD))
print("geladen:", H5AD, "| rows:", ad.n_obs)

# Zeiten zurückwandeln (nur falls später benötigt)
for c in ["Surgery_Start","Surgery_End","AKI_Start"]:
    if c in ad.obs.columns:
        ad.obs[c] = pd.to_datetime(ad.obs[c], errors="coerce")

# AKI prüfen/aufbauen
audit_aki(ad)

# ------------------ Plots ------------------
# Histogramme (einige Basismarker)
safe_hist(ad, "duration_hours", bins=30, fname="HIST_duration_hours.png")
safe_hist(ad, "crea_baseline",  bins=30, fname="HIST_crea_baseline.png")
safe_hist(ad, "cysc_baseline",  bins=30, fname="HIST_cysc_baseline.png")
safe_hist(ad, "vis_auc_0_24",   bins=30, fname="HIST_vis_auc_0_24.png")

# Violinplots nach AKI
safe_violin(ad, "crea_delta_0_48", "AKI_linked_0_7", fname="VIOLIN_crea_delta_vs_AKI.png")
safe_violin(ad, "cysc_delta_0_48", "AKI_linked_0_7", fname="VIOLIN_cysc_delta_vs_AKI.png")

# Barplots
if "Tx?" in ad.obs.columns:
    safe_bar(ad.obs["Tx?"], "Verteilung Tx?", "Tx?", "BAR_Tx.png")
safe_bar(ad.obs["AKI_linked_0_7"], "Häufigkeit AKI (0–7 Tage)", "AKI 0–7d", "BAR_AKI.png")

print("Fertig. Grafiken & Audit in:", OUTD)
