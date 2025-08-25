#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import anndata as ad
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
INP  = BASE / "h5ad" / "ops_with_patient_features.h5ad"
OUT  = BASE / "Audit" / "mannwhitney_lab_vis_stats.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

VARS = [
    "crea_baseline","crea_delta_0_48","crea_peak_0_48","crea_rate_0_48",
    "cysc_baseline","cysc_delta_0_48","cysc_peak_0_48","cysc_rate_0_48",
    "vis_auc_0_24","vis_auc_0_48","vis_max_0_24","vis_max_6_24","vis_mean_0_24",
    "duration_hours"
]

def derive_aki_column(df: pd.DataFrame) -> pd.Series:
    """
    Leite AKI (0/1) robust ab.
    Priorität: Stage -> AKI_any_0_7 -> AKI_linked_0_7 -> Zeitfenster.
    Wenn mehrere vorhanden: logisches OR.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # 1) Stage
    if "highest_AKI_stage_0_7" in df.columns:
        s = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
        out = ((s >= 1) | (out == 1)).astype("Int64")

    # 2) AKI_any_0_7 (bereits 0/1 oder True/False)
    if "AKI_any_0_7" in df.columns:
        s = pd.to_numeric(df["AKI_any_0_7"], errors="coerce")
        out = ((s > 0) | (out == 1)).astype("Int64")

    # 3) AKI_linked_0_7
    if "AKI_linked_0_7" in df.columns:
        s = pd.to_numeric(df["AKI_linked_0_7"], errors="coerce")
        out = ((s > 0) | (out == 1)).astype("Int64")

    # 4) Zeitfenster (falls noch NA und Zeiten da)
    needs_time = out.isna()
    if needs_time.any() and {"AKI_Start", "Surgery_End"}.issubset(df.columns):
        akis = pd.to_datetime(df.loc[needs_time, "AKI_Start"], errors="coerce")
        ends = pd.to_datetime(df.loc[needs_time, "Surgery_End"], errors="coerce")
        days = (akis - ends).dt.total_seconds() / 86400.0
        z = days.between(0, 7)
        out.loc[needs_time] = z.astype("Int64")

    # alles, was nicht 1 ist, als 0 setzen
    out = out.fillna(0).astype("Int64")
    out.name = "AKI"
    return out


def mw_test(x0: np.ndarray, x1: np.ndarray):
    stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
    n0, n1 = len(x0), len(x1)
    n = n0 + n1
    mu_U = n0 * n1 / 2
    sigma_U = np.sqrt(n0 * n1 * (n + 1) / 12)
    z = (stat - mu_U) / sigma_U
    r = abs(z) / np.sqrt(n)
    return stat, p, r

def main():
    adata = ad.read_h5ad(INP)
    df = adata.obs.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # AKI ableiten & anhängen
    df["AKI"] = derive_aki_column(df)
    # kurze Übersicht
    c0 = int((df["AKI"] == 0).sum())
    c1 = int((df["AKI"] == 1).sum())
    print(f"Quelle: {INP.name} | AKI=0: {c0} | AKI=1: {c1} | total: {len(df)}")

    rows = []
    for var in VARS:
        if var not in df.columns:
            print(f"[skip] {var} nicht vorhanden")
            continue
        sub = df[["AKI", var]].dropna()
        # numerisch machen
        sub[var] = pd.to_numeric(sub[var], errors="coerce")
        sub = sub.dropna()
        x0 = sub.loc[sub["AKI"] == 0, var].values
        x1 = sub.loc[sub["AKI"] == 1, var].values
        if len(x0) < 5 or len(x1) < 5:
            print(f"[skip] {var}: zu wenige Werte (AKI0={len(x0)}, AKI1={len(x1)})")
            continue
        stat, p, r = mw_test(x0, x1)
        rows.append({
            "Variable": var,
            "n_AKI0": len(x0), "n_AKI1": len(x1),
            "Median_AKI0": float(np.median(x0)),
            "Median_AKI1": float(np.median(x1)),
            "U_stat": float(stat),
            "p_value": float(p),
            "Effect_size_r": float(r)
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj_fdr"] = multipletests(res["p_value"], method="fdr_bh")[1]
        res = res.sort_values("p_adj_fdr")
        res.to_csv(OUT, index=False)
        print(f"Fertig. Ergebnisse gespeichert in: {OUT}")
        print(res.to_string(index=False))
    else:
        print("Keine Tests durchgeführt (prüfe Variablennamen und AKI-Spalte).")

if __name__ == "__main__":
    main()
