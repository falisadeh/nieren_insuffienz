##!/usr/bin/env python3
"""
Erstellt eine Table-1 mit gruppierten Kennzahlen (AKI 0–7 Tage: nein vs. ja) und statistischen Tests.
- Basierend auf dem bereits erzeugten H5AD: Daten/ops_with_patient_features.h5ad
- Nutzt, wenn verfügbar, ehrapy-Funktionen zum Setzen von Feature-Typen; ansonsten Fallback mit pandas.
- Für numerische Variablen (alle .X-Variablen):
    * Kennzahlen je Gruppe: count, mean, sd, median, IQR, min, max
    * Test: Mann-Whitney-U (robust) + Effektgröße (rank-biserial correlation)
- Für kategoriale Variablen aus .obs (Sex, AKI_Stage):
    * Häufigkeiten je Gruppe
    * Test: Chi² (allgemein) bzw. Fisher (2x2), Effekt: Cramér’s V (allgemein) bzw. Odds Ratio (2x2)
- Schreibt kompakte und vollständige CSVs in den Ordner „Daten“.

Ausgaben:
  - Daten/Table1_OP_level_with_stats.csv (voll)
  - Daten/Table1_OP_level_compact.csv (kompakt)
"""
from __future__ import annotations
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from anndata import read_h5ad, AnnData

H5 = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Daten/ops_with_patient_features.h5ad"
adata = read_h5ad(H5)
for c in ["AKI_linked_0_7", "AKI_Stage", "Sex"]:
    if c in adata.obs:
        adata.obs[c] = adata.obs[c].astype("category")
adata.write_h5ad(H5)

from scipy import stats

try:
    import ehrapy as ep  # type: ignore
except Exception:
    ep = None  # type: ignore

BASE_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_PATH = os.path.join(BASE_DIR, "Daten", "ops_with_patient_features.h5ad")
OUT_DIR = os.path.join(BASE_DIR, "Daten")
os.makedirs(OUT_DIR, exist_ok=True)

CAT_OBS_CANDIDATES = ["Sex", "AKI_Stage"]  # AKI_linked_0_7 ist die Gruppierungsvariable
GROUP_COL = "AKI_linked_0_7"


def safe_infer_feature_types(adata: AnnData) -> AnnData:
    """Setzt die gewünschten .obs-Spalten robust auf 'category'.
    Verzichtet auf ehrapy.replace_feature_types wegen versionsabhängiger Signaturen.
    Versucht optional ep.ad.infer_feature_types, falls vorhanden (ohne Nebenwirkungen).
    """
    # Primär: pandas-Casting (kompatibel zu allen Versionen)
    for c in CAT_OBS_CANDIDATES + [GROUP_COL]:
        if c in adata.obs.columns:
            try:
                adata.obs[c] = adata.obs[c].astype("category")
            except Exception:
                pass
    # Optional: ehrapy infer_feature_types (falls vorhanden)
    if ep is not None and hasattr(ep, "ad") and hasattr(ep.ad, "infer_feature_types"):
        try:
            ret = ep.ad.infer_feature_types(adata)
            if isinstance(ret, AnnData):
                return ret
        except Exception:
            pass
    return adata
    try:
        if hasattr(ep, "ad") and hasattr(ep.ad, "replace_feature_types"):
            ep.ad.replace_feature_types(
                adata,
                obs_types={
                    k: "categorical"
                    for k in CAT_OBS_CANDIDATES
                    if k in adata.obs.columns
                }
                | {GROUP_COL: "categorical"},
            )
        elif hasattr(ep, "ad") and hasattr(ep.ad, "infer_feature_types"):
            ret = ep.ad.infer_feature_types(adata)
            if isinstance(ret, AnnData):
                adata = ret
    except Exception:
        pass
    # Fallback pandas
    for c in CAT_OBS_CANDIDATES + [GROUP_COL]:
        if c in adata.obs.columns:
            try:
                adata.obs[c] = adata.obs[c].astype("category")
            except Exception:
                pass
    return adata


def summarize_numeric(x: np.ndarray) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
        }
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "median": float(np.median(x)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "iqr": float(q3 - q1),
    }


def mannwhitney_with_rbc(x0: np.ndarray, x1: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney-U-Test (zweiseitig) + rank-biserial correlation (RBC)."""
    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]
    if x0.size == 0 or x1.size == 0:
        return np.nan, np.nan
    U = stats.mannwhitneyu(x0, x1, alternative="two-sided").statistic
    n0, n1 = x0.size, x1.size
    # RBC = 1 - 2U/(n0*n1)
    rbc = 1.0 - 2.0 * (U / (n0 * n1))
    # p-Wert separat
    p = stats.mannwhitneyu(x0, x1, alternative="two-sided").pvalue
    return float(p), float(rbc)


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return float(
        np.sqrt((chi2 / n) / (min(r - 1, k - 1) if min(r - 1, k - 1) > 0 else np.nan))
    )


def build_table1(adata: AnnData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Gruppierungsmaske
    if GROUP_COL not in adata.obs.columns:
        raise ValueError(f"Gruppierungsvariable '{GROUP_COL}' fehlt in .obs")
    g = adata.obs[GROUP_COL]
    # robust 0/1
    try:
        g01 = g.astype(int)
    except Exception:
        g01 = (
            g.astype(str)
            .str.lower()
            .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
            .fillna(0)
            .astype(int)
        )

    idx0 = np.where(g01.values == 0)[0]
    idx1 = np.where(g01.values == 1)[0]

    # Numerische Matrix + Namen
    X = np.array(adata.X, dtype=float)
    cols = list(map(str, adata.var_names))

    rows = []
    rows_compact = []

    # Numerische Variablen
    for j, name in enumerate(cols):
        x = X[:, j]
        summ0 = summarize_numeric(x[idx0])
        summ1 = summarize_numeric(x[idx1])
        p, rbc = mannwhitney_with_rbc(x[idx0], x[idx1])
        row = {
            "Variable": name,
            "Group0_n": summ0["count"],
            "Group1_n": summ1["count"],
            "Group0_mean": summ0["mean"],
            "Group0_sd": summ0["sd"],
            "Group0_median": summ0["median"],
            "Group0_q1": summ0["q1"],
            "Group0_q3": summ0["q3"],
            "Group1_mean": summ1["mean"],
            "Group1_sd": summ1["sd"],
            "Group1_median": summ1["median"],
            "Group1_q1": summ1["q1"],
            "Group1_q3": summ1["q3"],
            "p_MWU": p,
            "RBC": rbc,
        }
        rows.append(row)
        # kompakte Darstellung: Median (IQR) + p
        rows_compact.append(
            {
                "Variable": name,
                "AKI 0–7 = 0": f"{summ0['median']:.3g} ({summ0['q1']:.3g}–{summ0['q3']:.3g})",
                "AKI 0–7 = 1": f"{summ1['median']:.3g} ({summ1['q1']:.3g}–{summ1['q3']:.3g})",
                "p (MWU)": f"{p:.3g}" if not np.isnan(p) else "",
            }
        )

    # Kategoriale Variablen aus .obs
    for c in CAT_OBS_CANDIDATES:
        if c not in adata.obs.columns:
            continue
        vc0 = adata.obs.iloc[idx0][c].value_counts(dropna=False)
        vc1 = adata.obs.iloc[idx1][c].value_counts(dropna=False)

        cats = sorted(
            set(vc0.index.tolist()) | set(vc1.index.tolist()), key=lambda x: str(x)
        )
        # Kontingenztafel
        table = np.array([[vc0.get(cat, 0), vc1.get(cat, 0)] for cat in cats])
        # Test
        p_cat = np.nan
        effect = np.nan
        if table.shape == (2, 2):
            try:
                odds, p_cat = stats.fisher_exact(table)
                effect = odds
            except Exception:
                p_cat = np.nan
                effect = np.nan
        else:
            try:
                chi2, p_cat, dof, _ = stats.chi2_contingency(table)
                effect = cramers_v(table)
            except Exception:
                p_cat = np.nan
                effect = np.nan
        # in Tabellen schreiben
        pretty_rows = []
        for cat in cats:
            n0 = int(vc0.get(cat, 0))
            n1 = int(vc1.get(cat, 0))
            pct0 = 100.0 * n0 / max(1, len(idx0))
            pct1 = 100.0 * n1 / max(1, len(idx1))
            label = f"{c} = {cat}"
            rows.append(
                {
                    "Variable": label,
                    "Group0_n": n0,
                    "Group1_n": n1,
                    "Group0_mean": np.nan,
                    "Group0_sd": np.nan,
                    "Group0_median": np.nan,
                    "Group0_q1": np.nan,
                    "Group0_q3": np.nan,
                    "Group1_mean": np.nan,
                    "Group1_sd": np.nan,
                    "Group1_median": np.nan,
                    "Group1_q1": np.nan,
                    "Group1_q3": np.nan,
                    "p_MWU": p_cat,
                    "RBC": effect,
                }
            )
            pretty_rows.append(
                {
                    "Variable": label,
                    "AKI 0–7 = 0": f"{n0} ({pct0:.1f}%)",
                    "AKI 0–7 = 1": f"{n1} ({pct1:.1f}%)",
                    "p (Cat)": f"{p_cat:.3g}" if not np.isnan(p_cat) else "",
                }
            )
        rows_compact.extend(pretty_rows)

    full = pd.DataFrame(rows)
    compact = pd.DataFrame(rows_compact)

    # Reihenfolge: zuerst wichtige Variablen, falls vorhanden
    order_hint = [
        "duration_hours",
        "age_years_at_op",
        "crea_baseline",
        "crea_peak_0_48",
        "crea_delta_0_48",
        "crea_rate_0_48",
        "cysc_baseline",
        "cysc_peak_0_48",
        "cysc_delta_0_48",
        "cysc_rate_0_48",
        "vis_auc_0_24",
        "vis_auc_0_48",
        "vis_max_0_24",
        "vis_max_6_24",
        "vis_mean_0_24",
    ]

    # Numerik zuerst in der gewünschten Reihenfolge, Rest hinten an
    def sort_key(name: str) -> Tuple[int, int]:
        return (0, order_hint.index(name)) if name in order_hint else (1, 0)

    # compact sortieren (nur numerische Namen aus order_hint + kategorische bleiben Wo sie sind)
    num_mask = compact["Variable"].isin(order_hint)
    compact_num = compact[num_mask].copy()
    compact_num["__order"] = compact_num["Variable"].map(lambda n: order_hint.index(n))
    compact_num = (
        compact_num.sort_values("__order").drop(columns=["__order"])
        if not compact_num.empty
        else compact_num
    )
    compact = pd.concat([compact_num, compact[~num_mask]], axis=0, ignore_index=True)

    return full, compact


def main():
    adata = read_h5ad(H5_PATH)
    adata = safe_infer_feature_types(adata)
    full, compact = build_table1(adata)
    full.to_csv(os.path.join(OUT_DIR, "Table1_OP_level_with_stats.csv"), index=False)
    compact.to_csv(os.path.join(OUT_DIR, "Table1_OP_level_compact.csv"), index=False)
    print("Table-1 erstellt in:", OUT_DIR)


if __name__ == "__main__":
    main()
