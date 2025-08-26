#!/usr/bin/env python3
"""
Visualisiert die Verteilung kontinuierlicher Features.
- Standard: Matplotlib mit Gruppierung nach AKI 0–7 (0 vs. 1)
- Verbesserungen: 99%-Perzentil-Kappung (Tails), gleiche Bins/x-Limits, Median-Linien je Gruppe, Gruppengrößen in Legende
- Ergänzt explizit: vis_auc_0_24 (falls vorhanden)
- Schließt aus: days_to_AKI, AKI_Duration_days (Outcome-nahe Variablen)

Eingabe:  Daten/ops_with_patient_features.h5ad
Ausgabe:  Diagramme/hist_<feature>.png (ein Plot pro Feature)
"""
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from anndata import read_h5ad

# ehrapy optional (für Settings/Typen)
try:
    import ehrapy as ep  # type: ignore
except Exception:
    ep = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUT = os.path.join(BASE, "Diagramme")
os.makedirs(OUT, exist_ok=True)

# Priorisierte Variablen zuerst
PRIO = [
    "duration_hours",
    "duration_minutes",
    "age_years_at_op",
    "vis_auc_0_24",
    "vis_auc_0_48",
    "vis_max_6_24",
    "vis_max_0_24",
    "vis_mean_0_24",
    "crea_delta_0_48",
    "crea_rate_0_48",
    "crea_peak_0_48",
    "cysc_delta_0_48",
    "cysc_rate_0_48",
    "cysc_peak_0_48",
]
# Variablen, die wir nicht plotten wollen (Outcome-nah)
EXCLUDE = {"days_to_AKI", "AKI_Duration_days"}

# Konfiguration
BINS = 30
PCTL_CAP = 99.0  # Kappung der x-Achse auf 99. Perzentil
ADD_MEDIAN_LINES = True
USE_LOGX = False  # bei extrem schiefen Verteilungen ggf. True setzen


def to_numpy1d(x) -> np.ndarray:
    arr = np.asarray(x).ravel()
    arr = arr.astype(float)
    return arr


def group_arrays(adata, key: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    x = to_numpy1d(adata[:, key].X)
    if "AKI_linked_0_7" in adata.obs.columns:
        g = adata.obs["AKI_linked_0_7"].astype(str)
        mask0 = g == "0"
        mask1 = g == "1"
    else:
        mask0 = np.ones_like(x, dtype=bool)
        mask1 = np.zeros_like(x, dtype=bool)
    x0 = x[mask0]
    x1 = x[mask1]
    # NaNs entfernen
    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]
    return x0, x1, x0.size, x1.size


def capped_limits(x0: np.ndarray, x1: np.ndarray, pctl: float) -> Tuple[float, float]:
    if x0.size + x1.size == 0:
        return 0.0, 1.0
    allx = np.concatenate([x0, x1])
    lo = np.nanpercentile(allx, 100 - pctl) if pctl > 50 else np.nanmin(allx)
    hi = np.nanpercentile(allx, pctl)
    if not np.isfinite(lo):
        lo = float(np.nanmin(allx))
    if not np.isfinite(hi):
        hi = float(np.nanmax(allx))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_histogram(adata, key: str):
    x0, x1, n0, n1 = group_arrays(adata, key)
    if n0 + n1 == 0:
        print(f"! {key}: keine gültigen Werte")
        return

    # x-Limits mit 99%-Perzentil kappeln
    xmin, xmax = capped_limits(x0, x1, PCTL_CAP)
    x0_cap = x0[(x0 >= xmin) & (x0 <= xmax)]
    x1_cap = x1[(x1 >= xmin) & (x1 <= xmax)]

    plt.close("all")
    fig = plt.figure(figsize=(7.5, 4.8))
    if USE_LOGX:
        # für log-x: Null/negativ filtern
        x0p = x0_cap[x0_cap > 0]
        x1p = x1_cap[x1_cap > 0]
        plt.hist(x0p, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 0 (n={n0})", log=False)
        plt.hist(x1p, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 1 (n={n1})", log=False)
        plt.xscale("log")
    else:
        plt.hist(x0_cap, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 0 (n={n0})")
        plt.hist(x1_cap, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 1 (n={n1})")

    # Median-Linien
    if ADD_MEDIAN_LINES:
        if x0_cap.size:
            m0 = float(np.median(x0_cap))
            plt.axvline(
                m0, linestyle="--", linewidth=1.5, label=f"Median AKI0 {m0:.2f}"
            )
        if x1_cap.size:
            m1 = float(np.median(x1_cap))
            plt.axvline(m1, linestyle=":", linewidth=1.8, label=f"Median AKI1 {m1:.2f}")

    plt.xlim(xmin, xmax)
    plt.title(f"Verteilung: {key} (stratifiziert nach AKI 0–7)")
    plt.xlabel(key)
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT, f"hist_{key}.png")
    plt.savefig(out_path, dpi=150)
    print(f"✔ gespeichert: {out_path}")


def main():
    adata = read_h5ad(H5)

    # Typen/Kategorien setzen
    for c in ("AKI_linked_0_7", "AKI_Stage", "Sex"):
        if c in adata.obs.columns:
            try:
                adata.obs[c] = adata.obs[c].astype("category")
            except Exception:
                pass

    # ehrapy Plot-Settings (nur DPI etc.)
    if ep is not None and hasattr(ep, "settings"):
        try:
            ep.settings.set_figure_params(dpi=150)
        except Exception:
            pass

    # Feature-Liste bauen
    num_keys: List[str] = list(map(str, adata.var_names))
    # vis_auc_0_24 explizit aufnehmen, falls vorhanden
    keys = [k for k in PRIO if k in num_keys]
    keys += [k for k in num_keys if (k not in keys and k not in EXCLUDE)]

    # Plot-Schleife
    for key in keys:
        try:
            if key in EXCLUDE:
                continue
            plot_histogram(adata, key)
        except Exception as e:
            print(f"! Fehler bei {key}: {e}")


if __name__ == "__main__":
    main()
