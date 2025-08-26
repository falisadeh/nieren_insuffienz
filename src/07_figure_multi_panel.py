#!/usr/bin/env python3
"""
Erstellt eine journal-taugliche Mehrfach-Abbildung (4–6 Subplots) mit Histogrammen der
wichtigsten Prädiktor-Variablen (Gruppen: AKI 0–7 = 0 vs. 1).

Features (anpassbar): duration_minutes, vis_auc_0_24, vis_auc_0_48,
crea_delta_0_48, crea_rate_0_48, vis_max_6_24.

Eigenschaften:
- 99. Perzentil-Kappung der x-Achse (robust gegen Ausreißer)
- Gleiche Bins je Subplot, Median-Linien je Gruppe, n in Legende
- Optionaler Mann-Whitney-U-p-Wert pro Panel (falls SciPy vorhanden)
- Export als PNG & PDF

Eingabe:  Daten/ops_with_patient_features.h5ad
Ausgabe:  Diagramme/FIG_risk_histograms_2x3.png / .pdf
"""
from __future__ import annotations
import os
import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from anndata import read_h5ad

# optional: SciPy für p-Werte
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUTD = os.path.join(BASE, "Diagramme")
os.makedirs(OUTD, exist_ok=True)

# ----- Konfiguration -----
FEATURES: List[str] = [
    "duration_minutes",
    "vis_auc_0_24",
    "vis_auc_0_48",
    "crea_delta_0_48",
    "crea_rate_0_48",
    "vis_max_6_24",
]
BINS = 30
PCTL_CAP = 99.0  # x-Achsenkappung
ADD_MEDIAN_LINES = True
FIGSIZE = (12, 8)  # (Breite, Höhe) in Zoll für 2x3 Layout


# ----- Helper -----
def to_1d(adata, var: str) -> np.ndarray:
    arr = np.asarray(adata[:, var].X).ravel()
    return arr.astype(float)


def group_arrays(adata, key: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    x = to_1d(adata, key)
    g = adata.obs.get("AKI_linked_0_7")
    if g is None:
        # alles in Gruppe 0, leere Gruppe 1
        return x[np.isfinite(x)], np.array([]), np.isfinite(x).sum(), 0
    g = g.astype(str)
    m0 = (g == "0").values
    m1 = (g == "1").values
    x0 = x[m0]
    x1 = x[m1]
    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]
    return x0, x1, x0.size, x1.size


def capped_limits(x0: np.ndarray, x1: np.ndarray, pctl: float) -> Tuple[float, float]:
    if x0.size + x1.size == 0:
        return 0.0, 1.0
    allx = np.concatenate([x0, x1])
    lo = float(np.nanmin(allx))
    hi = float(np.nanpercentile(allx, pctl))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(allx))
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "p = n. a."
    if p < 0.001:
        return "p < 0,001"
    return f"p = {p:.3f}".replace(".", ",")


# ----- Plot-Prozedur -----
def make_multi_figure(adata):
    # nur Features plotten, die auch existieren
    keys = [k for k in FEATURES if k in adata.var_names]
    n = len(keys)
    if n == 0:
        print("Keine der gewünschten Variablen vorhanden – bitte FEATURES prüfen.")
        return
    # Layout berechnen (max 6 → 2x3)
    rows = 2 if n > 3 else 1
    cols = min(3, n) if rows == 1 else 3
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE)
    axes = np.array(axes).reshape(-1)  # flach machen

    for i, key in enumerate(keys):
        ax = axes[i]
        x0, x1, n0, n1 = group_arrays(adata, key)
        xmin, xmax = capped_limits(x0, x1, PCTL_CAP)
        x0c = x0[(x0 >= xmin) & (x0 <= xmax)]
        x1c = x1[(x1 >= xmin) & (x1 <= xmax)]

        # Histogramme
        ax.hist(x0c, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 0 (n={n0})")
        ax.hist(x1c, bins=BINS, alpha=0.6, label=f"AKI 0–7 = 1 (n={n1})")

        # Mediane
        if ADD_MEDIAN_LINES:
            if x0c.size:
                m0 = float(np.median(x0c))
                ax.axvline(
                    m0, linestyle="--", linewidth=1.5, label=f"Median AKI0 {m0:.2f}"
                )
            if x1c.size:
                m1 = float(np.median(x1c))
                ax.axvline(
                    m1, linestyle=":", linewidth=1.8, label=f"Median AKI1 {m1:.2f}"
                )
                lab0 = f"Median AKI0 {m0:.2f}".replace(".", ",")
                lab1 = f"Median AKI1 {m1:.2f}".replace(".", ",")
                ax.axvline(m0, linestyle="--", linewidth=1.5, label=lab0)
                ax.axvline(m1, linestyle=":", linewidth=1.8, label=lab1)

        # p-Wert (Mann-Whitney U), falls SciPy verfügbar und beide Gruppen vorhanden
        p_txt = ""
        if mannwhitneyu is not None and x0c.size and x1c.size:
            try:
                stat, p = mannwhitneyu(x0c, x1c, alternative="two-sided")
                p_txt = "  (" + fmt_p(float(p)) + ")"
            except Exception:
                p_txt = ""

        ax.set_title(f"{key}{p_txt}")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(key)
        ax.set_ylabel("Häufigkeit")
        if i == 0:
            ax.legend()

    # leere Achsen ausblenden
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    fig.tight_layout()
    out_png = os.path.join(OUTD, "FIG_risk_histograms_2x3.png")
    out_pdf = os.path.join(OUTD, "FIG_risk_histograms_2x3.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("✔ gespeichert:", out_png)
    print("✔ gespeichert:", out_pdf)


def main():
    adata = read_h5ad(H5)
    # Kategorien sicherstellen
    if "AKI_linked_0_7" in adata.obs.columns:
        try:
            adata.obs["AKI_linked_0_7"] = adata.obs["AKI_linked_0_7"].astype("category")
        except Exception:
            pass
    make_multi_figure(adata)


if __name__ == "__main__":
    main()
