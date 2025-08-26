#!/usr/bin/env python3
"""
Age vs. Creatinin-Peak (nach AKI-Status) mit LOWESS-Trend und Spearman-Korrelationen.

Erzeugt zwei Abbildungen:
- Diagramme/scatter_age_vs_crea_peak_by_had_aki_lowess.png (Gesamtbereich, y auf 99. Perzentil gekappt)
- Diagramme/scatter_age_vs_crea_peak_by_had_aki_lowess_zoom_le_2y.png (Zoom: Alter ≤ 2 Jahre)

Sowie eine Ergebnis-Tabelle:
- Daten/spearman_age_crea_by_aki.csv (rho & p insgesamt und je Gruppe)

Voraussetzungen: H5AD aus dem Build-Skript, Spalten
- .X: age_years_at_op, crea_peak_0_48
- .obs: AKI_linked_0_7 (0/1). Falls "had_aki" fehlt, wird es aus AKI_linked_0_7 abgeleitet.
"""
from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import read_h5ad

# Try SciPy for Spearman
try:
    from scipy import stats
except Exception:
    stats = None  # type: ignore

# Try statsmodels for LOWESS
try:
    import statsmodels.api as sm  # type: ignore
except Exception:
    sm = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUTD = os.path.join(BASE, "Daten")
OUTP = os.path.join(BASE, "Diagramme")
os.makedirs(OUTD, exist_ok=True)
os.makedirs(OUTP, exist_ok=True)

XKEY = "age_years_at_op"
YKEY = "crea_peak_0_48"
GKEY = "had_aki"  # wird bei Bedarf aus AKI_linked_0_7 erzeugt

PCTL_Y_CAP = 99.0
LOWESS_FRAC = 0.3  # Glättungsfenster (0..1); bei viel Rauschen ggf. höher
POINT_SIZE = 14
ALPHA = 0.5


def to_1d(adata, var: str) -> np.ndarray:
    arr = np.asarray(adata[:, var].X).ravel()
    arr = arr.astype(float)
    return arr


def ensure_group(obs: pd.DataFrame, gkey: str = GKEY) -> pd.Series:
    if gkey in obs.columns:
        g = obs[gkey]
    elif "AKI_linked_0_7" in obs.columns:
        g = obs["AKI_linked_0_7"]
    else:
        g = pd.Series(np.nan, index=obs.index)
    # Normalisieren auf Strings "0"/"1"
    try:
        g = g.astype(int).astype(str)
    except Exception:
        g = g.astype(str)
    return g


def lowess_curve(
    x: np.ndarray, y: np.ndarray, frac: float = LOWESS_FRAC
) -> Tuple[np.ndarray, np.ndarray]:
    # Entferne NaNs
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    # Sortiert für saubere Linie
    order = np.argsort(x)
    x_ord, y_ord = x[order], y[order]
    if sm is not None:
        try:
            z = sm.nonparametric.lowess(y_ord, x_ord, frac=frac, return_sorted=True)
            return z[:, 0], z[:, 1]
        except Exception:
            pass
    # Fallback: gleitender Median
    n = max(5, int(frac * x_ord.size))
    if n % 2 == 0:
        n += 1
    half = n // 2
    xs, ys = [], []
    for i in range(x_ord.size):
        lo = max(0, i - half)
        hi = min(x_ord.size, i + half + 1)
        xs.append(x_ord[i])
        ys.append(np.median(y_ord[lo:hi]))
    return np.array(xs), np.array(ys)


def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or y.size < 3:
        return np.nan, np.nan
    if stats is not None:
        try:
            rho, p = stats.spearmanr(x, y)
            return float(rho), float(p)
        except Exception:
            pass
    # Fallback: Rangkorrelation via pandas, ohne p-Wert
    try:
        s = pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson")
        return float(s), np.nan
    except Exception:
        return np.nan, np.nan


def make_plot(df: pd.DataFrame, title: str, out_png: str, y_cap: float | None = None):
    plt.figure(figsize=(7.8, 5.0))
    for label, sub in df.groupby("g"):
        plt.scatter(
            sub["x"], sub["y"], s=POINT_SIZE, alpha=ALPHA, label=f"AKI 0–7 = {label}"
        )
        # LOWESS
        xs, ys = lowess_curve(sub["x"].values, sub["y"].values)
        if xs.size:
            plt.plot(xs, ys, linewidth=2.0, label=f"LOWESS AKI={label}")
    plt.xlabel("AgeAtFirstSurgery (Jahre)")
    plt.ylabel("peak_creatinine_post_op (µmol/l)")
    plt.title(title)
    if y_cap is not None:
        plt.ylim(None, y_cap)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("✔ gespeichert:", out_png)


def main():
    adata = read_h5ad(H5)

    # Datenframe bauen
    x = to_1d(adata, XKEY)
    y = to_1d(adata, YKEY)
    g = ensure_group(adata.obs, GKEY)
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()

    # y auf 99%-Perzentil begrenzen für besseren Überblick
    y_cap = float(np.nanpercentile(df["y"], PCTL_Y_CAP))

    # Spearman gesamt und je Gruppe
    results = []
    rho_all, p_all = spearman(df["x"].values, df["y"].values)
    results.append(
        {"group": "all", "rho_spearman": rho_all, "p": p_all, "n": int(df.shape[0])}
    )
    for label, sub in df.groupby("g"):
        rho, p = spearman(sub["x"].values, sub["y"].values)
        results.append(
            {
                "group": f"AKI={label}",
                "rho_spearman": rho,
                "p": p,
                "n": int(sub.shape[0]),
            }
        )
    res = pd.DataFrame(results)
    res_path = os.path.join(OUTD, "spearman_age_crea_by_aki.csv")
    res.to_csv(res_path, index=False)
    print("✔ geschrieben:", res_path)

    # Plot 1: Gesamt
    make_plot(
        df,
        "AgeAtFirstSurgery vs. peak_creatinine_post_op nach AKI-Status",
        os.path.join(OUTP, "scatter_age_vs_crea_peak_by_had_aki_lowess.png"),
        y_cap=y_cap,
    )

    # Plot 2: Zoom ≤ 2 Jahre
    df_zoom = df[df["x"] <= 2.0].copy()
    if not df_zoom.empty:
        y_cap_zoom = float(np.nanpercentile(df_zoom["y"], PCTL_Y_CAP))
        make_plot(
            df_zoom,
            "AgeAtFirstSurgery ≤ 2 Jahre vs. peak_creatinine_post_op (AKI-Farbe)",
            os.path.join(
                OUTP, "scatter_age_vs_crea_peak_by_had_aki_lowess_zoom_le_2y.png"
            ),
            y_cap=y_cap_zoom,
        )


if __name__ == "__main__":
    main()
