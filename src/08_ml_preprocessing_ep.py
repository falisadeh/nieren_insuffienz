#!/usr/bin/env python3
"""
ML-Vorverarbeitung mit ehrapy (so weit sinnvoll für EHR-Tabellen):
- Entfernt outcome-nahe Variablen aus .X (Leckage vermeiden)
- Log1p-Transformation für streng nichtnegative, rechtsschiefe Features (VIS, Dauer, Peaks, Baselines)
- (Optional) asinh für Variablen mit Vorzeichenwechsel (kompatibel mit negativen Deltas/Raten)
- Standardisierung/Skalierung aller Features (Mittel 0, Varianz 1) mit Kappung (max_value=10)
- Speichert neue H5AD

Hinweis: `normalize_total` ist für Zähldaten-Matrizen (z. B. scRNA) gedacht und
bei gemischt skalierten klinischen Merkmalen i. d. R. NICHT sinnvoll. Daher hier standardmäßig AUS.

Eingabe:  Daten/ops_with_patient_features.h5ad
Ausgabe:  Daten/ops_ml_processed.h5ad
Begleitdateien: Daten/ml_preproc_report.txt
"""
from __future__ import annotations
import os
from typing import List, Dict
import numpy as np
import pandas as pd
from anndata import read_h5ad

try:
    import ehrapy as ep
except Exception:
    ep = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
IN_H5 = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUT_H5 = os.path.join(BASE, "Daten", "ops_ml_processed.h5ad")
OUT_TXT = os.path.join(BASE, "Daten", "ml_preproc_report.txt")

# Outcome-/Leakage-Features in .X, die NICHT als Prädiktoren verwendet werden sollen
LEAKAGE = {"AKI_Duration_days", "days_to_AKI"}

# Falls du normalize_total TROTZDEM probieren willst, setze dies auf True
USE_NORMALIZE_TOTAL = False
NORMALIZE_TARGET_SUM = 1.0  # spielt nur bei USE_NORMALIZE_TOTAL=True

# asinh für vorzeichenbehaftete Variablen anwenden?
USE_ASINH_FOR_SIGNED = True


# ------------------------ Helper ------------------------
def to_np1d(adata, key: str) -> np.ndarray:
    return np.asarray(adata[:, key].X).ravel().astype(float)


# ------------------------- Main -------------------------
def main():
    ad = read_h5ad(IN_H5)

    # ----------------- Variablenliste vorbereiten -----------------
    all_vars: List[str] = list(map(str, ad.var_names))
    keep_vars: List[str] = [v for v in all_vars if v not in LEAKAGE]
    if len(keep_vars) < len(all_vars):
        ad = ad[:, keep_vars].copy()

    # ----------------- Nichtnegative vs. vorzeichenbehaftete -----------------
    nonneg_vars: List[str] = []
    signed_vars: List[str] = []
    for v in ad.var_names:
        x = to_np1d(ad, v)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        (nonneg_vars if np.nanmin(x) >= 0 else signed_vars).append(v)

    # ----------------- Optional: normalize_total (nicht empfohlen) -----------------
    if ep is not None and USE_NORMALIZE_TOTAL:
        try:
            ep.pp.normalize_total(ad, target_sum=NORMALIZE_TARGET_SUM)
        except Exception as e:
            print("normalize_total übersprungen:", e)

    # ----------------- Log1p für streng nichtnegative, rechtsschiefe Features -----------------
    # Auswahl: VIS*, duration_*, *_baseline, *_peak_*
    log_candidates = [
        v
        for v in nonneg_vars
        if (
            v.startswith("vis_")
            or v.startswith("duration_")
            or v.endswith("_baseline")
            or v.endswith("_peak_0_48")
        )
    ]
    # Anwenden (nur auf Teilmatrix)
    for v in log_candidates:
        x = to_np1d(ad, v)
        x = np.log1p(x)
        ad[:, v].X = x.reshape(-1, 1)

    # ----------------- asinh für vorzeichenbehaftete (Deltas/Raten) -----------------
    asinh_vars = []
    if USE_ASINH_FOR_SIGNED:
        for v in signed_vars:
            if v.endswith("_delta_0_48") or v.endswith("_rate_0_48"):
                asinh_vars.append(v)
        for v in asinh_vars:
            x = to_np1d(ad, v)
            ad[:, v].X = np.arcsinh(x).reshape(-1, 1)

    # ----------------- Standardisierung/Skalierung -----------------

    scaled_by = None

    # 1) ehrapy (falls vorhanden)
    if (ep is not None) and hasattr(ep, "pp") and hasattr(ep.pp, "scale"):
        try:
            ep.pp.scale(ad, max_value=10)
            scaled_by = "ehrapy.pp.scale"
            print("Skalierung: ehrapy.pp.scale")
        except Exception as e:
            print("Warnung: ehrapy.pp.scale fehlgeschlagen:", e)

    # 2) scanpy (Fallback, falls installiert)
    if scaled_by is None:
        try:
            import scanpy as sc  # type: ignore

            sc.pp.scale(ad, max_value=10)
            scaled_by = "scanpy.pp.scale"
            print("Skalierung: scanpy.pp.scale")
        except Exception as e:
            print("Hinweis: scanpy.pp.scale nicht verfügbar:", e)

    # 3) Manuell (immer verfügbar): z-Score + Kappung
    if scaled_by is None:
        X = ad.X.copy().astype(float)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        X = np.clip(X, -10, 10)
        ad.X = X
        scaled_by = "manual_zscore_clip"
        print("Skalierung: manuell (z-Score, Clip ±10)")

    # Info in den Report
    ad.uns["ml_preprocessing"] = {
        "removed_leakage_vars": sorted(list(LEAKAGE & set(all_vars))),
        "log1p_applied_vars": sorted(log_candidates),
        "asinh_applied_vars": sorted(asinh_vars),
        "scaled": True,
        "scaled_by": scaled_by,
        "normalize_total": USE_NORMALIZE_TOTAL,
    }

    # ----------------- Doku/Report -----------------
    ad.uns["ml_preprocessing"] = {
        "removed_leakage_vars": sorted(list(LEAKAGE & set(all_vars))),
        "log1p_applied_vars": sorted(log_candidates),
        "asinh_applied_vars": sorted(asinh_vars),
        "scaled": True,
        "scaled_by": scaled_by,
        "normalize_total": USE_NORMALIZE_TOTAL,
    }

    # ----------------- Doku/Report -----------------
    ad.uns["ml_preprocessing"] = {
        "removed_leakage_vars": sorted(list(LEAKAGE & set(all_vars))),
        "log1p_applied_vars": sorted(log_candidates),
        "asinh_applied_vars": sorted(asinh_vars),
        "scaled": True,
        "normalize_total": USE_NORMALIZE_TOTAL,
    }
    # kleiner Überblick
    n_obs, n_vars = ad.n_obs, ad.n_vars
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("ML-Preprocessing Report\n")
        f.write(f"n_obs={n_obs}, n_vars={n_vars}\n")
        for k, v in ad.uns["ml_preprocessing"].items():
            f.write(f"{k}: {v}\n")

    # ----------------- Speichern -----------------
    ad.write_h5ad(OUT_H5)
    print("✔ gespeichert:", OUT_H5)
    print("✔ Report:", OUT_TXT)


if __name__ == "__main__":
    main()
