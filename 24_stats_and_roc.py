#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_stats_and_roc.py
AKI (0–7 d) vs. Labor-/VIS-Parameter: nichtparametrische Tests + Effektstärken + univariate ROC/AUC.

Eingabe:
  h5ad/ops_with_patient_features.h5ad

Ausgabe:
  Audit/lab_vis_vs_AKI_stats_FDR.csv         (Tabellarische Tests + FDR)
  Audit/lab_vis_vs_AKI_auc.csv               (AUC + Schwellenwerte)
  Audit/lab_vis_vs_AKI_report.txt            (kurzer Textreport)
  Diagramme/ROC_<variable>.png               (optional pro Variable)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, roc_curve
import ehrapy as ep

# ------------------ Pfade ------------------
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5AD = BASE / "h5ad" / "ops_with_patient_features.h5ad"
OUTP = BASE / "Audit"
OUTF = BASE / "Diagramme"
OUTP.mkdir(parents=True, exist_ok=True)
OUTF.mkdir(parents=True, exist_ok=True)

# ------------------ Helper ------------------
def ensure_aki_flag(df: pd.DataFrame) -> pd.Series:
    """Robustes AKI-Flag (0/1) je OP."""
    if "highest_AKI_stage_0_7" in df.columns:
        st = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
        if st.notna().any():
            return (st >= 1).astype(int)

    if "AKI_linked_0_7" in df.columns:
        s = pd.to_numeric(df["AKI_linked_0_7"], errors="coerce")
        if s.notna().any():
            return (s > 0).astype(int)

    if {"AKI_Start", "Surgery_End"}.issubset(df.columns):
        ak = pd.to_datetime(df["AKI_Start"], errors="coerce")
        se = pd.to_datetime(df["Surgery_End"], errors="coerce")
        days = (ak - se).dt.total_seconds() / 86400.0
        return days.between(0, 7).fillna(False).astype(int)

    raise ValueError("Kein AKI-Feld ableitbar (weder Stage, Flag noch Zeiten).")

def iqr(a: np.ndarray) -> tuple[float, float]:
    return (np.nanpercentile(a, 25), np.nanpercentile(a, 75))

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # δ = ( #x>y - #x<y ) / (n_x * n_y); effiziente Approx. via Sortierung
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x_s = np.sort(x); y_s = np.sort(y)
    nx, ny = len(x_s), len(y_s)
    i = j = more = less = 0
    while i < nx and j < ny:
        if x_s[i] > y_s[j]:
            more += (ny - j)
            i += 1
        elif x_s[i] < y_s[j]:
            less += (ny - j)
            i += 1
        else:
            # gleiche Werte -> neutral; gehe in x weiter
            i += 1
    denom = nx * ny
    return (more - less) / denom if denom else np.nan

def hodges_lehmann_shift(x: np.ndarray, y: np.ndarray) -> float:
    """HL-Schätzer des Lageunterschieds: Median aller Paar-Differenzen (y - x)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Paar-Differenzen in Batches berechnen (Speicher-schonend)
    diffs = []
    chunk = 200  # genügt: 600x600 ~ 360k Diff -> ok; bei sehr großen n sonst weiter stückeln
    for i in range(0, len(y), chunk):
        yy = y[i:i+chunk][:, None]
        diffs.append((yy - x).ravel())
    diffs = np.concatenate(diffs)
    return float(np.median(diffs))

def summarize_and_test(df: pd.DataFrame, var: str, ycol: str = "AKI") -> dict | None:
    s = pd.to_numeric(df[var], errors="coerce")
    y = df[ycol].astype(int)
    a = s[y == 0].values
    b = s[y == 1].values
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) + len(b) == 0:
        return None

    # Mann–Whitney U (zweiseitig)
    p = np.nan
    if len(a) > 0 and len(b) > 0:
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan

    # Effektgrößen
    delta = cliffs_delta(a, b)
    hl = hodges_lehmann_shift(a, b)  # ~Median(B) - Median(A) (robust)

    res = dict(
        variable=var,
        n_total=int(np.sum(~np.isnan(s))),
        missing_pct=round(100 * np.mean(np.isnan(s)), 1),
        n_AKI0=int(len(a)),
        median_AKI0=float(np.median(a)) if len(a) else np.nan,
        iqr_AKI0_low=float(iqr(a)[0]) if len(a) else np.nan,
        iqr_AKI0_high=float(iqr(a)[1]) if len(a) else np.nan,
        n_AKI1=int(len(b)),
        median_AKI1=float(np.median(b)) if len(b) else np.nan,
        iqr_AKI1_low=float(iqr(b)[0]) if len(b) else np.nan,
        iqr_AKI1_high=float(iqr(b)[1]) if len(b) else np.nan,
        mannwhitney_p=float(p) if p == p else np.nan,
        cliffs_delta=float(delta) if delta == delta else np.nan,
        hodges_lehmann=float(hl) if hl == hl else np.nan,
    )
    return res

def do_roc(df: pd.DataFrame, var: str, ycol: str = "AKI", make_plot: bool = True):
    s = pd.to_numeric(df[var], errors="coerce")
    y = df[ycol].astype(int)
    mask = s.notna() & y.notna()
    s = s[mask]; y = y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)
    youden = tpr - fpr
    j_idx = int(np.argmax(youden))
    thr_opt = float(thr[j_idx])

    if make_plot:
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.2f}")
        plt.plot([0,1],[0,1], ls="--", lw=1)
        plt.scatter(fpr[j_idx], tpr[j_idx], s=40, label=f"Youden-J @ {thr_opt:.3g}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC – {var}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTF / f"ROC_{var}.png", dpi=300, bbox_inches="tight")
        plt.close()
    return float(auc), float(fpr[j_idx]), thr_opt

# ------------------ Daten laden ------------------
ad = ep.io.read_h5ad(str(H5AD))
df = ad.obs.copy()

# AKI-Flag sicherstellen
df["AKI"] = ensure_aki_flag(df)
vc = df["AKI"].value_counts().reindex([0,1], fill_value=0)
print(f"AKI=0: {int(vc.loc[0])} | AKI=1: {int(vc.loc[1])} | total: {len(df)}")

# ------------------ Feature-Liste ------------------
cands = [
    # Prä-OP
    "crea_baseline", "cysc_baseline",
    # 0–48h
    "crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
    "cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
    # HeMod/VIS & OP
    "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
    "duration_hours",
]
vars_present = [v for v in cands if v in df.columns]

# ------------------ Tests & Effektstärken ------------------
rows = []
for v in vars_present:
    r = summarize_and_test(df, v, ycol="AKI")
    if r: rows.append(r)

stats = pd.DataFrame(rows)
if not stats.empty:
    # FDR-Korrektur über alle p-Werte
    p = stats["mannwhitney_p"].fillna(1.0).values
    rej, p_adj, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    stats["p_fdr"] = p_adj
    stats["sig_fdr"] = rej
    # hübsche Spalten
    stats["median_IQR_AKI0"] = stats.apply(
        lambda r: f"{r['median_AKI0']:.3g} [{r['iqr_AKI0_low']:.3g}; {r['iqr_AKI0_high']:.3g}]" if pd.notna(r['median_AKI0']) else "",
        axis=1
    )
    stats["median_IQR_AKI1"] = stats.apply(
        lambda r: f"{r['median_AKI1']:.3g} [{r['iqr_AKI1_low']:.3g}; {r['iqr_AKI1_high']:.3g}]"
        if pd.notna(r['median_AKI1']) else "",
        axis=1
    )
    order_cols = [
        "variable","n_total","missing_pct",
        "n_AKI0","median_IQR_AKI0","n_AKI1","median_IQR_AKI1",
        "mannwhitney_p","p_fdr","cliffs_delta","hodges_lehmann","sig_fdr"
    ]
    stats = stats.sort_values(["sig_fdr","p_fdr"], ascending=[False, True])
    stats[order_cols].to_csv(OUTP / "lab_vis_vs_AKI_stats_FDR.csv", index=False)
    print("➜ gespeichert:", OUTP / "lab_vis_vs_AKI_stats_FDR.csv")
else:
    print("[WARN] Keine Variablen für Tests gefunden.")

# ------------------ ROC je Variable ------------------
roc_rows = []
for v in vars_present:
    auc, fpr_youden, thr = do_roc(df, v, ycol="AKI", make_plot=True)
    roc_rows.append(dict(variable=v, auc=auc, fpr_youden=fpr_youden, threshold_youden=thr))

roc = pd.DataFrame(roc_rows).sort_values("auc", ascending=False)
roc.to_csv(OUTP / "lab_vis_vs_AKI_auc.csv", index=False)
print("➜ gespeichert:", OUTP / "lab_vis_vs_AKI_auc.csv")

# ------------------ Kurzreport ------------------
with open(OUTP / "lab_vis_vs_AKI_report.txt", "w") as f:
    def w(s=""): f.write(s + "\n")
    w("AKI vs. Marker – Zusammenfassung")
    w(f"AKI=0: {int(vc.loc[0])} | AKI=1: {int(vc.loc[1])} | n={len(df)}\n")
    if not stats.empty:
        w("Signifikante Marker (FDR<0.05):")
        for _, r in stats[stats["sig_fdr"]].iterrows():
            w(f"  - {r['variable']}: p_fdr={r['p_fdr']:.3e}, Δ(HL)={r['hodges_lehmann']:.3g}, δ={r['cliffs_delta']:.3g}")
        w("")
    if not roc.empty:
        w("Top AUC (univariat):")
        for _, r in roc.head(8).iterrows():
            w(f"  - {r['variable']}: AUC={r['auc']:.3f}, Thr*={r['threshold_youden']:.3g}")
print("➜ Report:", OUTP / "lab_vis_vs_AKI_report.txt")

print("\nFertig.")
