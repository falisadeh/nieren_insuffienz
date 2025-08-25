#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_lab_vis_vs_AKI.py
Vergleich von Labor- & VIS-Parametern zwischen AKI 0–7d (0/1) und einfache Baseline-Modelle.

Eingabe:
  h5ad/ops_with_patient_features.h5ad

Ausgabe:
  Diagramme/*.png (Violinplots)
  Audit/lab_vis_vs_AKI_stats.csv (Statistik je Variable)
  Audit/lab_vis_vs_AKI_stats_FDR.csv (mit FDR-korrigierten p-Werten)
  Audit/lab_vis_vs_AKI_models.txt (AUCs & Koeffizienten)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # Headless (verhindert macOS NSSavePanel-Warnung)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ehrapy as ep
from anndata import AnnData

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Optional für Mini-Modelle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ------------------ Pfade ------------------
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5AD = BASE / "h5ad" / "ops_with_patient_features.h5ad"
OUTP = BASE / "Audit"
OUTF = BASE / "Diagramme"
OUTP.mkdir(parents=True, exist_ok=True)
OUTF.mkdir(parents=True, exist_ok=True)

# ------------------ Helper ------------------
def booleanize01(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["1","true","wahr","ja","yes"]).astype(int)

def ensure_aki_flag(df: pd.DataFrame) -> pd.Series:
    # 1) aus Stufe (robust, hat bei dir funktioniert)
    if "highest_AKI_stage_0_7" in df.columns:
        st = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
        if st.notna().any():
            return (st >= 1).astype(int)
    # 2) vorhandenes Flag
    if "AKI_linked_0_7" in df.columns:
        return booleanize01(df["AKI_linked_0_7"])
    # 3) aus Zeitdifferenz, falls vorhanden
    if {"AKI_Start","Surgery_End"}.issubset(df.columns):
        ak = pd.to_datetime(df["AKI_Start"], errors="coerce")
        se = pd.to_datetime(df["Surgery_End"], errors="coerce")
        days = (ak - se).dt.total_seconds()/86400.0
        return days.between(0,7).fillna(False).astype(int)
    raise ValueError("Kein AKI-Feld ableitbar (weder Stage, Flag noch Zeiten).")

def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Effektgröße für ordinal/nichtparametrische Daten
    # δ = ( # x>y - # x<y ) / (n_x * n_y)
    # effizient über Sortierung
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    i = j = more = less = 0
    nx, ny = len(x_sorted), len(y_sorted)
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            more += ny - j
            i += 1
        elif x_sorted[i] < y_sorted[j]:
            less += ny - j
            i += 1
        else:
            # bei Gleichheit: gehe über alle gleichen y weiter
            y_val = y_sorted[j]
            k = j
            while k < ny and y_sorted[k] == y_val:
                k += 1
            # gleiche Werte zählen weder zu more noch less
            i += 1
    denom = nx * ny
    return (more - less) / denom if denom > 0 else np.nan

def group_summary(df: pd.DataFrame, var: str, yflag: str = "AKI_linked_0_7") -> dict:
    s = pd.to_numeric(df[var], errors="coerce")
    g = pd.to_numeric(df[yflag], errors="coerce").astype(int)
    a = s[g == 0].dropna().values
    b = s[g == 1].dropna().values
    if len(a) == 0 and len(b) == 0:
        return None
    # Mann–Whitney U (zweiseitig)
    p = np.nan
    if len(a) > 0 and len(b) > 0:
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
    # Effektgröße
    delta = cliff_delta(a, b)
    return {
        "variable": var,
        "n_total": int(s.notna().sum()),
        "missing_pct": round(100 * s.isna().mean(), 1),
        "n_AKI0": int((~np.isnan(a)).sum()),
        "n_AKI1": int((~np.isnan(b)).sum()),
        "median_AKI0": np.nanmedian(a) if len(a) else np.nan,
        "iqr_AKI0": (np.nanpercentile(a, 25) if len(a) else np.nan,
                     np.nanpercentile(a, 75) if len(a) else np.nan),
        "mean_AKI0": float(np.nanmean(a)) if len(a) else np.nan,
        "sd_AKI0": float(np.nanstd(a, ddof=1)) if len(a) > 1 else np.nan,
        "median_AKI1": np.nanmedian(b) if len(b) else np.nan,
        "iqr_AKI1": (np.nanpercentile(b, 25) if len(b) else np.nan,
                     np.nanpercentile(b, 75) if len(b) else np.nan),
        "mean_AKI1": float(np.nanmean(b)) if len(b) else np.nan,
        "sd_AKI1": float(np.nanstd(b, ddof=1)) if len(b) > 1 else np.nan,
        "p_mannwhitney": float(p) if p == p else np.nan,
        "cliffs_delta": float(delta) if delta == delta else np.nan,
    }

def violin_plot(ad: AnnData, key: str, groupby: str, out_png: Path):
    if key not in ad.obs.columns or groupby not in ad.obs.columns:
        print(f"[skip] {key} oder {groupby} fehlt")
        return
    # ehrapy erwartet category
    ad.obs[groupby] = ad.obs[groupby].astype("category")
    try:
        ep.pl.violin(ad, keys=key, groupby=groupby)
        plt.gcf().savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] ehrapy.violin {key}~{groupby}: {e}")
        # Fallback Boxplot
        tmp = ad.obs[[key, groupby]].copy()
        tmp[key] = pd.to_numeric(tmp[key], errors="coerce")
        tmp = tmp.dropna()
        groups = [g[key].values for _, g in tmp.groupby(groupby)]
        labels = [str(l) for l in tmp[groupby].cat.categories]
        plt.figure()
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.xlabel(groupby); plt.ylabel(key)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

def run_cv_auc(df: pd.DataFrame, ycol: str, features: list[str], seed: int = 42) -> tuple[float, float, pd.Series]:
    cols = [c for c in features if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").values
    y = pd.to_numeric(df[ycol], errors="coerce").astype(int).values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2 or len(y) < 50:
        return np.nan, np.nan, pd.Series(dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    coefs = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced"
            ))
        ])
        pipe.fit(X[tr], y[tr])
        prob = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        # Koeffizienten zurückskalieren (nur ungefähre Wichtigkeit)
        clf = pipe.named_steps["clf"]
        sc  = pipe.named_steps["scaler"]
        coefs += clf.coef_.ravel() / 5.0
    return float(np.nanmean(aucs)), float(np.nanstd(aucs)), pd.Series(coefs, index=cols).sort_values(key=np.abs, ascending=False)

# ------------------ Daten laden ------------------
ad = ep.io.read_h5ad(str(H5AD))
df = ad.obs.copy()
print("geladen:", H5AD, "| rows:", len(df))

# AKI sicherstellen
df["AKI_linked_0_7"] = ensure_aki_flag(df)
ad.obs["AKI_linked_0_7"] = df["AKI_linked_0_7"]
ad.obs["AKI_linked_0_7"] = ad.obs["AKI_linked_0_7"].astype("category")
vc = df["AKI_linked_0_7"].value_counts().sort_index()
print("AKI=0:", int(vc.get(0,0)), "| AKI=1:", int(vc.get(1,0)))

# ------------------ Variablenblöcke ------------------
preop_vars = [
    "crea_baseline", "cysc_baseline"
]
early_postop_vars = [
    "crea_peak_0_48","crea_delta_0_48","crea_rate_0_48",
    "cysc_peak_0_48","cysc_delta_0_48","cysc_rate_0_48",
    "vis_max_0_24","vis_mean_0_24","vis_max_6_24","vis_auc_0_24","vis_auc_0_48",
    "duration_hours"
]
all_vars = [v for v in preop_vars + early_postop_vars if v in df.columns]

# ------------------ Statistik je Variable ------------------
rows = []
for var in all_vars:
    res = group_summary(df, var, yflag="AKI_linked_0_7")
    if res is not None:
        rows.append(res)

stats = pd.DataFrame(rows)
if not stats.empty:
    # IQR hübsch machen
    def fmt_iqr(t):
        if isinstance(t, tuple):
            return f"[{t[0]:.2f}; {t[1]:.2f}]"
        return ""
    stats["median_IQR_AKI0"] = stats.apply(lambda r: f"{r['median_AKI0']:.2f} {fmt_iqr(r['iqr_AKI0'])}" if pd.notna(r["median_AKI0"]) else "", axis=1)
    stats["median_IQR_AKI1"] = stats.apply(lambda r: f"{r['median_AKI1']:.2f} {fmt_iqr(r['iqr_AKI1'])}" if pd.notna(r["median_AKI1"]) else "", axis=1)
    # FDR
    pvals = stats["p_mannwhitney"].replace({np.nan:1.0}).values
    rej, p_adj, *_ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    stats["p_adj_fdr"] = p_adj
    stats["signif_fdr"] = rej
    # Ordnung: signifikant zuerst
    stats = stats.sort_values(["signif_fdr","p_adj_fdr"], ascending=[False, True])

    stats_out = stats[[
        "variable","n_total","missing_pct",
        "n_AKI0","median_IQR_AKI0","mean_AKI0","sd_AKI0",
        "n_AKI1","median_IQR_AKI1","mean_AKI1","sd_AKI1",
        "p_mannwhitney","p_adj_fdr","cliffs_delta","signif_fdr"
    ]]
    stats_out.to_csv(OUTP / "lab_vis_vs_AKI_stats.csv", index=False)
    stats_out.to_csv(OUTP / "lab_vis_vs_AKI_stats_FDR.csv", index=False)
    print("Statistiken gespeichert:", OUTP / "lab_vis_vs_AKI_stats_FDR.csv")
else:
    print("[WARN] Keine Zahlenvariablen gefunden → keine Statistik geschrieben.")

# ------------------ Violinplots ------------------
for var in all_vars:
    out_png = OUTF / f"VIOLIN_{var}_by_AKI.png"
    violin_plot(ad, var, "AKI_linked_0_7", out_png)

print("Violinplots gespeichert in:", OUTF)

# ------------------ Mini-Modelle ------------------
# 1) Prä-OP (nur baseline)
pre_feat = [v for v in preop_vars if v in df.columns]
pre_auc, pre_sd, pre_imp = run_cv_auc(df, "AKI_linked_0_7", pre_feat)

# 2) Frühe Post-OP (ohne baseline)
post_feat = [v for v in early_postop_vars if v in df.columns]
post_auc, post_sd, post_imp = run_cv_auc(df, "AKI_linked_0_7", post_feat)

# 3) Kombiniert
comb_feat = [v for v in pre_feat + post_feat]
comb_auc, comb_sd, comb_imp = run_cv_auc(df, "AKI_linked_0_7", comb_feat)

with open(OUTP / "lab_vis_vs_AKI_models.txt", "w") as f:
    def w(line=""):
        f.write(line + "\n")
    w("Mini-Modelle (Logistic Regression, 5-fold Stratified CV, ROC-AUC):")
    w(f"  Prä-OP  (n_feat={len(pre_feat)}):  AUC={pre_auc:.3f} ± {pre_sd:.3f}")
    w(f"  Post-OP (n_feat={len(post_feat)}): AUC={post_auc:.3f} ± {post_sd:.3f}")
    w(f"  Kombiniert (n_feat={len(comb_feat)}): AUC={comb_auc:.3f} ± {comb_sd:.3f}\n")
    if not pre_imp.empty:
        w("Top Koeffizienten Prä-OP:")
        w(pre_imp.head(10).to_string())
        w("")
    if not post_imp.empty:
        w("Top Koeffizienten Post-OP:")
        w(post_imp.head(10).to_string())
        w("")
    if not comb_imp.empty:
        w("Top Koeffizienten Kombiniert:")
        w(comb_imp.head(15).to_string())
        w("")
print("Model-Report gespeichert:", OUTP / "lab_vis_vs_AKI_models.txt")

print("\nFertig.")
