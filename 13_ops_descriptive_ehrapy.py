# 13_ops_descriptive_ehrapy.py
import ehrapy as ep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== Pfad anpassen ====
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5 = BASE / "ops_with_crea_cysc_vis_features.h5ad"
OUT = BASE / "fig_ops_desc"
OUT.mkdir(parents=True, exist_ok=True)

# ==== Daten laden (ehrapy / AnnData) ====
adata = ep.io.read_h5ad(str(H5))
df = adata.obs.copy()

print(adata)
print("Spalten (obs):", df.columns.tolist())

# ==== Schnelle Übersicht ====
# numerische und kategoriale Variablen trennen
num_cols = df.select_dtypes(include=[np.number, "float64", "int64"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

print("\n--- Numerische Features (erste 12) ---")
print(df[num_cols].describe().T.head(12))

print("\n--- Kategoriale Features (erste 12) ---")
for c in cat_cols[:12]:
    vc = df[c].value_counts(dropna=False).head(6)
    print(f"{c} ->\n{vc}\n")

# ==== Welche Kernfeatures plotten? ====
core_feats = [
    # Kreatinin (mg/dL)
    "crea_baseline", "crea_peak_0_48", "crea_delta_0_48", "crea_rate_0_48",
    # Cystatin C (mg/L)
    "cysc_baseline", "cysc_peak_0_48", "cysc_delta_0_48", "cysc_rate_0_48",
    # VIS
    "vis_max_0_24", "vis_mean_0_24", "vis_max_6_24", "vis_auc_0_24", "vis_auc_0_48",
    # ggf. OP-Dauer
    "duration_hours",
]
core_feats = [c for c in core_feats if c in df.columns]

# ==== Helfer: einfache Histogramme speichern ====
def save_hist(series: pd.Series, title: str, fname: Path, bins=40, logx=False):
    s = series.dropna()
    if s.empty:
        print(f"[skip] {title}: keine Daten")
        return
    plt.figure()
    if logx:
        # bei stark rechtsschiefen Verteilungen optional log10
        s = np.log10(s[s > 0])
        plt.xlabel("log10")
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ==== Plots: unstratifiziert ====
for feat in core_feats:
    logx = feat.endswith("_auc_0_24") or feat.endswith("_auc_0_48") or feat.startswith("vis_max")
    save_hist(df[feat], f"{feat} (Histogramm)", OUT / f"hist_{feat}.png", bins=40, logx=logx)

# ==== Optional: Stratifizierung nach AKI (falls vorhanden) ====
aki_col = None
for cand in ["AKI_linked_0_7", "aki_0_7", "AKI"]:
    if cand in df.columns:
        aki_col = cand
        break

if aki_col:
    print(f"\nStratifiziere nach {aki_col}...")
    # Boxplots pro Feature nach AKI
    for feat in core_feats:
        sub = df[[aki_col, feat]].dropna()
        if sub.empty:
            continue
        plt.figure()
        groups = [sub.loc[sub[aki_col]==val, feat] for val in sorted(sub[aki_col].unique())]
        plt.boxplot(groups, labels=[str(v) for v in sorted(sub[aki_col].unique())], showfliers=False)
        plt.title(f"{feat} nach {aki_col}")
        plt.xlabel(aki_col)
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(OUT / f"box_{feat}_by_{aki_col}.png", dpi=150)
        plt.close()

    # kleine tabellarische Übersicht
    summary = (
        df.groupby(aki_col)[core_feats]
          .agg(["count","median","mean"])
    )
    summary.to_csv(OUT / f"summary_by_{aki_col}.csv")
    print(f"Gespeichert: {OUT / f'summary_by_{aki_col}.csv'}")
else:
    print("\nHinweis: Keine AKI-Spalte gefunden – stratifizierte Plots übersprungen.")

# ==== Korrelationen der numerischen Features ====
if core_feats:
    corr = df[core_feats].corr(method="spearman", min_periods=50)  # robuster bei Ausreißern
    corr.to_csv(OUT / "corr_core_feats_spearman.csv")
    # simple Heatmap ohne Styling
    plt.figure(figsize=(max(6, 0.5*len(core_feats)), max(5, 0.5*len(core_feats))))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(core_feats)), core_feats, rotation=90)
    plt.yticks(range(len(core_feats)), core_feats)
    plt.title("Spearman-Korrelation (core features)")
    plt.tight_layout()
    plt.savefig(OUT / "corr_core_feats_spearman.png", dpi=150)
    plt.close()
    print(f"Gespeichert: {OUT / 'corr_core_feats_spearman.png'}")
else:
    print("Keine core_feats gefunden – Korrelationen übersprungen.")

print("\nFertig. Grafiken liegen unter:", OUT)
