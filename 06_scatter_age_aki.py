# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import read_h5ad

# %% Pfade/Datei
H5AD = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
OUTDIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUTDIR, exist_ok=True)

# %% Daten laden (falls adata noch nicht existiert)
try:
    adata
except NameError:
    adata = read_h5ad(H5AD)

# ---- Konfig: Spaltennamen prüfen
age_cols = [c for c in adata.obs.columns if c in ("age_years_at_op","age_days_at_op")]
if not age_cols:
    raise KeyError("Keine Alters-Spalte gefunden. Erwartet: 'age_years_at_op' oder 'age_days_at_op'.")
AGE = age_cols[0]  # nimmt, was vorhanden ist (Jahre bevorzugt)

AKI = "AKI_linked_0_7"
if AKI not in adata.obs.columns:
    raise KeyError(f"{AKI} nicht in adata.obs gefunden.")

# %% 1) Scatter: Alter vs. AKI (0/1) mit leichtem Jitter
df = adata.obs[[AGE, AKI]].copy()
df[AGE] = pd.to_numeric(df[AGE], errors="coerce")
df[AKI] = pd.to_numeric(df[AKI], errors="coerce")
df = df.dropna()

# Jitter nur auf y (0/1), damit Punkte nicht exakt übereinander liegen
rng = np.random.default_rng(42)
y = df[AKI].values.astype(float) + rng.uniform(-0.05, 0.05, size=len(df))

plt.figure(figsize=(7,4), dpi=160)
plt.scatter(df[AGE].values, y, s=10, alpha=0.6)
plt.yticks([0,1], ["AKI=0","AKI=1"])
plt.xlabel("Alter bei OP" + (" (Jahre)" if AGE.endswith("_years_at_op") else " (Tage)"))
plt.ylabel("AKI 0–7 Tage (mit Jitter)")
plt.title("Streudiagramm: Alter vs. AKI (0–7 Tage)")
plt.tight_layout()
out1 = os.path.join(OUTDIR, "scatter_age_vs_AKI_0_7.png")
plt.savefig(out1, bbox_inches="tight")
plt.close()
print("Gespeichert:", out1)

# %% 2) Optional: Nur AKI-Fälle – Alter vs. Tage bis AKI
if "days_to_AKI" in adata.obs.columns:
    dfa = adata.obs[[AGE, "days_to_AKI", AKI]].copy()
    dfa[AGE] = pd.to_numeric(dfa[AGE], errors="coerce")
    dfa["days_to_AKI"] = pd.to_numeric(dfa["days_to_AKI"], errors="coerce")
    dfa = dfa.dropna()
    dfa = dfa[dfa[AKI] == 1]

    plt.figure(figsize=(7,4), dpi=160)
    plt.scatter(dfa[AGE].values, dfa["days_to_AKI"].values, s=12, alpha=0.7)
    plt.xlabel("Alter bei OP" + (" (Jahre)" if AGE.endswith("_years_at_op") else " (Tage)"))
    plt.ylabel("Tage bis AKI")
    plt.title("Nur AKI-Fälle: Alter vs. Tage bis AKI (0–7)")
    plt.tight_layout()
    out2 = os.path.join(OUTDIR, "scatter_age_vs_days_to_AKI.png")
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print("Gespeichert:", out2)
else:
    print("Hinweis: 'days_to_AKI' nicht vorhanden – Plot 2 übersprungen.")
#----------------------------
# %% Extras: Trendkurven für Streudiagramme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: scikit-learn LogReg + Polynomfeatures (für glatte, nichtlineare Kurve)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional: LOESS (falls statsmodels da ist); sonst wird automatisch übersprungen
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOESS = True
except Exception:
    HAS_LOESS = False

# --- Hilfsfunktionen ---------------------------------------------------
def add_logistic_trend(ax, x, y, degree=3, n_grid=300):
    """Legt eine glatte p(AKI|Alter)-Kurve über den Scatter (0/1)."""
    # Nur gültige Zahlen
    m = np.isfinite(x) & np.isfinite(y)
    x_fit = x[m].reshape(-1, 1)
    y_fit = y[m].astype(int)

    # Pipeline: Polynomfeatures -> LogReg
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("logreg", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(x_fit, y_fit)

    # Gleichmäßig feinmaschiges Alter-Raster
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid).reshape(-1, 1)
    p = pipe.predict_proba(x_grid)[:, 1]  # p(AKI=1)

    # Trendkurve zeichnen (keine Farbangabe -> Matplotlib-Default)
    ax.plot(x_grid.ravel(), p, linewidth=2, label=f"Logistische Trendkurve (Grad {degree})")

def add_loess_trend(ax, x, y, frac=0.25):
    """LOESS-Glättung der 0/1-Zielvariablen (zeigt mittlere p(AKI|Alter))."""
    if not HAS_LOESS:
        return
    m = np.isfinite(x) & np.isfinite(y)
    lo = lowess(y[m].astype(float), x[m], frac=frac, return_sorted=True)
    ax.plot(lo[:, 0], lo[:, 1], linewidth=2, linestyle="--", label=f"LOESS (frac={frac})")

def add_binned_rate(ax, x, y, bins=20):
    """Empirische AKI-Rate je Alters-Bin (mit Punkten)."""
    dfb = pd.DataFrame({"x": x, "y": y}).dropna()
    # Quantile-Bins funktionieren gut bei schiefen Verteilungen
    dfb["bin"] = pd.qcut(dfb["x"], q=bins, duplicates="drop")
    grp = dfb.groupby("bin")
    x_mid = grp["x"].median().values
    rate = grp["y"].mean().values
    ax.scatter(x_mid, rate, s=30, marker="s", label=f"AKI-Rate je Quantil-Bin (n={len(x_mid)})")

# --- 1) Alter vs. AKI (0/1) mit Trendkurven ----------------------------
df_plot = adata.obs[[AGE, AKI]].copy()
df_plot[AGE] = pd.to_numeric(df_plot[AGE], errors="coerce")
df_plot[AKI] = pd.to_numeric(df_plot[AKI], errors="coerce")
df_plot = df_plot.dropna()

# y-Jitter für sichtbare Punkte
rng = np.random.default_rng(42)
y_scatter = df_plot[AKI].values.astype(float) + rng.uniform(-0.05, 0.05, size=len(df_plot))

fig = plt.figure(figsize=(7, 4), dpi=160)
ax = plt.gca()
ax.scatter(df_plot[AGE].values, y_scatter, s=10, alpha=0.6, label="OP-Episoden")

# Trendkurven (mind. eine aktiv lassen)
add_logistic_trend(ax, df_plot[AGE].values, df_plot[AKI].values, degree=3, n_grid=400)
if HAS_LOESS:
    add_loess_trend(ax, df_plot[AGE].values, df_plot[AKI].values, frac=0.25)
add_binned_rate(ax, df_plot[AGE].values, df_plot[AKI].values, bins=20)

# Achsen/Labels
xlabel = "Alter bei OP (Jahre)" if AGE.endswith("_years_at_op") else "Alter bei OP (Tage)"
ax.set_xlabel(xlabel)
ax.set_ylabel("AKI 0–7 Tage (Jitter) / p(AKI)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["0", "1"])
ax.set_title("Alter vs. AKI (0–7 Tage) mit Trend")
ax.legend(loc="best")
plt.tight_layout()

out_trend1 = os.path.join(OUTDIR, "scatter_age_vs_AKI_0_7_trend.png")
plt.savefig(out_trend1, bbox_inches="tight")
plt.close()
print("Gespeichert:", out_trend1)

# --- 2) Nur AKI-Fälle: Alter vs. Tage bis AKI + LOESS ------------------
if "days_to_AKI" in adata.obs.columns:
    dfa = adata.obs[[AGE, "days_to_AKI", AKI]].copy()
    dfa[AGE] = pd.to_numeric(dfa[AGE], errors="coerce")
    dfa["days_to_AKI"] = pd.to_numeric(dfa["days_to_AKI"], errors="coerce")
    dfa = dfa.dropna()
    dfa = dfa[dfa[AKI] == 1]

    fig = plt.figure(figsize=(7, 4), dpi=160)
    ax = plt.gca()
    ax.scatter(dfa[AGE].values, dfa["days_to_AKI"].values, s=12, alpha=0.7, label="AKI-Fälle")
    if HAS_LOESS:
        lo = lowess(dfa["days_to_AKI"].values, dfa[AGE].values, frac=0.3, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], linewidth=2, linestyle="--", label="LOESS (Tage bis AKI)")

    xlabel = "Alter bei OP (Jahre)" if AGE.endswith("_years_at_op") else "Alter bei OP (Tage)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tage bis AKI (0–7)")
    ax.set_title("Nur AKI-Fälle: Alter vs. Tage bis AKI")
    ax.legend(loc="best")
    plt.tight_layout()

    out_trend2 = os.path.join(OUTDIR, "scatter_age_vs_days_to_AKI_loess.png")
    plt.savefig(out_trend2, bbox_inches="tight")
    plt.close()
    print("Gespeichert:", out_trend2)
else:
    print("Hinweis: 'days_to_AKI' nicht vorhanden – Plot 2 (LOESS) übersprungen.")
