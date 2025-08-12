#%%
"""
07_age_trends_splines_FIXED.py
Finale Abbildungen (Publikations-/Jury-Qualität) aus aki_ops_master_S1_survival.h5ad

Erzeugt:
- age_vs_AKI_spline_years_ci.png               (Splines + 95%-CI, Jahre)
- age_days_vs_AKI_spline_0_2y_ci.png           (Splines + 95%-CI, 0–2 Jahre in Tagen)
- age_vs_hours_to_AKI_loess.png                (Nur AKI-Fälle: Stunden bis AKI + LOESS)
- age_vs_AKI_spline_by_reop_ci.png (optional)  (Splines + 95%-CI, stratifiziert nach Re-OP)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOESS = True
except Exception:
    HAS_LOESS = False

# ------------------------- Pfade -------------------------
H5AD = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
OUT  = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUT, exist_ok=True)

# --------------------- Daten laden -----------------------
from anndata import read_h5ad
adata = read_h5ad(H5AD)
obs = adata.obs

# ------------------- Grund-DataFrame ---------------------
# Alter in Jahren/Tagen robust ableiten
if "age_years_at_op" in obs.columns:
    age_years = pd.to_numeric(obs["age_years_at_op"], errors="coerce").astype(float)
else:
    if "age_days_at_op" not in obs.columns:
        raise KeyError("Keine Alters-Spalte gefunden (age_years_at_op / age_days_at_op)")
    age_years = pd.to_numeric(obs["age_days_at_op"], errors="coerce").astype(float) / 365.25

if "age_days_at_op" in obs.columns:
    age_days = pd.to_numeric(obs["age_days_at_op"], errors="coerce").astype(float)
else:
    age_days = pd.to_numeric(obs["age_years_at_op"], errors="coerce").astype(float) * 365.25

# AKI-Spalte robust finden
aki_candidates = ["AKI", "AKI_linked_0_7", "aki_linked_0_7"]
AKI_COL = next((c for c in aki_candidates if c in obs.columns), None)
if AKI_COL is None:
    raise KeyError(f"Keine AKI-Spalte im Datensatz gefunden. Verfügbar: {list(obs.columns)}")

AKI_vals = pd.to_numeric(obs[AKI_COL], errors="coerce").astype(float)

# df bauen (einheitlicher Name 'AKI')
df = pd.DataFrame({
    "AKI": AKI_vals,
    "age_years": age_years,
    "age_days": age_days,
}).replace([np.inf, -np.inf], np.nan)

print(f"adata geladen: n_obs={adata.n_obs}, df-Spalten={list(df.columns)}")
print("AKI-Quelle:", AKI_COL)

# ----------------- Plot A: Splines (Jahre) -----------------
dfA = df.dropna(subset=["AKI", "age_years"]).copy()
df_spline = 5  # Freiheitsgrade (4–6 sinnvoll)
X = dmatrix(f"cr(age_years, df={df_spline})", {"age_years": dfA["age_years"]}, return_type="dataframe")
y = dfA["AKI"].astype(int).values

model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
res   = model.fit()

grid = np.linspace(dfA["age_years"].min(), dfA["age_years"].max(), 400)
Xg   = dmatrix(f"cr(age_years, df={df_spline})", {"age_years": grid}, return_type="dataframe")
pred = res.get_prediction(sm.add_constant(Xg)).summary_frame()
p_mean, p_low, p_high = pred["mean"], pred["mean_ci_lower"], pred["mean_ci_upper"]

# Binned-Rate (Quantile-Bins)
df_b = dfA.copy()
df_b["bin"] = pd.qcut(df_b["age_years"], q=20, duplicates="drop")
b_x = df_b.groupby("bin", observed=True)["age_years"].median().values
b_p = df_b.groupby("bin", observed=True)["AKI"].mean().values

plt.figure(figsize=(8,5), dpi=180)
plt.plot(grid, p_mean, linewidth=2, label=f"Natürliche kubische Splines (df={df_spline})")
plt.fill_between(grid, p_low, p_high, alpha=0.25, label="95%-Konfidenzband")
plt.scatter(b_x, b_p, s=40, marker="s", label=f"AKI-Rate je Quantil-Bin (n={len(b_x)})")
plt.xlabel("Alter bei OP (Jahre)")
plt.ylabel("p(AKI 0–7 Tage)")
plt.title("Alter vs. AKI (0–7 Tage): Natürliche Splines + 95%-CI")
plt.ylim(-0.02, 1.02); plt.legend(); plt.tight_layout()
outA = os.path.join(OUT, "age_vs_AKI_spline_years_ci.png")
plt.savefig(outA, bbox_inches="tight"); plt.close()
print("Gespeichert:", outA)

# ------------- Plot B: Splines (0–2 Jahre in Tagen) --------------
mask_days = df["age_days"].notna() & (df["age_days"] >= 0) & (df["age_days"] <= 730) & df["AKI"].notna()
d2 = df.loc[mask_days, ["age_days", "AKI"]].copy()

if len(d2) >= 30:
    df_spline_days = 5
    Xd = dmatrix(f"cr(age_days, df={df_spline_days})", {"age_days": d2["age_days"]}, return_type="dataframe")
    yd = d2["AKI"].astype(int).values
    md = sm.GLM(yd, sm.add_constant(Xd), family=sm.families.Binomial()).fit()

    grid_d = np.linspace(d2["age_days"].min(), d2["age_days"].max(), 400)
    Xgd    = dmatrix(f"cr(age_days, df={df_spline_days})", {"age_days": grid_d}, return_type="dataframe")
    sf     = md.get_prediction(sm.add_constant(Xgd)).summary_frame()
    pm, lo, hi = sf["mean"], sf["mean_ci_lower"], sf["mean_ci_upper"]

    d2b = d2.copy()
    d2b["bin"] = pd.qcut(d2b["age_days"], q=20, duplicates="drop")
    b_x = d2b.groupby("bin", observed=True)["age_days"].median().values
    b_p = d2b.groupby("bin", observed=True)["AKI"].mean().values

    plt.figure(figsize=(8,5), dpi=180)
    plt.plot(grid_d, pm, linewidth=2, label=f"Natürliche Splines (df={df_spline_days})")
    plt.fill_between(grid_d, lo, hi, alpha=0.25, label="95%-Konfidenzband")
    plt.scatter(b_x, b_p, s=40, marker="s", label=f"AKI-Rate je Quantil-Bin (n={len(b_x)})")
    plt.xlabel("Alter bei OP (Tage, 0–2 Jahre)")
    plt.ylabel("p(AKI 0–7 Tage)")
    plt.title("Alter (Tage) vs. AKI (0–7 Tage): Zoom 0–2 Jahre")
    plt.ylim(-0.02, 1.02); plt.legend(); plt.tight_layout()
    outB = os.path.join(OUT, "age_days_vs_AKI_spline_0_2y_ci.png")
    plt.savefig(outB, bbox_inches="tight"); plt.close()
    print("Gespeichert:", outB)
else:
    print(f"0–2 Jahre: zu wenig n (n={len(d2)}) – Plot übersprungen.")

# --------- Plot C: Nur AKI-Fälle – Stunden bis AKI + LOESS ---------
hours = None
if "AKI_Start" in obs.columns and "Surgery_End" in obs.columns:
    dt_aki = pd.to_datetime(obs["AKI_Start"], errors="coerce")
    dt_end = pd.to_datetime(obs["Surgery_End"], errors="coerce")
    hours = (dt_aki - dt_end).dt.total_seconds() / 3600.0
elif "days_to_AKI" in obs.columns:
    hours = pd.to_numeric(obs["days_to_AKI"], errors="coerce") * 24.0

if hours is not None:
    dfa = pd.DataFrame({"age_years": df["age_years"], "AKI": df["AKI"], "hours": hours}).dropna()
    dfa = dfa[dfa["AKI"] == 1].copy()
    dfa["hours"] = dfa["hours"].clip(lower=0, upper=168)

    plt.figure(figsize=(8,5), dpi=180)
    plt.scatter(dfa["age_years"], dfa["hours"], s=14, alpha=0.8, label="AKI-Fälle")
    if HAS_LOESS and len(dfa) >= 30:
        lo = lowess(dfa["hours"].values, dfa["age_years"].values, frac=0.3, return_sorted=True)
        plt.plot(lo[:,0], lo[:,1], linewidth=2, linestyle="--", label="LOESS (Stunden bis AKI)")
    plt.xlabel("Alter bei OP (Jahre)")
    plt.ylabel("Stunden bis AKI (0–168)")
    plt.title("Nur AKI-Fälle: Stunden bis AKI nach OP")
    plt.ylim(-5, 175); plt.legend(); plt.tight_layout()
    outC = os.path.join(OUT, "age_vs_hours_to_AKI_loess.png")
    plt.savefig(outC, bbox_inches="tight"); plt.close()
    print("Gespeichert:", outC)
else:
    print("Hinweis: hours_to_AKI konnte nicht berechnet werden (fehlende Spalten).")

# ---- Plot D: Stratifiziert nach Re-OP (wenn vorhanden) ----
if "is_reop" in obs.columns:
    df2 = pd.DataFrame({
        "age_years": df["age_years"],
        "AKI": df["AKI"],
        "is_reop": pd.to_numeric(obs["is_reop"], errors="coerce"),
    }).dropna()

    done = False
    for grp, label in [(0, "Erst-OP"), (1, "Re-OP")]:
        dfg = df2[df2["is_reop"] == grp]
        if len(dfg) < 30:
            continue
        Xg = dmatrix("cr(age_years, df=5)", {"age_years": dfg["age_years"]}, return_type="dataframe")
        mg = sm.GLM(dfg["AKI"].astype(int).values, sm.add_constant(Xg), family=sm.families.Binomial()).fit()
        grid = np.linspace(dfg["age_years"].min(), dfg["age_years"].max(), 300)
        Xp   = dmatrix("cr(age_years, df=5)", {"age_years": grid}, return_type="dataframe")
        sf   = mg.get_prediction(sm.add_constant(Xp)).summary_frame()
        if not done:
            plt.figure(figsize=(8,5), dpi=180); done = True
        plt.plot(grid, sf["mean"], linewidth=2, label=label)
        plt.fill_between(grid, sf["mean_ci_lower"], sf["mean_ci_upper"], alpha=0.20)
    if done:
        plt.xlabel("Alter bei OP (Jahre)"); plt.ylabel("p(AKI 0–7 Tage)")
        plt.title("Alter vs. AKI: Splines + 95%-CI (stratifiziert nach Re-OP)")
        plt.ylim(-0.02, 1.02); plt.legend(); plt.tight_layout()
        outD = os.path.join(OUT, "age_vs_AKI_spline_by_reop_ci.png")
        plt.savefig(outD, bbox_inches="tight"); plt.close()
        print("Gespeichert:", outD)
    else:
        print("Stratifizierung: zu wenig n in mindestens einer Gruppe – Plot übersprungen.")
else:
    print("'is_reop' nicht gefunden – Stratifikationsplot übersprungen.")

print("Fertig.")

# %%
