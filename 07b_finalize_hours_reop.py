import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from anndata import read_h5ad
import statsmodels.api as sm
from patsy import dmatrix
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOESS = True
except Exception:
    HAS_LOESS = False

H5AD = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
OUT  = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUT, exist_ok=True)

adata = read_h5ad(H5AD)
obs = adata.obs

# --- Robust: Altersvektor in Jahren ---
if "age_years_at_op" in obs:
    age_years = pd.to_numeric(obs["age_years_at_op"], errors="coerce").astype(float)
else:
    age_years = pd.to_numeric(obs["age_days_at_op"], errors="coerce").astype(float) / 365.25

AKI = pd.to_numeric(obs["AKI_linked_0_7"], errors="coerce").astype(float)

# ================== Plot 1: Stunden bis AKI ==================
hours_to_AKI = None
if "AKI_Start" in obs.columns and "Surgery_End" in obs.columns:
    dt_aki = pd.to_datetime(obs["AKI_Start"], errors="coerce")
    dt_end = pd.to_datetime(obs["Surgery_End"], errors="coerce")
    hours_to_AKI = (dt_aki - dt_end).dt.total_seconds() / 3600.0
elif "days_to_AKI" in obs.columns:
    hours_to_AKI = pd.to_numeric(obs["days_to_AKI"], errors="coerce") * 24.0

if hours_to_AKI is not None:
    dfa = pd.DataFrame({"age_years": age_years, "AKI": AKI, "hours": hours_to_AKI}).dropna()
    dfa = dfa[dfa["AKI"] == 1].copy()
    # Datenqualität: negative/überlange Werte kappen
    dfa["hours"] = dfa["hours"].clip(lower=0, upper=168)

    plt.figure(figsize=(7.5,4.5), dpi=180)
    plt.scatter(dfa["age_years"], dfa["hours"], s=12, alpha=0.8, label="AKI-Fälle")
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

# =========== Plot 2: Splines + CI, stratifiziert nach Re-OP ===========
if "is_reop" in obs.columns:
    df2 = pd.DataFrame({
        "age_years": age_years,
        "AKI": AKI,
        "is_reop": pd.to_numeric(obs["is_reop"], errors="coerce")
    }).dropna()
    done = False
    for grp, label in [(0, "Erst-OP"), (1, "Re-OP")]:
        dfg = df2[df2["is_reop"] == grp]
        if len(dfg) < 30:  # Mindest-n
            continue
        Xg = dmatrix("cr(age_years, df=5)", {"age_years": dfg["age_years"]}, return_type="dataframe")
        mg = sm.GLM(dfg["AKI"].astype(int).values, sm.add_constant(Xg), family=sm.families.Binomial()).fit()
        grid = np.linspace(dfg["age_years"].min(), dfg["age_years"].max(), 300)
        Xp   = dmatrix("cr(age_years, df=5)", {"age_years": grid}, return_type="dataframe")
        sf   = mg.get_prediction(sm.add_constant(Xp)).summary_frame()
        if not done:
            plt.figure(figsize=(7.5,4.5), dpi=180); done = True
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
