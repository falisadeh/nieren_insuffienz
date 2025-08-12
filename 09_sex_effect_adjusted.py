# 09_sex_effect_adjusted.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from anndata import read_h5ad
import statsmodels.api as sm
from patsy import dmatrix, dmatrices

# --------- Pfade anpassen ---------
H5AD = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
PATIENT_MASTER = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv"
OUT  = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUT, exist_ok=True)

# --------- Helfer ---------
def norm(s):
    s = str(s).replace("\ufeff","").strip().lower()
    return " ".join(s.split())

def find_col(cols, cands):
    low = {norm(c): c for c in cols}
    for cand in cands:
        if cand in low: return low[cand]
    # contains-suche
    for k,v in low.items():
        if any(sub in k for sub in cands): return v
    return None

def load_csv_smart(path):
    for sep in [None, ";", ","]:
        for enc in ["utf-8-sig","latin1"]:
            try:
                return pd.read_csv(path, sep=sep, engine="python", encoding=enc)
            except Exception:
                pass
    raise FileNotFoundError(f"CSV nicht lesbar: {path}")

# --------- Daten laden ---------
adata = read_h5ad(H5AD); obs = adata.obs

aki_col = find_col(obs.columns, ["aki_linked_0_7","aki","akilinked07"])
age_col = find_col(obs.columns, ["age_years_at_op","age_years"])
reop_col= find_col(obs.columns, ["is_reop","reop","isreop"])
dur_col = find_col(obs.columns, ["duration_hours","op_duration_hours","duration"])
sex_col = find_col(obs.columns, ["sex","gender","geschlecht"])
pmid_col= find_col(obs.columns, ["pmid","patientmasterindx","patientmasterindex"])

if dur_col is None:
    # Versuche Dauer aus Start/Ende zu bauen
    s_start = find_col(obs.columns, ["surgery_start","start of surgery","surgery start"])
    s_end   = find_col(obs.columns, ["surgery_end","end of surgery","surgery end"])
    if s_start and s_end:
        dh = (pd.to_datetime(obs[s_end], errors="coerce") - pd.to_datetime(obs[s_start], errors="coerce")).dt.total_seconds()/3600.0
        obs = obs.copy(); obs["duration_hours_built"] = dh
        dur_col = "duration_hours_built"

# Sex besorgen (obs oder Patient Master)
if sex_col is None:
    if pmid_col is None: raise KeyError("Kein Geschlecht in obs und keine PMID zum Join vorhanden.")
    pm = load_csv_smart(PATIENT_MASTER)
    pm_pmid = find_col(pm.columns, ["pmid","patientmasterindx","patientmasterindex"])
    pm_sex  = find_col(pm.columns, ["sex","gender","geschlecht"])
    if pm_pmid is None or pm_sex is None: raise KeyError("In Patient Master fehlen PMID/Sex.")
    tmp = pd.DataFrame({"PMID": obs[pmid_col].values})
    sex_series = tmp.merge(pm[[pm_pmid, pm_sex]].rename(columns={pm_pmid:"PMID", pm_sex:"sex_raw"}),
                           on="PMID", how="left")["sex_raw"].astype(str)
else:
    sex_series = obs[sex_col].astype(str)

# DataFrame bauen
df = pd.DataFrame({
    "AKI": pd.to_numeric(obs[aki_col], errors="coerce"),
    "age_years": pd.to_numeric(obs[age_col], errors="coerce"),
    "is_reop": pd.to_numeric(obs[reop_col], errors="coerce") if reop_col else 0.0,
    "duration_hours": pd.to_numeric(obs[dur_col], errors="coerce") if dur_col else np.nan,
    "sex_raw": sex_series.str.strip().str.lower()
})

# Sex normalisieren
map_sex = {"f":"f","female":"f","w":"f","weiblich":"f","m":"m","male":"m","männlich":"m"}
df["sex"] = df["sex_raw"].map(map_sex)
df.loc[df["sex"].isna() & df["sex_raw"].isin(["0","1"]), "sex"] = df["sex_raw"].map({"0":"m","1":"f"})

# Clean
df = df.dropna(subset=["AKI","age_years"])
if df["sex"].isin(["f","m"]).sum() < 50:
    raise ValueError("Zu wenige valide Sex-Werte nach Normalisierung.")

# Fehlen OP-Dauer? -> Median-Imputation (nur für Modell; Transparenz!)
if df["duration_hours"].isna().mean() > 0:
    med_dur = df["duration_hours"].median(skipna=True)
    df["duration_hours"] = df["duration_hours"].fillna(med_dur)

# Binär-Codierung
df = df[df["sex"].isin(["f","m"])].copy()
df["SexM"] = (df["sex"]=="m").astype(int)   # Referenz: weiblich (0)
df["is_reop"] = (df["is_reop"]>0).astype(int)
df["AKI"] = df["AKI"].astype(int)

print(f"n={len(df)} | AKI-Rate={df['AKI'].mean():.3f} | m-Anteil={df['SexM'].mean():.3f}")

# --------- Modelle (GLM Logit) ---------
# Basis ohne Geschlecht
X0 = dmatrix("cr(age_years, df=5) + duration_hours + is_reop", df, return_type="dataframe")
m0 = sm.GLM(df["AKI"], sm.add_constant(X0), family=sm.families.Binomial()).fit(cov_type="HC3")

# + Haupt­effekt Geschlecht
X1 = dmatrix("cr(age_years, df=5) + duration_hours + is_reop + SexM", df, return_type="dataframe")
m1 = sm.GLM(df["AKI"], sm.add_constant(X1), family=sm.families.Binomial()).fit(cov_type="HC3")

# + Interaktion Geschlecht × Alter-Splines
X2 = dmatrix("cr(age_years, df=5) * SexM + duration_hours + is_reop", df, return_type="dataframe")  # * = main + interaction
m2 = sm.GLM(df["AKI"], sm.add_constant(X2), family=sm.families.Binomial()).fit(cov_type="HC3")

# Likelihood-Ratio-Tests (gegenüber verschachtelten Modellen)
import scipy.stats as st
def lrtest(m_small, m_big, df_diff):
    LR = 2*(m_big.llf - m_small.llf)
    p = 1 - st.chi2.cdf(LR, df=df_diff)
    return LR, p

LR_sex, p_sex = lrtest(m0, m1, df_diff=(m1.df_model - m0.df_model))
LR_inter, p_inter = lrtest(m1, m2, df_diff=(m2.df_model - m1.df_model))

print("\n=== Tests ===")
print(f"Haupteffekt Geschlecht (LRT): LR={LR_sex:.2f}, p={p_sex:.4g}")
# Wald-p für SexM im m1:
p_wald_sex = m1.pvalues.get("SexM", np.nan)
print(f"Haupteffekt Geschlecht (Wald in m1): p={p_wald_sex:.4g}")
print(f"Interaktion Geschlecht×Alter (LRT): LR={LR_inter:.2f}, p={p_inter:.4g}")

# --------- Adjustierte Kurven plotten ---------
grid = np.linspace(df["age_years"].min(), df["age_years"].max(), 350)
def predict_curve(sexM, reop=0, dur=None):
    if dur is None: dur = float(df["duration_hours"].median())
    new = pd.DataFrame({"age_years":grid, "SexM":sexM, "is_reop":reop, "duration_hours":dur})
    Xg = dmatrix("cr(age_years, df=5) * SexM + duration_hours + is_reop", new, return_type="dataframe")
    sf = m2.get_prediction(sm.add_constant(Xg)).summary_frame()
    return sf["mean"].values, sf["mean_ci_lower"].values, sf["mean_ci_upper"].values

colors = {0:"#0EA5A4", 1:"#2563EB"}  # 0=f, 1=m
plt.figure(figsize=(8,4.8), dpi=180)
for sexM,label in [(0,"weiblich"),(1,"männlich")]:
    pm, lo, hi = predict_curve(sexM=sexM, reop=0)  # Erst-OP
    plt.plot(grid, pm, color=colors[sexM], lw=2, label=label + " (Erst-OP)")
    plt.fill_between(grid, lo, hi, color=colors[sexM], alpha=0.22)

plt.ylim(-0.02,1.02); plt.xlabel("Alter bei OP (Jahre)"); plt.ylabel("p(AKI 0–7 Tage)")
plt.title("Adjustierte p(AKI): Alter-Splines · Dauer (Median) · Re-OP=0")
plt.legend(); plt.tight_layout()
out = os.path.join(OUT, "age_vs_AKI_adj_by_sex_ci.png")
plt.savefig(out, bbox_inches="tight"); plt.close()
print("Gespeichert:", out)

# Optional: zweite Grafik für Re-OP=1
plt.figure(figsize=(8,4.8), dpi=180)
for sexM,label in [(0,"weiblich"),(1,"männlich")]:
    pm, lo, hi = predict_curve(sexM=sexM, reop=1)
    plt.plot(grid, pm, color=colors[sexM], lw=2, label=label + " (Re-OP)")
    plt.fill_between(grid, lo, hi, color=colors[sexM], alpha=0.22)
plt.ylim(-0.02,1.02); plt.xlabel("Alter bei OP (Jahre)"); plt.ylabel("p(AKI 0–7 Tage)")
plt.title("Adjustierte p(AKI): Alter-Splines · Dauer (Median) · Re-OP=1")
plt.legend(); plt.tight_layout()
out2 = os.path.join(OUT, "age_vs_AKI_adj_by_sex_ci_reop.png")
plt.savefig(out2, bbox_inches="tight"); plt.close()
print("Gespeichert:", out2)
title_add = f" · p(Sex)={p_wald_sex:.3g} · p(Sex×Alter)={p_inter:.3g}"
plt.title("Adjustierte p(AKI): Alter-Splines · Dauer (Median) · Re-OP=0" + title_add)
# und analog für Re-OP=1
with open(os.path.join(OUT, "sex_effect_tests.txt"), "w") as f:
    f.write(f"n={len(df)}\n")
    f.write(f"Haupteffekt Geschlecht (LRT): LR={LR_sex:.2f}, p={p_sex:.4g}\n")
    f.write(f"Haupteffekt Geschlecht (Wald in m1): p={p_wald_sex:.4g}\n")
    f.write(f"Interaktion Geschlecht×Alter (LRT): LR={LR_inter:.2f}, p={p_inter:.4g}\n")
