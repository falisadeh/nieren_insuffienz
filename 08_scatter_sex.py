# 08_scatter_sex.py  (robust: smarter CSV-Reader + flexible Spaltenfindung)
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from anndata import read_h5ad
import statsmodels.api as sm
from patsy import dmatrix

H5AD = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
PATIENT_MASTER = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv"
OUT  = r"/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUT, exist_ok=True)

# ---------- Helpers ----------
def norm(s: str) -> str:
    s = str(s)
    return (s.replace("\ufeff","").strip()
            .replace("(", " ").replace(")", " ")
            .replace(".", "").replace("-", " ")
            .lower())

def load_csv_smart(path):
    # Versuche, Separator & Encoding automatisch zu erkennen
    for sep, enc in [(None, "utf-8-sig"), (";", "utf-8-sig"), (",", "utf-8-sig"), (";", "latin1")]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise FileNotFoundError(f"Konnte CSV nicht lesen: {path}")
    # Spalten normalisieren
    new_cols = []
    for c in df.columns:
        c2 = norm(c)
        c2 = " ".join(c2.split())           # Mehrfach-Leerzeichen -> einfach
        c2 = c2.replace(" ", "")            # Leerzeichen entfernen
        new_cols.append(c2)
    df.columns = new_cols
    return df

def build_norm_map(cols):
    # Map von normalisiert -> Originalname
    m = {}
    for c in cols:
        key = norm(c)
        key = " ".join(key.split())
        key = key.replace(" ", "")
        m[key] = c
    return m

def find_col(norm_map, candidates, contains_any=None):
    # exakte Kandidaten zuerst
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # sonst: contains-Suche (z. B. "geschlecht" in "geschlechtmf")
    if contains_any:
        for k in norm_map:
            if any(sub in k for sub in contains_any):
                return norm_map[k]
    return None

# ---------- Daten laden ----------
adata = read_h5ad(H5AD)
obs = adata.obs

# Norm-Map für obs
obs_map = build_norm_map(obs.columns)

aki_obs  = find_col(obs_map, ["aki","akilinked07","aki_linked_0_7"])
age_obs  = find_col(obs_map, ["age_years_at_op","ageyearsatop","age_years","ageyears"])
sex_obs  = find_col(obs_map, ["sex","gender","geschlecht"], contains_any=["sex","gender","geschlecht"])
pmid_obs = find_col(obs_map, ["pmid","patientmasterindx","patientmasterindex","patientmasterindx"])

if not aki_obs or not age_obs:
    raise KeyError(f"AKI/Alter in adata.obs nicht gefunden (AKI={aki_obs}, Alter={age_obs}).")

# --- Sex aus obs oder via Patient Master join ---
if sex_obs is not None:
    sex_series = obs[sex_obs].astype(str)
    source = "obs"
else:
    # Patient Master lesen & normalisieren
    pm = load_csv_smart(PATIENT_MASTER)
    pm_map = build_norm_map(pm.columns)
    pmid_pm = find_col(pm_map, ["pmid","patientmasterindx","patientmasterindex","patientmasterindx"])
    sex_pm  = find_col(pm_map, ["sex","gender","geschlecht"], contains_any=["sex","gender","geschlecht"])
    if not pmid_pm or not sex_pm or not pmid_obs:
        raise KeyError(
            f"In Patient Master oder obs fehlen Schlüssel (pmid_obs={pmid_obs}, pmid_pm={pmid_pm}, sex_pm={sex_pm}).\n"
            f"Obs-Spalten (normiert): {list(obs_map.keys())[:12]}\n"
            f"PM-Spalten (normiert): {list(pm_map.keys())[:12]}"
        )
    # Join über PMID
    left = pd.DataFrame({"PMID": obs[pmid_obs].values})
    left["__row__"] = np.arange(len(left))
    right = pm[[pmid_pm, sex_pm]].rename(columns={pmid_pm: "PMID", sex_pm: "sex_raw"})
    merged = left.merge(right, on="PMID", how="left").sort_values("__row__")
    sex_series = merged["sex_raw"].astype(str).fillna("")
    source = "patient_master"

print(f"Sex-Quelle: {source}  |  AKI-Spalte: {aki_obs}  |  Alter-Spalte: {age_obs}")

# ---------- DataFrame bauen ----------
df = pd.DataFrame({
    "AKI": pd.to_numeric(obs[aki_obs], errors="coerce").astype(float),
    "age_years": pd.to_numeric(obs[age_obs], errors="coerce").astype(float),
    "sex_raw": sex_series.str.strip().str.lower()
}).dropna(subset=["AKI","age_years"])

# Geschlecht normalisieren
map_sex = {"f":"f","female":"f","w":"f","weiblich":"f","m":"m","male":"m","männlich":"m"}
df["sex"] = df["sex_raw"].map(map_sex)
# Falls codiert mit 0/1 oder '1'/'0'
df.loc[df["sex"].isna() & df["sex_raw"].isin(["0","1"]), "sex"] = df["sex_raw"].map({"0":"m","1":"f"})
df = df[df["sex"].isin(["f","m"])].copy()

# ----------------- Plot 1: Scatter -----------------
rng = np.random.default_rng(42)
colors = {"f":"#0EA5A4", "m":"#2563EB"}
plt.figure(figsize=(8,4.8), dpi=180)
for s in ["f","m"]:
    d = df[df["sex"]==s]
    if len(d)==0: continue
    jitter = (rng.random(len(d)) - 0.5) * 0.12
    plt.scatter(d["age_years"], d["AKI"] + jitter, s=12, alpha=0.7,
                label=("weiblich" if s=="f" else "männlich"), c=colors[s])
plt.yticks([0,1], ["kein AKI","AKI"])
plt.xlabel("Alter bei OP (Jahre)"); plt.ylabel("AKI (0–7 Tage)")
plt.title("Alter vs. AKI (0/1) – farblich nach Geschlecht"); plt.legend()
plt.tight_layout()
out1 = os.path.join(OUT, "scatter_age_vs_AKI_by_sex.png")
plt.savefig(out1, bbox_inches="tight"); plt.close(); print("Gespeichert:", out1)

# -------- Plot 2: Splines je Geschlecht ----------
plt.figure(figsize=(8,4.8), dpi=180)
for s in ["f","m"]:
    d = df[df["sex"]==s].dropna(subset=["AKI","age_years"])
    if len(d) < 40:  # kleine Gruppen auslassen
        continue
    X  = dmatrix("cr(age_years, df=5)", {"age_years": d["age_years"]}, return_type="dataframe")
    y  = d["AKI"].astype(int).values
    res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    grid = np.linspace(d["age_years"].min(), d["age_years"].max(), 350)
    Xg   = dmatrix("cr(age_years, df=5)", {"age_years": grid}, return_type="dataframe")
    sf   = res.get_prediction(sm.add_constant(Xg)).summary_frame()
    plt.plot(grid, sf["mean"], color=colors[s], linewidth=2,
             label=("weiblich" if s=="f" else "männlich"))
    plt.fill_between(grid, sf["mean_ci_lower"], sf["mean_ci_upper"], alpha=0.25, color=colors[s])
plt.ylim(-0.02, 1.02)
plt.xlabel("Alter bei OP (Jahre)"); plt.ylabel("p(AKI 0–7 Tage)")
plt.title("Alter vs. p(AKI) – Splines + 95%-CI nach Geschlecht"); plt.legend()
plt.tight_layout()
out2 = os.path.join(OUT, "age_vs_AKI_spline_by_sex_ci.png")
plt.savefig(out2, bbox_inches="tight"); plt.close(); print("Gespeichert:", out2)

# Zusammen mit Streudiagramm

# Kombi: Scatter (jitter) + Splines je Geschlecht
plt.figure(figsize=(8,4.8), dpi=180)
rng = np.random.default_rng(7)
colors = {"f":"#0EA5A4", "m":"#2563EB"}

# Scatter
for s in ["f","m"]:
    d = df[df["sex"]==s]
    if len(d)==0: continue
    jitter = (rng.random(len(d)) - 0.5) * 0.12
    plt.scatter(d["age_years"], d["AKI"] + jitter, s=10, alpha=0.35, c=colors[s])

# Splines drüber
for s in ["f","m"]:
    d = df[df["sex"]==s]
    if len(d) < 40: continue
    X  = dmatrix("cr(age_years, df=5)", {"age_years": d["age_years"]}, return_type="dataframe")
    y  = d["AKI"].astype(int).values
    res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    grid = np.linspace(d["age_years"].min(), d["age_years"].max(), 350)
    Xg   = dmatrix("cr(age_years, df=5)", {"age_years": grid}, return_type="dataframe")
    sf   = res.get_prediction(sm.add_constant(Xg)).summary_frame()
    plt.plot(grid, sf["mean"], color=colors[s], linewidth=2, label=("weiblich" if s=="f" else "männlich"))

plt.yticks([0,1], ["kein AKI","AKI"])
plt.ylim(-0.02, 1.02)
plt.xlabel("Alter bei OP (Jahre)")
plt.ylabel("AKI (0–7 Tage)")
plt.title("Alter vs. AKI – Rohpunkte + Splines nach Geschlecht")
plt.legend()
plt.tight_layout()
out3 = os.path.join(OUT, "scatter_plus_splines_by_sex.png")
plt.savefig(out3, bbox_inches="tight"); plt.close()
print("Gespeichert:", out3)



