# op_dauer_from_supp_csv.py
#%%
import matplotlib
matplotlib.use("Agg")  # blockiert nicht im Terminal
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE); (OUT / "Diagramme").mkdir(exist_ok=True)

# 1) Supplement laden & säubern
df = pd.read_csv(f"{BASE}/Procedure Supplement.csv", sep=";", parse_dates=["Timestamp"])
df.columns = df.columns.str.strip()
for c in ["PMID", "SMID", "Procedure_ID", "TimestampName"]:
    df[c] = df[c].astype(str).str.strip()

# 2) Nur OP-Start/Ende
keys = ["PMID", "SMID", "Procedure_ID"]
df_se = df[df["TimestampName"].isin(["Start of surgery", "End of surgery"])][keys + ["TimestampName","Timestamp"]]

# 3) Pro Eingriff: frühester Start, spätestes Ende
starts = (df_se[df_se["TimestampName"]=="Start of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].min()
          .rename(columns={"Timestamp":"Start_supp"}))
ends   = (df_se[df_se["TimestampName"]=="End of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].max()
          .rename(columns={"Timestamp":"End_supp"}))

ops = starts.merge(ends, on=keys, how="outer")

# 4) Dauer berechnen (Zeitverschiebung egal!)
ops["duration_minutes"] = (ops["End_supp"] - ops["Start_supp"]).dt.total_seconds() / 60
ops["duration_hours"]   = ops["duration_minutes"] / 60

# 5) Qualität & Filter
missing_pairs = ops["Start_supp"].isna().sum() + ops["End_supp"].isna().sum()
negatives     = (ops["duration_minutes"] < 0).sum()

# Nur valide behalten: beide Zeiten vorhanden & Dauer >= 0
ops_valid = ops.dropna(subset=["Start_supp","End_supp"]).copy()
ops_valid = ops_valid[ops_valid["duration_minutes"] >= 0]

# Optional: unrealistisch lange OPs kappen (je nach Klinikrealität anpassen)
# z.B. nur <= 24h behalten:
# ops_valid = ops_valid[ops_valid["duration_hours"] <= 24]

print(f"Gesamt Eingriffe (unique): {ops.shape[0]}")
print(f"Fehlende Start/Ende: {missing_pairs}")
print(f"Negative Dauern (vor Filter): {negatives}")
print("\nKurzstatistik (Stunden) auf gültigen Fällen:")
print(ops_valid["duration_hours"].describe())

# 6) Speichern
out_csv = OUT / "op_durations_from_supp.csv"
ops_valid.to_csv(out_csv, sep=";", index=False)
print(f"\nGespeichert: {out_csv}")

# 7) Histogramm (Stunden)
plt.figure(figsize=(6,4))
plt.hist(ops_valid["duration_hours"].dropna(), bins=30)
plt.xlabel("OP-Dauer (Stunden)")
plt.ylabel("Anzahl Eingriffe")
plt.title("Verteilung der OP-Dauer (aus Procedure Supplement)")
plt.tight_layout()
plt.savefig(OUT / "Diagramme" / "OP_Dauer_Supp.png", dpi=300)
plt.close()
#
# eindeutige ID pro OP bauen
ops_valid['OP_ID'] = (
    ops_valid['PMID'].astype(str) + '_' +
    ops_valid['SMID'].astype(str) + '_' +
    ops_valid['Procedure_ID'].astype(str)
)
ops_valid.set_index('OP_ID', inplace=True)

# A) Direkt mit ehrapy einlesen (einfach)
ops_valid.to_csv(f"{BASE}/op_durations_from_supp.csv", sep=';', index=True)
import ehrapy as ep
adata = ep.io.read_csv(f"{BASE}/op_durations_from_supp.csv", sep=';')
# jetzt hast du alles in adata (ehrapy-konform)

# B) Oder manuell als AnnData bauen (stabil, ohne ep-Abhängigkeit)
from anndata import AnnData
import numpy as np
adata = AnnData(X=np.empty((ops_valid.shape[0], 0)))  # leere Matrix
adata.obs = ops_valid.copy()                           # alle Spalten in obs
import pandas as pd
import numpy as np
from anndata import AnnData

# Annahme: ops_valid existiert schon und enthält mind.:
# ['PMID','SMID','Procedure_ID','Start_supp','End_supp','duration_hours','duration_minutes']

# 1) Eindeutige OP-ID + Index
ops_valid = ops_valid.copy()
ops_valid['OP_ID'] = (
    ops_valid['PMID'].astype(str) + '_' +
    ops_valid['SMID'].astype(str) + '_' +
    ops_valid['Procedure_ID'].astype(str)
)
ops_valid = ops_valid.set_index('OP_ID')

# 2) AnnData bauen: X = numerische Dauer, obs = alle Metadaten
X = ops_valid[['duration_hours']].to_numpy(dtype=float)  # <— numerisch!
obs = ops_valid.drop(columns=['duration_hours']).copy()

adata = AnnData(X=X, obs=obs)
adata.var_names = ['duration_hours']

# 3) Schnelle Checks/„Ergebnis ausgeben“
print(adata)                                 # Form etc.
print(adata.obs.head(3))                      # Metadaten
print(pd.Series(adata.X.ravel()).describe())  # Kennzahlen der Dauer

# 4) (Optional) als .h5ad speichern
#adata.write_h5ad("/…/op_durations_from_supp.h5ad")
#%%
# EIne Tabelle für Op und AKI-Labels
# 00_build_analytic_ops_master.py
import pandas as pd
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE)

# ---------- 1) Load & clean ----------
op = pd.read_csv(f"{BASE}/HLM Operationen.csv", sep=";", parse_dates=["Start of surgery","End of surgery"])
aki = pd.read_csv(f"{BASE}/AKI Label.csv",     sep=";", parse_dates=["Start"])
pat = pd.read_csv(f"{BASE}/Patient Master Data.csv", sep=";")
supp= pd.read_csv(f"{BASE}/Procedure Supplement.csv", sep=";", parse_dates=["Timestamp"])

for df in (op, aki, pat, supp):
    df.columns = df.columns.str.strip()
    if "PMID" in df: df["PMID"] = df["PMID"].astype(str).str.strip()
if "SMID" in op: op["SMID"] = op["SMID"].astype(str).str.strip()
if "Procedure_ID" in op: op["Procedure_ID"] = op["Procedure_ID"].astype(str).str.strip()
for c in ("SMID","Procedure_ID","TimestampName"):
    if c in supp: supp[c] = supp[c].astype(str).str.strip()

def norm_sex(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip().lower()
    if s in {"m","male","männlich"}: return "m"
    if s in {"f","w","female","weiblich"}: return "f"
    return pd.NA
if "Sex" in pat:
    pat["Sex_norm"] = pat["Sex"].apply(norm_sex)
pat = pat.drop_duplicates(subset="PMID", keep="first")

# ---------- 2) Surgery times from Supplement (min Start, max End per OP) ----------
keys = ["PMID","SMID","Procedure_ID"]
se = supp[supp["TimestampName"].isin(["Start of surgery","End of surgery"])]

starts = (se[se["TimestampName"]=="Start of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].min()
          .rename(columns={"Timestamp":"Start_supp"}))
ends   = (se[se["TimestampName"]=="End of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].max()
          .rename(columns={"Timestamp":"End_supp"}))
supp_se = starts.merge(ends, on=keys, how="outer")

# ---------- 3) Combine HLM & Supplement into a single surgery window ----------
# Coalesce: nimm HLM-Zeit, sonst Supplement
for k in ("SMID","Procedure_ID"):
    if k not in op.columns:
        # Falls nicht vorhanden, trotzdem joinen nur mit vorhandenen Keys
        keys = [x for x in keys if x in op.columns]
        break

ops = op.merge(supp_se, on=[k for k in ("PMID","SMID","Procedure_ID") if k in op.columns], how="left")

ops["Surgery_Start"] = ops["Start of surgery"].where(ops["Start of surgery"].notna(), ops["Start_supp"])
ops["Surgery_End"]   = ops["End of surgery"].where(ops["End of surgery"].notna(),   ops["End_supp"])

# Dauer
ops["duration_minutes"] = (ops["Surgery_End"] - ops["Surgery_Start"]).dt.total_seconds()/60
ops["duration_hours"]   = ops["duration_minutes"]/60

# Gültig: beide Zeiten vorhanden & Dauer >= 0
ops = ops.dropna(subset=["Surgery_Start","Surgery_End"]).copy()
ops = ops[ops["duration_minutes"] >= 0].copy()

# Eindeutige OP-ID
ops["OP_ID"] = (
    ops["PMID"].astype(str) + "_" +
    ops.get("SMID","").astype(str) + "_" +
    ops.get("Procedure_ID","").astype(str)
)

# ---------- 4) Earliest AKI per patient & prior mapping ----------
aki_first = (aki.dropna(subset=["Start"])
             .sort_values(["PMID","Start"])
             .groupby("PMID", as_index=False).first()
             .rename(columns={"Start":"AKI_Start"}))

# Tabellen zum Merge vorbereiten & sortieren
ops_for_join  = ops[["PMID","SMID","Procedure_ID","Surgery_Start","OP_ID"]].sort_values(["PMID","Surgery_Start"]).reset_index(drop=True)
akis_for_join = aki_first[["PMID","AKI_Start"]].sort_values(["PMID","AKI_Start"]).reset_index(drop=True)

# prior mapping: letzte OP ≤ AKI je PMID
# --- Typen & Zeiten harmonisieren ---
ops_for_join = (
    ops[["PMID","SMID","Procedure_ID","Surgery_Start","OP_ID"]]
    .dropna(subset=["Surgery_Start"])
    .assign(
        PMID=lambda d: d["PMID"].astype(str).str.strip(),
        SMID=lambda d: d["SMID"].astype(str).str.strip(),
        Procedure_ID=lambda d: d["Procedure_ID"].astype(str).str.strip(),
        Surgery_Start=lambda d: pd.to_datetime(d["Surgery_Start"], errors="coerce").dt.tz_localize(None),
    )
    .dropna(subset=["Surgery_Start"])
    .sort_values(["PMID","Surgery_Start"], kind="mergesort")
    .reset_index(drop=True)
)

akis_for_join = (
    aki_first[["PMID","AKI_Start"]]
    .dropna(subset=["AKI_Start"])
    .assign(
        PMID=lambda d: d["PMID"].astype(str).str.strip(),
        AKI_Start=lambda d: pd.to_datetime(d["AKI_Start"], errors="coerce").dt.tz_localize(None),
    )
    .dropna(subset=["AKI_Start"])
    .sort_values(["PMID","AKI_Start"], kind="mergesort")
    .reset_index(drop=True)
)

###### --- prior mapping: letzte OP ≤ AKI je PMID ---
parts = []
for pmid, g_aki in akis_for_join.groupby("PMID", sort=False):
    g_ops = ops_for_join.loc[ops_for_join["PMID"] == pmid]
    if g_ops.empty:
        continue
    m = pd.merge_asof(
        g_aki.sort_values("AKI_Start"),
        g_ops.sort_values("Surgery_Start"),
        left_on="AKI_Start", right_on="Surgery_Start",
        direction="backward",
        allow_exact_matches=True,
    )
    m["PMID"] = pmid  # wichtig, damit die Spalte sicher vorhanden ist
    parts.append(m)

prior_map = pd.concat(parts, ignore_index=True)
prior_map["days_to_AKI"] = (prior_map["AKI_Start"] - prior_map["Surgery_Start"]).dt.days


# Delta-Tage
prior_map["days_to_AKI"] = (prior_map["AKI_Start"] - prior_map["Surgery_Start"]).dt.days

# ---------- 5) Build OP-level master ----------
master = ops.merge(
    prior_map[["PMID","SMID","Procedure_ID","OP_ID","AKI_Start","days_to_AKI"]],
    on=["PMID","SMID","Procedure_ID","OP_ID"],
    how="left"
).merge(
    pat[["PMID","Sex_norm"]],
    on="PMID", how="left"
)

# Label, ob diese OP die prior-OP vor AKI ist
master["AKI_linked"] = master["AKI_Start"].notna().astype("int64")
# Optional: Fenster 0–7 Tage (nur für Analysen)
master["AKI_linked_0_7"] = ((master["AKI_linked"]==1) & (master["days_to_AKI"].between(0,7))).astype("int64")

# ---------- 6) (Optional) Patient-level summary ----------
pat_summary = (master.groupby("PMID")
               .agg(
                   n_ops            = ("OP_ID","count"),
                   total_op_hours   = ("duration_hours","sum"),
                   mean_op_hours    = ("duration_hours","mean"),
                   max_op_hours     = ("duration_hours","max"),
                   earliest_op      = ("Surgery_Start","min"),
                   latest_op        = ("Surgery_Start","max"),
                   AKI_Start        = ("AKI_Start","max"),   # earliest AKI per prior_map (identisch pro Patient)
               )
               .reset_index()
               .merge(pat[["PMID","Sex_norm"]], on="PMID", how="left"))

# ---------- 7) Save ----------
master_cols_order = [
    "OP_ID","PMID","SMID","Procedure_ID",
    "Surgery_Start","Surgery_End","duration_hours","duration_minutes",
    "AKI_Start","days_to_AKI","AKI_linked","AKI_linked_0_7",
    "Sex_norm"
]
master = master[[c for c in master_cols_order if c in master.columns]]

master_path = OUT / "analytic_ops_master.csv"
pat_path    = OUT / "analytic_patient_summary.csv"
master.to_csv(master_path, sep=";", index=False)
pat_summary.to_csv(pat_path, sep=";", index=False)

print("Geschrieben:")
print(" -", master_path)
print(" -", pat_path)
print("\nChecks:")
print("OPs gesamt:", master.shape[0])
print("linked prior-OPs:", int(master["AKI_linked"].sum()))
print("linked 0–7:", int(master["AKI_linked_0_7"].sum()))

print(master["duration_hours"].describe())
#Ehrapy AnnData export
#%%
import ehrapy as ep
from anndata import AnnData
import pandas as pd
import numpy as np
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
path = Path(BASE) / "analytic_ops_master.csv"

# CSV laden; Datumsfelder tolerant parsen
date_cols = ["Surgery_Start","Surgery_End","AKI_Start"]
df = pd.read_csv(path, sep=";")
for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# OP_ID sicherstellen
if "OP_ID" not in df.columns:
    df["OP_ID"] = (
        df["PMID"].astype(str) + "_" +
        df.get("SMID","").astype(str) + "_" +
        df.get("Procedure_ID","").astype(str)
    )

# Muss-Spalte prüfen
assert "duration_hours" in df.columns, f"'duration_hours' fehlt. Spalten: {df.columns.tolist()}"

# AnnData bauen: X = Dauer (float), obs = Rest
df = df.set_index("OP_ID")
X   = df[["duration_hours"]].to_numpy(dtype=float)
obs = df.drop(columns=["duration_hours"])
adata = AnnData(X=X, obs=obs)
adata.var_names = ["duration_hours"]

print(adata)  # z.B. AnnData object with n_obs × n_vars = 1209 × 1


# %%
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

obs = adata.obs.copy()
obs["duration_hours"] = np.asarray(adata.X).ravel().astype(float)

def med_iqr(s):
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return q2, q1, q3

x = obs.loc[obs["AKI_linked"]==1, "duration_hours"].dropna()
y = obs.loc[obs["AKI_linked"]==0, "duration_hours"].dropna()

mx, q1x, q3x = med_iqr(x)
my, q1y, q3y = med_iqr(y)
U, p = mannwhitneyu(x, y, alternative="two-sided")

print(f"AKI_linked=1 (n={len(x)}): Median {mx:.2f} h (IQR {q1x:.2f}–{q3x:.2f})")
print(f"AKI_linked=0 (n={len(y)}): Median {my:.2f} h (IQR {q1y:.2f}–{q3y:.2f})")
print(f"Mann–Whitney: U={U:.0f}, p={p:.4g}")

#
import numpy as np
import pandas as pd
import scipy.sparse as sp

def get_duration_hours(adata):
    # Fall 1: Dauer liegt in obs
    if "duration_hours" in adata.obs.columns:
        return pd.to_numeric(adata.obs["duration_hours"], errors="coerce").to_numpy()

    # Fall 2: Dauer liegt in X (über var_names identifizierbar)
    vars_ = [str(v) for v in adata.var_names]
    if "duration_hours" in vars_:
        j = vars_.index("duration_hours")
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        col = X[:, j]                      # <-- bereits NumPy, kein .to_numpy()
        return pd.to_numeric(col, errors="coerce")

    # Fallback: nimm die erste Spalte von X
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    return pd.to_numeric(X[:, 0], errors="coerce")
obs = adata.obs.copy()
obs["duration_hours"] = get_duration_hours(adata).astype(float)
print(obs["duration_hours"].describe())
# --- Boxplots aus ehrapy/AnnData ---

# Arbeits-DataFrame
obs = adata.obs.copy()
obs["duration_hours"] = get_duration_hours(adata)
obs["AKI_linked"] = pd.to_numeric(obs["AKI_linked"], errors="coerce").astype("Int64")
obs["AKI_linked_0_7"] = pd.to_numeric(obs["AKI_linked_0_7"], errors="coerce").astype("Int64")

# 1) Boxplot: OP-Dauer vs. AKI_linked (0/1)
plt.figure(figsize=(5,4))
obs.boxplot(column="duration_hours", by="AKI_linked", grid=False)
plt.xlabel("AKI_linked (0 = nein, 1 = prior-OP)")
plt.ylabel("OP-Dauer (Stunden)")
plt.title("OP-Dauer nach AKI-Zuordnung")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUT / "OP_Dauer_vs_AKI_linked.png", dpi=300)
plt.close()

# 2) Boxplot: nur prior-OPs im 0–7-Tage-Fenster vs. andere (0/1)
plt.figure(figsize=(5,4))
obs.boxplot(column="duration_hours", by="AKI_linked_0_7", grid=False)
plt.xlabel("AKI_linked_0_7 (0 = nein, 1 = prior-OP mit 0–7 Tagen)")
plt.ylabel("OP-Dauer (Stunden)")
plt.title("OP-Dauer – prior-OPs im 0–7-Tage-Fenster")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUT / "OP_Dauer_vs_AKI_linked_0_7.png", dpi=300)
plt.close()

# 3) Optional: nur verlinkte OPs, Boxplot nach Geschlecht
if "Sex_norm" in obs.columns:
    linked = obs[obs["AKI_linked"] == 1].copy()
    if not linked.empty:
        plt.figure(figsize=(6,4))
        linked.boxplot(column="duration_hours", by="Sex_norm", grid=False)
        plt.xlabel("Geschlecht")
        plt.ylabel("OP-Dauer (Stunden)")
        plt.title("OP-Dauer (nur prior-OPs) nach Geschlecht")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(OUT / "OP_Dauer_prior_by_sex.png", dpi=300)
        plt.close()

print("Gespeichert:",
      OUT / "OP_Dauer_vs_AKI_linked.png",
      OUT / "OP_Dauer_vs_AKI_linked_0_7.png",
      "(+ OP_Dauer_prior_by_sex.png falls Sex_norm vorhanden)")
#%%
# ehrapy-first Boxplot
# ehrapy-first, robustes Abholen von duration_hours & AKI_linked und Boxplot speichern
import matplotlib
matplotlib.use("Agg")
import ehrapy as ep
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.sparse as sp
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE) / "Diagramme"; OUT.mkdir(exist_ok=True)

# 1) CSV in ehrapy laden
adata = ep.io.read_csv(f"{BASE}/analytic_ops_master.csv", sep=";")
if "OP_ID" in adata.obs.columns:
    adata.obs_names = adata.obs["OP_ID"].astype(str)
    adata.obs.drop(columns=["OP_ID"], inplace=True)

# 2) Helfer: hole Spalte 'name' aus obs ODER aus X/var_names (auch mit Trim/Lower)
def get_col(adata, name, dtype=float):
    # direkte Namen
    if name in adata.obs.columns:
        return pd.to_numeric(adata.obs[name], errors="coerce").astype(dtype)
    vars_ = [str(v) for v in adata.var_names]
    if name in vars_:
        j = vars_.index(name)
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        return pd.to_numeric(X[:, j], errors="coerce").astype(dtype)

    # fuzzy match (getrimmt/klein)
    target = name.strip().lower()
    for c in adata.obs.columns:
        if c.strip().lower() == target:
            return pd.to_numeric(adata.obs[c], errors="coerce").astype(dtype)
    vars_map = {c: c.strip().lower() for c in vars_}
    for c, low in vars_map.items():
        if low == target:
            j = vars_.index(c)
            X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            return pd.to_numeric(X[:, j], errors="coerce").astype(dtype)

    raise KeyError(f"'{name}' nicht gefunden. obs={list(adata.obs.columns)[:10]}, vars={vars_[:10]}")

# 3) Arbeits-DataFrame für Plot
obs = adata.obs.copy()
obs["duration_hours"] = get_duration_hours(adata).astype(float)

# AKI_linked sauber als pandas-Integer (0/1) anlegen
aki_vals = get_col(adata, "AKI_linked", dtype=float)       # kann ndarray oder Series sein
aki_series = pd.Series(aki_vals, index=obs.index)          # -> garantiert pandas Series
aki_series = pd.to_numeric(aki_series, errors="coerce").round()

# bevorzugt nullable-Integer:
try:
    obs["AKI_linked"] = aki_series.astype("Int64")
except TypeError:
    # Fallback, falls deine pandas-Version "Int64" nicht kennt
    obs["AKI_linked"] = aki_series.fillna(0).astype("int64")

# 4) Boxplot speichern
plt.figure(figsize=(6,4))
obs.boxplot(column="duration_hours", by="AKI_linked", grid=False)
plt.xlabel("AKI_linked (0 = nein, 1 = prior-OP)")
plt.ylabel("OP-Dauer (Stunden)")
plt.title("OP-Dauer nach AKI-Zuordnung")
plt.suptitle("")
plt.tight_layout()
out_path = OUT / "OP_Dauer_vs_AKI_linked.png"
plt.savefig(out_path, dpi=300)
plt.close()

print("Gespeichert:", out_path)
# Optional: schneller Check
print("obs cols (Ausschnitt):", list(adata.obs.columns)[:12])
print("vars (Ausschnitt):", [str(v) for v in adata.var_names[:12]])
print(obs[["duration_hours","AKI_linked"]].dropna().head())

# %%
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

obs = adata.obs.copy()
dur = np.asarray(adata.X).ravel().astype(float)
obs["duration_hours"] = dur

prior = obs[obs["AKI_linked"]==1].copy()
g = prior.groupby("Sex_norm")["duration_hours"]

print("n prior-OPs:", len(prior), " | n_f:", (prior["Sex_norm"]=="f").sum(), "n_m:", (prior["Sex_norm"]=="m").sum())
print("Median/IQR f:", g.get_group("f").median(), g.get_group("f").quantile([0.25,0.75]).tolist())
print("Median/IQR m:", g.get_group("m").median(), g.get_group("m").quantile([0.25,0.75]).tolist())

x = g.get_group("f").dropna(); y = g.get_group("m").dropna()
U, p = mannwhitneyu(x, y, alternative="two-sided")
print(f"Mann–Whitney (f vs m): U={U:.0f}, p={p:.4g}")


