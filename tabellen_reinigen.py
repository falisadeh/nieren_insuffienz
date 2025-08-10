import re
import pandas as pd
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"

# ---------- Hilfsfunktionen ----------
def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def to_str_strip(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def parse_dates(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def normalize_sex(s):
    if pd.isna(s): return pd.NA
    x = str(s).strip().lower()
    if x in {"m", "male", "männlich"}: return "m"
    if x in {"f", "w", "female", "weiblich"}: return "f"
    return pd.NA

# ---------- 1) HLM Operationen ----------
df_op = pd.read_csv(f"{BASE}/HLM Operationen.csv", sep=";")
strip_columns(df_op)
to_str_strip(df_op, ["PMID", "SMID", "Procedure_ID"])
parse_dates(df_op, ["Start of surgery", "End of surgery"])

# optional: komplett leere/kaputte OP-Zeilen entfernen
df_op = df_op[~df_op["Start of surgery"].isna()].copy()

# ---------- 2) Patient Master Data ----------
df_pat = pd.read_csv(f"{BASE}/Patient Master Data.csv", sep=";")
strip_columns(df_pat)
to_str_strip(df_pat, ["PMID", "Sex"])
parse_dates(df_pat, ["DateOfBirth", "DateOfDie"])
df_pat["Sex"] = df_pat["Sex"].apply(normalize_sex)
# Doppelte Stammdatensätze auflösen
df_pat = df_pat.drop_duplicates(subset="PMID", keep="first").copy()

# ---------- 3) AKI Label ----------
df_aki = pd.read_csv(f"{BASE}/AKI Label.csv", sep=";")
strip_columns(df_aki)
# Tippfehler korrigieren (Duartion -> Duration), nur wenn vorhanden
if "Duartion" in df_aki.columns and "Duration" not in df_aki.columns:
    df_aki = df_aki.rename(columns={"Duartion": "Duration"})
to_str_strip(df_aki, ["PMID", "Decision"])
parse_dates(df_aki, ["Start", "End"])

# AKI-Flags (optional, falls später gebraucht)
# AKI (0/1) und Stufe aus Decision extrahieren, z.B. "AKI 2" -> 2
df_aki["AKI"] = df_aki["Decision"].notna().astype("Int64")
def parse_stage(x):
    if pd.isna(x): return pd.NA
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else pd.NA
df_aki["AKI_Stufe"] = df_aki["Decision"].apply(parse_stage).astype("Int64")

# Pro Patient das früheste AKI-Datum (für Latenz-Analyse)
df_aki_first = (
    df_aki.sort_values("Start")
          .dropna(subset=["Start"])
          .groupby("PMID", as_index=False)
          .first()[["PMID", "Start"]]
          .rename(columns={"Start": "AKI_Start"})
)

# ---------- 4) Procedure Supplement (für spätere Zwecke; AKI steht hier wohl nicht drin) ----------
df_supp = pd.read_csv(f"{BASE}/Procedure Supplement.csv", sep=";")
strip_columns(df_supp)
to_str_strip(df_supp, ["PMID", "SMID", "Procedure_ID", "TimestampName"])
parse_dates(df_supp, ["Timestamp"])

# ---------- 5) Integritäts-Checks (Keys) ----------
# PMIDs als Strings harmonisieren
df_op["PMID"]  = df_op["PMID"].astype(str).str.strip()
df_pat["PMID"] = df_pat["PMID"].astype(str).str.strip()
df_aki_first["PMID"] = df_aki_first["PMID"].astype(str).str.strip()

pmids_only_in_op  = set(df_op["PMID"])  - set(df_pat["PMID"])
pmids_only_in_pat = set(df_pat["PMID"]) - set(df_op["PMID"])

print(f"PMIDs nur in OP   (Beispiel max 10): {list(pmids_only_in_op)[:10]}")
print(f"PMIDs nur in PAT  (Beispiel max 10): {list(pmids_only_in_pat)[:10]}")

# ---------- 6) OPs nach Geschlecht zählen (m/f/unbekannt) ----------
merged = pd.merge(df_op, df_pat[["PMID", "Sex"]], on="PMID", how="left")
total_ops = len(df_op)
by_sex = merged["Sex"].value_counts(dropna=False)
m_count = int(by_sex.get("m", 0))
f_count = int(by_sex.get("f", 0))
na_count = int(by_sex.get(pd.NA, 0))

print("\n=== OP-Zählung ===")
print("Total OPs:", total_ops)                 # sollte 1209 sein
print("m:", m_count, " | f:", f_count, " | unbekannt:", na_count)
print("Summe:", m_count + f_count + na_count)  # muss == total_ops sein

# Plausibilitäts-Assertion
assert (m_count + f_count + na_count) == total_ops, "Summe m+f+unbekannt != Gesamt-OPs"

# ---------- 7) Latenz: AKI nach erster OP (Patienten-Ebene) ----------
first_op = (
    df_op.sort_values("Start of surgery")
         .groupby("PMID", as_index=False)["Start of surgery"]
         .first()
         .rename(columns={"Start of surgery": "First_OP"})
)

df_patient = pd.merge(first_op, df_aki_first, on="PMID", how="inner")
df_patient["days_to_AKI"] = (df_patient["AKI_Start"] - df_patient["First_OP"]).dt.days

# 0–30 Tage (oder 0–7, je nach Protokoll) filtern
df_all = df_patient.copy()
df_filtered = df_all[(df_all["days_to_AKI"] >= 0) & (df_all["days_to_AKI"] <= 30)]
df_outliers = df_all[(df_all["days_to_AKI"] < 0) | (df_all["days_to_AKI"] > 30)]

print("\n=== AKI-Latenz (Patienten-Ebene) ===")
print("Vor Filter:", df_all.shape[0], " | Nach Filter:", df_filtered.shape[0],
      " | Outlier:", df_outliers.shape[0])
print(df_filtered["days_to_AKI"].describe())

# ---------- 8) Saubere CSVs speichern ----------
outdir = Path(BASE)
df_op.to_csv(outdir / "HLM Operationen_clean.csv", sep=";", index=False)
df_pat.to_csv(outdir / "Patient Master Data_clean.csv", sep=";", index=False)
df_aki_first.to_csv(outdir / "AKI Label_first_clean.csv", sep=";", index=False)
df_supp.to_csv(outdir / "Procedure Supplement_clean.csv", sep=";", index=False)

#%% (AKI zum jeweils letzten OP vor AKI statt „erste OP“) Ergebnisse speichern (CSV + Histogramm)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE) / "Diagramme"

# Annahme: df_all = First_OP vs. AKI_Start (vorher berechnet), df_filtered = 0–7 Tage
df_all.to_csv(Path(BASE) / "AKI_Latenz_alle.csv", index=False, sep=";")
df_outliers = df_all[(df_all['days_to_AKI'] < 0) | (df_all['days_to_AKI'] > 7)]
df_outliers.to_csv(Path(BASE) / "AKI_Latenz_outlier.csv", index=False, sep=";")
df_filtered.to_csv(Path(BASE) / "AKI_Latenz_0_7.csv", index=False, sep=";")

# Histogramm 0–7 Tage
OUT.mkdir(exist_ok=True)
plt.figure(figsize=(6,4))
plt.hist(df_filtered['days_to_AKI'], bins=range(0,8))
plt.xlim(0,7)
plt.xlabel('Tage bis AKI nach erster OP')
plt.ylabel('Anzahl der Patienten')
plt.title('Verteilung der Tage bis zum AKI-Beginn (0–7)')
plt.tight_layout()
plt.savefig(OUT / "AKI_Latenz_Hist_0-7.png", dpi=300)
plt.show()
#Robustheits-Check: AKI zum jeweils letzten OP vor AKI
ops = (df_op[['PMID','Start of surgery']]
       .dropna()
       .assign(PMID=lambda d: d['PMID'].astype(str).str.strip())
       .rename(columns={'Start of surgery':'Surgery_Start'}))

akis = (df_aki[['PMID','Start']]
        .dropna()
        .assign(PMID=lambda d: d['PMID'].astype(str).str.strip())
        .rename(columns={'Start':'AKI_Start'}))

parts = []
for pmid, g_aki in akis.groupby('PMID'):
    g_ops = ops[ops['PMID'] == pmid]
    if g_ops.empty:
        continue
    g_aki = g_aki.sort_values('AKI_Start')
    g_ops = g_ops.sort_values('Surgery_Start')
    m = pd.merge_asof(
        g_aki, g_ops,
        left_on='AKI_Start',
        right_on='Surgery_Start',
        direction='backward',
        allow_exact_matches=True
    )
    m['PMID'] = pmid
    parts.append(m)

res = pd.concat(parts, ignore_index=True)
res = res.dropna(subset=['Surgery_Start'])

# frühestes AKI pro Patient
res_first = (res.sort_values('AKI_Start')
               .groupby('PMID', as_index=False)
               .first())

res_first['days_to_AKI'] = (res_first['AKI_Start'] - res_first['Surgery_Start']).dt.days
res_filt = res_first[(res_first['days_to_AKI'] >= 0) & (res_first['days_to_AKI'] <= 7)]

print("n nach Filter:", res_filt.shape[0])
print(res_filt['days_to_AKI'].describe())

import pandas as pd

# -------------------------------------------------------
# 0) Falls df_op / df_aki noch nicht existieren, lade sie so:
# df_op  = pd.read_csv("HLM Operationen.csv", sep=";", parse_dates=["Start of surgery"])
# df_aki = pd.read_csv("AKI Label.csv",     sep=";", parse_dates=["Start"])
# df_op.columns  = df_op.columns.str.strip()
# df_aki.columns = df_aki.columns.str.strip()
# -------------------------------------------------------

# Einheitliche Typen/Trim
df_op  = df_op.copy()
df_aki = df_aki.copy()
df_op["PMID"]  = df_op["PMID"].astype(str).str.strip()
df_aki["PMID"] = df_aki["PMID"].astype(str).str.strip()

# ============================
# A) Variante "erste OP" → df_first
# ============================
# 1) erste OP pro Patient
first_op = (
    df_op[["PMID","Start of surgery"]]
      .dropna()
      .sort_values("Start of surgery")
      .groupby("PMID", as_index=False)
      .first()
      .rename(columns={"Start of surgery":"First_OP"})
)

# 2) frühestes AKI pro Patient
df_aki_first = (
    df_aki[["PMID","Start"]]
      .dropna()
      .sort_values("Start")
      .groupby("PMID", as_index=False)
      .first()
      .rename(columns={"Start":"AKI_Start"})
)

# 3) mergen & Differenz
df_first = pd.merge(first_op, df_aki_first, on="PMID", how="inner")
df_first["days_first"] = (df_first["AKI_Start"] - df_first["First_OP"]).dt.days

# 4) Filter 0–7 Tage
df_first = df_first[(df_first["days_first"] >= 0) & (df_first["days_first"] <= 7)].copy()

print("df_first (erste OP)  n:", df_first.shape[0])
print(df_first["days_first"].describe())

# ============================
# B) Variante "letzte OP vor AKI" → df_prior
# ============================
# 1) alle OPs & alle AKIs je Patient, sauber sortiert
# ===== Variante B: merge_asof je Patient (robust) =====
# Vorbereitung: nur benötigte Spalten, NaN raus, IDs säubern
ops = (df_op[['PMID','Start of surgery']].dropna()
       .rename(columns={'Start of surgery':'Surgery_Start'}).copy())
ops['PMID'] = ops['PMID'].astype(str).str.strip()

akis = (df_aki[['PMID','Start']].dropna()
        .rename(columns={'Start':'AKI_Start'}).copy())
akis['PMID'] = akis['PMID'].astype(str).str.strip()

parts = []
for pmid, g_aki in akis.groupby('PMID', sort=False):
    g_ops = ops.loc[ops['PMID'] == pmid]
    if g_ops.empty:
        continue
    # innerhalb der PMID sauber sortieren (wichtig!)
    g_aki = g_aki.sort_values('AKI_Start')
    g_ops = g_ops.sort_values('Surgery_Start')

    m = pd.merge_asof(
        g_aki, g_ops,
        left_on='AKI_Start', right_on='Surgery_Start',
        direction='backward',
        allow_exact_matches=True
    )
    m['PMID'] = pmid
    parts.append(m)

# zusammenführen und nur AKIs mit vorheriger OP behalten
res = pd.concat(parts, ignore_index=True)
res = res.dropna(subset=['Surgery_Start']).copy()

# pro Patient das früheste AKI behalten
res_first = (res.sort_values(['PMID','AKI_Start'])
               .groupby('PMID', as_index=False)
               .first())

# Latenz berechnen + 0–7 Tage filtern
res_first['days_prior'] = (res_first['AKI_Start'] - res_first['Surgery_Start']).dt.days
df_prior = res_first[(res_first['days_prior'] >= 0) & (res_first['days_prior'] <= 7)].copy()

print("df_prior (letzte OP vor AKI)  n:", df_prior.shape[0])
print(df_prior['days_prior'].describe())

# ============================
# C) Vergleich beider Ansätze
# ============================
# Merge auf PMID + AKI_Start (beide haben je Patient das früheste AKI)
import pandas as pd

# Falls noch nicht gestrippt/geladen:
# df_op  = pd.read_csv("HLM Operationen.csv", sep=";", parse_dates=["Start of surgery"])
# df_aki = pd.read_csv("AKI Label.csv",     sep=";", parse_dates=["Start"])
df_op  = df_op.copy();  df_aki = df_aki.copy()
df_op.columns  = df_op.columns.str.strip()
df_aki.columns = df_aki.columns.str.strip()
df_op["PMID"]  = df_op["PMID"].astype(str).str.strip()
df_aki["PMID"] = df_aki["PMID"].astype(str).str.strip()

# ============== A) ERSTE OP pro Patient → df_first ==============
first_op = (df_op[["PMID","Start of surgery"]]
            .dropna()
            .sort_values("Start of surgery")
            .groupby("PMID", as_index=False)
            .first()
            .rename(columns={"Start of surgery":"OP"}))

aki_first = (df_aki[["PMID","Start"]]
             .dropna()
             .sort_values("Start")
             .groupby("PMID", as_index=False)
             .first()
             .rename(columns={"Start":"AKI_Start"}))

df_first = pd.merge(first_op, aki_first, on="PMID", how="inner")
df_first["days_first"] = (df_first["AKI_Start"] - df_first["OP"]).dt.days
df_first = df_first[(df_first["days_first"] >= 0) & (df_first["days_first"] <= 7)].copy()

print("df_first n:", df_first.shape[0])
print(df_first["days_first"].describe())

# ============== B) LETZTE OP VOR AKI je Patient → df_prior ==============
# robust: merge_asof pro Patient (um Sortierungsfehler zu vermeiden)
ops = (df_op[["PMID","Start of surgery"]]
       .dropna()
       .rename(columns={"Start of surgery":"OP"}))
akis = (df_aki[["PMID","Start"]]
        .dropna()
        .rename(columns={"Start":"AKI_Start"}))

parts = []
for pmid, g_aki in akis.groupby("PMID", sort=False):
    g_ops = ops.loc[ops["PMID"] == pmid]
    if g_ops.empty:
        continue
    g_aki = g_aki.sort_values("AKI_Start")
    g_ops = g_ops.sort_values("OP")
    m = pd.merge_asof(
        g_aki, g_ops,
        left_on="AKI_Start", right_on="OP",
        direction="backward", allow_exact_matches=True
    )
    m["PMID"] = pmid
    parts.append(m)

df_prior = pd.concat(parts, ignore_index=True)
df_prior = df_prior.dropna(subset=["OP"]).copy()

# frühestes AKI pro Patient behalten
df_prior = (df_prior.sort_values(["PMID","AKI_Start"])
                    .groupby("PMID", as_index=False)
                    .first())

df_prior["days_prior"] = (df_prior["AKI_Start"] - df_prior["OP"]).dt.days
df_prior = df_prior[(df_prior["days_prior"] >= 0) & (df_prior["days_prior"] <= 7)].copy()

print("\ndf_prior n:", df_prior.shape[0])
print(df_prior["days_prior"].describe())

# ============== C) VERGLEICH beider Ansätze ==============
cmp = df_first.merge(
    df_prior, on=["PMID","AKI_Start"], how="outer",
    suffixes=("_first","_prior"), indicator=True
)

only_prior = cmp[cmp["_merge"] == "right_only"]
only_first = cmp[cmp["_merge"] == "left_only"]
both       = cmp[cmp["_merge"] == "both"].copy()

# hat sich die zugeordnete OP geändert?
if not both.empty:
    both["changed_op"] = (both["OP_first"] != both["OP_prior"]).fillna(False)
    changed = int(both["changed_op"].sum())
else:
    changed = 0

print("\nNur in prior-Variante:", len(only_prior))
print("Nur in first-Variante :", len(only_first))
print("Bei beiden, aber andere OP:", changed)

# Sets bilden (du hast 'cmp' bereits)
only_prior = cmp[cmp['_merge']=='right_only'].copy()
both       = cmp[cmp['_merge']=='both'].copy()

# Differenzen anschauen
only_prior[['PMID','OP_prior','AKI_Start','days_prior']].to_csv("only_prior_19.csv", index=False, sep=";")

both['delta_days'] = (both['AKI_Start'] - both['OP_prior']).dt.days - (both['AKI_Start'] - both['OP_first']).dt.days
both[['PMID','OP_first','OP_prior','AKI_Start','days_first','days_prior','delta_days']].to_csv("both_changed_11.csv", index=False, sep=";")

print("only_prior – Verteilung days_prior:")
print(only_prior['days_prior'].describe())
print("\nboth_changed – Differenz (prior näher negativ):")
print(both['delta_days'].describe())
prio_counts = df_prior['days_prior'].value_counts().sort_index()
print(prio_counts.to_string())
# ggf. als Prozent: (prio_counts / df_prior.shape[0] * 100).round(1)
# angenommen: df_prior hat Spalte 'Sex' aus Patient Master
import pandas as pd

# 1) Patientendaten laden (falls noch nicht vorhanden)
df_pat = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv",
    sep=";"
)
df_pat.columns = df_pat.columns.str.strip()

# 2) Geschlecht normalisieren
def norm_sex(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip().lower()
    if s in {"m", "male", "männlich"}: return "m"
    if s in {"f", "w", "female", "weiblich"}: return "f"
    return pd.NA

df_pat["Sex_norm"] = df_pat["Sex"].apply(norm_sex)

# 3) Keys harmonisieren
df_pat["PMID"]   = df_pat["PMID"].astype(str).str.strip()
df_prior["PMID"] = df_prior["PMID"].astype(str).str.strip()

# 4) Geschlecht in df_prior mergen
df_prior = df_prior.merge(df_pat[["PMID","Sex_norm"]], on="PMID", how="left")

# 5) Optional: unbekannt markieren statt NA
df_prior["Sex_norm"] = df_prior["Sex_norm"].fillna("unbekannt")

# 6) Kreuztabelle (Anzahl)
ct_counts = pd.crosstab(df_prior["days_prior"], df_prior["Sex_norm"], dropna=False)
print(ct_counts)

# 7) Kreuztabelle als Prozent je Geschlecht
ct_pct = (ct_counts / ct_counts.sum(axis=0) * 100).round(1)

#%%

# 04_aki_latenz_ehrapy.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
import ehrapy as ep
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE) / "Diagramme"
OUT.mkdir(exist_ok=True)

# ---------------------------
# 1) Rohdaten einlesen
# ---------------------------
df_op   = pd.read_csv(f"{BASE}/HLM Operationen.csv",      sep=";", parse_dates=["Start of surgery"])
df_aki  = pd.read_csv(f"{BASE}/AKI Label.csv",            sep=";", parse_dates=["Start"])
df_pat  = pd.read_csv(f"{BASE}/Patient Master Data.csv",  sep=";")
df_supp = pd.read_csv(f"{BASE}/Procedure Supplement.csv", sep=";", parse_dates=["Timestamp"])

for df in (df_op, df_aki, df_pat, df_supp):
    df.columns = df.columns.str.strip()

# Keys und Typen harmonisieren
for df in (df_op, df_aki, df_pat):
    df["PMID"] = df["PMID"].astype(str).str.strip()


# Geschlecht normieren
def norm_sex(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip().lower()
    if s in {"m", "male", "männlich"}: return "m"
    if s in {"f", "w", "female", "weiblich"}: return "f"
    return pd.NA

df_pat["Sex"] = df_pat["Sex"].apply(norm_sex)
df_pat = df_pat.drop_duplicates(subset="PMID", keep="first")

#
from anndata import AnnData

# ehrapy_df mit Sex_norm statt Sex
ehrapy_df = (
    df_prior[["PMID", "AKI_Start", "OP", "days_prior"]]
      .merge(df_first[["PMID", "days_first"]], on="PMID", how="left")
      .merge(df_pat[["PMID", "Sex_norm"]],      on="PMID", how="left")
      .rename(columns={"OP": "OP_prior", "days_prior": "days_to_AKI"})
)

# AnnData erstellen
adata = AnnData(ehrapy_df.set_index("PMID"))

# Kategorie setzen (auf Sex_norm)
if "Sex_norm" in adata.obs.columns:
    adata.obs["Sex_norm"] = adata.obs["Sex_norm"].astype("category")
else:
    print("Warnung: 'Sex_norm' fehlt in adata.obs")

print("adata.obs.columns:", adata.obs.columns.tolist())

# ---------------------------
# 2) Latenz berechnen (2 Varianten)
# ---------------------------

# A) Erste OP pro Patient
first_op = (df_op[["PMID","Start of surgery"]]
            .dropna()
            .sort_values("Start of surgery")
            .groupby("PMID", as_index=False)
            .first()
            .rename(columns={"Start of surgery":"OP_first"}))

aki_first = (df_aki[["PMID","Start"]]
             .dropna()
             .sort_values("Start")
             .groupby("PMID", as_index=False)
             .first()
             .rename(columns={"Start":"AKI_Start"}))

df_first = first_op.merge(aki_first, on="PMID", how="inner")
df_first["days_first"] = (df_first["AKI_Start"] - df_first["OP_first"]).dt.days
df_first = df_first[(df_first["days_first"] >= 0) & (df_first["days_first"] <= 7)].copy()

# B) Letzte OP vor AKI (prior) – robust pro Patient
ops  = (df_op[["PMID","Start of surgery"]].dropna()
        .rename(columns={"Start of surgery":"OP"}))
akis = (df_aki[["PMID","Start"]].dropna()
        .rename(columns={"Start":"AKI_Start"}))

parts = []
for pmid, g_aki in akis.groupby("PMID", sort=False):
    g_ops = ops.loc[ops["PMID"] == pmid]
    if g_ops.empty:
        continue
    g_aki = g_aki.sort_values("AKI_Start")
    g_ops = g_ops.sort_values("OP")
    m = pd.merge_asof(g_aki, g_ops, left_on="AKI_Start", right_on="OP",
                      direction="backward", allow_exact_matches=True)
    m["PMID"] = pmid
    parts.append(m)

df_prior = pd.concat(parts, ignore_index=True)
df_prior = df_prior.dropna(subset=["OP"]).copy()
df_prior = (df_prior.sort_values(["PMID","AKI_Start"])
                    .groupby("PMID", as_index=False)
                    .first())
df_prior["days_prior"] = (df_prior["AKI_Start"] - df_prior["OP"]).dt.days
df_prior = df_prior[(df_prior["days_prior"] >= 0) & (df_prior["days_prior"] <= 7)].copy()

# ---------------------------
# 3) ehrapy/AnnData-Objekt bauen
#    (Primäranalyse = prior; Sensitivität = first als Zusatzspalte)
# ---------------------------
ehrapy_df = (df_prior[["PMID","AKI_Start","OP","days_prior"]]
             .merge(df_first[["PMID","days_first"]], on="PMID", how="left")
             .merge(df_pat[["PMID","Sex"]], on="PMID", how="left"))

ehrapy_df = ehrapy_df.rename(columns={
    "OP": "OP_prior",
    "days_prior": "days_to_AKI",   # Primärvariable
})

# IDs als Index für AnnData
adata = AnnData(ehrapy_df.set_index("PMID"))
adata.obs["Sex"] = adata.obs["Sex"].astype("category")

# Metadaten in .uns (nützlich für Reproduzierbarkeit)
adata.uns["analysis"] = {
    "window_days": [0,7],
    "method_primary": "prior (last surgery before AKI via merge_asof)",
    "method_sensitivity": "first (first surgery per patient)",
    "n_primary": int((~adata.obs["days_to_AKI"].isna()).sum()),
}

# ---------------------------
# 4) Deskriptive Statistik & Plot aus adata.obs
# ---------------------------
desc = adata.obs["days_to_AKI"].describe()
print(desc)

plt.figure(figsize=(6,4))
plt.hist(adata.obs["days_to_AKI"].dropna(), bins=range(0,8))
plt.xlim(0,7)
plt.xlabel("Tage bis AKI (prior)")
plt.ylabel("Anzahl der Patienten")
plt.title("AKI-Latenz (0–7 Tage) – prior-Variante")
plt.tight_layout()
plt.savefig(OUT / "AKI_Latenz_prior_0-7.png", dpi=300)
plt.show()

# ---------------------------
# 5) Speichern: CSV + .h5ad (ehrapy/AnnData)
# ---------------------------
ehrapy_df.to_csv(f"{BASE}/ehrapy_input_AKI_latenz_prior.csv", index=False, sep=";")
adata.write_h5ad(f"{BASE}/aki_latenz_prior.h5ad")

# Optional: Einlesen über ehrapy (zeigt, dass es „in ehrapy“ läuft)
# adata2 = ep.io.read_csv(f"{BASE}/ehrapy_input_AKI_latenz_prior.csv")
# adata2.write_h5ad(f"{BASE}/aki_latenz_prior_from_csv.h5ad")

# %%
