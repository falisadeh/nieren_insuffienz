import pandas as pd, numpy as np
import anndata as ad
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_OPS = f"{BASE}/h5ad/ops_with_patient_features.h5ad"
CSV_HLM = f"{BASE}/Original Daten/HLM Operationen.csv"


def norm_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))  # "00009"->"9", "9.0"->"9"
    except:
        return s.lstrip("0") or "0"


# --- OP-Daten (AnnData) laden + normalisieren ---
ops = ad.read_h5ad(H5_OPS)
ops_df = ops.obs.reset_index(drop=True).copy()

# sicherstellen: keine Categoricals für IDs
for c in ["PMID", "SMID"]:
    if c in ops_df.columns:
        ops_df[c] = pd.Series(ops_df[c]).astype("string")

ops_df["PMID_norm"] = ops_df["PMID"].map(norm_id)
ops_df["SMID_norm"] = ops_df["SMID"].map(norm_id)


# --- HLM Operationen.csv robust einlesen ---
def read_semicolon(path):
    try:
        return pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")


hlm = read_semicolon(CSV_HLM)
# Spalten säubern + umbenennen (für spätere Einheitlichkeit)
hlm.columns = hlm.columns.str.strip()
hlm = hlm.rename(
    columns={
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "Tx?": "Tx",
    }
)

# IDs normalisieren
hlm["PMID_norm"] = hlm["PMID"].map(norm_id)
hlm["SMID_norm"] = hlm["SMID"].map(norm_id)

# eindeutiges SMID->PMID Mapping
map_smid_to_pmid = hlm.dropna(subset=["SMID_norm"]).drop_duplicates("SMID_norm")[
    ["SMID_norm", "PMID_norm"]
]

# --- PMID via SMID nachfüllen (ohne Categorical-Fehler) ---
ops_df = ops_df.merge(
    map_smid_to_pmid, on="SMID_norm", how="left", suffixes=("", "_from_HLM")
)

# unbedingt String-Typ vor fillna
ops_df["PMID_norm"] = ops_df["PMID_norm"].astype("string")
ops_df["PMID_norm_from_HLM"] = ops_df["PMID_norm_from_HLM"].astype("string")

ops_df["PMID_final"] = (
    ops_df["PMID_norm"].fillna(ops_df["PMID_norm_from_HLM"]).astype("string")
)

print("PMID non-null vor Füllung :", ops.obs["PMID"].notna().sum())
print("PMID non-null nach Füllung:", ops_df["PMID_final"].notna().sum())
print("Eindeutige Kinder (final) :", ops_df["PMID_final"].nunique())

# ==== ab hier ANSCHLUSS an  aktuellen Stand mit ops_df (enthält PMID_final)! ====
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
CSV_PAT = f"{BASE}/Original Daten/Patient Master Data.csv"
OUTD = Path(BASE) / "Diagramme"
OUTD.mkdir(parents=True, exist_ok=True)


def read_semicolon(path):
    try:
        return pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def map_sex(x):
    if pd.isna(x):
        return "Unbekannt"
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return "Unbekannt"


# --- Patient Master Data laden & normierte PMID erzeugen (wie zuvor) ---
pat = read_semicolon(CSV_PAT)
pat.columns = pat.columns.str.strip()
pmid_col = next(c for c in pat.columns if c.lower() == "pmid")
sex_col = next(c for c in pat.columns if c.lower() in ("sex", "geschlecht"))


def norm_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except:
        return s.lstrip("0") or "0"


pat["PMID_norm"] = pat[pmid_col].map(norm_id)
pat["Geschlecht"] = pat[sex_col].map(map_sex)
pat = pat[["PMID_norm", "Geschlecht"]].drop_duplicates("PMID_norm")

# --- Patienten-Ebene (einmal pro Kind) ---
patients = (
    ops_df.dropna(subset=["PMID_final"])
    .drop_duplicates("PMID_final")[["PMID_final"]]
    .merge(pat, left_on="PMID_final", right_on="PMID_norm", how="left")
)

n_expected = patients["PMID_final"].nunique()  # sollte 1067 sein
counts_pat = (
    patients["Geschlecht"]
    .value_counts()
    .reindex(["Männlich", "Weiblich"], fill_value=0)
    .astype(int)
)
known_pat = int(counts_pat.sum())
perc_pat = (counts_pat / known_pat * 100).round(1) if known_pat else counts_pat * 0

print(f"Eindeutige Kinder in OP-Kohorte (final): {n_expected}")
print(
    f"Mit Geschlecht gefunden:                {known_pat}  (fehlend: {n_expected-known_pat})"
)
print(pd.DataFrame({"n": counts_pat, "%": perc_pat}))

# --- Plot Patienten-Ebene ---
plt.figure(figsize=(7, 7))
bars = plt.bar(
    ["Männlich", "Weiblich"], counts_pat.values, color=["#1f77b4", "#ff69b4"]
)
for b, p in zip(bars, perc_pat.values):
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() * 1.01,
        f"{p:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
    )
plt.title(
    f"Geschlechterverteilung (Patienten mit HLM-OP)\nBekannt: {known_pat} / {n_expected}"
)
plt.ylabel("Anzahl der Kinder")
plt.tight_layout()
plt.savefig(OUTD / "geschlechterverteilung_hlm_patienten_final.png", dpi=200)

# --- OP-Ebene (alle OPs zählen) ---
ops_with_sex = ops_df.merge(pat, left_on="PMID_final", right_on="PMID_norm", how="left")
counts_ops = (
    ops_with_sex["Geschlecht"]
    .value_counts()
    .reindex(["Männlich", "Weiblich"], fill_value=0)
    .astype(int)
)
known_ops = int(counts_ops.sum())
perc_ops = (counts_ops / known_ops * 100).round(1) if known_ops else counts_ops * 0


# --- Zusammenfassung als CSV ---
summary = pd.DataFrame(
    {
        "Ebene": ["Patienten (einmal pro Kind)", "Operationen (alle OPs)"],
        "Männlich_n": [counts_pat["Männlich"], counts_ops["Männlich"]],
        "Männlich_%": [perc_pat["Männlich"], perc_ops["Männlich"]],
        "Weiblich_n": [counts_pat["Weiblich"], counts_ops["Weiblich"]],
        "Weiblich_%": [perc_pat["Weiblich"], perc_ops["Weiblich"]],
        "Bekannt_n": [known_pat, known_ops],
        "Gesamt_n": [n_expected, len(ops_df)],
        "Unbekannt_n": [n_expected - known_pat, len(ops_df) - known_ops],
    }
)
summary.to_csv(OUTD / "geschlechterverteilung_pat_vs_ops_final.csv", index=False)
print("CSV gespeichert:", OUTD / "geschlechterverteilung_pat_vs_ops_final.csv")


# Kontrolle#######################
# 1) Rohdaten laden (Semikolon + BOM-fest)
def read_semicolon(path):
    import pandas as pd

    try:
        return pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")


pat_raw = read_semicolon(CSV_PAT)  # <-- unverändert lassen
pat_raw.columns = pat_raw.columns.str.strip()


# 2) Normierte PMID in der Roh-Tabelle anlegen
def norm_id(x):
    import numpy as np

    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except:
        return s.lstrip("0") or "0"


pmid_raw_col = next(c for c in pat_raw.columns if c.lower() == "pmid")
pat_raw["PMID_norm"] = pat_raw[pmid_raw_col].map(norm_id)

# 3) Welche Spalte ist das Roh-Geschlecht?
sex_raw_col = next(
    (c for c in pat_raw.columns if c.lower() in ("sex", "geschlecht")), None
)
assert (
    sex_raw_col is not None
), "In Patient Master Data gibt es weder 'Sex' noch 'Geschlecht'."

# 4) Rohverteilung NACH Zuschnitt auf deine OP-Kohorte (einmal pro PMID)
pmids_kohorte = set(ops_df["PMID_final"].dropna().astype(str).unique())
vc_raw = pat_raw.loc[
    pat_raw["PMID_norm"].isin(pmids_kohorte), sex_raw_col
].value_counts(dropna=False)
print("Rohwerte im Stammdatensatz (Kohorte):\n", vc_raw)


# 5) Mapping zu 'Geschlecht' (Männlich/Weiblich) und finale Counts
def map_sex(x):
    import numpy as np

    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return np.nan


pat_mapped = pat_raw.loc[
    pat_raw["PMID_norm"].isin(pmids_kohorte), ["PMID_norm", sex_raw_col]
].copy()
pat_mapped["Geschlecht"] = pat_mapped[sex_raw_col].map(map_sex)

# pro Patient auf einen Wert bringen (Mode)
pat_mode = (
    pat_mapped.groupby("PMID_norm", as_index=True)["Geschlecht"].agg(
        lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else np.nan
    )
).reset_index()

final_counts = (
    pat_mode["Geschlecht"]
    .value_counts(dropna=False)
    .reindex(["Männlich", "Weiblich"], fill_value=0)
)
print("Final (Patientenebene):\n", final_counts, "\nSumme:", int(final_counts.sum()))
