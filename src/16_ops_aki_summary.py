import pandas as pd
import numpy as np
from pathlib import Path

# === Pfade anpassen, falls nötig ===
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
OPS_CSV = BASE / "HLM Operationen.csv"
PAT_CSV = BASE / "Patient Master Data.csv"
AKI_CSV = BASE / "AKI Label.csv"

OUT_DIR = BASE / "Daten"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Hilfsfunktionen ----------
def to_dt(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def years_between(start, end):
    # robuste Jahresdifferenz (Tage / 365.25)
    return (end - start).dt.total_seconds() / (365.25 * 24 * 3600)


def map_decision_to_flag(x: pd.Series) -> pd.Series:
    """
    Grobe Fallback-Logik:
    - Positiv, wenn Hinweise auf AKI 1/2/3 oder 'aki' vorhanden und nicht 'keine/0/nein'
    - Negativ bei 'keine', '0', 'nein', 'no'
    - NaN bleibt NaN
    """

    def _one(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip().lower()
        neg = any(k in s for k in ["keine", "nein", "no", "0"])
        pos = any(
            k in s for k in ["aki 1", "aki1", "aki 2", "aki2", "aki 3", "aki3", "aki"]
        )
        if pos and not neg:
            return 1
        if neg and not pos:
            return 0
        return pd.NA

    return x.map(_one)


# ---------- Daten laden & vorbereiten ----------
ops = pd.read_csv(OPS_CSV)
pat = pd.read_csv(PAT_CSV)
aki = pd.read_csv(AKI_CSV)

# Spaltennamen säubern/vereinheitlichen
ops.columns = ops.columns.str.strip()
pat.columns = pat.columns.str.strip()
aki.columns = aki.columns.str.strip()

# Umbenennen wie vereinbart
ops = ops.rename(
    columns={"Start of surgery": "Surgery_Start", "End of surgery": "Surgery_End"}
)
aki = aki.rename(
    columns={"Start": "AKI_Start", "End": "AKI_End", "Decision": "AKI_Decision"}
)

# Datumsfelder parsen
ops["Surgery_Start"] = to_dt(ops.get("Surgery_Start"))
ops["Surgery_End"] = to_dt(ops.get("Surgery_End"))
pat["DateOfBirth"] = to_dt(pat.get("DateOfBirth"))
aki["AKI_Start"] = to_dt(aki.get("AKI_Start"))
aki["AKI_End"] = to_dt(aki.get("AKI_End"))

# Grundchecks
for col in ["PMID", "Surgery_Start", "Surgery_End"]:
    if col not in ops:
        raise ValueError(f"Spalte '{col}' fehlt in HLM Operationen.csv")

if "PMID" not in pat or "DateOfBirth" not in pat:
    raise ValueError(
        "Spalten 'PMID' und/oder 'DateOfBirth' fehlen in Patient Master Data.csv"
    )

# ---------- Alter bei OP & OP-Index ----------
# mit DOB mergen
ops = ops.merge(pat[["PMID", "DateOfBirth"]], on="PMID", how="left")

# sortieren und OP-Index (1,2,3,...) pro Patient
ops = ops.sort_values(["PMID", "Surgery_Start", "Surgery_End"]).reset_index(drop=True)
ops["op_idx"] = ops.groupby("PMID").cumcount() + 1

# Alter in Jahren zum OP-Beginn
ops["Age_years_at_op"] = years_between(ops["DateOfBirth"], ops["Surgery_Start"])

# ---------- AKI zeitlich linken (0–7 Tage nach OP-Ende) ----------
# Fallback-Flag aus Decision (auf Patientenebene)
aki["AKI_patient_flag"] = map_decision_to_flag(aki.get("AKI_Decision"))

# Für asof-Join: je Patient sortiert
aki_sorted = aki.sort_values(["PMID", "AKI_Start"]).reset_index(drop=True)
ops_sorted = ops.sort_values(["PMID", "Surgery_End"]).reset_index(drop=True)

# asof-Merge: erste AKI_Start NACH Surgery_End (direction="forward"), Toleranz 7 Tage
linked = pd.merge_asof(
    left=ops_sorted,
    right=aki_sorted,
    by="PMID",
    left_on="Surgery_End",
    right_on="AKI_Start",
    direction="forward",
    tolerance=pd.Timedelta(days=7),
)

# AKI-Verknüpfung innerhalb 0–7 Tage
linked["AKI_linked_0_7"] = linked["AKI_Start"].notna().astype(int)

# Falls kein zeitlicher Link, optional Patienten-Flag als Zusatzinfo (nicht für 0–7 zählen!)
linked["AKI_patient_flag"] = linked["AKI_patient_flag"].astype("float").astype("Int64")

# ---------- Q1: Kinder mit >=2 OPs ----------
n_ops_per_patient = linked.groupby("PMID")["op_idx"].max().rename("n_ops").reset_index()
total_patients = n_ops_per_patient["PMID"].nunique()
n_ge2 = (n_ops_per_patient["n_ops"] >= 2).sum()
pct_ge2 = 100 * n_ge2 / total_patients if total_patients else np.nan

# ---------- Q2: Alter bei 1., 2., 3. ... OP ----------
age_summary = (
    linked.dropna(subset=["Age_years_at_op"])
    .groupby("op_idx")["Age_years_at_op"]
    .agg(
        n="count",
        median_years=lambda s: np.median(s),
        q1_years=lambda s: np.percentile(s, 25),
        q3_years=lambda s: np.percentile(s, 75),
        min_years="min",
        max_years="max",
    )
    .reset_index()
)

# ---------- Q3: AKI – bei welcher OP? ----------
aki_by_opidx = (
    linked[linked["AKI_linked_0_7"] == 1]
    .groupby("op_idx")
    .size()
    .rename("AKI_cases")
    .reset_index()
    .sort_values("op_idx")
)

# Anteil AKI pro OP-Index (bezogen auf Anzahl OPs dieses Index)
ops_by_idx = linked.groupby("op_idx").size().rename("n_ops_at_idx").reset_index()
aki_rate_by_idx = aki_by_opidx.merge(ops_by_idx, on="op_idx", how="right")
aki_rate_by_idx["AKI_rate_%"] = (
    100 * aki_rate_by_idx["AKI_cases"].fillna(0) / aki_rate_by_idx["n_ops_at_idx"]
)

# ---------- Ergebnisse ausgeben/speichern ----------
summary_lines = [
    f"Gesamtzahl Patienten: {total_patients}",
    f"Patienten mit ≥2 OPs: {n_ge2} ({pct_ge2:.1f}%)",
]
print("\n".join(summary_lines))

print("\nAlter nach OP-Index (Jahre):")
print(age_summary.to_string(index=False))

print("\nAKI (0–7 Tage) nach OP-Index:")
print(aki_rate_by_idx.fillna(0).to_string(index=False))

# CSV-Exporte
n_ops_per_patient.to_csv(OUT_DIR / "ops_per_patient.csv", index=False)
age_summary.to_csv(OUT_DIR / "age_by_op_index.csv", index=False)
aki_rate_by_idx.to_csv(OUT_DIR / "aki_by_op_index_0_7.csv", index=False)

# Optional: verlinkte Tabelle als Grundlage fürs H5AD
linked_out_cols = [
    "PMID",
    "SMID",
    "Procedure_ID",
    "Surgery_Start",
    "Surgery_End",
    "op_idx",
    "Age_years_at_op",
    "AKI_Start",
    "AKI_linked_0_7",
    "AKI_patient_flag",
]
for c in linked_out_cols:
    if c not in linked.columns:
        linked[c] = pd.NA
linked[linked_out_cols].to_csv(OUT_DIR / "ops_linked_with_aki_0_7.csv", index=False)
