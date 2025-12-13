import pandas as pd
import numpy as np
from pathlib import Path
from paths import cs_transfer_path, ORIGINAL_DATA_DIR

# Schalter & Helper ganz oben (nach Imports)
CHECKS = True


def check(name: str, cond: bool, msg_if_fail: str):
    if CHECKS:
        if cond:
            print(f"CHECK OK – {name}")
        else:
            raise AssertionError(f"CHECK FAIL – {name}: {msg_if_fail}")


# ================== Pfade ==================
BASE = cs_transfer_path()
ORIG_DIR = ORIGINAL_DATA_DIR
OPS_CSV = ORIG_DIR / "HLM Operationen.csv"
PAT_CSV = ORIG_DIR / "Patient Master Data.csv"
AKI_CSV = ORIG_DIR / "AKI Label.csv"
OUT_DIR = BASE / "Daten"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = BASE / "Diagramme"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# ================== Helfer ==================
def read_csv_auto(path: Path) -> pd.DataFrame:
    """Robustes CSV-Lesen: erkennt ;/,, entfernt BOM."""
    for kwargs in [
        dict(sep=None, engine="python", encoding="utf-8-sig"),
        dict(sep=";", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8-sig"),
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )
    return df


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def map_decision_to_flag(x: pd.Series) -> pd.Series:
    """Ganz grob: 'AKI 1/2/3' → 1, 'keine/nein/0' → 0, sonst NA."""

    def _one(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).lower().strip()
        neg = any(k in s for k in ["keine", "nein", "no", "0"])
        pos = any(
            k in s for k in ["aki 1", "aki1", "aki 2", "aki2", "aki 3", "aki3", " aki"]
        )
        if pos and not neg:
            return 1
        if neg and not pos:
            return 0
        return pd.NA

    return x.map(_one)


# ================== Laden ==================
ops = read_csv_auto(OPS_CSV)
pat = read_csv_auto(PAT_CSV)
aki = read_csv_auto(AKI_CSV)

print("OPS-Spalten:", list(ops.columns))
print("PAT-Spalten:", list(pat.columns))
print("AKI-Spalten:", list(aki.columns))

# Umbenennungen (vereinheitlichen)
ops = ops.rename(
    columns={"Start of surgery": "Surgery_Start", "End of surgery": "Surgery_End"}
)
aki = aki.rename(
    columns={
        "Duartion": "Duration",
        "Start": "AKI_Start",
        "End": "AKI_End",
        "Decision": "AKI_Decision",
    }
)

# Pflichtspalten prüfen
for need in ["PMID", "Surgery_Start", "Surgery_End"]:
    if need not in ops.columns:
        raise ValueError(
            f"HLM Operationen.csv fehlt Spalte '{need}'. Vorhanden: {list(ops.columns)}"
        )
for need in ["PMID", "DateOfBirth"]:
    if need not in pat.columns:
        raise ValueError(
            f"Patient Master Data.csv fehlt Spalte '{need}'. Vorhanden: {list(pat.columns)}"
        )
for need in ["PMID", "AKI_Start"]:
    if need not in aki.columns:
        # wir können mit Decision-Fallback leben, aber warnen:
        print(f"Warnung: AKI-CSV ohne '{need}'. Vorhanden: {list(aki.columns)}")

# ================== Typen & Merges ==================
ops["PMID"] = ops["PMID"].astype(str).str.strip()
pat["PMID"] = pat["PMID"].astype(str).str.strip()
aki["PMID"] = aki["PMID"].astype(str).str.strip()

# Datumsfelder
ops["Surgery_Start"] = to_dt(ops["Surgery_Start"])
ops["Surgery_End"] = to_dt(ops["Surgery_End"])
pat["DateOfBirth"] = to_dt(pat["DateOfBirth"])
aki["AKI_Start"] = to_dt(aki.get("AKI_Start"))
aki["AKI_End"] = to_dt(aki.get("AKI_End"))

# --- Deduplizierung der OP-Zeilen (wichtig!) ---
if "Procedure_ID" in ops.columns:
    dedup_keys = ["PMID", "Procedure_ID", "Surgery_Start", "Surgery_End"]
else:
    dedup_keys = ["PMID", "Surgery_Start", "Surgery_End"]

before = len(ops)
ops = ops.drop_duplicates(subset=dedup_keys).copy()
print(f"OP-Zeilen vor/nach Deduplizierung: {before}/{len(ops)}")
# --- OP-Index je Patient + Patientenzahlen ---
ops = ops.sort_values(["PMID", "Surgery_Start", "Surgery_End"]).reset_index(drop=True)
ops["op_idx"] = ops.groupby("PMID").cumcount() + 1
# === MINI-CHECKS A: OP-Ebene ===
check(
    "Patientenzahl = Anzahl erster OPs",
    ops["PMID"].nunique() == int((ops["op_idx"] == 1).sum()),
    "Erste OPs ≠ Patientenzahl → Duplikate oder op_idx falsch.",
)

check(
    "Anzahl Patient:innen mit ≥2 OPs korrekt",
    int((ops.groupby("PMID")["op_idx"].max() >= 2).sum())
    == int((ops["op_idx"] == 2).sum()),
    "Zweite OPs ≠ Zahl Patient:innen mit ≥2 OPs.",
)


total_patients = int(ops["PMID"].nunique())
n_ops_per_patient = ops.groupby("PMID")["op_idx"].max()
n_ge2 = int((n_ops_per_patient >= 2).sum())
pct_ge2 = 100 * n_ge2 / total_patients

# Merge DOB in OPs
ops = ops.merge(pat[["PMID", "DateOfBirth"]], on="PMID", how="left")

# OP-Index je Patient
ops = ops.sort_values(["PMID", "Surgery_Start", "Surgery_End"]).reset_index(drop=True)
ops["op_idx"] = ops.groupby("PMID").cumcount() + 1

# Patientenzahlen korrekt aus der OP-Tabelle
total_patients = int(ops["PMID"].nunique())
n_ops_per_patient = ops.groupby("PMID")["op_idx"].max()
n_ge2 = int((n_ops_per_patient >= 2).sum())
pct_ge2 = 100 * n_ge2 / total_patients


# Alter bei OP (Jahre)
ops["Age_years_at_op"] = (
    ops["Surgery_Start"] - ops["DateOfBirth"]
).dt.total_seconds() / (365.25 * 24 * 3600)

# Patienten-Flag aus Decision (nur Info)
aki["AKI_patient_flag"] = (
    map_decision_to_flag(aki.get("AKI_Decision")) if "AKI_Decision" in aki else pd.NA
)

# ================== Robustes merge_asof (TZ-naiv + int64 Keys, gruppenweise) ==================
# 1) Zeitspalten vereinheitlichen (UTC -> tz-naiv)
ops["Surgery_End"] = pd.to_datetime(
    ops["Surgery_End"], errors="coerce", utc=True
).dt.tz_localize(None)
aki["AKI_Start"] = pd.to_datetime(
    aki["AKI_Start"], errors="coerce", utc=True
).dt.tz_localize(None)

# 2) Nur Zeilen mit Schlüsselwerten
ops_m = ops.dropna(subset=["PMID", "Surgery_End"]).copy()
aki_m = aki.dropna(subset=["PMID", "AKI_Start"]).copy()

# 3) Merge-Keys als int64 (ns)
ops_m["key_ns"] = ops_m["Surgery_End"].astype("int64")
aki_m["key_ns"] = aki_m["AKI_Start"].astype("int64")

# 4) Toleranz in ns
tol_ns = pd.Timedelta(days=7).value


def merge_asof_by_pid(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, lgrp in left.groupby("PMID", sort=False):
        # rechte PMID entfernen, damit im Ergebnis NUR die linke 'PMID' bleibt
        rgrp = right[right["PMID"] == pid].drop(columns=["PMID"], errors="ignore")
        if rgrp.empty:
            tmp = lgrp.copy()
            tmp["AKI_Start"] = pd.NaT
            out.append(tmp)
            continue
        l = lgrp.sort_values("key_ns", kind="mergesort")
        r = rgrp.sort_values("key_ns", kind="mergesort")
        m = pd.merge_asof(
            l,
            r,
            on="key_ns",
            direction="forward",
            tolerance=tol_ns,
            suffixes=("", "_aki"),
        )
        out.append(m)
    return pd.concat(out, ignore_index=True)


linked = merge_asof_by_pid(ops_m, aki_m)
linked["AKI_linked_0_7"] = linked["AKI_Start"].notna().astype("int8")
# --- PMID vereinheitlichen (nur eine Spalte behalten) ---
if "PMID" not in linked.columns or linked["PMID"].isna().all():
    for c in ["PMID_x", "PMID_y"]:
        if c in linked and linked[c].notna().any():
            linked["PMID"] = linked[c]
            break
linked["PMID"] = linked["PMID"].astype(str).str.strip()
# === MINI-CHECKS B: Merge/Flags ===
check(
    "AKI_linked_0_7 ist binär",
    set(linked["AKI_linked_0_7"].unique()) <= {0, 1},
    "AKI_linked_0_7 enthält andere Werte als 0/1.",
)

check(
    "PMID in linked befüllt",
    "PMID" in linked.columns and linked["PMID"].notna().any(),
    "Keine gültige PMID-Spalte in linked – nach Merge PMID_x/_y prüfen.",
)


# --- Patient:innen mit AKI (0–7 Tage) bei ≥1 OP ---
patients_with_aki = int(linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique())
any_aki_rate = 100 * patients_with_aki / total_patients

# --- PMID vereinheitlichen (nur eine Spalte behalten) ---
if "PMID" not in linked.columns or linked["PMID"].isna().all():
    for c in ["PMID_x", "PMID_y"]:
        if c in linked and linked[c].notna().any():
            linked["PMID"] = linked[c]
            break
linked["PMID"] = linked["PMID"].astype(str).str.strip()

# --- Patient:innen mit AKI (0–7 Tage) bei ≥1 OP ---
patients_with_aki = int(linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique())
any_aki_rate = 100 * patients_with_aki / total_patients
print(
    f"\nPatient:innen mit AKI (0–7 Tage, bei ≥1 OP): {patients_with_aki} ({any_aki_rate:.1f}%)"
)


# Nach dem Merge sicherstellen, dass 'PMID' existiert und befüllt ist
if "PMID" not in linked or linked["PMID"].isna().all():
    if "PMID_x" in linked:
        linked["PMID"] = linked["PMID_x"]
    elif "PMID_y" in linked:
        linked["PMID"] = linked["PMID_y"]

linked["PMID"] = linked["PMID"].astype(str).str.strip()

# PMID vereinheitlichen
if "PMID_x" in linked.columns and "PMID" not in linked.columns:
    linked.rename(columns={"PMID_x": "PMID"}, inplace=True)
if "PMID_y" in linked.columns:
    if "PMID" in linked.columns:
        linked.drop(columns=["PMID_y"], inplace=True)
    else:
        linked.rename(columns={"PMID_y": "PMID"}, inplace=True)

linked["PMID"] = linked["PMID"].astype(str).str.strip()


# AKI-Flag sicher setzen (int 0/1)
linked["AKI_linked_0_7"] = linked["AKI_Start"].notna().astype("int8")
total_patients = int(ops["PMID"].nunique())

# patients_with_aki = int(linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique())
# any_aki_rate = 100 * patients_with_aki / total_patients
# print(f"\nPatient:innen mit AKI (0–7 Tage, bei ≥1 OP): {patients_with_aki} ({any_aki_rate:.1f}%)")


# ================== Q1–Q3 Auswertungen ==================
# Q1: Kinder mit >=2 OPs
# KORREKT: Patientenzahlen aus der OP-Tabelle (ops), nicht aus linked
n_ops_per_patient = ops.groupby("PMID")["op_idx"].max().rename("n_ops").reset_index()

# Sanity-Checks (optional, aber hilfreich)
assert total_patients == int(
    (ops["op_idx"] == 1).sum()
), "Erste OPs ≠ Patientenzahl – es gibt noch Duplikate."
assert n_ge2 == int(
    (ops["op_idx"] == 2).sum()
), "Zweite OPs ≠ Anzahl Patient:innen mit ≥2 OPs."
# Kinder mit AKI
patients_with_aki = linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique()
aki_rate_patients = 100 * patients_with_aki / total_patients


# Q2: Alter nach OP-Index
age_summary = (
    linked.dropna(subset=["Age_years_at_op"])
    .groupby("op_idx")["Age_years_at_op"]
    .agg(
        n="count",
        median_years=lambda s: float(np.median(s)),
        q1_years=lambda s: float(np.percentile(s, 25)),
        q3_years=lambda s: float(np.percentile(s, 75)),
        min_years=lambda s: float(np.min(s)),
        max_years=lambda s: float(np.max(s)),
    )
    .reset_index()
)

# Q3: AKI bei welcher OP?
aki_by_opidx = (
    linked[linked["AKI_linked_0_7"] == 1]
    .groupby("op_idx")
    .size()
    .rename("AKI_cases")
    .reset_index()
    .sort_values("op_idx")
)
ops_by_idx = linked.groupby("op_idx").size().rename("n_ops_at_idx").reset_index()
aki_rate_by_idx = ops_by_idx.merge(aki_by_opidx, on="op_idx", how="left")
aki_rate_by_idx["AKI_cases"] = aki_rate_by_idx["AKI_cases"].fillna(0).astype(int)
aki_rate_by_idx["AKI_rate_%"] = (
    100 * aki_rate_by_idx["AKI_cases"] / aki_rate_by_idx["n_ops_at_idx"]
)

# ================== Ausgabe ==================
# ================== FINAL-REPORT (einmalige Ausgabe + CSVs) ==================
# Patient:innen mit AKI (0–7) – robust (PMID sicher vorhanden)
if "PMID" not in linked or linked["PMID"].isna().all():
    for c in ["PMID_x", "PMID_y"]:
        if c in linked and linked[c].notna().any():
            linked["PMID"] = linked[c]
            break
linked["PMID"] = linked["PMID"].astype(str).str.strip()

patients_with_aki = int(linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique())
any_aki_rate = 100 * patients_with_aki / int(ops["PMID"].nunique())

# Patienten-Summary
first_aki_opidx = (
    linked.loc[linked["AKI_linked_0_7"] == 1, ["PMID", "op_idx"]]
    .sort_values(["PMID", "op_idx"])
    .drop_duplicates(subset=["PMID"])
    .set_index("PMID")["op_idx"]
)

patient_summary = (
    ops.groupby("PMID")["op_idx"]
    .max()
    .to_frame("n_ops")
    .join(first_aki_opidx.rename("first_aki_op_idx"))
    .assign(any_aki=lambda d: d["first_aki_op_idx"].notna().astype(int))
    .reset_index()
)

only_later_aki = int(((patient_summary["first_aki_op_idx"] > 1).fillna(False)).sum())
aki_counts_per_patient = (
    linked.loc[linked["AKI_linked_0_7"] == 1].groupby("PMID").size()
)
patients_aki_multi_ops = int((aki_counts_per_patient >= 2).sum())
# === MINI-CHECKS C: Konsistenz Tabellen ===
# Summe AKI-Events (zeilenbasiert) = Summe aus der OP-Index-Tabelle
check(
    "Summe AKI-Events passt zur OP-Index-Tabelle",
    int(linked["AKI_linked_0_7"].sum()) == int(aki_rate_by_idx["AKI_cases"].sum()),
    "linked-Summe ≠ Summe AKI_cases je OP-Index.",
)

# Patientenbasiert: any_aki aus patient_summary = Unique PMIDs mit AKI in linked
patients_with_aki_linked = int(
    linked.loc[linked["AKI_linked_0_7"] == 1, "PMID"].nunique()
)
check(
    "Patient:innen mit AKI konsistent",
    int(patient_summary["any_aki"].sum()) == patients_with_aki_linked,
    "patient_summary.any_aki ≠ unique PMIDs mit AKI in linked.",
)

print("\n===== KOHORTEN-REPORT =====")
print(f"Patient:innen gesamt: {int(ops['PMID'].nunique())}")
print(
    f"Patient:innen mit ≥2 OPs: {int((ops.groupby('PMID')['op_idx'].max() >= 2).sum())} ({100*int((ops.groupby('PMID')['op_idx'].max() >= 2).sum())/int(ops['PMID'].nunique()):.1f}%)"
)
print(
    f"Patient:innen mit AKI (0–7) bei ≥1 OP: {patients_with_aki} ({any_aki_rate:.1f}%)"
)
print(
    f"  davon erstes AKI bei 1. OP: {int((patient_summary['first_aki_op_idx'] == 1).sum())}"
)
print(f"  davon erstes AKI bei OP >1: {only_later_aki}")
print(
    f"Patient:innen mit AKI nach ≥2 OPs (mehrere AKI-Events): {patients_aki_multi_ops}"
)

print("\nAKI (0–7) nach OP-Index (zeilenbasiert):")
print(aki_rate_by_idx.to_string(index=False))

print("\nAlter nach OP-Index (Jahre):")
print(age_summary.to_string(index=False))

# CSVs speichern
OUT_DIR.mkdir(parents=True, exist_ok=True)
patient_summary.to_csv(OUT_DIR / "patient_summary_ops_aki.csv", index=False)
ops.groupby("PMID")["op_idx"].max().rename("n_ops").reset_index().to_csv(
    OUT_DIR / "ops_per_patient.csv", index=False
)
age_summary.to_csv(OUT_DIR / "age_by_op_index.csv", index=False)
aki_rate_by_idx.to_csv(OUT_DIR / "aki_by_op_index_0_7.csv", index=False)
linked[
    [
        "PMID",
        "SMID",
        "Procedure_ID",
        "Surgery_Start",
        "Surgery_End",
        "op_idx",
        "Age_years_at_op",
        "AKI_Start",
        "AKI_linked_0_7",
    ]
].to_csv(OUT_DIR / "ops_linked_with_aki_0_7.csv", index=False)


# Balkenplot: AKI-Rate je OP-Index (optional)
# ===== PLOTS (nach linked/aki_rate_by_idx, vor CSV-Exports) =====
import numpy as np
import matplotlib.pyplot as plt

# --- AKI-Rate je OP-Index (95%-Wilson), CI für n<5 ausblenden ---
# --- EIN Plot-Block mit Modus: 'dim' | 'hide' | 'show' ---
MODE = (
    "dim"  # 'dim' = n<5 ohne CI + abgeblendet, 'hide' = n<5 raus, 'show' = alles mit CI
)
N_CUT = 5  # Schwelle für "kleines n"

df = aki_rate_by_idx.copy()
if MODE == "hide":
    df = df[df["n_ops_at_idx"] >= N_CUT]

x = df["op_idx"].astype(int).to_numpy()
n = df["n_ops_at_idx"].astype(int).to_numpy()
k = df["AKI_cases"].astype(int).to_numpy()
p = k / np.where(n == 0, 1, n)

# Wilson 95%-CI um beobachtete Rate p
z = 1.96
den = 1 + z**2 / n
center = (p + z**2 / (2 * n)) / den
half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
lower = np.clip(center - half, 0, 1)
upper = np.clip(center + half, 0, 1)
heights = p * 100
yerr = np.vstack(((p - lower) * 100, (upper - p) * 100))

# bei MODE 'dim': CI für n<N_CUT ausblenden
if MODE == "dim":
    mask_ci = n >= N_CUT
    yerr[:, ~mask_ci] = 0.0

fig = plt.figure()
bars = plt.bar(x, heights, yerr=yerr, capsize=4, linewidth=1)

# bei MODE 'dim': Balken mit n<N_CUT abblenden
if MODE == "dim":
    for b, ni in zip(bars, n):
        if ni < N_CUT:
            b.set_alpha(0.35)

# Beschriftungen
for xi, hi, ki, ni in zip(x, heights, k, n):
    plt.text(xi, hi + 1.2, f"{hi:.1f}% ({ki}/{ni})", ha="center", va="bottom")

plt.xticks(x, [f"{xi}\n(n={ni})" for xi, ni in zip(x, n)])
plt.xlabel("OP-Index")
plt.ylabel("AKI-Rate (%)")
plt.title("AKI (0–7 Tage) nach OP-Index")
plt.ylim(0, max((heights + yerr[1]) * 1.05))
plt.gca().yaxis.grid(True, linestyle="--", alpha=0.3)

subtitle = {
    "dim": "Fehlerbalken: 95%-Wilson; Kategorien mit n<5 ohne CI und abgeblendet.",
    "hide": "Fehlerbalken: 95%-Wilson; Kategorien mit n<5 nicht dargestellt.",
    "show": "Fehlerbalken: 95%-Wilson; alle Kategorien mit CI.",
}[MODE]
plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(DIAG_DIR / "AKI_rate_by_opindex.png", dpi=300)
plt.close(fig)


# --- Boxplot: Alter je OP-Index mit n-Labels ---
groups, labels = [], []
for k_idx in sorted(linked["op_idx"].dropna().unique().astype(int)):
    vals = linked.loc[linked["op_idx"] == k_idx, "Age_years_at_op"].dropna().values
    if len(vals) == 0:
        continue
    groups.append(vals)
    labels.append(f"{k_idx}\n(n={len(vals)})")
fig2 = plt.figure()
plt.boxplot(groups, labels=labels, showfliers=False)  # <- keine Marker für Ausreißer
plt.xlabel("OP-Index")
plt.ylabel("Alter (Jahre)")
plt.title("Alter bei OP nach OP-Index")
plt.tight_layout()
plt.ylim(0, max(v.max() for v in groups) * 1.05)
fig2.savefig(DIAG_DIR / "age_box_by_opindex.png", dpi=300)
plt.close(fig2)
