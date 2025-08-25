"""
15_causal_analysis.py
Kausale Analyse (ehrapy): OP-Dauer → AKI ≤7 Tage

Voraussetzungen:
- 'ops_with_aki.csv' aus 14_merge_aki_into_ops.py existiert.
- Optional (empfohlen): 'Patient Master Data.csv' mit Spalten wie PMID, Sex, DateOfBirth
"""

import ehrapy as ep
import pandas as pd
import numpy as np
from anndata import AnnData
from pathlib import Path

# ------------------ Pfade ------------------
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
OPS_CSV   = BASE / "ops_with_aki.csv"
PAT_CSV   = BASE / "Patient Master Data.csv"     # optional
OUTD      = BASE / "Diagramme"
OUTD.mkdir(parents=True, exist_ok=True)
DAG_PNG   = OUTD / "DAG_duration_to_AKI.png"
ANALYSIS_CSV = OUTD / "causal_dataset_op_level.csv"
RESULTS_TXT  = OUTD / "causal_results.txt"

# ------------- Hilfsfunktionen -------------
_num = lambda x: pd.to_numeric(x, errors="coerce")

def build_duration_hours(df: pd.DataFrame) -> pd.Series:
    if "duration_hours" in df:
        return _num(df["duration_hours"])
    if "duration_minutes" in df:
        return _num(df["duration_minutes"]) / 60.0
    if {"Surgery_Start", "Surgery_End"}.issubset(df.columns):
        ss = pd.to_datetime(df["Surgery_Start"], errors="coerce")
        se = pd.to_datetime(df["Surgery_End"],   errors="coerce")
        return (se - ss).dt.total_seconds() / 3600.0
    if "duration" in df:
        return _num(df["duration"])
    return pd.Series(np.nan, index=df.index)

def build_outcome_aki7(df: pd.DataFrame) -> pd.Series:
    aki_cols = [c for c in df.columns if "aki" in c.lower()]
    print("AKI-bezogene Spalten erkannt:", aki_cols)

    for cand in ["AKI_linked_0_7", "AKI_final_0_7", "AKI_time_0_7"]:
        if cand in df.columns:
            s = df[cand]
            if s.dtype == bool:
                return s.astype(int)
            if s.dtype == object:
                sl = s.astype(str).str.lower()
                mapped = sl.map({"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0})
                if mapped.notna().any():
                    return mapped.fillna(0).astype(int)
            return _num(s).fillna(0).astype(int)

    if {"AKI_Start", "Surgery_End"}.issubset(df.columns):
        se = pd.to_datetime(df["Surgery_End"], errors="coerce")
        ak = pd.to_datetime(df["AKI_Start"],    errors="coerce")
        days = (ak - se).dt.total_seconds() / 86400.0
        return days.between(0, 7).fillna(False).astype(int)

    raise ValueError("Kein Outcome für AKI ≤7d ableitbar.")

def normalize_sex(s: pd.Series) -> pd.Series:
    sl = s.astype(str).str.lower().str.strip()
    out = np.where(
        sl.isin({"f","w","female","weiblich"}), 1,
        np.where(sl.isin({"m","male","männlich"}), 0, np.nan)
    )
    return pd.Series(out, index=s.index).astype("float")

# ------------- 1) CSVs laden --------------
adata_raw = ep.io.read_csv(str(OPS_CSV))
ops = adata_raw.obs.copy()
print("CSV geladen:", OPS_CSV, "| Zeilen:", len(ops))
# === PATCH: PMID robust sicherstellen =====================================

from pathlib import Path

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    return df

def _find_pmid_col(columns) -> str | None:
    """Suche eine Spalte, die wie PMID aussieht (case-/whitespace-/underscore-robust)."""
    cols = [c for c in columns]
    # 1) exakte & einfache Varianten zuerst
    candidates_exact = {"PMID", "pmid", "Pmid"}
    for c in cols:
        if c in candidates_exact:
            return c
    # 2) casefold + pattern
    lc = {c: c.lower().replace(" ", "").replace("_", "") for c in cols}
    # Priorität: beginnt mit 'pmid'
    for c, norm in lc.items():
        if norm.startswith("pmid"):
            return c
    
    # fallback: patient id-like
    for c, norm in lc.items():
        if any(k in norm for k in ["patientid", "patid", "subjectid", "personid"]):
            return c
    return None
def read_any_csv(path: Path) -> pd.DataFrame:
    # 1) Versuch: Auto-Delimiter (python engine kann sniffen)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)  # Fallback

    # Prüfen, ob nur 1 Spalte und Semikolons im Header -> neu einlesen mit sep=';'
    if df.shape[1] == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    # BOM entfernen und Header trimmen
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df

def ensure_pmid(ops: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    ops = _normalize_headers(ops)

    # 1) Direkte PMID-Spalte vorhanden?
    pmid_col = _find_pmid_col(ops.columns)
    if pmid_col:
        if pmid_col != "PMID":
            ops = ops.rename(columns={pmid_col: "PMID"})
        ops["PMID"] = ops["PMID"].astype(str).str.strip()
        print("PMID gefunden als:", pmid_col, "→ normalisiert zu 'PMID'")
        return ops

    # 2) Zeitbasiertes Matching: Kandidatenquellen
    sources = [
        base_dir / "analytic_ops_master_ehrapy.csv",
        base_dir / "HLM Operationen.csv",
    ]

    # Helper: Zeiten parsen + Round-to-minute Keys
    def prep_times(df, prefix=""):
        df = df.copy()
        for c in ["Surgery_Start", "Surgery_End"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        # runde auf volle Minute ⇒ robust gg. Sekunden/Millis
        for c in ["Surgery_Start", "Surgery_End"]:
            col = prefix + c if prefix else c
            if c in df.columns:
                df[f"{col}_rmin"] = df[c].dt.floor("min")
        return df

    # 2a) exakter Minutenschlüssel: Start_rmin & End_rmin
    # 2b) Fallback merge_asof mit Toleranz
    for src in sources:
        if not src.exists():
            continue
        try:
            aux = pd.read_csv(src)
            aux = _normalize_headers(aux)

            # Braucht PMID + OP-Zeiten
            if "PMID" not in aux.columns:
                print(f"Hinweis: {src.name} hat keine 'PMID'-Spalte (Spalten: {list(aux.columns)[:10]} ...)")
                continue
            if not {"Surgery_Start", "Surgery_End"}.issubset(aux.columns):
                print(f"Hinweis: {src.name} hat keine OP-Zeiten (erwarte Surgery_Start & Surgery_End).")
                continue

            ops2 = prep_times(ops)
            aux2 = prep_times(aux)

            # --- 2a) try exact minute match on start & end ---
            if {"Surgery_Start_rmin","Surgery_End_rmin"}.issubset(ops2.columns) and \
               {"Surgery_Start_rmin","Surgery_End_rmin"}.issubset(aux2.columns):

                keys = ["Surgery_Start_rmin","Surgery_End_rmin"]
                aux_min = aux2[["PMID"] + keys].drop_duplicates()
                before = ops2.get("PMID", pd.Series(index=ops2.index, dtype=object)).notna().sum()

                merged = ops2.merge(aux_min, on=keys, how="left")
                after = merged["PMID"].notna().sum()
                print(f"{src.name}: Zeit-Match (exakt/min) → PMID non-null vorher: {before}, nachher: {after}")

                if after > 0:
                    merged["PMID"] = merged["PMID"].astype(str).str.strip()
                    return merged.drop(columns=[k for k in merged.columns if k.endswith("_rmin")])

            # --- 2b) fallback: merge_asof auf Start & Prüfen End-Toleranz ---
            # Sortieren für asof
            ops_asof = ops2.sort_values("Surgery_Start")
            aux_asof = aux2.sort_values("Surgery_Start")

            # asof (nearest) mit Toleranz ±10 Minuten
            tol = pd.Timedelta("10min")
            m = pd.merge_asof(
                ops_asof,
                aux_asof[["Surgery_Start","Surgery_End","PMID"]].rename(
                    columns={"Surgery_Start":"aux_Surgery_Start","Surgery_End":"aux_Surgery_End"}
                ).sort_values("aux_Surgery_Start"),
                left_on="Surgery_Start",
                right_on="aux_Surgery_Start",
                direction="nearest",
                tolerance=tol
            )

            # Endzeit-Abweichung prüfen (auch ≤10 min)
            valid = (m["Surgery_End"].notna() & m["aux_Surgery_End"].notna() &
                     ( (m["Surgery_End"] - m["aux_Surgery_End"]).abs() <= tol ))
            matched = m[valid].copy()
            got = matched["PMID"].notna().sum()
            print(f"{src.name}: Zeit-Match (asof ±10min, Start & End geprüft) → Treffer: {got}")

            if got > 0:
                matched["PMID"] = matched["PMID"].astype(str).str.strip()
                # Auf Original-Reihenfolge zurück, unnötige aux-Spalten weg
                cols_drop = [c for c in matched.columns if c.endswith("_rmin") or c.startswith("aux_")]
                matched = matched.drop(columns=cols_drop)
                # Nicht gematchte Zeilen beibehalten (PMID NaN)
                # (Wir lassen es durchlaufen – spätere Konfounder-Selektion dropt ggf.)
                return matched

        except Exception as e:
            print(f"Warnung: Zeitbasiertes Mapping mit {src.name} fehlgeschlagen: {e}")

    # Wenn alles fehlschlägt:
    print("Spalten in ops_with_aki.csv:", list(ops.columns))
    raise ValueError(
        "Keine 'PMID' auffindbar. Zeitbasiertes Mapping via analytic_ops_master_ehrapy.csv / HLM Operationen.csv "
        "hat nicht genügend gemeinsame Zeitspalten. Bitte prüfe, ob diese Dateien Surgery_Start/End enthalten."
    )

# --- Anwenden:
ops = ensure_pmid(ops, BASE)


# Optional: Patientenstamm laden (für Alter/Geschlecht)
pat = None
if PAT_CSV.exists():
    try:
        pat = pd.read_csv(PAT_CSV)
        pat.columns = pat.columns.str.strip()
        # gängige Namensvarianten
        pat = pat.rename(columns={
            "PMID":"PMID",
            "Sex":"Sex",
            "sex":"Sex",
            "Geschlecht":"Sex",
            "DateOfBirth":"DateOfBirth",
            "Geburtsdatum":"DateOfBirth",
            "BirthDate":"DateOfBirth"
        })
        keep_cols = [c for c in ["PMID","Sex","DateOfBirth"] if c in pat.columns]
        pat = pat[keep_cols].drop_duplicates(subset=["PMID"])
        print("Patientenstamm geladen:", PAT_CSV, "| Zeilen:", len(pat))
    except Exception as e:
        print("Warnung: Patient Master Data konnte nicht genutzt werden:", e)
#
# --- direkt NACH dem Einlesen von ops und (optional) pat hinzufügen ---

def _norm_pmid(s: pd.Series) -> pd.Series:
    # alles zu String, trimmen, führende Nullen nicht anfassen (falls numerisch → string)
    return s.astype(str).str.strip()

# Spalten vereinheitlichen
ops.columns = ops.columns.str.strip()
ops = ops.rename(columns={
    "Start of surgery": "Surgery_Start",
    "End of surgery": "Surgery_End"
})
if "PMID" not in ops.columns:
    raise ValueError("In ops_with_aki.csv fehlt die Spalte 'PMID'.")

# WICHTIG: PMID normalisieren
ops["PMID"] = _norm_pmid(ops["PMID"])

# Datumsfelder robust parsen (ops)
for c in ["Surgery_Start", "Surgery_End", "AKI_Start"]:
    if c in ops.columns:
        ops[c] = pd.to_datetime(ops[c], errors="coerce", utc=True)

print("OPS: Spalten", list(ops.columns))
print("OPS: non-null Surgery_Start:", ops["Surgery_Start"].notna().sum() if "Surgery_Start" in ops else 0)
print("OPS: non-null Surgery_End:", ops["Surgery_End"].notna().sum() if "Surgery_End" in ops else 0)

# Patiententabelle robust behandeln (falls vorhanden)
if pat is not None:
    # Normalisieren & umbenennen
    pat = pat.rename(columns={
        "sex": "Sex", "Geschlecht": "Sex",
        "Geburtsdatum": "DateOfBirth", "BirthDate": "DateOfBirth"
    })
    if "PMID" not in pat.columns:
        raise ValueError("In Patient Master Data fehlt 'PMID'.")
    pat["PMID"] = _norm_pmid(pat["PMID"])
    if "DateOfBirth" in pat.columns:
        pat["DateOfBirth"] = pd.to_datetime(pat["DateOfBirth"], errors="coerce", utc=True)

    # Merge (left, nur benötigte Spalten)
    keep_cols = [c for c in ["PMID","Sex","DateOfBirth"] if c in pat.columns]
    pat = pat[keep_cols].drop_duplicates(subset=["PMID"])
    ops = ops.merge(pat, on="PMID", how="left")

print("Nach Merge: ops-Zeilen", len(ops))
print("Nach Merge: non-null DateOfBirth:", ops["DateOfBirth"].notna().sum() if "DateOfBirth" in ops else 0)
print("Nach Merge: non-null Sex:", ops["Sex"].notna().sum() if "Sex" in ops else 0)

# Dauer neu/robust
ops["duration_hours"] = build_duration_hours(ops)
print("non-null duration_hours:", ops["duration_hours"].notna().sum())

# Outcome robust
ops["AKI_linked_0_7"] = build_outcome_aki7(ops)
print("Outcome (1) count:", int((ops["AKI_linked_0_7"] == 1).sum()))

# Alter in Jahren berechnen, unrealistische Werte filtern
if {"DateOfBirth","Surgery_Start"}.issubset(ops.columns):
    age_days = (ops["Surgery_Start"] - ops["DateOfBirth"]).dt.total_seconds() / (3600*24)
    # plausibel: 0 bis 21 Jahre; alles andere → NaN
    ops["age_years_at_op"] = (age_days / 365.25).where((age_days >= 0) & (age_days <= 21*365.25))
else:
    # Fallbacks
    for c in ["age_years_at_op","Age_years","AgeYears"]:
        if c in ops.columns:
            ops["age_years_at_op"] = pd.to_numeric(ops[c], errors="coerce")
            break

print("non-null age_years_at_op:", ops["age_years_at_op"].notna().sum() if "age_years_at_op" in ops else 0)

# Geschlecht → Sex_norm (0=♂, 1=♀)
if "Sex" in ops.columns:
    ops["Sex_norm"] = normalize_sex(ops["Sex"])
print("non-null Sex_norm:", ops["Sex_norm"].notna().sum() if "Sex_norm" in ops else 0)

# Re-OP definieren (chronologische OP-Nummer pro PMID)
if {"PMID","Surgery_Start"}.issubset(ops.columns):
    ops = ops.sort_values(["PMID","Surgery_Start"])
    ops["op_index"] = ops.groupby("PMID").cumcount() + 1
    ops["is_reop"] = (ops["op_index"] > 1).astype(float)
else:
    ops["is_reop"] = np.nan

print("non-null is_reop:", ops["is_reop"].notna().sum())

# --- Confounder-Auswahl anpassen: temporär 70% ---
conf_cands = [c for c in ["age_years_at_op","is_reop","Sex_norm"] if c in ops.columns]
coverage = {c: float(1 - ops[c].isna().mean()) for c in conf_cands}
good = [c for c in conf_cands if coverage[c] >= 0.70]  # TEMP: 70%

print("\nConfounder-Abdeckung (nach Fix):")
for c in conf_cands:
    print(f"  {c}: {coverage[c]*100:.1f}% non-NA", "✔" if c in good else "")

req_cols = ["duration_hours","AKI_linked_0_7"] + good
df = ops[req_cols].dropna(subset=req_cols).copy()
print("Zeilen nach Drop-NA:", len(df), "| genutzte Variablen:", req_cols)

# ------------- 2) Basis-Variablen bauen --------
ops.columns = ops.columns.str.strip()
# Surgery_* in Standard bringen
rename_ops = {
    "Start of surgery": "Surgery_Start",
    "End of surgery":   "Surgery_End"
}
ops = ops.rename(columns={k:v for k,v in rename_ops.items() if k in ops.columns})

ops["duration_hours"]  = build_duration_hours(ops)
ops["AKI_linked_0_7"]  = build_outcome_aki7(ops)

# Alter berechnen: bevorzugt aus Patiententabelle
if pat is not None and {"PMID","DateOfBirth"}.issubset(pat.columns) and "Surgery_Start" in ops.columns:
    ops = ops.merge(pat[["PMID","DateOfBirth"]], on="PMID", how="left")
    dob  = pd.to_datetime(ops["DateOfBirth"], errors="coerce")
    sst  = pd.to_datetime(ops["Surgery_Start"], errors="coerce")
    ops["age_years_at_op"] = (sst - dob).dt.total_seconds() / (3600*24*365.25)
else:
    # fallback: falls bereits vorhanden
    for c in ["age_years_at_op","Age_years","AgeYears"]:
        if c in ops.columns:
            ops["age_years_at_op"] = _num(ops[c])
            break

# Geschlecht normalisieren (vorzugsweise aus Patiententabelle)
if pat is not None and {"PMID","Sex"}.issubset(pat.columns):
    ops = ops.merge(pat[["PMID","Sex"]], on="PMID", how="left", suffixes=("","_pat"))
    src = ops["Sex_pat"] if "Sex_pat" in ops.columns else ops.get("Sex")
    if src is not None:
        ops["Sex_norm"] = normalize_sex(src)
else:
    if "Sex" in ops.columns:
        ops["Sex_norm"] = normalize_sex(ops["Sex"])

# Re-OP ableiten: pro Patient chronologisch sortieren
if {"PMID","Surgery_Start"}.issubset(ops.columns):
    ops = ops.sort_values(["PMID","Surgery_Start"])
    ops["op_index"] = ops.groupby("PMID").cumcount() + 1
    ops["is_reop"]  = (ops["op_index"] > 1).astype(float)
else:
    # Fallback, falls explizite Spalte existiert
    if "is_reop" in ops.columns:
        ops["is_reop"] = _num(ops["is_reop"]).round().clip(0,1)
    else:
        ops["is_reop"] = np.nan

# Typen säubern
for c in ["duration_hours","AKI_linked_0_7","age_years_at_op","is_reop","Sex_norm"]:
    if c in ops.columns:
        ops[c] = _num(ops[c])

# Pflichtspalten prüfen
must_have = ["duration_hours","AKI_linked_0_7"]
if ops[must_have].isna().any().any():
    print("\nWARNUNG: NAs in Pflichtspalten – betroffene Zeilen werden gedroppt.")

# ------------- 3) Confounder-Auswahl -----
conf_cands = [c for c in ["age_years_at_op","is_reop","Sex_norm"] if c in ops.columns]
coverage = {c: float(1 - ops[c].isna().mean()) for c in conf_cands}
good = [c for c in conf_cands if coverage[c] >= 0.80]  # ≥80% vorhanden

print("\nConfounder-Abdeckung:")
for c in conf_cands:
    print(f"  {c}: {coverage[c]*100:.1f}% non-NA", "✔" if c in good else "")

req_cols = must_have + good
df = ops[req_cols].dropna(subset=req_cols).copy()
n0 = len(df)
print("Zeilen nach Drop-NA (nur verwendete Spalten):", n0)
if n0 == 0:
    raise ValueError("Alle Zeilen wurden entfernt. Bitte Spalten prüfen / Konfounder reduzieren.")

# ------------- 4) AnnData + DAG ----------
ad = AnnData(X=df.to_numpy())
ad.obs_names = df.index.astype(str)
ad.var_names = req_cols

# DAG dynamisch mit networkx
import networkx as nx
G = nx.DiGraph()
for c in good:
    G.add_edge(c, "duration_hours")
    G.add_edge(c, "AKI_linked_0_7")
G.add_edge("duration_hours", "AKI_linked_0_7")

# Optional: PNG zeichnen
try:
    from graphviz import Source
    dot_lines = ["digraph G {", "  rankdir=LR;"] + \
        [f"  {u} -> {v};" for u, v in G.edges()] + ["}"]
    Source("\n".join(dot_lines)).render(
        filename=str(DAG_PNG.with_suffix("")),
        format="png",
        cleanup=True
    )
    print("DAG gespeichert:", DAG_PNG)
except Exception as e:
    print("DAG-Bild optional, Rendering übersprungen:", e)

# Sanity-Check: Knoten ⊆ Daten
df_in = ad.to_df()
missing_nodes = [n for n in G.nodes if n not in df_in.columns]
if missing_nodes:
    raise ValueError(f"Diese DAG-Knoten fehlen in den Daten: {missing_nodes}")

# ------------- 5) Schätzung ---------------
# (A) Backdoor: lineare Regression (effizient, baseline)
est_lin = ep.tl.causal_inference(
    adata=ad,
    graph=G,
    treatment="duration_hours",
    outcome="AKI_linked_0_7",
    estimation_method="backdoor.linear_regression",
    refute_methods=[],
    print_causal_estimate=True,
    print_summary=False,
    show_graph=False,
    show_refute_plots=False,
    return_as="estimate",
)

# (B) Propensity-Score-Stratifizierung (robuster bei binärem Outcome)
# (Wir diskretisieren Behandlung grob in Quantile, wie es DoWhy intern handhabt)
try:
    est_pss = ep.tl.causal_inference(
        adata=ad,
        graph=G,
        treatment="duration_hours",
        outcome="AKI_linked_0_7",
        estimation_method="backdoor.propensity_score_stratification",
        refute_methods=[],
        print_causal_estimate=True,
        print_summary=False,
        show_graph=False,
        show_refute_plots=False,
        return_as="estimate",
    )
except Exception as e:
    est_pss = None
    print("Hinweis: PSS-Schätzer nicht gelaufen:", e)

# ------------- 6) Ergebnisse & Export -----
df.to_csv(ANALYSIS_CSV, index=False)
with open(RESULTS_TXT, "w") as f:
    f.write(f"Datensatz: {ANALYSIS_CSV}\n")
    f.write(f"Variablen: {req_cols}\n\n")
    f.write("Schätzer A (linear_regression):\n")
    try:
        f.write(f"ATE pro zusätzliche Stunde: {float(est_lin.value):.6f}\n")
    except Exception:
        f.write("ATE konnte nicht gelesen werden.\n")
    if est_pss is not None:
        f.write("\nSchätzer B (propensity_score_stratification):\n")
        try:
            f.write(f"ATE pro zusätzliche Stunde: {float(est_pss.value):.6f}\n")
        except Exception:
            f.write("ATE konnte nicht gelesen werden.\n")

print("\nATE (linear, pro zusätzlicher OP-Stunde):", float(est_lin.value))
if est_pss is not None:
    print("ATE (PSS, pro zusätzlicher OP-Stunde):", float(est_pss.value))
print("Analyse-Datensatz gespeichert:", ANALYSIS_CSV)
print("Ergebnisse gespeichert:", RESULTS_TXT)

# t-test fuer VIS genge AKI:
if "vis_auc_0_24" in df.columns:
    aki_vis = df[df["AKI_linked_0_7"] == 1]["vis_auc_0_24"]
    no_aki_vis = df[df["AKI_linked_0_7"] == 0]["vis_auc_0_24"]
    if not aki_vis.empty and not no_aki_vis.empty:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(aki_vis, no_aki_vis, equal_var=False)
        print(f"T-Test VIS AUC 0-24 (AKI vs. kein AKI): t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
    else:
        print("Nicht genügend Daten für T-Test VIS AUC 0-24.")

