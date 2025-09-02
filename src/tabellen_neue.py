# -*- coding: utf-8 -*-
"""
Ziel:
- Eine einzige Master-H5AD-Tabelle auf OP-Ebene bauen (jede Zeile = 1 Operation)
- Alle zentralen Spalten zusammenführen:
  * OP-Basis (ops_with_patient_features.h5ad)
  * HLM Operationen.csv (SMID→PMID + OP-Start/Ende)
  * Patient Master Data.csv (Geschlecht usw.)
  * AKI Label.csv (AKI-Zeiten + Stage) — zeitbasiertes Linking via merge_asof
  * optional: Zusatzfeatures aus ops_with_patient_features_ehrapy_enriched.h5ad

Wichtig:
- IDs normalisieren (PMID/SMID): "00009","9","9.0" → "9"
- Keine Categoricals für IDs; alles als string (verhindert fillna/merge-Fehler)
- Vor merge_asof: streng sortieren (by=Patient-ID, on=Zeit)

Ausgabe:
- h5ad/ops_master_all.h5ad
- Daten/ops_master_all_preview.csv (ersten 50 Zeilen als Vorschau)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import datetime as dt
import warnings

warnings.filterwarnings("ignore", message=".*Transforming to str index.*")


# ===================== Pfade anpassen =====================
BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
H5AD_OPS = BASE / "h5ad" / "ops_with_patient_features.h5ad"  # 1209 OPs
H5AD_ENR = BASE / "h5ad" / "ops_with_patient_features_ehrapy_enriched.h5ad"  # optional
CSV_HLM = BASE / "Original Daten" / "HLM Operationen.csv"  # ; getrennt
CSV_PAT = BASE / "Original Daten" / "Patient Master Data.csv"  # ; getrennt
CSV_AKI = BASE / "Original Daten" / "AKI Label.csv"  # ; getrennt

OUT_H5AD = BASE / "h5ad" / "ops_master_all.h5ad"
OUT_CSV = BASE / "Daten" / "ops_master_all_preview.csv"
OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ===================== Helper-Funktionen =====================
def read_csv_smart(path: Path) -> pd.DataFrame:
    """Robustes CSV-Reading: Semikolon (';'), BOM-Fallback; falls doch komma-getrennt, erkennen."""
    try:
        df = pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # Falls dennoch alles in einer Spalte gelandet ist, probiere Standard-Komma
    if df.shape[1] == 1 and (
        "," in str(df.iloc[0, 0]) or ";" not in str(df.iloc[0, 0])
    ):
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def norm_id(x):
    """ID normalisieren: '00009'→'9', '9.0'→'9'; None bleibt NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except:
        s2 = s.lstrip("0")
        return s2 if s2 else "0"


def map_sex_to_full(x):
    """Rohwerte 'm/f/1/0/female/...' auf 'Männlich/Weiblich' mappen; sonst NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "mann", "männlich", "1"}:
        return "Männlich"
    if s in {"f", "female", "frau", "weiblich", "0"}:
        return "Weiblich"
    return np.nan


def to_dt(s):
    """Nach datetime64[ns] parsen (naiv, ohne tzinfo)."""
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


from pandas.api.types import CategoricalDtype


def sanitize_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = (
                pd.to_datetime(s, errors="coerce")
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M:%S")
                .astype(object)
            )  # <- object, nicht "string"
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("int8")
        elif isinstance(s.dtype, CategoricalDtype):
            out[c] = s.astype(object)  # <- object statt "string"
        elif pd.api.types.is_object_dtype(s):
            # schon ok (object); wenn du sicher auf str willst:
            out[c] = s.astype(object)
    return out


def pick_col(df: pd.DataFrame, *candidates):
    """Erste vorhandene Spalte aus Kandidatenliste zurückgeben (case-insensitiv)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c is not None:
            return c
    return None


# ===================== 1) OP-Basis laden =====================
print(f"Lade OP-H5AD: {H5AD_OPS}")
A = ad.read_h5ad(H5AD_OPS)
ops = A.obs.reset_index(drop=True).copy()
ops.columns = ops.columns.str.strip()

# IDs zu 'string' (keine Categoricals); Normalisierung anlegen
for c in ["PMID", "SMID", "Procedure_ID"]:
    if c in ops.columns:
        ops[c] = pd.Series(ops[c]).astype("string")
ops["PMID_norm"] = ops["PMID"].map(norm_id) if "PMID" in ops.columns else np.nan
ops["SMID_norm"] = ops["SMID"].map(norm_id) if "SMID" in ops.columns else np.nan

# ===================== 2) HLM laden & SMID→PMID mappen =====================
print(f"Lade HLM: {CSV_HLM}")
hlm = read_csv_smart(CSV_HLM)
# Spalten aufräumen/vereinheitlichen
hlm = hlm.rename(
    columns={
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "Tx?": "Tx",
    }
)
for c in ["PMID", "SMID", "Procedure_ID"]:
    if c in hlm.columns:
        hlm[c] = pd.Series(hlm[c]).astype("string")
hlm["PMID_norm"] = hlm["PMID"].map(norm_id) if "PMID" in hlm.columns else np.nan
hlm["SMID_norm"] = hlm["SMID"].map(norm_id) if "SMID" in hlm.columns else np.nan
hlm["Surgery_Start"] = to_dt(hlm.get("Surgery_Start"))
hlm["Surgery_End"] = to_dt(hlm.get("Surgery_End"))

# eindeutiges Mapping SMID→PMID
map_smid_to_pmid = hlm.dropna(subset=["SMID_norm"]).drop_duplicates("SMID_norm")[
    ["SMID_norm", "PMID_norm"]
]

# PMID auffüllen (Categorical-Falle vermeiden → alles 'string')
ops = ops.merge(
    map_smid_to_pmid, on="SMID_norm", how="left", suffixes=("", "_from_HLM")
)
ops["PMID_norm"] = ops["PMID_norm"].astype("string")
ops["PMID_norm_from_HLM"] = ops["PMID_norm_from_HLM"].astype("string")
ops["PMID_final"] = ops["PMID_norm"].fillna(ops["PMID_norm_from_HLM"]).astype("string")

print("Diagnose IDs:")
print("  PMID non-null (vor Füllung):", pd.notna(ops["PMID_norm"]).sum())
print("  PMID non-null (nach Füllung):", pd.notna(ops["PMID_final"]).sum())
print("  Eindeutige Kinder (PMID_final):", ops["PMID_final"].nunique())

# Surgeryzeiten ggf. aus HLM übernehmen
if "Surgery_End" not in ops.columns or ops["Surgery_End"].isna().all():
    ops = ops.merge(
        hlm[["SMID_norm", "Surgery_Start", "Surgery_End"]], on="SMID_norm", how="left"
    )
ops["Surgery_Start"] = to_dt(ops.get("Surgery_Start"))
ops["Surgery_End"] = to_dt(ops.get("Surgery_End"))

# Dauer berechnen, falls fehlt
if "duration_hours" not in ops.columns and {"Surgery_Start", "Surgery_End"} <= set(
    ops.columns
):
    ops["duration_hours"] = (
        ops["Surgery_End"] - ops["Surgery_Start"]
    ).dt.total_seconds() / 3600.0
if "duration_minutes" not in ops.columns and "duration_hours" in ops.columns:
    ops["duration_minutes"] = ops["duration_hours"] * 60.0

# ===================== 3) Patient Master Data mergen (Geschlecht) =====================
print(f"Lade Patienten: {CSV_PAT}")
pat = read_csv_smart(CSV_PAT)
pmid_pat = pick_col(pat, "PMID")
sex_pat = pick_col(pat, "Sex", "Geschlecht")
assert (
    pmid_pat is not None and sex_pat is not None
), "Patient Master Data: 'PMID' oder 'Sex/Geschlecht' fehlt."
pat["PMID_norm"] = pat[pmid_pat].map(norm_id)
pat["Sex_raw"] = pat[sex_pat].astype("string")
pat["Geschlecht"] = pat["Sex_raw"].map(map_sex_to_full)
pat_keep = pat[["PMID_norm", "Geschlecht", "Sex_raw"]].drop_duplicates("PMID_norm")

ops = ops.merge(
    pat_keep,
    left_on="PMID_final",
    right_on="PMID_norm",
    how="left",
    suffixes=("", "_pat"),
)
if "PMID_norm_pat" in ops.columns:
    ops.drop(columns=["PMID_norm_pat"], inplace=True)

# ===================== 4) AKI Label mergen (zeitbasiert via merge_asof) =====================
print(f"Lade AKI: {CSV_AKI}")
aki = read_csv_smart(CSV_AKI)
# Spalten flexibel erkennen
pmid_aki = pick_col(aki, "PMID") or pick_col(aki, "pmid")
start_aki = pick_col(aki, "AKI_Start", "Start")
end_aki = pick_col(aki, "AKI_End", "End")
dec_aki = pick_col(aki, "AKI_Decision", "Decision")
assert pmid_aki is not None, "AKI Label: PMID-Spalte nicht gefunden."

aki["PMID_norm"] = aki[pmid_aki].map(norm_id)
aki["AKI_Start"] = to_dt(aki[start_aki]) if start_aki else pd.NaT
aki["AKI_End"] = to_dt(aki[end_aki]) if end_aki else pd.NaT
aki["AKI_Decision_raw"] = (
    aki[dec_aki].astype("string")
    if dec_aki
    else pd.Series(pd.NA, index=aki.index, dtype="string")
)


def decision_to_stage(s):
    if pd.isna(s):
        return np.nan
    t = str(s).lower().strip()
    if t in {"3", "aki 3"}:
        return 3
    if t in {"2", "aki 2"}:
        return 2
    if t in {"1", "aki 1"}:
        return 1
    if t in {"0", "kein", "none", "no aki"}:
        return 0
    # Freitext: versuche Muster
    if "aki 3" in t:
        return 3
    if "aki 2" in t:
        return 2
    if "aki 1" in t:
        return 1
    return np.nan


aki["AKI_stage"] = aki["AKI_Decision_raw"].map(decision_to_stage)

# -------- merge_asof erfordert: gleiche ID-Spalte + sortiert nach (by, time) --------
# === AKI zeitbasiert linken: patientenweise asof (robust, kein "left keys must be sorted") ===
import numpy as np
import pandas as pd

# 0) Typen harmonisieren (IDs als string, Zeiten als naive datetime ohne tzinfo)
ops["PMID_final"] = ops["PMID_final"].astype("string")
ops["Surgery_End"] = pd.to_datetime(ops["Surgery_End"], errors="coerce").dt.tz_localize(
    None
)

aki["PMID_norm"] = aki["PMID_norm"].astype("string")
aki["AKI_Start"] = pd.to_datetime(aki["AKI_Start"], errors="coerce").dt.tz_localize(
    None
)
aki["AKI_End"] = pd.to_datetime(aki["AKI_End"], errors="coerce").dt.tz_localize(None)


# 1) Hilfsfunktion: pro Patient sortieren und asof-joinen
def asof_per_patient(ops_grp: pd.DataFrame, aki_grp: pd.DataFrame) -> pd.DataFrame:
    L = ops_grp.sort_values("Surgery_End").copy()
    R = aki_grp.sort_values("AKI_Start").copy()
    if R.empty:
        L["AKI_Start"] = pd.NaT
        L["AKI_End"] = pd.NaT
        L["AKI_stage"] = np.nan
        return L
    out = pd.merge_asof(
        left=L,
        right=R[["AKI_Start", "AKI_End", "AKI_stage"]],
        left_on="Surgery_End",
        right_on="AKI_Start",
        direction="forward",
        allow_exact_matches=True,
    )
    return out


# 2) Patientenweise anwenden (PMID_final ist unsere Patienten-ID auf OP-Seite)
pieces = []
for pid, ops_grp in ops.dropna(subset=["PMID_final"]).groupby("PMID_final", sort=False):
    aki_grp = aki[aki["PMID_norm"] == pid]
    pieces.append(asof_per_patient(ops_grp, aki_grp))

linked = pd.concat(pieces, ignore_index=True)

# 3) Metriken berechnen
linked["days_to_AKI"] = (
    linked["AKI_Start"] - linked["Surgery_End"]
).dt.total_seconds() / (3600 * 24)
linked["AKI_linked"] = (~linked["AKI_Start"].isna()).astype(int)
linked["AKI_linked_0_7"] = (
    (linked["AKI_linked"] == 1)
    & (linked["days_to_AKI"] >= 0)
    & (linked["days_to_AKI"] <= 7)
).astype(int)

# 4) Ergebnis zurück in 'ops'
ops = linked
# Optional: Prüfen, ob innerhalb einzelner Patientengruppen rückwärtslaufende Zeiten vorkommen
bad_left = ops.groupby("PMID_final")["Surgery_End"].apply(
    lambda s: (s.diff() < pd.Timedelta(0)).any()
)
print("Patienten mit unsortierten OP-Endzeiten:", bad_left[bad_left].index.tolist())


# ===================== 5) optionale Zusatzfeatures aus enriched.h5ad =====================
extra_cols = []
if H5AD_ENR.exists():
    print(f"Lade Zusatzfeatures: {H5AD_ENR}")
    A_enr = ad.read_h5ad(H5AD_ENR)
    enr = A_enr.obs.reset_index(drop=True).copy()
    for c in ["PMID", "SMID", "Procedure_ID"]:
        if c in enr.columns:
            enr[c] = pd.Series(enr[c]).astype("string")
    enr["PMID_norm"] = enr.get("PMID").map(norm_id) if "PMID" in enr.columns else np.nan
    enr["SMID_norm"] = enr.get("SMID").map(norm_id) if "SMID" in enr.columns else np.nan

    # Füge hier gewünschte Feature-Spalten hinzu, die vorhanden sind
    candidate_features = [
        "crea_baseline",
        "crea_peak_0_48",
        "crea_delta_0_48",
        "crea_rate_0_48",
        "cysc_baseline",
        "cysc_peak_0_48",
        "cysc_delta_0_48",
        "cysc_rate_0_48",
        "vis_max_0_24",
        "vis_mean_0_24",
        "vis_max_6_24",
        "vis_auc_0_24",
        "vis_auc_0_48",
        "age_years_at_first_op",
        "age_group_pediatric",
    ]
    present = [c for c in candidate_features if c in enr.columns]
    extra_cols = present
    keep = ["PMID_norm", "SMID_norm", "Procedure_ID"] + present
    before_cols = set(ops.columns)
    ops = ops.merge(
        enr[keep],
        on=["PMID_norm", "SMID_norm", "Procedure_ID"],
        how="left",
        suffixes=("", "_enr"),
    )
    added = [c for c in ops.columns if c not in before_cols]
    print("Zusatzspalten übernommen:", [c for c in added if c in extra_cols])

# ===================== 6) Sanitize & Speichern =====================
# H5AD-sichere Typen
ops_clean = sanitize_for_h5ad(ops)
# Vor dem Bauen des AnnData-Objekts:
ops_clean = sanitize_for_h5ad(ops)
ops_clean = ops_clean.reset_index(drop=True)
ops_clean.index = ops_clean.index.astype(str)  # <- gegen ImplicitModificationWarning

X = np.empty((len(ops_clean), 0))
adata = ad.AnnData(X=X, obs=ops_clean)
print("\n=== Schnell-Checks ===")
print("n_obs:", adata.n_obs, "| obs-Spalten:", adata.obs.shape[1])
print("Eindeutige Kinder:", pd.Series(adata.obs["PMID_final"]).nunique())
if "AKI_linked" in adata.obs:
    print("AKI_linked=1:", int((adata.obs["AKI_linked"].astype("Int64") == 1).sum()))
if "AKI_linked_0_7" in adata.obs:
    print(
        "AKI_linked_0_7=1:",
        int((adata.obs["AKI_linked_0_7"].astype("Int64") == 1).sum()),
    )
import numpy as np
import pandas as pd
from pandas import StringDtype

# 1) Alle Pandas-StringDtype-Spalten zurück auf klassisches object (Python-str)
for col in ops_clean.columns:
    if isinstance(ops_clean[col].dtype, StringDtype):
        ops_clean[col] = ops_clean[col].astype(object)  # kein Pandas-StringDtype mehr

# 2) (schön) Index konsistent setzen – vermeidet spätere Warnungen
ops_clean = ops_clean.reset_index(drop=True)
ops_clean.index = ops_clean.index.astype(str)
######################
# === Drop-in Patch: H5AD-kompatibel speichern ===
import numpy as np
import pandas as pd
import anndata as ad
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype

# 1) Kopie von 'ops' als Ausgangspunkt (falls du bereits 'ops_clean' hast, nimm das)
ops_out = ops.copy()

# 2) Datentypen vereinheitlichen:
#    - Datetime -> ISO-String
#    - Text/sonstige object -> Pandas StringDtype (nullable)
#    - Numerik/Bool bleibt numerisch/bool
for c in ops_out.columns:
    s = ops_out[c]
    if is_datetime64_any_dtype(s):
        ops_out[c] = (
            pd.to_datetime(s, errors="coerce")
            .dt.tz_localize(None)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    elif not is_numeric_dtype(s) and not is_bool_dtype(s):
        # alles Nicht-Numerische (außer Datetime) konsequent als Pandas-String führen
        ops_out[c] = s.astype("string")

# 3) Index bewusst setzen (vermeidet implizite Umwandlungswarnungen)
ops_out = ops_out.reset_index(drop=True)
ops_out.index = ops_out.index.astype(str)

# 4) AnnData: Schreiben von Pandas-StringSpalten erlauben
if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True

# 5) AnnData bauen & speichern
X = np.empty((len(ops_out), 0))
adata = ad.AnnData(X=X, obs=ops_out)
adata.write_h5ad(OUT_H5AD)
print("Gespeichert:", OUT_H5AD)
####################################
# 3) AnnData bauen & schreiben


# direkt vor dem AnnData-Konstruktor
ops_clean = ops_clean.reset_index(drop=True)
ops_clean.index = ops_clean.index.astype(str)

X = np.empty((len(ops_clean), 0))
adata = ad.AnnData(X=X, obs=ops_clean)

# AnnData aufbauen (X leer; alles in obs)
X = np.empty((len(ops_clean), 0))
adata = ad.AnnData(X=X, obs=ops_clean)

# Provenienz in .uns
adata.uns["provenance"] = {
    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    "base_files": {
        "ops_h5ad": str(H5AD_OPS),
        "hlm_csv": str(CSV_HLM),
        "patient_master_csv": str(CSV_PAT),
        "aki_label_csv": str(CSV_AKI),
        "enriched_h5ad": str(H5AD_ENR) if H5AD_ENR.exists() else None,
    },
    "notes": "PMIDs via SMID ergänzt; AKI via merge_asof (erstes Event ≥ OP-Ende); Datentypen für H5AD sanitizt.",
}
import anndata as ad

# Opt-in erlauben (wenn Setting existiert)
if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True

adata.write_h5ad(OUT_H5AD)

adata.write_h5ad(OUT_H5AD)
ops_clean.head(50).to_csv(OUT_CSV, index=False)

# ===================== 7) kleine Diagnosen =====================
print("\n=== FERTIG ===")
print(f"Gespeichert H5AD: {OUT_H5AD}")
print(f"Preview CSV:      {OUT_CSV}")
print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
print("Spalten in obs:", adata.obs.shape[1])

# Schnell-Checks:
print("\nSchnell-Checks:")
print("  Eindeutige Kinder:", pd.Series(adata.obs["PMID_final"]).nunique())
print(
    "  AKI_linked=1 (insg.):",
    (
        int((adata.obs["AKI_linked"].astype("Int64") == 1).sum())
        if "AKI_linked" in adata.obs.columns
        else "NA"
    ),
)
print(
    "  AKI_linked_0_7=1:",
    (
        int((adata.obs["AKI_linked_0_7"].astype("Int64") == 1).sum())
        if "AKI_linked_0_7" in adata.obs.columns
        else "NA"
    ),
)
