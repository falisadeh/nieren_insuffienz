#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_ml_pipeline_aki.py (ehrapy + sklearn)

- Lädt AnnData via ehrapy
- Wählt Zielvariable robust (bevorzugt AKI_linked_0_7); leitet sie sonst aus vorhandenen Spalten ab
- GroupKFold nach PMID: ROC-AUC, PR-AUC, Brier
- SimpleImputer (Median) + Missing-Indicator
- Permutation Importance (finaler Fit)
- Optional: SHAP Summary (falls installiert)
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import ehrapy as ep
from anndata import AnnData

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

print(">>> start 25_ml_pipeline_aki.py", flush=True)

# -------------------------
# Konfiguration
# -------------------------
H5AD_PATH = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad/ops_with_patient_features.h5ad"
)
OUT_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CANDIDATES: List[str] = ["AKI_linked_0_7", "AKI"]  # Priorität links
GROUP_COL = "PMID"

FEATURES_WISHED: List[str] = [
    "crea_baseline", "crea_delta_0_48", "crea_peak_0_48", "crea_rate_0_48",
    "cysc_baseline", "cysc_delta_0_48", "cysc_peak_0_48", "cysc_rate_0_48",
    "vis_max_0_24", "vis_mean_0_24", "vis_max_6_24",
    "vis_auc_0_24", "vis_auc_0_48",
    "duration_hours", "age_years_at_first_op", "n_ops", "Sex_norm",
]

# -------------------------
# Hilfsfunktionen
# -------------------------
def coerce_sex_to_numeric(s: pd.Series) -> pd.Series:
    """Mappt Sex_norm robust auf 0/1 (m/f), sonst NaN (wird imputiert)."""
    mapping = {"m": 0, "male": 0, "mann": 0, "0": 0, 0: 0,
               "f": 1, "w": 1, "weiblich": 1, "female": 1, "1": 1, 1: 1}
    return s.astype(str).str.strip().str.lower().map(mapping).astype("float64")

def pick_target_with_classes(df: pd.DataFrame, candidates: List[str]) -> Tuple[str, pd.Series]:
    """Wählt die erste Zielspalte aus candidates, die >1 Klasse hat. Castet robust nach 0/1-Int."""
    tried = []
    for c in candidates:
        if c in df.columns:
            y_raw = df[c]
            if y_raw.dtype == bool:
                y = y_raw.astype(int)
            else:
                y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)
            nuniq = y.nunique(dropna=False)
            tried.append((c, nuniq, int((y == 1).sum())))
            if nuniq >= 2:
                return c, y
    msg = "Keine gültige Zielspalte mit >1 Klasse gefunden."
    if tried:
        msg += " Geprüft: " + ", ".join([f"{c} (unique={u}, pos={p})" for c, u, p in tried])
    else:
        msg += " Keine Kandidaten vorhanden."
    raise ValueError(msg)

def derive_aki0_7(df: pd.DataFrame) -> pd.Series:
    """
    Leitet AKI_linked_0_7 robust her in folgender Priorität:
    1) days_to_AKI_calc (0..7)  -> 1
    2) days_to_AKI      (0..7)  -> 1
    3) AKI_linked_0_7_time: 0/1 oder 0..7 -> 1
    4) highest_AKI_stage_0_7 > 0 -> 1
    5) Fallback: 0 <= (AKI_Start - Surgery_End) <= 7 Tage
    Fehlt alles -> 0
    """
    y = pd.Series(0, index=df.index, dtype=int)

    # 1) days_to_AKI_calc
    if "days_to_AKI_calc" in df.columns:
        d = pd.to_numeric(df["days_to_AKI_calc"], errors="coerce")
        y1 = ((d >= 0) & (d <= 7)).astype(int)
        if y1.sum() > 0:
            return y1.fillna(0).astype(int)

    # 2) days_to_AKI
    if "days_to_AKI" in df.columns:
        d = pd.to_numeric(df["days_to_AKI"], errors="coerce")
        y2 = ((d >= 0) & (d <= 7)).astype(int)
        if y2.sum() > 0:
            return y2.fillna(0).astype(int)

    # 3) AKI_linked_0_7_time
    if "AKI_linked_0_7_time" in df.columns:
        t = pd.to_numeric(df["AKI_linked_0_7_time"], errors="coerce")
        uniq = set(t.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            y3 = t.fillna(0).astype(int)
            if y3.sum() > 0:
                return y3
        y3b = ((t >= 0) & (t <= 7)).astype(int)
        if y3b.sum() > 0:
            return y3b.fillna(0).astype(int)

    # 4) highest_AKI_stage_0_7 > 0
    if "highest_AKI_stage_0_7" in df.columns:
        st = pd.to_numeric(df["highest_AKI_stage_0_7"], errors="coerce")
        y4 = (st > 0).astype(int)
        if y4.sum() > 0:
            return y4.fillna(0).astype(int)

    # 5) Fallback: Zeitstempel
    end_candidates = ["Surgery_End", "surgery_end", "Surgery_End_ts", "end_of_surgery", "OP_Ende"]
    start_candidates = ["AKI_Start", "aki_start", "AKI_Start_ts"]
    end_col = next((c for c in end_candidates if c in df.columns), None)
    start_col = next((c for c in start_candidates if c in df.columns), None)
    if (end_col is not None) and (start_col is not None):
        se = pd.to_datetime(df[end_col], errors="coerce", utc=True)
        ak = pd.to_datetime(df[start_col], errors="coerce", utc=True)
        delta_days = (ak - se).dt.total_seconds() / 86400.0
        y5 = ((delta_days >= 0) & (delta_days <= 7)).astype(int).fillna(0).astype(int)
        if y5.sum() > 0:
            return y5

    return y.fillna(0).astype(int)

def get_transformed_feature_names(preprocessor: ColumnTransformer, num_cols: List[str]) -> List[str]:
    """Spaltennamen nach Transformation (inkl. Missing-Indicator)."""
    names = list(num_cols)
    num_trf: SimpleImputer = preprocessor.named_transformers_["num"]
    if hasattr(num_trf, "indicator_") and getattr(num_trf, "add_indicator", False):
        for idx in num_trf.indicator_.features_:
            names.append(f"{num_cols[idx]}__missing")
    return names

def derive_sex_norm(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Sucht nach Spalten, die 'sex', 'gender', 'geschl' o.ä. im Namen tragen,
    und mappt deren Werte robust auf 0/1 (m=0, f=1). Gibt None zurück, wenn nichts Brauchbares.
    """
    import re
    pat = re.compile(r"(sex|gender|geschl|geschlecht|genus|sexo)", re.IGNORECASE)
    cand_cols = [c for c in df.columns if pat.search(str(c))]
    if not cand_cols:
        return None

    male_vals = {
        "m", "male", "mann", "masculino", "masculin",
        "männlich", "männl", "boy", "b", "knabe", "maskulin", "herr"
    }
    female_vals = {
        "f", "female", "weiblich", "weibl", "feminin",
        "féminin", "girl", "g", "mädchen", "frau"
    }

    for col in cand_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().mean() >= 0.05:
                sn = pd.to_numeric(s, errors="coerce").where(lambda x: x.isin([0, 1])).astype("float64")
                if sn.notna().sum() > 0:
                    print(f"Sex_norm aus numerischer Spalte '{col}' abgeleitet.")
                    return sn

        st = (
            s.astype(str).str.strip().str.lower()
             .replace({"nan": None, "none": None, "": None})
        )
        st_num = pd.to_numeric(st, errors="coerce")
        if st_num.notna().sum() > 0 and set(st_num.dropna().unique()).issubset({0, 1}):
            sn = st_num.astype("float64")
            if sn.notna().sum() > 0:
                print(f"Sex_norm aus binärer String-Spalte '{col}' abgeleitet.")
                return sn

        mapped = st.map(lambda v: (0.0 if v in male_vals else (1.0 if v in female_vals else np.nan)))
        if mapped.notna().mean() >= 0.05:
            print(f"Sex_norm aus kategorialer Spalte '{col}' abgeleitet.")
            return mapped.astype("float64")

    return None

# -------------------------
# Hauptlogik
# -------------------------
def main() -> int:
    print(">>> calling main()", flush=True)

    # Existenzcheck
    if not os.path.exists(H5AD_PATH):
        print(f"Datei nicht gefunden: {H5AD_PATH}", file=sys.stderr)
        return 1

    # === Laden via ehrapy ===
    adata: AnnData = ep.io.read_h5ad(H5AD_PATH)
    df = adata.obs.copy()
    print(f"Geladen (ehrapy): {H5AD_PATH} | rows: {len(df)}")
    print("AKI-ähnliche Spalten:", [c for c in df.columns if "aki" in c.lower()])

    # Debug: was steckt in Sex_norm?
    if "Sex_norm" in df.columns:
        print(">>> Rohwerte Sex_norm (Top 20):")
        print(df["Sex_norm"].astype(str).str.strip().str.lower().value_counts(dropna=False).head(20))

    # --- Sex_norm ableiten, wenn leer/fehlend ---
    make_sex = ("Sex_norm" not in df.columns) or (df["Sex_norm"].isna().all())
    if make_sex:
        sn = derive_sex_norm(df)
        if sn is not None:
            df["Sex_norm"] = sn
            print(">>> Sex_norm wurde neu erzeugt (0=m, 1=w).")
        else:
            print(">>> Konnte Sex_norm nicht ableiten (keine passende Spalte gefunden/gefüllt).")

    # --- Zielspalte robust bestimmen/ableiten ---
    try:
        target_col, y = pick_target_with_classes(df, TARGET_CANDIDATES)
        if y.nunique() < 2:
            raise ValueError(f"Ziel '{target_col}' hat nur eine Klasse.")
    except Exception as e:
        print(f"Warnung: {e} -> leite Label aus vorhandenen Spalten ab ...")
        y = derive_aki0_7(df)
        df["AKI_linked_0_7"] = y
        target_col = "AKI_linked_0_7"

    print(f"Zielspalte: {target_col}")
    print("AKI-Verteilung:", y.value_counts().to_dict())
    if y.nunique() < 2:
        for col in ["days_to_AKI_calc", "days_to_AKI", "AKI_linked_0_7_time",
                    "highest_AKI_stage_0_7", "AKI_Start", "Surgery_End"]:
            if col in df.columns:
                print(f"DEBUG {col}: non-null={df[col].notna().sum()}, unique≈{df[col].nunique()}")
                print(df[col].dropna().astype(str).head(5))
        raise RuntimeError("Zielvariable hat weiterhin nur eine Klasse. Bitte DEBUG-Ausgaben prüfen.")

    # === Gruppen (PMID) ===
    if GROUP_COL not in df.columns:
        raise ValueError(f"Gruppenspalte '{GROUP_COL}' fehlt in den Daten.")
    groups = df[GROUP_COL]

    # === Features auswählen ===
    features = [c for c in FEATURES_WISHED if c in df.columns]
    missing = [c for c in FEATURES_WISHED if c not in df.columns]
    if missing:
        print("Warnung – folgende gewünschten Features fehlen:", missing)

    X = df[features].copy()

    # Sex_norm nach 0/1 konvertieren (falls vorhanden)
    if "Sex_norm" in X.columns:
        X["Sex_norm"] = coerce_sex_to_numeric(X["Sex_norm"])

    # Vollständig-NaN numerische Spalten droppen (z. B. Sex_norm, wenn Mapping nicht griff)
    drop_all_nan = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].isna().all()]
    if drop_all_nan:
        print("Dropping fully-NaN numeric features:", drop_all_nan)
        X = X.drop(columns=drop_all_nan)

    # Nur numerische Features behalten
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("Keine numerischen Features gefunden.")
    X_num = X[num_cols].copy()

    # === Pipeline ===
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median", add_indicator=True), num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)
    clf = Pipeline([("pre", pre), ("rf", rf)])

    # === Cross-Validation (GroupKFold) ===
    cv = GroupKFold(n_splits=5)
    roc_list, pr_list, brier_list = [], [], []
    valid_folds = 0

    for fold, (tr, te) in enumerate(cv.split(X_num, y, groups=groups), start=1):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            print(f"[Fold {fold}] übersprungen (zu wenige Klassen in Train/Test).")
            continue

        clf.fit(X_num.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X_num.iloc[te])
        prob = proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(te))

        roc = roc_auc_score(y.iloc[te], prob)
        pr = average_precision_score(y.iloc[te], prob)
        brier = brier_score_loss(y.iloc[te], prob)

        roc_list.append(roc); pr_list.append(pr); brier_list.append(brier)
        valid_folds += 1
        print(f"[Fold {fold}] ROC-AUC={roc:.3f} | PR-AUC={pr:.3f} | Brier={brier:.3f}")

    if valid_folds == 0:
        raise RuntimeError("Keine gültigen CV-Folds (Klassenproblem). Bitte Label/Groups prüfen.")

    print("\nROC-AUC Mittelwert:", round(np.mean(roc_list), 3))
    print("PR-AUC  Mittelwert:", round(np.mean(pr_list), 3))
    print("Brier   Mittelwert:", round(np.mean(brier_list), 3))

    # === Finaler Fit + Permutation Importance ===
    clf.fit(X_num, y)
    result = permutation_importance(
        clf, X_num, y, scoring="roc_auc",
        n_repeats=20, random_state=42, n_jobs=-1
    )
    # Permutation Importance bezieht sich auf INPUT-Features -> num_cols!
    names_for_perm = list(num_cols)
    perm_mean = pd.Series(result.importances_mean, index=names_for_perm).sort_values(ascending=False)
    print("\nPermutation Importance (Finaler Fit) – Top 15:")
    print(perm_mean.head(15))
    perm_mean.to_csv(os.path.join(OUT_DIR, "PermutationImportance_final.csv"))

    # === Optional: SHAP Summary ===
    try:
        import shap
        import matplotlib.pyplot as plt
        rf_est = clf.named_steps["rf"]
        Xt = clf.named_steps["pre"].transform(X_num)
        try:
            trans_names = clf.named_steps["pre"].get_feature_names_out()
        except Exception:
            trans_names = list(num_cols)  # Fallback
        shap_values = shap.TreeExplainer(rf_est).shap_values(Xt)
        shap.summary_plot(shap_values[1], Xt, feature_names=trans_names, show=False)
        out_png = os.path.join(OUT_DIR, "SHAP_summary.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"SHAP Summary gespeichert: {out_png}")
    except Exception as e:
        print(f"SHAP konnte nicht ausgeführt werden (optional): {e}")

    print("\nFertig mit ehrapy-ML-Pipeline.")
    return 0

# -------------------------
if __name__ == "__main__":
    try:
        rc = main()
        print(f">>> main() finished with rc={rc}", flush=True)
        sys.exit(rc)
    except Exception:
        import traceback
        print(">>> Uncaught exception in main():", flush=True)
        traceback.print_exc()
        sys.exit(1)

