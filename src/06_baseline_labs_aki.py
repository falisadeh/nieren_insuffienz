#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_baseline_labs_aki.py

Ziel
----
Baseline-Laborwerte (vor OP) und frühe Post-OP-Parameter auf prädiktiven
Wert für AKI (0–7 Tage nach OP-Ende) testen – reproduzierbar, ohne Leakage.

Was es tut
----------
1) Lädt eine OP-Ereignis-Tabelle (bevorzugt: ops_with_aki.csv; Fallback:
   analytic_ops_master_ehrapy.csv + AKI-Infos, falls vorhanden).
2) Vereinheitlicht Spaltennamen (strip/rename) und Zielvariable `AKI_linked_0_7`.
3) Definiert Feature-Sets (abhängig davon, was in den Daten existiert):
   - BASELINE_ONLY: nur echte Baselines (z. B. crea_baseline, cysc_baseline)
   - EARLY_POSTOP: frühe Post-OP-Fenster (z. B. vis_mean_0_24, *_0_24, *_0_48)
   - COMBINED: BASELINE_ONLY ∪ EARLY_POSTOP
4) Verwendet GroupKFold (Gruppen = PMID), um Patient-Leakage zu vermeiden.
5) Pipeline: Imputation (Median) + Skalierung + Logistische Regression.
6) Metriken: ROC-AUC, PR-AUC, Brier; zusätzlich Konfusionsmatrix @0.5.
7) Speichert Ergebnisse (JSON + CSV) und Plots (ROC/PR) nach
   `Diagramme/` bzw. `Daten/`.

Pfadkonstanten bitte bei Bedarf anpassen (base_dir).

Voraussetzungen
---------------
- Python ≥ 3.10
- pandas, numpy, scikit-learn, matplotlib, joblib

Ausführen
---------
$ /opt/miniconda3/envs/ehrapy_env/bin/python 06_baseline_labs_aki.py

Hinweis
-------
Dieses Skript nutzt absichtlich scikit-learn/AnnData-nahen Stil ohne
spezifische ehrapy-APIs, da einige ep.io/plotting-Funktionen in deiner
Version nicht verfügbar waren. Die Ergebnisse sind dennoch 1:1 in die
Bachelorarbeit integrierbar (Methodik/Ergebnisse/Diskussion).
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# ---------------------- Pfade & Ordner ----------------------
base_dir = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
path_ops_pref = os.path.join(base_dir, "ops_with_aki.csv")
path_ops_fallback = os.path.join(base_dir, "analytic_ops_master_ehrapy.csv")
# Ausgaben
out_dir_fig = os.path.join(base_dir, "Diagramme")
out_dir_dat = os.path.join(base_dir, "Daten")
os.makedirs(out_dir_fig, exist_ok=True)
os.makedirs(out_dir_dat, exist_ok=True)


# ---------------------- Utility -----------------------------
def strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


@dataclass
class FeatureSets:
    baseline_only: List[str]
    early_postop: List[str]

    @property
    def combined(self) -> List[str]:
        return sorted(list({*self.baseline_only, *self.early_postop}))


def find_existing(columns: List[str], df: pd.DataFrame) -> List[str]:
    return [c for c in columns if c in df.columns]


def load_ops_table() -> pd.DataFrame:
    """Lädt bevorzugt ops_with_aki.csv, sonst analytic_ops_master_ehrapy.csv.
    Vereinheitlicht Spalten und Zielvariable.
    """
    if os.path.isfile(path_ops_pref):
        df = pd.read_csv(path_ops_pref, encoding="utf-8")
        df = strip_cols(df)
        print(f"Geladen: {path_ops_pref} | rows: {len(df)}")
    elif os.path.isfile(path_ops_fallback):
        df = pd.read_csv(path_ops_fallback, encoding="utf-8")
        df = strip_cols(df)
        print(f"Geladen (Fallback): {path_ops_fallback} | rows: {len(df)}")
    else:
        raise FileNotFoundError(
            "Weder 'ops_with_aki.csv' noch 'analytic_ops_master_ehrapy.csv' gefunden."
        )

    # Spalten normalisieren
    rename_map = {
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "DateofBirth": "DateOfBirth",
        "DateofDie": "DateOfDie",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Geschlecht normalisieren
    if "Sex_norm" in df.columns and "Sex" not in df.columns:
        df["Sex"] = df["Sex_norm"]

    # Zielvariable sicherstellen
    if "AKI_linked_0_7" not in df.columns:
        # Falls nur AKI_Start vorhanden ist, baut man Dummy (0/1) auf Basis days_to_AKI
        if "days_to_AKI" in df.columns:
            df["AKI_linked_0_7"] = (df["days_to_AKI"].between(0, 7)).astype(int)
            print("Hinweis: AKI_linked_0_7 aus days_to_AKI (0–7) abgeleitet.")
        else:
            raise ValueError(
                "Spalte 'AKI_linked_0_7' fehlt und konnte nicht abgeleitet werden."
            )

    # Datentypen säubern
    if df["AKI_linked_0_7"].dtype != int:
        df["AKI_linked_0_7"] = df["AKI_linked_0_7"].astype(int)

    # Stringify IDs
    for id_col in ["PMID", "SMID", "Procedure_ID"]:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)

    return df


def define_feature_sets(df: pd.DataFrame) -> FeatureSets:
    # Kandidatenlisten – werden anschließend auf Existenz gefiltert
    candidates_baseline = [
        "crea_baseline",
        "cysc_baseline",
        # ggf. mehr Baseline-Labore ergänzen, sobald verfügbar
    ]
    candidates_early = [
        # frühe Post-OP-Fenster (nur falls vorhanden; sonst leer)
        "vis_mean_0_24",
        "vis_max_0_24",
        "vis_max_6_24",
        "vis_auc_0_24",
        "vis_auc_0_48",
        "crea_peak_0_48",
        "crea_delta_0_48",
        "crea_rate_0_48",
        "cysc_peak_0_48",
        "cysc_delta_0_48",
        "cysc_rate_0_48",
    ]

    baseline = find_existing(candidates_baseline, df)
    early = find_existing(candidates_early, df)

    print("Feature-Set BASELINE_ONLY:", baseline)
    print("Feature-Set EARLY_POSTOP:", early)

    return FeatureSets(baseline_only=baseline, early_postop=early)


def run_cv(
    df: pd.DataFrame,
    features: List[str],
    label: str = "AKI_linked_0_7",
    groups_col: str = "PMID",
    add_demo: bool = True,
) -> Dict[str, dict]:
    """Cross-Validation mit GroupKFold. Option: demografische Kovariaten (Alter, Sex) hinzufügen,
    sofern vorhanden.
    """
    X_cols = list(features)
    demo_cols = []
    if add_demo:
        if "age_years_at_op" in df.columns:
            demo_cols.append("age_years_at_op")
        elif "age_years" in df.columns:
            demo_cols.append("age_years")
        if "Sex" in df.columns:
            demo_cols.append("Sex")
        X_cols = X_cols + demo_cols

    # Spalten, die existieren
    X_cols = [c for c in X_cols if c in df.columns]
    if not X_cols:
        raise ValueError("Keine gültigen Feature-Spalten gefunden.")

    y = df[label].values.astype(int)
    groups = df[groups_col].values if groups_col in df.columns else None

    # Numerisch vs. kategorial
    num_cols = [c for c in X_cols if c != "Sex"]
    cat_cols = [c for c in X_cols if c == "Sex"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    cv = GroupKFold(
        n_splits=min(5, len(np.unique(groups)) if groups is not None else 5)
    )

    fold_metrics = []
    y_true_all, y_proba_all = [], []

    for i, (tr, te) in enumerate(cv.split(df, y, groups)):
        X_tr, X_te = df.iloc[tr][X_cols], df.iloc[te][X_cols]
        y_tr, y_te = y[tr], y[te]

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_te, proba)
        pr = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

        fold_metrics.append(
            {
                "fold": int(i + 1),
                "n_test": int(len(te)),
                "AUROC": float(roc),
                "AUPRC": float(pr),
                "brier": float(brier),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        y_true_all.append(y_te)
        y_proba_all.append(proba)

    y_true_all = np.concatenate(y_true_all)
    y_proba_all = np.concatenate(y_proba_all)

    # Gesamtmetriken (out-of-fold)
    metrics_overall = {
        "AUROC": float(roc_auc_score(y_true_all, y_proba_all)),
        "AUPRC": float(average_precision_score(y_true_all, y_proba_all)),
        "brier": float(brier_score_loss(y_true_all, y_proba_all)),
    }

    # Plots speichern
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    prec, rec, _ = precision_recall_curve(y_true_all, y_proba_all)

    tag = ("+demo" if add_demo else "no-demo") + f"__nfeat{len(features)}"

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={metrics_overall['AUROC']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC – {tag}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(out_dir_fig, f"ROC_baseline_labs_{tag}.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"AP={metrics_overall['AUPRC']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR – {tag}")
    plt.legend(loc="lower left")
    pr_path = os.path.join(out_dir_fig, f"PR_baseline_labs_{tag}.png")
    plt.savefig(pr_path, bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "overall": metrics_overall,
        "folds": fold_metrics,
        "plots": {"roc": roc_path, "pr": pr_path},
        "features_used": X_cols,
        "label": label,
        "groups": groups_col,
    }


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fit_full_and_export_coefs(
    df: pd.DataFrame, features: List[str], add_demo: bool, out_csv: str
):
    X_cols = list(features)
    if add_demo:
        if "age_years_at_op" in df.columns:
            X_cols.append("age_years_at_op")
        elif "age_years" in df.columns:
            X_cols.append("age_years")
        if "Sex" in df.columns:
            X_cols.append("Sex")
    X_cols = [c for c in X_cols if c in df.columns]

    num_cols = [c for c in X_cols if c != "Sex"]
    cat_cols = [c for c in X_cols if c == "Sex"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    y = df["AKI_linked_0_7"].astype(int).values
    pipe.fit(df[X_cols], y)

    # Koeffizienten extrahieren (nur als grobe Orientierung – skaliert!)
    # Mapping der Feature-Namen aus dem ColumnTransformer:
    ohe = (
        pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        if cat_cols
        else None
    )

    feature_names = []
    # Numeric
    feature_names.extend(num_cols)
    # Categorical (OneHot)
    if ohe is not None:
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(ohe_names)

    coef = pipe.named_steps["clf"].coef_.ravel()
    coefs = pd.DataFrame({"feature": feature_names, "coef_scaled": coef}).sort_values(
        "coef_scaled", ascending=False
    )
    coefs.to_csv(out_csv, index=False)
    print(f"Koeffizienten gespeichert: {out_csv}")


if __name__ == "__main__":
    df = load_ops_table()
    fs = define_feature_sets(df)

    results: Dict[str, dict] = {}

    # 1) BASELINE_ONLY
    if fs.baseline_only:
        res_base = run_cv(df, fs.baseline_only, add_demo=True)
        results["BASELINE_ONLY+demo"] = res_base
        # optional ohne Demografie
        res_base_nodemo = run_cv(df, fs.baseline_only, add_demo=False)
        results["BASELINE_ONLY"] = res_base_nodemo
    else:
        print("WARNUNG: Keine Baseline-Labore in den Daten gefunden.")

    # 2) EARLY_POSTOP (falls vorhanden)
    if fs.early_postop:
        res_early = run_cv(df, fs.early_postop, add_demo=True)
        results["EARLY_POSTOP+demo"] = res_early
    else:
        print("Hinweis: Keine frühen Post-OP-Features gefunden – übersprungen.")

    # 3) COMBINED
    if fs.baseline_only or fs.early_postop:
        res_comb = run_cv(df, fs.combined, add_demo=True)
        results["COMBINED+demo"] = res_comb

    # Gesamtergebnisse schreiben
    json_path = os.path.join(out_dir_dat, "ML_baseline_labs_metrics.json")
    save_json(results, json_path)
    print(f"Metriken gespeichert: {json_path}")

    # Vollmodell-Koeffizienten exportieren (für das stärkste Set – hier COMBINED+demo, falls vorhanden)
    strongest_key = None
    best_auc = -1.0
    for k, v in results.items():
        auc = v.get("overall", {}).get("AUROC", -1)
        if auc > best_auc:
            best_auc = auc
            strongest_key = k
    if strongest_key is not None:
        features_used = results[strongest_key]["features_used"]
        coef_csv = os.path.join(
            out_dir_dat, f"coefs_{strongest_key.replace('+','_')}.csv"
        )
        fit_full_and_export_coefs(df, features_used, add_demo=False, out_csv=coef_csv)

    print("Fertig.")
