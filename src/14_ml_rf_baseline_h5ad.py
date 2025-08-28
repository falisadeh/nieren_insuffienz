#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/14_ml_rf_baseline_h5ad.py

Baseline-only Vorhersage von AKI (0–7 Tage) aus einer h5ad-Gesamttabelle.
- Verwendet ausschließlich präoperative Baselines ("*_baseline" bzw. preop-Synonyme) + Alter + optional Geschlecht.
- Verhindert Patient-Leakage: gruppensichere Splits (Gruppen = PMID) für Train/Val/Test und GridSearch-CV.
- Modell: RandomForestClassifier (class_weight='balanced').
- Metriken: AUROC, AUPRC, Brier; Confusion @0.5/@F1/@Youden; Kalibration; Importances.
- Artefakte: Diagramme/ML_baseline_only_rf_*.png, Daten/ML_baseline_only_rf_*.json/.csv

Ausführen:
  conda activate ehrapy_ml
  /opt/miniconda3/envs/ehrapy_ml/bin/python -u src/14_ml_rf_baseline_h5ad.py
"""
from __future__ import annotations
import os, re, json, warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")  # nicht-interaktiv, sicher im Skriptlauf
import matplotlib.pyplot as plt

from anndata import read_h5ad

# Kompatible Imports für verschiedene sklearn-Versionen
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
except Exception:
    StratifiedGroupKFold = None  # type: ignore

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score,
)
from sklearn.calibration import calibration_curve

# -------------------- Pfade & Konstanten --------------------
BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5AD_CANDIDATES = [
    os.path.join(BASE, "h5ad", "ops_with_patient_features.h5ad"),
    os.path.join(BASE, "Daten", "ops_ml_processed.h5ad"),
    os.path.join(BASE, "ops_with_patient_features.h5ad"),
]
OUT_DIR_PLOTS = os.path.join(BASE, "Diagramme")
OUT_DIR_DATA = os.path.join(BASE, "Daten")
os.makedirs(OUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUT_DIR_DATA, exist_ok=True)

RANDOM_STATE = 42
# Optional: GridSearch überspringen, wenn DO_GS=0 in der Umgebung gesetzt ist
DO_GS = os.environ.get("DO_GS", "1") == "1"

# -------------------- Utilities --------------------

def find_h5ad() -> str:
    for p in H5AD_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Keine h5ad-Datei gefunden in Kandidatenpfaden: " + ", ".join(H5AD_CANDIDATES))


def pick_y(obs: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """Wählt die erste Zielspalte aus, die *beide* Klassen enthält (0/1).
    Skipt Kandidaten, die leer oder eindimensional sind (z. B. alles 0)."""
    candidates = [
        # Bevorzugt solche, die in der Praxis meist gefüllt sind
        "had_aki",
        "AKI_linked_0_7",
        "aki_linked_0_7",
        "AKI",
        "aki",
    ]
    for c in candidates:
        if c in obs.columns:
            y_raw = (
                obs[c]
                .astype(str)
                .replace({"True": "1", "False": "0", "yes": "1", "no": "0", "nan": np.nan})
            )
            y = pd.to_numeric(y_raw, errors="coerce").astype(float)
            # Nur 0/1 zulassen
            y = y.where((y.isin([0.0, 1.0])), np.nan)
            y = y.fillna(0.0).astype(int).values  # fehlend als 0 behandeln (konservativ)
            uniq = np.unique(y)
            if len(uniq) == 2:
                return c, y
            else:
                print(f"Warnung: Zielkandidat '{c}' enthält nur eine Klasse ({uniq}). Überspringe.", flush=True)
    raise KeyError("Keine geeignete Zielspalte mit beiden Klassen gefunden (hat alles 0?).").")


def sanitize_sex(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"f", "w", "female", "weiblich"}: return "f"
    if s in {"m", "male", "männlich"}: return "m"
    return s


def detect_age_col(obs: pd.DataFrame) -> str | None:
    for c in ["age_years_at_op", "age_years", "Age_years_at_op", "Age_years"]:
        if c in obs.columns:
            return c
    return None


def collect_baseline_cols(obs: pd.DataFrame) -> List[str]:
    cols = obs.columns.str.strip().tolist()

    def looks_like_baseline(name: str) -> bool:
        n = name.lower()
        positive = any(p in n for p in ("baseline", "base", "preop", "pre_op", "pre-op"))
        if not positive:
            return False
        negative = [
            "delta", "rate", "peak", "auc", "max", "min", "mean",
            "0_24", "0-24", "0_48", "0-48", "1_24", "6_24",
            "vis", "duration", "hours", "minutes",
        ]
        if any(tok in n for tok in negative):
            return False
        return True

    cand = [c for c in cols if looks_like_baseline(c)]
    # bekannte Aliasse ergänzen
    for k in ["crea_baseline", "cysc_baseline", "crea_preop", "cysc_preop", "crea_base", "cysc_base"]:
        if k in cols and k not in cand:
            cand.append(k)

    # nur numerische behalten (mind. 1 gültiger Wert)
    keep: List[str] = []
    for c in cand:
        s = pd.to_numeric(obs[c], errors="coerce")
        if s.notna().any():
            keep.append(c)

    keep = sorted(dict.fromkeys(keep))
    print(f"Gefundene Baselines: {keep if keep else '— keine —'}", flush=True)
    return keep


def thresholds_from_validation(p_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = tpr - fpr
    thr_youden = float(thr[np.argmax(j)]) if len(thr) else 0.5

    prec, rec, thr_pr = precision_recall_curve(y_val, p_val)
    f1s = []
    for t in thr_pr:
        yhat = (p_val >= t).astype(int)
        f1s.append(f1_score(y_val, yhat))
    thr_f1 = float(thr_pr[int(np.argmax(f1s))]) if len(thr_pr) else 0.5
    return thr_youden, thr_f1


def plot_cm(cm: np.ndarray, title: str, out_png: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_group_cv(y_train: np.ndarray, groups_train: np.ndarray, n_splits: int, seed: int):
    """
    Liefert eine CV, die Gruppen respektiert und (soweit möglich) in train/test
    beide Klassen enthält. Fällt auf GroupKFold zurück, wenn StratifiedGroupKFold fehlt.
    """
    # bevorzugt: stratifiziert + gruppiert (falls verfügbar)
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Fallback: GroupKFold mit Filter auf „brauchbare“ Folds
    gkf = GroupKFold(n_splits=max(2, n_splits))
    folds = []
    X_dummy = np.zeros((len(y_train), 1))
    for tr, te in gkf.split(X_dummy, y_train, groups_train):
        if len(np.unique(y_train[tr])) == 2 and len(np.unique(y_train[te])) == 2:
            folds.append((tr, te))
    if len(folds) >= 2:
        return folds
    return list(GroupKFold(n_splits=2).split(X_dummy, y_train, groups_train))

# -------------------- Hauptprogramm --------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    h5_path = find_h5ad()
    print(f"Lade: {h5_path}", flush=True)
    adata = read_h5ad(h5_path)
    obs = adata.obs.copy()
    obs.columns = obs.columns.str.strip()

    # Zielvariable
    y_name, y = pick_y(obs)
    print(f"Ziel: {y_name} | Positiv={int(np.nansum(y))} / {len(y)} (Prävalenz={float(np.nanmean(y)):.1%})", flush=True)

    # Gruppen
    if "PMID" not in obs.columns:
        raise KeyError("Spalte 'PMID' nicht in .obs gefunden – notwendig für gruppensichere Splits.")
    groups_all = obs["PMID"].astype(str).values

    # Features
    baseline_cols = collect_baseline_cols(obs)
    if not baseline_cols:
        raise RuntimeError("Keine Baseline-Spalten gefunden. Bitte prüfe, dass z. B. 'crea_baseline'/'cysc_baseline' vorhanden sind.")

    age_col = detect_age_col(obs)
    sex_col = "Sex" if "Sex" in obs.columns else None

    if sex_col:
        obs[sex_col] = obs[sex_col].apply(sanitize_sex)
        if not obs[sex_col].notna().any():
            print("Hinweis: Sex ist komplett fehlend -> wird nicht verwendet.", flush=True)
            sex_col = None

    X_cols = list(baseline_cols)
    if age_col: X_cols.append(age_col)
    if sex_col: X_cols.append(sex_col)

    X = obs[X_cols].copy()

    # Gruppensichere Splits: 70% Train, 15% Val, 15% Test (robust, bis beide Klassen in allen Splits vorhanden sind)
    def grouped_split_with_retry(y: np.ndarray, groups: np.ndarray, test_prop=0.30, val_prop=0.50, max_tries=200, seed=RANDOM_STATE):
        rng = np.random.RandomState(seed)
        X_dummy = np.zeros((len(y), 1))
        def ok(arr):
            u = np.unique(arr)
            return len(u) == 2
        tries = 0
        for t in range(max_tries):
            rs1 = int(rng.randint(0, 10_000))
            gss1 = GroupShuffleSplit(n_splits=1, test_size=test_prop, random_state=rs1)
            tr_idx, temp_idx = next(gss1.split(X_dummy, y, groups))
            if not (ok(y[tr_idx]) and ok(y[temp_idx])):
                continue
            rs2 = int(rng.randint(0, 10_000))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=rs2)
            val_rel, test_rel = next(gss2.split(X_dummy[temp_idx], y[temp_idx], groups[temp_idx]))
            val_idx = temp_idx[val_rel]
            test_idx = temp_idx[test_rel]
            if ok(y[val_idx]) and ok(y[test_idx]):
                return tr_idx, val_idx, test_idx, t+1
        # Letzter Fallback: kleinere Test-/Val-Anteile
        for tp in (0.25, 0.20):
            for t in range(max_tries):
                rs1 = int(rng.randint(0, 10_000))
                gss1 = GroupShuffleSplit(n_splits=1, test_size=tp, random_state=rs1)
                tr_idx, temp_idx = next(gss1.split(X_dummy, y, groups))
                if not (ok(y[tr_idx]) and ok(y[temp_idx])):
                    continue
                for vp in (0.50, 0.40):
                    rs2 = int(rng.randint(0, 10_000))
                    gss2 = GroupShuffleSplit(n_splits=1, test_size=vp, random_state=rs2)
                    val_rel, test_rel = next(gss2.split(X_dummy[temp_idx], y[temp_idx], groups[temp_idx]))
                    val_idx = temp_idx[val_rel]
                    test_idx = temp_idx[test_rel]
                    if ok(y[val_idx]) and ok(y[test_idx]):
                        return tr_idx, val_idx, test_idx, t+1
        raise RuntimeError("Konnte keine gruppensicheren Splits mit beiden Klassen finden. Outcome ist evtl. extrem selten oder leer.")

    train_idx, val_idx, test_idx, ntries = grouped_split_with_retry(y, groups_all)
    print(f"Splits gefunden nach {ntries} Versuch(en).", flush=True)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y[val_idx]
    X_test,  y_test  = X.iloc[test_idx],  y[test_idx]

    # Split-Info
    for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        uniq, cnts = np.unique(yy, return_counts=True)
        dist = ", ".join(f"{int(u)}:{int(c)}" for u,c in zip(uniq, cnts))
        print(f"{name} Größe={len(yy)} | Klassenverteilung: {dist}", flush=True)

    # Preprocessing & Modell
    num_cols = [c for c in X_cols if c != sex_col]
    cat_cols = [sex_col] if sex_col else []

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)

    pipe = Pipeline([("pre", pre), ("clf", rf)])

    # Hyperparameter-Raster
    param_grid = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2"],
    }

    # --- GridSearch nur auf dem Trainingssatz ---
    groups_train = groups_all[train_idx]
    cls_counts = np.bincount(y_train)
    if cls_counts.size < 2 or cls_counts.min() == 0:
        raise RuntimeError("Train-Split enthält nur eine Klasse. Bitte RANDOM_STATE/Splits anpassen oder Anteil ändern.")

    n_splits = int(min(5, np.unique(groups_train).size, cls_counts[0], cls_counts[1]))
    if n_splits < 2:
        n_splits = 2

    cv = make_group_cv(y_train, groups_train, n_splits, RANDOM_STATE)
    cv_kind = "StratifiedGroupKFold" if (StratifiedGroupKFold is not None and not isinstance(cv, list)) else "GroupKFold-filtered"
    print(f"CV: {cv_kind}, n_splits={n_splits}", flush=True)

    best = None
    if DO_GS:
        gs = GridSearchCV(
            pipe,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
        gs.fit(X_train, y_train, **({"groups": groups_train} if StratifiedGroupKFold is not None and not isinstance(cv, list) else {}))

        # Prüfen, ob CV-Scores brauchbar sind; sonst Fallback ohne GS
        try:
            cv_mean = np.array(gs.cv_results_["mean_test_score"], dtype=float)
            if np.all(np.isnan(cv_mean)):
                print("Warnung: Alle CV-Scores sind NaN -> Fallback ohne GridSearch.", flush=True)
            else:
                best = gs.best_estimator_
                print(f"BestParams: {getattr(gs, 'best_params_', None)}", flush=True)
        except Exception:
            print("Warnung: GridSearch-Ergebnisse nicht verfügbar -> Fallback ohne GridSearch.", flush=True)
    else:
        print("DO_GS=0 -> überspringe GridSearch, nutze Default-RF.", flush=True)

    if best is None:
        best = pipe.set_params(
            clf__n_estimators=600,
            clf__max_depth=None,
            clf__min_samples_leaf=1,
            clf__max_features="sqrt",
        )
        best.fit(X_train, y_train)

    # Vorhersagen auf Validation mit aktuellem (nur-Train) Modell
    p_val = best.predict_proba(X_val)[:, 1]
    thr_y, thr_f1 = thresholds_from_validation(p_val, y_val)

    # Bestes Modell auf Train+Val neu fitten und dann Test vorhersagen
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = np.concatenate([y_train, y_val])
    best.fit(X_trval, y_trval)

    p_test = best.predict_proba(X_test)[:, 1]

    def metrics_at(th: float) -> Dict[str, float | List[List[int]]]:
        yhat = (p_test >= th).astype(int)
        cm = confusion_matrix(y_test, yhat)
        return {
            "threshold": float(th),
            "AUROC": float(roc_auc_score(y_test, p_test)),
            "AUPRC": float(average_precision_score(y_test, p_test)),
            "brier": float(brier_score_loss(y_test, p_test)),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        }

    m05 = metrics_at(0.5)
    myj = metrics_at(thr_y)
    mf1 = metrics_at(thr_f1)

    # Plots
    pref = os.path.join(OUT_DIR_PLOTS, "ML_baseline_only_rf")

    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUROC={m05['AUROC']:.3f})")
    plt.tight_layout(); plt.savefig(pref+"_roc.png", dpi=200, bbox_inches="tight"); plt.close()

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUPRC={m05['AUPRC']:.3f})")
    plt.tight_layout(); plt.savefig(pref+"_pr.png", dpi=200, bbox_inches="tight"); plt.close()

    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(prob_pred, prob_true)
    plt.plot([0,1],[0,1],"--", linewidth=1)
    plt.xlabel("Vorhergesagte Wahrscheinlichkeit"); plt.ylabel("Beobachteter Anteil")
    plt.title("Kalibrationskurve – Baseline-only RF")
    plt.tight_layout(); plt.savefig(pref+"_calibration.png", dpi=200, bbox_inches="tight"); plt.close()

    plot_cm(confusion_matrix(y_test, (p_test>=0.5).astype(int)), "Confusion @0.5", pref+"_cm_0p5.png")
    plot_cm(confusion_matrix(y_test, (p_test>=thr_y).astype(int)), f"Confusion @Youden ({thr_y:.2f})", pref+f"_cm_youden_{thr_y:.2f}.png")
    plot_cm(confusion_matrix(y_test, (p_test>=thr_f1).astype(int)), f"Confusion @F1 ({thr_f1:.2f})", pref+f"_cm_f1_{thr_f1:.2f}.png")

    # Importances (Gini)
    clf: RandomForestClassifier = best.named_steps["clf"]
    importances = clf.feature_importances_

    # Feature-Namen rekonstruieren (nach Preprocessing)
    feat_names = []
    feat_names.extend([c for c in X_cols if c != sex_col])
    if sex_col:
        ohe: OneHotEncoder = best.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        feat_names.extend(list(ohe.get_feature_names_out([sex_col])))

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR_DATA, "ML_baseline_only_rf_importance_gini.csv"), index=False)

    # Balkenplot Top-12
    topk = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(6.0, 4.5))
    plt.barh(topk["feature"], topk["importance"])
    plt.title("RF Feature Importance – Baseline-only (Top 12)")
    plt.tight_layout(); plt.savefig(pref+"_importance_gini.png", dpi=200, bbox_inches="tight"); plt.close()

    # Report speichern
    out = {
        "y_col": y_name,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features": X_cols,
        "baseline_features": baseline_cols,
        "age_col": age_col,
        "sex_col": sex_col,
        "thresholds": {"youden": thr_y, "f1": thr_f1, "fixed_0.5": 0.5},
        "metrics_test@0.5": m05,
        "metrics_test@youden": myj,
        "metrics_test@f1": mf1,
        "best_params": getattr(gs, "best_params_", None),
        "h5ad": h5_path,
    }
    with open(os.path.join(OUT_DIR_DATA, "ML_baseline_only_rf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Fertig. Artefakte unter:")
    print("- Daten/ML_baseline_only_rf_metrics.json, *_importance_gini.csv")
    print("- Diagramme/ML_baseline_only_rf_[roc|pr|calibration|cm_*|importance_*].png")


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except Exception as e:
        print("! ERROR:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
