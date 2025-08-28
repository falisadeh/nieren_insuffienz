#!/usr/bin/env python3
"""
Kalibration der LogReg-Baseline (Platt/Sigmoid und Isotonic) auf ops_ml_processed.h5ad

- Verwendet nach Möglichkeit die in 11_ml_baseline.py gespeicherten Features und Zielspalte
  (Daten/ML_baseline_metrics.json). Fallback: identische Feature-Auswahl-Logik.
- Splits wie zuvor: 70/15/15 (Train/Val/Test), stratifiziert, random_state=42
- Base-Estimator: Logistische Regression (class_weight='balanced'); HP-Tuning per 5-fold CV
- Kalibration: CalibratedClassifierCV (cv='prefit') mit method ∈ { 'sigmoid', 'isotonic' } auf der Val-Menge
- Evaluation auf Test: AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier-Score; Confusion@0.5/@Youden/@F1
- Plots: ROC, PR, Kalibration (Reliability), Confusion-Matrizen
- Artefakte unter cs-transfer/Daten bzw. cs-transfer/Diagramme
"""
from __future__ import annotations
import os, json, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from anndata import read_h5ad

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Optional: Spearman für Redundanzfilter
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_MAIN = os.path.join(BASE, "Daten", "ops_ml_processed.h5ad")
H5_FALL = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUT_DIR_PLOTS = os.path.join(BASE, "Diagramme")
OUT_DIR_DATA = os.path.join(BASE, "Daten")
os.makedirs(OUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUT_DIR_DATA, exist_ok=True)

Y_CANDIDATES = ["had_aki", "AKI_linked_0_7", "AKI", "aki", "aki_linked_0_7"]
DEFAULT_FEATURES = [
    "vis_auc_0_48",
    "vis_auc_0_24",
    "crea_delta_0_48",
    "crea_rate_0_48",
    "duration_minutes",
]


# -------------------- Utils --------------------
def load_adata():
    path = H5_MAIN if os.path.exists(H5_MAIN) else H5_FALL
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Keine H5AD gefunden (Daten/ops_ml_processed.h5ad oder Daten/ops_with_patient_features.h5ad)"
        )
    return read_h5ad(path)


def pick_y(obs: pd.DataFrame) -> Tuple[str, np.ndarray]:
    for c in Y_CANDIDATES:
        if c in obs.columns:
            y_raw = (
                obs[c]
                .astype(str)
                .replace({"True": "1", "False": "0", "yes": "1", "no": "0"})
            )
            y = y_raw.astype(int).values
            return c, y
    raise KeyError(
        "Keine geeignete Zielspalte (had_aki / AKI_linked_0_7) in .obs gefunden."
    )


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def load_rank_table() -> pd.DataFrame | None:
    cand = [
        os.path.join(OUT_DIR_DATA, "ranked_features_table.csv"),
        os.path.join(OUT_DIR_DATA, "rank_features_ttest.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            if "variable" in df.columns and "feature" not in df.columns:
                df = df.rename(columns={"variable": "feature"})
            if "q" in df.columns and "qval" not in df.columns:
                df = df.rename(columns={"q": "qval"})
            return df
    return None


def pick_features(adata, y_name: str, max_k: int = 8) -> List[str]:
    # 1) Falls 11_ml_baseline-Metrik existiert → gleiche Features nutzen
    mpath = os.path.join(OUT_DIR_DATA, "ML_baseline_metrics.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                m = json.load(f)
            feats = [
                f for f in m.get("features", []) if f in list(map(str, adata.var_names))
            ]
            if feats:
                return feats[:max_k]
        except Exception:
            pass

    # 2) Manuelle Liste (optional)
    pref = os.path.join(OUT_DIR_DATA, "recommended_baseline_features.txt")
    feats = read_lines(pref)
    feats = [f for f in feats if f in list(map(str, adata.var_names))]
    if feats:
        return feats[:max_k]

    # 3) Ranking-Tabellen
    df = load_rank_table()
    if df is not None and not df.empty:
        key = (
            "qval"
            if "qval" in df.columns
            else ("pval" if "pval" in df.columns else None)
        )
        if key is not None:
            d = df.dropna(subset=[key]).sort_values(key)
            cand = [
                f
                for f in d["feature"].astype(str).tolist()
                if f in list(map(str, adata.var_names))
            ]
            # Redundanzfilter
            sel: List[str] = []
            if spearmanr is None:
                for f in cand:
                    base = f.split("_0_24")[0].split("_0_48")[0]
                    if any(base in s or s in base for s in sel):
                        continue
                    sel.append(f)
                    if len(sel) >= max_k:
                        break
                return sel
            else:
                Xfull = adata.to_df()
                for f in cand:
                    if f not in Xfull.columns:
                        continue
                    keep = True
                    for g in sel:
                        try:
                            rho, _ = spearmanr(
                                Xfull[f].values, Xfull[g].values, nan_policy="omit"
                            )
                            if rho is not None and abs(rho) > 0.85:
                                keep = False
                                break
                        except Exception:
                            pass
                    if keep:
                        sel.append(f)
                    if len(sel) >= max_k:
                        break
                if sel:
                    return sel

    # 4) Fallback
    return [f for f in DEFAULT_FEATURES if f in list(map(str, adata.var_names))][:max_k]


# -------------------- Plots --------------------
def plot_roc_pr(y_true, y_prob, out_prefix: str, title_suffix: str = ""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUROC={auroc:.3f}) {title_suffix}")
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUPRC={auprc:.3f}) {title_suffix}")
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, y_prob, out_png: str, title: str):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("Vorhergesagte Wahrscheinlichkeit")
    plt.ylabel("Beobachteter Anteil")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(cm: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------- Main --------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    adata = load_adata()
    y_name, y = pick_y(adata.obs)

    Xdf = adata.to_df()
    feats = pick_features(adata, y_name, max_k=8)
    if not feats:
        raise RuntimeError("Keine geeigneten Features gefunden.")
    X = Xdf[feats].copy()

    # Split 70/15/15 wie zuvor
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Basismodell + HP-Tuning
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", solver="liblinear"
                ),
            ),
        ]
    )
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    base = gs.best_estimator_
    # Base auf Train+Val neu fitten? Für Kalibration brauchen wir ein "prefit" nur mit Train
    # → Wir fitten base NUR auf Train, kalibrieren auf Val.
    base = gs.best_estimator_.set_params()
    base.fit(X_train, y_train)

    # Kalibratoren (cv='prefit' → erwartet bereits gefitteten Estimator)
    cal_sig = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal_iso = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal_sig.fit(X_val, y_val)
    cal_iso.fit(X_val, y_val)

    # Wahrscheinlichkeiten auf Test
    p_test_base = base.predict_proba(X_test)[:, 1]
    p_test_sig = cal_sig.predict_proba(X_test)[:, 1]
    p_test_iso = cal_iso.predict_proba(X_test)[:, 1]

    # Schwellen via Validierung (pro Kalibrator separat)
    def thresholds_from_val(estimator, Xv, yv):
        pv = estimator.predict_proba(Xv)[:, 1]
        fpr, tpr, thr = roc_curve(yv, pv)
        youden = np.argmax(tpr - fpr)
        thr_y = float(thr[youden]) if youden < len(thr) else 0.5
        prec, rec, thr_pr = precision_recall_curve(yv, pv)
        f1s = []
        for t in thr_pr:
            yhat = (pv >= t).astype(int)
            f1s.append(f1_score(yv, yhat))
        t_idx = int(np.argmax(f1s)) if len(f1s) > 0 else 0
        thr_f1 = float(thr_pr[t_idx]) if len(thr_pr) > 0 else 0.5
        return thr_y, thr_f1

    thrY_base, thrF1_base = thresholds_from_val(base, X_val, y_val)
    thrY_sig, thrF1_sig = thresholds_from_val(cal_sig, X_val, y_val)
    thrY_iso, thrF1_iso = thresholds_from_val(cal_iso, X_val, y_val)

    # Bewertung auf Test
    def metrics(y_true, p_prob, thr_list):
        auroc = float(roc_auc_score(y_true, p_prob))
        auprc = float(average_precision_score(y_true, p_prob))
        out = {}
        for name, th in thr_list:
            y_pred = (p_prob >= th).astype(int)
            out[name] = {
                "threshold": float(th),
                "AUROC": auroc,
                "AUPRC": auprc,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "brier": float(brier_score_loss(y_true, p_prob)),
                "cm": confusion_matrix(y_true, y_pred).tolist(),
            }
        return out

    met_base = metrics(
        y_test,
        p_test_base,
        [("@0.5", 0.5), ("@youden", thrY_base), ("@f1", thrF1_base)],
    )
    met_sig = metrics(
        y_test, p_test_sig, [("@0.5", 0.5), ("@youden", thrY_sig), ("@f1", thrF1_sig)]
    )
    met_iso = metrics(
        y_test, p_test_iso, [("@0.5", 0.5), ("@youden", thrY_iso), ("@f1", thrF1_iso)]
    )

    # Plots
    pref_base = os.path.join(OUT_DIR_PLOTS, "ML_calibrated_base")
    pref_sig = os.path.join(OUT_DIR_PLOTS, "ML_calibrated_sigmoid")
    pref_iso = os.path.join(OUT_DIR_PLOTS, "ML_calibrated_isotonic")
    plot_roc_pr(y_test, p_test_base, pref_base, "(uncalibrated)")
    plot_roc_pr(y_test, p_test_sig, pref_sig, "(Platt)")
    plot_roc_pr(y_test, p_test_iso, pref_iso, "(Isotonic)")

    plot_calibration(
        y_test, p_test_base, pref_base + "_calibration.png", "Kalibration: unkalibriert"
    )
    plot_calibration(
        y_test,
        p_test_sig,
        pref_sig + "_calibration.png",
        "Kalibration: Platt (sigmoid)",
    )
    plot_calibration(
        y_test, p_test_iso, pref_iso + "_calibration.png", "Kalibration: Isotonic"
    )

    # Confusion Plots
    def plot_cm_set(mdict, out_prefix, tag):
        for k, m in mdict.items():
            cm = np.array(m["cm"])  # 2x2
            plot_confusion(
                cm,
                f"{out_prefix}_cm_{tag}_{k.replace('@','')}.png",
                f"Confusion {tag} {k}",
            )

    plot_cm_set(met_base, pref_base, "base")
    plot_cm_set(met_sig, pref_sig, "sigmoid")
    plot_cm_set(met_iso, pref_iso, "isotonic")

    # Gesamt-Report
    report = {
        "y_col": y_name,
        "features": feats,
        "thresholds": {
            "base": {"youden": thrY_base, "f1": thrF1_base},
            "sigmoid": {"youden": thrY_sig, "f1": thrF1_sig},
            "isotonic": {"youden": thrY_iso, "f1": thrF1_iso},
        },
        "metrics_base": met_base,
        "metrics_sigmoid": met_sig,
        "metrics_isotonic": met_iso,
    }
    with open(
        os.path.join(OUT_DIR_DATA, "ML_calibrated_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Fertig. Artefakte gespeichert unter:")
    print("- Daten/ML_calibrated_metrics.json")
    print("- Diagramme/ML_calibrated_*[roc|pr|calibration|cm_*].png")
    print("Features:", feats)


if __name__ == "__main__":
    main()
