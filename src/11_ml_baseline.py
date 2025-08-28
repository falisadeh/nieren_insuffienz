#!/usr/bin/env python3
"""
Baseline-Modelle für AKI (0–7 Tage) auf ops_ml_processed.h5ad

- Ziel (y): bevorzugt 'had_aki', sonst 'AKI_linked_0_7' (in .obs); nach 0/1 gemappt
- Features (X):
    1) Wenn vorhanden: Daten/recommended_baseline_features.txt (eine pro Zeile)
    2) Sonst: aus Daten/ranked_features_table.csv bzw. rank_features_ttest.csv (Top-N, FDR)
       + Redundanzfilter via Spearman (|rho|>0.85)
    3) Fallback: ['vis_auc_0_48','vis_auc_0_24','crea_delta_0_48','crea_rate_0_48','duration_minutes']
- Modell: Logistische Regression (class_weight='balanced'), Hyperparameter per 5-fold Stratified CV
- Splits: 70/15/15 (Train/Val/Test), stratifiziert
- Metriken: AUROC, AUPRC, Accuracy, Recall, Precision, F1
- Schwellenwahl: Youden-J (ROC) + bestes F1 (validierungsbasiert) → Confusion-Matrizen auf Test
- Plots: ROC, PR, Kalibration, Koeffizienten-Balken
- Artefakte: Modelle/Plots/Tabellen unter cs-transfer/Daten bzw. cs-transfer/Diagramme

Voraussetzungen: anndata, numpy, pandas, scikit-learn, matplotlib, scipy (optional)
"""
from __future__ import annotations
import os, json, math, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from anndata import read_h5ad

# Matplotlib headless
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# Sklearn
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
from sklearn.calibration import calibration_curve

# Optional: Spearman
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
def load_adata() -> "AnnData":
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
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    except Exception:
        return []


def load_rank_table() -> pd.DataFrame | None:
    cand = [
        os.path.join(OUT_DIR_DATA, "ranked_features_table.csv"),
        os.path.join(OUT_DIR_DATA, "rank_features_ttest.csv"),
        os.path.join(BASE, "Daten", "ranked_features_table.csv"),
        os.path.join(BASE, "Daten", "rank_features_ttest.csv"),
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
    # 1) Manueller Wunsch: recommended_baseline_features.txt
    pref = os.path.join(OUT_DIR_DATA, "recommended_baseline_features.txt")
    feats = read_lines(pref)
    if feats:
        feats = [f for f in feats if f in list(map(str, adata.var_names))]
        if feats:
            return feats[:max_k]

    # 2) Ranking-Tabellen nutzen
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
            # Redundanzfilter via Spearman
            sel: List[str] = []
            if spearmanr is None:
                # einfacher Domain-Filter: vermeide 2x sehr ähnliche Namen
                for f in cand:
                    base = f.split("_0_24")[0].split("_0_48")[0]
                    if any(base in s or s in base for s in sel):
                        continue
                    sel.append(f)
                    if len(sel) >= max_k:
                        break
                return sel
            else:
                # Greedy Auswahl auf Basis |rho|<0.85
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

    # 3) Fallback
    return [f for f in DEFAULT_FEATURES if f in list(map(str, adata.var_names))][:max_k]


# -------------------- Plots --------------------
def plot_roc_pr(y_true, y_prob, out_prefix: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUROC={auroc:.3f})")
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUPRC={auprc:.3f})")
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, y_prob, out_png: str):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("Vorhergesagte Wahrscheinlichkeit")
    plt.ylabel("Beobachteter Anteil")
    plt.title("Kalibrationskurve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_coefficients(names: List[str], coefs: np.ndarray, out_png: str):
    idx = np.argsort(np.abs(coefs))
    names_s = [names[i] for i in idx]
    coefs_s = coefs[idx]
    plt.figure(figsize=(6.5, max(3, 0.25 * len(names_s) + 1)))
    plt.barh(names_s, coefs_s)
    plt.xlabel("Koeffizient (standardisiert)")
    plt.title("LogReg-Koeffizienten")
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

    # DataFrame aus .X (bereits skaliert in ops_ml_processed.h5ad)
    Xdf = adata.to_df()  # (n_obs x n_vars) mit var_names

    # Feature-Auswahl
    feats = pick_features(adata, y_name, max_k=8)
    if not feats:
        raise RuntimeError("Keine geeigneten Features gefunden.")
    X = Xdf[feats].copy()

    # Imputer + LogReg Pipeline (Skalierung ist bereits erfolgt → keine weitere Skalierung nötig)
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

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Hyperparameter-Tuning auf Train via 5-fold CV
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X_train, y_train)

    # Bestes Modell auf Train+Val neu fitten
    best = gs.best_estimator_
    best.fit(pd.concat([X_train, X_val], axis=0), np.concatenate([y_train, y_val]))

    # Wahrscheinlichkeiten
    p_val = best.predict_proba(X_val)[:, 1]
    p_test = best.predict_proba(X_test)[:, 1]

    # Schwellen von Validierung ableiten
    fpr, tpr, thr = roc_curve(y_val, p_val)
    youden = np.argmax(tpr - fpr)
    thr_youden = float(thr[youden]) if youden < len(thr) else 0.5

    prec, rec, thr_pr = precision_recall_curve(y_val, p_val)
    f1s = []
    for t in thr_pr:
        y_hat = (p_val >= t).astype(int)
        f1s.append(f1_score(y_val, y_hat))
    t_idx = int(np.argmax(f1s)) if len(f1s) > 0 else 0
    thr_f1 = float(thr_pr[t_idx]) if len(thr_pr) > 0 else 0.5

    # Test-Metriken @0.5, @Youden, @F1
    def metrics_at(th):
        y_pred = (p_test >= th).astype(int)
        return {
            "threshold": float(th),
            "AUROC": float(roc_auc_score(y_test, p_test)),
            "AUPRC": float(average_precision_score(y_test, p_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "brier": float(brier_score_loss(y_test, p_test)),
            "cm": confusion_matrix(y_test, y_pred).tolist(),
        }

    metrics_05 = metrics_at(0.5)
    metrics_yj = metrics_at(thr_youden)
    metrics_f1 = metrics_at(thr_f1)

    # Plots
    out_prefix = os.path.join(OUT_DIR_PLOTS, "ML_baseline")
    plot_roc_pr(y_test, p_test, out_prefix)
    plot_calibration(y_test, p_test, out_prefix + "_calibration.png")

    # Confusion-Matrizen
    plot_confusion(
        np.array(metrics_05["cm"]), out_prefix + "_cm_0p5.png", "Confusion (th=0.5)"
    )
    plot_confusion(
        np.array(metrics_yj["cm"]),
        out_prefix + f"_cm_youden_{thr_youden:.2f}.png",
        "Confusion (Youden)",
    )
    plot_confusion(
        np.array(metrics_f1["cm"]),
        out_prefix + f"_cm_f1_{thr_f1:.2f}.png",
        "Confusion (F1)",
    )

    # Koeffizienten
    clf = best.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
        coef_df = pd.DataFrame({"feature": feats, "coef": coefs}).sort_values("coef")
        coef_df.to_csv(
            os.path.join(OUT_DIR_DATA, "ML_baseline_coefficients.csv"), index=False
        )
        plot_coefficients(
            feats, coefs, os.path.join(OUT_DIR_PLOTS, "ML_baseline_coefficients.png")
        )

    # Artefakte + Report
    out_json = {
        "y_col": y_name,
        "features": feats,
        "best_params": gs.best_params_,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "metrics_test@0.5": metrics_05,
        "metrics_test@youden": metrics_yj,
        "metrics_test@f1": metrics_f1,
    }
    with open(
        os.path.join(OUT_DIR_DATA, "ML_baseline_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print("Fertig. Artefakte gespeichert unter:")
    print("- Daten/ML_baseline_metrics.json, ML_baseline_coefficients.csv")
    print("- Diagramme/ML_baseline_*.png")
    print("Features:", feats)


if __name__ == "__main__":
    main()
