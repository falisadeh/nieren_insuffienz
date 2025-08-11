import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import read_h5ad

H5 = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
SAVE_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(SAVE_DIR, exist_ok=True)

import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def prepare_df(adata):
    need = ["PMID","event_idx","duration_hours","is_reop","Sex_norm"]
    for c in need:
        if c not in adata.obs.columns:
            adata.obs[c] = np.nan
    df = adata.obs[need].copy()
    df["event_idx"] = pd.to_numeric(df["event_idx"], errors="coerce").astype("Int64")
    df = df[df["event_idx"].notna() & df["PMID"].notna()].copy()
    df["event_idx"] = df["event_idx"].astype(int)
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["is_reop"] = pd.to_numeric(df["is_reop"], errors="coerce").fillna(0).astype(int)
    df["Sex_norm"] = df["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")
    return df

def fit_glm_clustered(df):
    y = df["event_idx"].values
    X = df[["duration_hours","is_reop","Sex_norm"]].copy()
    X["duration_hours"] = X["duration_hours"].fillna(X["duration_hours"].median())
    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = sm.add_constant(X_dm, has_constant="add")
    model = sm.GLM(y, X_dm, family=sm.families.Binomial())
    res = model.fit()
    res_rob = res.get_robustcov_results(cov_type="cluster", groups=df["PMID"], use_correction=True)
    params, se, z, p = res_rob.params, res_rob.bse, res_rob.tvalues, res_rob.pvalues
    ci_lo, ci_hi = params - 1.96*se, params + 1.96*se
    or_tab = pd.DataFrame({
        "Term": params.index,
        "Coef": params.values,
        "OR": np.exp(params.values),
        "CI_low": np.exp(ci_lo.values),
        "CI_high": np.exp(ci_hi.values),
        "z": z.values, "p": p.values,
    })
    intercept = or_tab[or_tab["Term"]=="const"]
    body = or_tab[or_tab["Term"]!="const"].sort_values("Term")
    out = pd.concat([intercept, body], axis=0) if not intercept.empty else body
    path = os.path.join(SAVE_DIR, "S4_glm_cluster_or.csv")
    out.to_csv(path, index=False)
    print("GLM (cluster-robust) OR-Tabelle gespeichert:", path)
    return path

def run_cv(df, n_splits=5):
    y = df["event_idx"].values
    groups = df["PMID"].values
    X = df[["duration_hours","is_reop","Sex_norm"]]
    num_feats = ["duration_hours","is_reop"]
    cat_feats = ["Sex_norm"]
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
    ])
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs"))])
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    proba = np.empty_like(y, dtype=float)
    for i,(tr,te) in enumerate(gkf.split(X,y,groups),1):
        clf.fit(X.iloc[tr], y[tr])
        proba[te] = clf.predict_proba(X.iloc[te])[:,1]
        print(f"Fold {i} train={len(tr)} test={len(te)}")

    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)

    fpr,tpr,_ = roc_curve(y, proba)
    plt.figure(figsize=(6,5)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("S4 ROC (GroupKFold)"); plt.tight_layout()
    roc_path = os.path.join(SAVE_DIR,"S4_ROC.png"); plt.savefig(roc_path,dpi=150); plt.close()

    prec,rec,_ = precision_recall_curve(y, proba)
    plt.figure(figsize=(6,5)); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("S4 Precision-Recall (GroupKFold)"); plt.tight_layout()
    pr_path = os.path.join(SAVE_DIR,"S4_PR.png"); plt.savefig(pr_path,dpi=150); plt.close()

    prob_true, prob_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5)); plt.plot(prob_pred, prob_true); plt.plot([0,1],[0,1],'--'); plt.xlabel("Vorhergesagte Wahrscheinlichkeit"); plt.ylabel("Beobachteter Anteil"); plt.title("S4 Kalibration"); plt.tight_layout()
    cal_path = os.path.join(SAVE_DIR,"S4_Calibration.png"); plt.savefig(cal_path,dpi=150); plt.close()

    metrics = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier": float(brier),
               "roc_path": roc_path, "pr_path": pr_path, "calibration_path": cal_path}
    pd.DataFrame([metrics]).to_csv(os.path.join(SAVE_DIR,"S4_cv_metrics.csv"), index=False)
    print(f"CV-Metriken: ROC_AUC={roc_auc:.3f}, PR_AUC={pr_auc:.3f}, Brier={brier:.3f}")
    return metrics

def main():
    adata = read_h5ad(H5)
    df = prepare_df(adata)
    or_csv = fit_glm_clustered(df)
    metrics = run_cv(df)
    adata.uns["S4_glm"] = {"or_table_path": or_csv, "n": int(df.shape[0]), "k": int(len(pd.read_csv(or_csv)))}
    adata.uns["S4_cv"] = metrics
    adata.write_h5ad(H5)
    print("AnnData (S4) gespeichert:", H5)

if __name__ == "__main__":
    main()
