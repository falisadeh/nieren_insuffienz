# 16_ehrapy_op_level.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ehrapy as ep

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# Einstellungen / Pfade
# =========================
EH_CSV = "ehrapy_input_index_ops.csv"
os.makedirs("Diagramme", exist_ok=True)

# =========================
# 1) Daten in ehrapy laden
# =========================
adata = ep.io.read_csv(EH_CSV)

# Feature-Typen bestimmen (Info) und ggf. korrigieren
ep.ad.infer_feature_types(adata)
to_fix = {
    "n_ops": "numeric",           # deine ehrapy-Version nutzt 'numeric'/'categorical'/'date'
    "AKI_any_0_7": "categorical",
    "highest_AKI_stage_0_7": "categorical",
}
for feat, ftype in to_fix.items():
    if feat in adata.to_df().columns:
        try:
            ep.ad.replace_feature_types(adata, feat, ftype)
        except TypeError:
            ep.ad.replace_feature_types(adata=adata, feature=feat, corrected_type=ftype)

print("Featuretypen (nach Korrektur):", adata.uns.get("feature_types", {}))

# =========================================
# 2) ML-Setup: Leak vermeiden, Datentypen
# =========================================
df_ml = adata.to_df()
# Label früh puffern
label_series = df_ml["AKI_any_0_7"].astype(int)              # Ziel als Serie merken
sex_series = df_ml["Sex_norm"].astype(str) if "Sex_norm" in df_ml else None


# Ziel & Leckage-Spalten entfernen
y = df_ml["AKI_any_0_7"].astype(int)
leak_cols = [c for c in ["AKI_any_0_7", "highest_AKI_stage_0_7"] if c in df_ml.columns]
X = df_ml.drop(columns=leak_cols, errors="ignore").copy()

# Gewünschte numerische Prädiktoren (nur nehmen, wenn vorhanden)
num_wanted = [
    "age_years_at_first_op", "n_ops", "duration_hours",
    "crea_baseline", "crea_peak_0_48", "crea_delta_0_48", "crea_rate_0_48",
    "cysc_baseline", "cysc_peak_0_48", "cysc_delta_0_48", "cysc_rate_0_48",
    "vis_max_0_24", "vis_mean_0_24", "vis_max_6_24", "vis_auc_0_24", "vis_auc_0_48",
]
num_cols = [c for c in num_wanted if c in X.columns]

# Kategoriale Prädiktoren (nur Sex_norm)
cat_cols = [c for c in ["Sex_norm"] if c in X.columns]

# Numerik wirklich numerisch machen
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Unerwünschte object-Spalten (außer Sex_norm) droppen
extra_obj = [c for c in X.columns if X[c].dtype == "object" and c not in cat_cols]
if extra_obj:
    X = X.drop(columns=extra_obj)

print("Numerische Features:", num_cols)
print("Kategoriale Features:", cat_cols)

# =========================================
# 3) Pipeline (Impute/Scale/OneHot + Modell)
# =========================================
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# =========================
# 4) Train/Test + Bewertung
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== Classification Report (Test) ===")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC (Test):", roc_auc_score(y_test, y_proba))

# (Optional) 5-fold CV auf Gesamtdaten
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_cv = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV ROC-AUC 5-fold: {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")

# =========================
# 5) Feature-Importances
# =========================
try:
    # Namen nach One-Hot rekonstruieren
    oh = clf.named_steps["prep"].named_transformers_["cat"].named_steps["oh"] if cat_cols else None
    num_names = num_cols
    cat_names = list(oh.get_feature_names_out(cat_cols)) if oh is not None else []
    feat_names = num_names + cat_names

    importances = clf.named_steps["rf"].feature_importances_
    order = np.argsort(importances)[::-1]
    top = min(15, len(importances))

    print("\nTop-Features:")
    for i in range(top):
        j = order[i]
        print(f"{i+1:>2}. {feat_names[j]}  |  {importances[j]:.4f}")
except Exception as e:
    print("Feature-Importances nicht darstellbar:", e)
#Label in obs schreiben (nachdem du es aus var_names entfernt hast)
# a) sicherstellen, dass AKI_any_0_7 NICHT mehr in var_names ist
if "AKI_any_0_7" in adata.var_names:
    adata = adata[:, [v for v in adata.var_names if v != "AKI_any_0_7"]].copy()

# b) nun das zuvor gepufferte Label in obs spiegeln
adata.obs["AKI_any_0_7"] = label_series.astype("category").values
if sex_series is not None:
    adata.obs["Sex_norm"] = sex_series.astype("category").values

# Kontrolle (sollte False / True drucken)
print("'AKI_any_0_7' in var_names? ", "AKI_any_0_7" in list(adata.var_names))
print("'AKI_any_0_7' in obs? ", "AKI_any_0_7" in adata.obs.columns)



# =========================================
# 6) Plot-Galerie (ehrapy) – mit Konflikt-Fix
# =========================================
# a) Zielvariable darf NICHT in var_names stehen (sonst Konflikt in groupby)
if "AKI_any_0_7" in adata.var_names:
    keep_vars = [v for v in adata.var_names if v != "AKI_any_0_7"]
    adata = adata[:, keep_vars].copy()

# b) Ziel/Grouping in obs spiegeln
df_all = adata.to_df()
if "AKI_any_0_7" in df_all.columns:
    adata.obs["AKI_any_0_7"] = df_all["AKI_any_0_7"].astype("category")
if "Sex_norm" in df_all.columns and "Sex_norm" not in adata.obs.columns:
    adata.obs["Sex_norm"] = df_all["Sex_norm"].astype("category")

print("'AKI_any_0_7' in var_names? ", "AKI_any_0_7" in list(adata.var_names))
print("'AKI_any_0_7' in obs? ", "AKI_any_0_7" in adata.obs.columns)


# ---- ERSETZT deinen df_all-Block + die zwei Boxplots mit seaborn ----
# --- Korrelationsmatrix (Seaborn statt ep.pl.correlation_matrix) ---
# --- sauberes DF für Plots zusammenstellen ---
df_all = adata.to_df().copy()

# Label/Grouping aus obs ergänzen (ohne Spaltenkollision)
if "AKI_any_0_7" in adata.obs.columns:
    df_all["AKI_any_0_7"] = adata.obs["AKI_any_0_7"].astype(int).values
if ("Sex_norm" not in df_all.columns) and ("Sex_norm" in adata.obs.columns):
    df_all["Sex_norm"] = adata.obs["Sex_norm"].astype(str).values

# Nur Zeilen mit benötigten Spalten behalten
df_box1 = df_all[["AKI_any_0_7", "duration_hours"]].dropna()
df_box2 = df_all[["AKI_any_0_7", "crea_delta_0_48"]].dropna()

import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("Diagramme", exist_ok=True)

# --- Boxplot: OP-Dauer nach AKI ---
plt.figure(figsize=(5,4))
sns.boxplot(x="AKI_any_0_7", y="duration_hours", data=df_box1, showfliers=False)
sns.stripplot(x="AKI_any_0_7", y="duration_hours", data=df_box1, color="0.3", size=2, alpha=0.35)
plt.title("OP-Dauer nach AKI (0–7 Tage)")
plt.xlabel("AKI 0–7 Tage (0/1)")
plt.ylabel("Dauer (h)")
plt.tight_layout()
plt.savefig("Diagramme/box_duration_AKI.png", dpi=300)
plt.close()

# --- Boxplot: Kreatinin-Delta 0–48h nach AKI ---
plt.figure(figsize=(5,4))
sns.boxplot(x="AKI_any_0_7", y="crea_delta_0_48", data=df_box2, showfliers=False)
sns.stripplot(x="AKI_any_0_7", y="crea_delta_0_48", data=df_box2, color="0.3", size=2, alpha=0.35)
plt.title("Kreatinin-Δ (0–48 h) nach AKI")
plt.xlabel("AKI 0–7 Tage (0/1)")
plt.ylabel("Δ Kreatinin (Einheit)")
plt.tight_layout()
plt.savefig("Diagramme/box_crea_delta_AKI.png", dpi=300)
plt.close()
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("Diagramme", exist_ok=True)

# ----------- Violinplot (ehrapy) -----------
try:
    ep.pl.violin(adata, keys=["age_years_at_first_op"], groupby="AKI_any_0_7")
    plt.title("Altersverteilung nach AKI (0–7 Tage)")
    plt.savefig("Diagramme/violin_age_AKI.png", dpi=300)
    plt.close()
    print("✅ Violinplot gespeichert: Diagramme/violin_age_AKI.png")
except Exception as e:
    print("Violinplot-Fehler:", e)

# ----------- Boxplots (Seaborn) -----------
df_all = adata.to_df().copy()
if "AKI_any_0_7" in adata.obs.columns:
    df_all["AKI_any_0_7"] = adata.obs["AKI_any_0_7"].astype(int).values

# OP-Dauer nach AKI
plt.figure(figsize=(5,4))
sns.boxplot(x="AKI_any_0_7", y="duration_hours", data=df_all, showfliers=False)
sns.stripplot(x="AKI_any_0_7", y="duration_hours", data=df_all,
              color="0.3", size=2, alpha=0.35)
plt.title("OP-Dauer nach AKI (0–7 Tage)")
plt.savefig("Diagramme/box_duration_AKI.png", dpi=300)
plt.close()

# Kreatinin-Delta nach AKI
plt.figure(figsize=(5,4))
sns.boxplot(x="AKI_any_0_7", y="crea_delta_0_48", data=df_all, showfliers=False)
sns.stripplot(x="AKI_any_0_7", y="crea_delta_0_48", data=df_all,
              color="0.3", size=2, alpha=0.35)
plt.title("Kreatinin-Δ (0–48 h) nach AKI")
plt.savefig("Diagramme/box_crea_delta_AKI.png", dpi=300)
plt.close()

