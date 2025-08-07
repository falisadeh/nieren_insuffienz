import os
os.environ["SCIPY_ARRAY_API"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ==========================
# 1. DATEN EINLESEN UND ÜBERPRÜFEN
# ==========================
df = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/full_patient_data.csv")
df.columns = df.columns.str.strip()
print("Spaltenübersicht:", df.columns.tolist())

# Zielvariable setzen
df["target"] = df["verstorben"]

# ==========================
# 2. FEATURE ENGINEERING
# ==========================
#df["age_months"] = df["Age"] * 12

# ==========================
# 3. FEATURES & ZIEL
# ==========================
features = [
    "Age", "Sex", "OP_Anzahl"
]
X = df[features]
y = df["target"]

# ==========================
# 4. TRAIN-TEST-SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# 5. PREPROCESSING PIPELINE
# ==========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
])

# ==========================
# 6. MODELL + GRIDSEARCH + SMOTE
# ==========================
model = RandomForestClassifier(random_state=42)

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5]
}

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", model)
])

clf = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
clf.fit(X_train, y_train)
# Für importances
# Sicherstellen, dass das Modell trainiert wurde
best_model = clf.best_estimator_.named_steps["classifier"]

# Feature Importances berechnen
importances = best_model.feature_importances_

used_features = features[:len(importances)]
print("Verwendete Merkmale im Modell:", used_features)


# ==========================
# 7. EVALUATION
# ==========================
y_probs = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probs)
print(f"\nROC AUC Score: {roc_auc:.3f}")

# Schwellenwert-Anpassung
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

from sklearn.metrics import f1_score

# Precision, Recall, Thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# Optimaler Threshold (höchster F1-Score)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
plt.plot(thresholds, f1_scores[:-1], label="F1-Score", linewidth=2, linestyle="--")

# Optimaler Punkt markieren
plt.axvline(x=optimal_threshold, color="gray", linestyle="dotted", label=f"Optimaler Threshold: {optimal_threshold:.2f}")
plt.scatter(optimal_threshold, optimal_f1, color="red", label=f"Max. F1: {optimal_f1:.2f}")

# Layout
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Metrik-Wert", fontsize=12)
plt.title("Precision, Recall und F1-Score in Abhängigkeit vom Schwellenwert", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_f1_curve.png", dpi=300)  # optional für Bachelorarbeit
plt.show()


# Schwellenwert festlegen (z. B. 0.4)
threshold = 0.4
y_pred = (y_probs > threshold).astype(int)

print("\nClassification Report (angepasster Threshold):")
print(classification_report(y_test, y_pred))

# ==========================
# 8. FEATURE IMPORTANCE
# ==========================
# Nach dem GridSearchCV-Fit
best_model = clf.best_estimator_.named_steps["classifier"]
importances = best_model.feature_importances_

# Nur so viele Features verwenden, wie der Klassifikator tatsächlich verarbeitet hat
used_features = features[:len(importances)]

# Plotten
feat_imp = pd.Series(importances, index=used_features)
feat_imp = feat_imp.sort_values()
feat_imp.plot(kind="barh", title="Wichtigste Merkmale zur Vorhersage")
plt.xlabel("Wichtigkeit")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================
# 9. Beste Modellparameter
# ==========================
print("Beste Parameterkombination:", clf.best_params_)
from sklearn.metrics import f1_score

# Precision, Recall, Thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# Optimaler Threshold (höchster F1-Score)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
plt.plot(thresholds, f1_scores[:-1], label="F1-Score", linewidth=2, linestyle="--")

# Optimaler Punkt markieren
plt.axvline(x=optimal_threshold, color="gray", linestyle="dotted", label=f"Optimaler Threshold: {optimal_threshold:.2f}")
plt.scatter(optimal_threshold, optimal_f1, color="red", label=f"Max. F1: {optimal_f1:.2f}")

# Layout
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Metrik-Wert", fontsize=12)
plt.title("Precision, Recall und F1-Score in Abhängigkeit vom Schwellenwert", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_f1_curve.png", dpi=300)  # optional für Bachelorarbeit
plt.show()

