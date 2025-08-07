import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
os.environ["SCIPY_ARRAY_API"] = "1"

# ==========================
# 1. DATEN EINLESEN UND VORBEREITUNG
# ==========================
df = pd.read_csv("full_patient_data.csv")
df.columns = df.columns.str.strip()  # falls Leerzeichen

# Beispielhafte Zielvariable
df["target"] = df["verstorben"]  # oder wie deine Variable heißt

# ==========================
# 2. FEATURE ENGINEERING
# ==========================
df["age_months"] = df["age"] * 12
df["cpb_per_op"] = df["total_cpb_time_min"] / (df["num_ops"] + 1)
df["creat_age_ratio"] = df["creatinine_first"] / (df["age"] + 0.1)

# ==========================
# 3. FEATURES & ZIEL
# ==========================
features = [
    "age", "gender_num", "total_cpb_time_min", "num_ops", 
    "age_months", "cpb_per_op", "creat_age_ratio", 
    # Füge hier weitere wichtige Spalten ein!
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
# 5. PIPELINE
# ==========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Du kannst später auch kategoriale Merkmale ergänzen
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

# ==========================
# 7. EVALUATION
# ==========================
y_probs = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC: {roc_auc:.3f}")

# Schwellenwert-Anpassung
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Optional: Threshold-Plot
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision/Recall vs Threshold")
plt.show()

# Schwellenwert wählen (z. B. 0.4)
threshold = 0.4
y_pred = (y_probs > threshold).astype(int)

print("\nClassification Report (angepasster Threshold):")
print(classification_report(y_test, y_pred))

# ==========================
# 8. FEATURE IMPORTANCE (optional)
# ==========================
best_model = clf.best_estimator_.named_steps["classifier"]
feature_names = numeric_features  # falls nur numerische Features
importances = best_model.feature_importances_

feat_imp = pd.Series(importances, index=feature_names)
feat_imp.nlargest(10).plot(kind="barh")
plt.title("Top 10 Merkmale")
plt.xlabel("Wichtigkeit")
plt.show()
