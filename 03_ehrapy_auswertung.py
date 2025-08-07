#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import anndata  # aus der anndata-Bibliothek
from sklearn.metrics import roc_auc_score

# 1. CSV laden
df = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/full_patient_data.csv")
# Nach dem Einlesen der CSV PMDI entfernen
df = df.drop(columns=["PMID"])

#1.2 Zielvariable vorbereiten
df["verstorben"] = df["verstorben"].astype(int) # Sicherstellen, dass Zielspalte numerisch ist (0 oder 1)

# 2. Spaltennamen bereinigen
df.columns = df.columns.str.strip()  # Entfernt Leerzeichen rund um Spaltennamen
print("Originaldaten geladen:", df.shape)

  # Sicherstellen, dass Zielspalte numerisch ist (0 oder 1)

#  4. Spalten klassifizieren
flag_cols = []  # z. B. ["some_flag_column"] – falls binäre Spalten 
num_as_cat = ["Sex"]  # numerisch kodiert, aber kategorial gemeint (1 = männlich, 2 = weiblich)

# Kategorische Spalten automatisch erkennen
obj_cats = df.select_dtypes(include=["object"]).columns.tolist()

# Gesamtübersicht der kategorischen Spalten
categorical_cols = sorted(set(flag_cols + num_as_cat + obj_cats))

# Numerische Spalten (außer Zielvariable und Kategorisches)
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in categorical_cols and c != "verstorben"]

# 5. Zielvariable trennen
y = df["verstorben"]
X = df.drop(columns=["verstorben"])

# 6. Eigener Winsorizer zur Ausreißerbehandlung
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

#  7. Pipelines definieren
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("winsor", Winsorizer()),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"))
])

# Gesamte Vorverarbeitung zusammenfassen
preprocess = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, categorical_cols)
])

#  8. Vorverarbeitung anwenden
X_transformed = preprocess.fit_transform(X)

# 9. Feature-Namen extrahieren
# Kategorische Namen holen aus OneHotEncoder
cat_features = preprocess.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
feature_names = numeric_cols + list(cat_features)

# 10. Neuen DataFrame erstellen
df_ready = pd.DataFrame(X_transformed, columns=feature_names)
df_ready["verstorben"] = y.values

# 11. AnnData erzeugen (für ehrapy-kompatible Form)
adata = anndata.AnnData(X=X_transformed)
adata.obs = df_ready[["verstorben"]].copy()
adata.obs["label"] = adata.obs["verstorben"].astype("category")

# 12. Modelltraining mit Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
# Vorhersage & Auswertung
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeit für Klasse 1 (verstorben)
#  13. Evaluation
print("\n🔍 Klassifikationsbericht:")
print(classification_report(y_test, y_pred))

#  14. Wichtigste Merkmale für Vorhersage
importances = rf.feature_importances_
top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC-Score: {roc_auc:.3f}")
importances = rf.feature_importances_
top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)


print("\n Top 10 wichtigste Merkmale für Sterblichkeit:")
print(top_features)

# 15. Optional: Export der bereinigten Daten
df_ready.to_csv("full_patient_data_clean_ready.csv", index=False)
print("Bereinigte Datei gespeichert.")

# Einen neuen Schritt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, classification_report

# Wahrscheinlichkeiten vorhersagen (für Klasse 1 = verstorben)
y_proba = rf.predict_proba(X_test)[:, 1]

# Precision, Recall, Thresholds berechnen
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1-Score für alle Schwellen berechnen
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Beste Schwelle finden (höchster F1-Score)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1 = f1_scores[best_threshold_index]

print(f" Bester Schwellenwert: {best_threshold:.2f}")
print(f" Höchster F1-Score: {best_f1:.3f}")

# Neue Vorhersagen mit optimierter Schwelle
y_pred_thresh = (y_proba >= best_threshold).astype(int)

# Bericht ausgeben
print("\n Klassifikationsbericht mit optimiertem Threshold:")
print(classification_report(y_test, y_pred_thresh))

# Plot: Precision, Recall, F1 vs. Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1-Score")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Beste Schwelle ({best_threshold:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Wert")
plt.title("Precision, Recall & F1-Score vs. Threshold")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
#Klassifikationsbericht mit optimiertem Threshold:
# precision    recall  f1-score   support

#0       0.99      0.99      0.99       210
# 1       0.25      0.25      0.25         4
#accuracy                           0.97       214 
# macro avg       0.62      0.62      0.62       214
#weighted avg       0.97      0.97      0.97       214
# Interpretation: Recall 1 = 0.25: Das Modell erkennt 25 % der verstorbenen Patienten – besser als vorher (0.00).
#,Precision 1 = 0.25: Wenn das Modell „verstorben“ sagt, liegt es zu 25 % richtig.
#F1-Score 1 = 0.25: Noch schwach, aber besser als 0.
#Accuracy = 0.97: Sehr hoch, aber wegen der starken Klassen-Ungleichheit (nur 4 von 214 = 1.8 %) nur begrenzt aussagekräftig.