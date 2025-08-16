#!/usr/bin/env python3
"""
Decision Tree + Random Forest für AKI (0–7 Tage) auf deinem Datensatz.

• Lädt standardmäßig deine H5AD-Datei und greift auf adata.obs zu.
• Alternativ kann eine CSV geladen werden.
• Trainiert einen kleinen, gut visualisierbaren Decision Tree (Graphviz) und
  einen robusten Random Forest (mit Feature Importances).
• Speichert alle Outputs in deinen Diagramme-Ordner.

Ausführen (Beispiele):
  python 08_tree_and_forest_analysis.py \
      --input \
      "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad" \
      --input-type h5ad

  python 08_tree_and_forest_analysis.py \
      --input \
      "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/ehrapy_input.csv" \
      --input-type csv

Voraussetzungen:
  pip install scikit-learn pandas numpy anndata graphviz
  (Graphviz-Systempaket muss installiert sein, z. B. via brew/apt/conda)
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Optional: H5AD einlesen
try:
    import anndata as ad
except Exception:
    ad = None

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt

# Versuche Graphviz (Python-Bindings); wenn nicht vorhanden, DOT-Datei trotzdem ausgeben
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False


DEFAULT_OUTPUT_DIR = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard-Pfade aus deinem Projektkontext
DEFAULT_H5AD = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad")
DEFAULT_CSV = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/ehrapy_input.csv")

# Erwartete Spaltennamen in adata.obs oder CSV
TARGET_COL = "AKI_linked_0_7"  # binär 0/1
CANDIDATE_FEATURES = [
    "duration_hours",      # OP-Dauer (h)
    "age_years_at_op",     # Alter in Jahren bei OP
    "is_reop",             # Re-OP (0/1)
    "Sex_norm",            # Geschlecht (z. B. 'm'/'w')
]


def load_data(input_path: Path, input_type: str) -> pd.DataFrame:
    """Lädt Daten aus H5AD (adata.obs) oder CSV und gibt DataFrame zurück.
    Wir prüfen defensiv auf vorhandene Spalten.
    """
    if input_type == "h5ad":
        if ad is None:
            raise RuntimeError("anndata ist nicht installiert, kann H5AD nicht lesen.")
        adata = ad.read_h5ad(str(input_path))
        df = adata.obs.copy()
    elif input_type == "csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("input_type muss 'h5ad' oder 'csv' sein")

    # Spaltennamen säubern (Sicherheitsnetz)
    df.columns = df.columns.str.strip()

    # Prüfe, ob Zielspalte existiert; wenn nicht, versuche sinnvolle Alternativen
    if TARGET_COL not in df.columns:
        # Fallbacks/Heuristiken (falls abweichende Benennung):
        for alt in ["AKI", "AKI_0_7", "AKI_linked", "aki_0_7", "aki"]:
            if alt in df.columns:
                warnings.warn(f"Zielspalte '{TARGET_COL}' fehlt – nutze stattdessen '{alt}'.")
                df[TARGET_COL] = df[alt]
                break
        else:
            raise KeyError(
                f"Zielvariable '{TARGET_COL}' wurde nicht gefunden. Verfügbare Spalten: {list(df.columns)[:30]}..."
            )

    # Wähle Feature-Spalten, die tatsächlich existieren
    available_features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not available_features:
        raise KeyError(
            "Keine der erwarteten Feature-Spalten gefunden. Bitte prüfe die Spaltennamen in deinem Datensatz.\n"
            f"Erwartet: {CANDIDATE_FEATURES}\nGefunden: {list(df.columns)[:30]}..."
        )

    # Nur relevante Spalten behalten (Features + Target)
    cols = available_features + [TARGET_COL]
    df_small = df[cols].copy()

    # Konvertiere Zielvariable robust zu 0/1
    df_small[TARGET_COL] = (
        df_small[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": 1, "false": 0, "ja": 1, "nein": 0})
    )
    # Falls noch Strings drin sind, versuche numerisch zu werden; alles andere → NaN
    df_small[TARGET_COL] = pd.to_numeric(df_small[TARGET_COL], errors="coerce")

    # Nur Zeilen behalten, wo Ziel nicht fehlend ist
    before = len(df_small)
    df_small = df_small.dropna(subset=[TARGET_COL])
    after = len(df_small)
    if after < before:
        warnings.warn(f"{before - after} Zeilen entfernt, weil Zielvariable fehlte.")

    return df_small


def build_preprocessor(df: pd.DataFrame, features: list[str]):
    """Erzeuge ColumnTransformer mit Imputation/Scaling/OneHot für numerische und kategoriale Variablen."""
    # Erkennung numerisch vs. kategorial
    numeric_features = []
    categorical_features = []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    # Häufig ist is_reop als bool/0-1 kodiert → numerisch behandeln
    # Häufig ist Sex_norm als 'm'/'w' → kategorial behandeln

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def plot_and_save_curves(y_true, y_proba, outdir: Path, prefix: str):
    # ROC
    roc = RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(f"ROC-Kurve – {prefix}")
    roc_path = outdir / f"{prefix}_ROC.png"
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    # PR
    pr = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(f"Precision–Recall – {prefix}")
    pr_path = outdir / f"{prefix}_PR.png"
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()

    return str(roc_path), str(pr_path)


def train_decision_tree(preprocessor, X_train, y_train, max_depth=3, random_state=42):
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model = Pipeline(steps=[("prep", preprocessor), ("tree", clf)])
    model.fit(X_train, y_train)
    return model


def export_tree_graphviz(model: Pipeline, feature_names: list[str], outdir: Path, filename_prefix: str):
    """Exportiert den Baum als .dot und rendert (falls möglich) als .png."""
    # Hole die echten Feature-Namen nach dem Preprocessing (inkl. OneHot-Spalten)
    prep: ColumnTransformer = model.named_steps["prep"]
    tree: DecisionTreeClassifier = model.named_steps["tree"]

    # Numerische Features bleiben wie sie sind
    num_features = prep.transformers_[0][2]

    # Kategoriale Spalten werden ge-onehotted → Namen extrahieren
    cat_features = prep.transformers_[1][2]
    ohe: OneHotEncoder = prep.named_transformers_["cat"].named_steps["onehot"]
    try:
        ohe_names = ohe.get_feature_names_out(cat_features).tolist()
    except Exception:
        # Fallback (ältere sklearn)
        ohe_names = [f"{c}_{i}" for c in cat_features for i in range(1000)]  # overshoot, wir kürzen später

    all_feature_names = list(num_features) + list(ohe_names)

    dot_path = outdir / f"{filename_prefix}.dot"
    png_path = outdir / f"{filename_prefix}.png"

    export_graphviz(
        tree,
        out_file=str(dot_path),
        feature_names=all_feature_names[: tree.n_features_in_],
        class_names=["kein AKI", "AKI"],
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True,
    )

    if GRAPHVIZ_AVAILABLE:
        try:
            src = graphviz.Source(dot_path.read_text())
            src.format = "png"
            src.render(filename=str(outdir / filename_prefix), cleanup=True)
            print(f"Decision Tree PNG gespeichert: {png_path}")
        except Exception as e:
            print(f"Graphviz-Rendering fehlgeschlagen ({e}). DOT-Datei liegt vor: {dot_path}")
    else:
        print(f"Graphviz (Python) nicht verfügbar. DOT-Datei gespeichert: {dot_path}")

    return str(dot_path), str(png_path)


def train_random_forest(preprocessor, X_train, y_train, n_estimators=400, max_depth=None, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model = Pipeline(steps=[("prep", preprocessor), ("rf", rf)])
    model.fit(X_train, y_train)
    return model


def extract_feature_importances(model: Pipeline, preprocessor: ColumnTransformer, outdir: Path, prefix: str) -> pd.DataFrame:
    rf: RandomForestClassifier = model.named_steps["rf"]

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    try:
        ohe_names = ohe.get_feature_names_out(cat_features).tolist()
    except Exception:
        ohe_names = [f"{c}_{i}" for c in cat_features for i in range(1000)]

    feature_names = list(num_features) + list(ohe_names)
    importances = rf.feature_importances_[: len(feature_names)]
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = outdir / f"{prefix}_feature_importances.csv"
    fi.to_csv(csv_path, index=False)
    print(f"Feature Importances gespeichert: {csv_path}")
    return fi


def main():
    parser = argparse.ArgumentParser(description="Decision Tree & Random Forest für AKI (0–7 Tage)")
    parser.add_argument("--input", type=str, default=None, help="Pfad zu H5AD oder CSV")
    parser.add_argument("--input-type", type=str, choices=["h5ad", "csv"], default=None, help="Dateityp")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output-Verzeichnis für Plots & CSVs")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximale Tiefe des Decision Trees (für Visualisierung)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Input ableiten, falls nicht explizit angegeben
    input_path = None
    input_type = None
    if args.input and args.input_type:
        input_path = Path(args.input)
        input_type = args.input_type
    else:
        # Bevorzugt H5AD, sonst CSV
        if DEFAULT_H5AD.exists():
            input_path = DEFAULT_H5AD
            input_type = "h5ad"
        elif DEFAULT_CSV.exists():
            input_path = DEFAULT_CSV
            input_type = "csv"
        else:
            print("Kein Eingabepfad angegeben und keine Default-Datei gefunden.")
            sys.exit(1)

    print(f"Lade Daten aus: {input_path} (Typ: {input_type})")
    df = load_data(input_path, input_type)

    # Features bestimmen, die wirklich vorhanden sind
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    print(f"Verwendete Features: {features}")
    X = df[features]
    y = df[TARGET_COL].astype(int)

    # Preprocessor
    preprocessor, num_feats, cat_feats = build_preprocessor(df, features)
    print(f"Numerische Features: {num_feats}")
    print(f"Kategoriale Features: {cat_feats}")

    # Train/Test Split (stratifiziert, damit Klassenverhältnis erhalten bleibt)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ------------------------- Decision Tree -------------------------
    tree_model = train_decision_tree(preprocessor, X_train, y_train, max_depth=args.max_depth)
    y_proba_tree = tree_model.predict_proba(X_test)[:, 1]
    y_pred_tree = (y_proba_tree >= 0.5).astype(int)

    roc_tree = roc_auc_score(y_test, y_proba_tree)
    pr_tree = average_precision_score(y_test, y_proba_tree)
    print(f"Decision Tree – ROC-AUC: {roc_tree:.3f} | PR-AUC: {pr_tree:.3f}")
    print("Decision Tree – Klassifikationsbericht (Threshold 0.5):")
    print(classification_report(y_test, y_pred_tree, target_names=["kein AKI", "AKI"]))

    # Visualisierung exportieren
    dot_path, png_path = export_tree_graphviz(tree_model, features, outdir, "DecisionTree_AKI")

    # Kurven speichern
    roc_png, pr_png = plot_and_save_curves(y_test, y_proba_tree, outdir, "DecisionTree_AKI")

    # ------------------------- Random Forest -------------------------
    rf_model = train_random_forest(preprocessor, X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_proba_rf >= 0.5).astype(int)

    roc_rf = roc_auc_score(y_test, y_proba_rf)
    pr_rf = average_precision_score(y_test, y_proba_rf)
    print(f"Random Forest – ROC-AUC: {roc_rf:.3f} | PR-AUC: {pr_rf:.3f}")
    print("Random Forest – Klassifikationsbericht (Threshold 0.5):")
    print(classification_report(y_test, y_pred_rf, target_names=["kein AKI", "AKI"]))

    # PR/ROC speichern
    roc_png_rf, pr_png_rf = plot_and_save_curves(y_test, y_proba_rf, outdir, "RandomForest_AKI")

    # Feature Importances extrahieren & speichern
    fi = extract_feature_importances(rf_model, rf_model.named_steps["prep"], outdir, "RandomForest_AKI")

    # Kurzer Abschlussbericht als JSON
    report = {
        "input": str(input_path),
        "features": features,
        "tree": {
            "roc_auc": float(roc_tree),
            "pr_auc": float(pr_tree),
            "roc_png": roc_png,
            "pr_png": pr_png,
            "graphviz_png": png_path if os.path.exists(png_path) else None,
            "dot": dot_path,
        },
        "random_forest": {
            "roc_auc": float(roc_rf),
            "pr_auc": float(pr_rf),
            "roc_png": roc_png_rf,
            "pr_png": pr_png_rf,
            "feature_importances_csv": str(DEFAULT_OUTPUT_DIR / "RandomForest_AKI_feature_importances.csv"),
        },
    }
    report_path = outdir / "tree_forest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Bericht gespeichert: {report_path}")


if __name__ == "__main__":
    main()
