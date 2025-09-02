# === ehrapy Plots für pädiatrische Altersgruppen ===
# -*- coding: utf-8 -*-
# Präsentationsplots – ehrapy-first

import os
import pandas as pd
import numpy as np

# Matplotlib headless für Datei-Export
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ehrapy as ep

# ---------------- CONFIG ----------------
BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = f"{BASE}/h5ad/ops_with_patient_features.h5ad"
OUT = f"{BASE}/Diagramme"
os.makedirs(OUT, exist_ok=True)
# ---------------------------------------

# 1) Laden & Feature-Typen für ehrapy
adata = ep.io.read_h5ad(H5)

# robuste Typsetzung (version-unabhängig)
cat_cols = [
    "PMID",
    "SMID",
    "Procedure_ID",
    "Sex",
    "age_group_pediatric",
    "AKI_linked_0_7",
    "is_reop",
    "Tx?",
]
num_cols = ["duration_hours", "age_years_at_first_op", "days_to_AKI"]
date_cols = ["Surgery_Start", "Surgery_End", "AKI_Start", "AKI_End"]

for c in cat_cols:
    if c in adata.obs:
        adata.obs[c] = adata.obs[c].astype("category")
for c in num_cols:
    if c in adata.obs:
        adata.obs[c] = pd.to_numeric(adata.obs[c], errors="coerce")
for c in date_cols:
    if c in adata.obs:
        adata.obs[c] = pd.to_datetime(adata.obs[c], errors="coerce")

# ehrapy erkennt anhand der Pandas-Dtypen
ep.ad.infer_feature_types(adata)

# 2) Reihenfolge der pädiatrischen Gruppen fixieren (für Achsen)
group_order = [
    "Neonates (0–28 T.)",
    "Infants (1–12 Mon.)",
    "Toddlers (1–3 J.)",
    "Preschool (3–5 J.)",
    "School-age (6–12 J.)",
    "Adolescents (13–18 J.)",
]
if "age_group_pediatric" not in adata.obs:
    raise ValueError("Spalte 'age_group_pediatric' fehlt im H5AD.")

adata.obs["age_group_pediatric"] = adata.obs["age_group_pediatric"].cat.set_categories(
    group_order, ordered=True
)

# 3) Metrik wählen (OP-Dauer)
metric = None
if "duration_hours" in adata.obs.columns:
    metric = "duration_hours"
elif "duration_minutes" in adata.obs.columns:
    adata.obs["duration_hours"] = (
        pd.to_numeric(adata.obs["duration_minutes"], errors="coerce") / 60.0
    )
    metric = "duration_hours"

# 4) Verteilung der Altersgruppen (Balken) – kurzer Matplotlib-Plot
counts = adata.obs["age_group_pediatric"].value_counts().sort_index()
counts.to_csv(os.path.join(OUT, "AgeGroups_pediatric_counts_from_h5ad.csv"))

plt.figure(figsize=(7.5, 4.2))
counts.plot(kind="bar", color="#6aaed6")
plt.title("Altersgruppen (pädiatrisch) – Anzahl")
plt.xlabel("Gruppe")
plt.ylabel("n")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "AgeGroups_pediatric_counts_bar.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 5) Violinplot (ehrapy) – OP-Dauer nach Altersgruppen
if metric is not None:
    # leichte Bereinigung für Plausibilität
    m = pd.to_numeric(adata.obs[metric], errors="coerce")
    adata.obs[metric] = m.where((m >= 0) & (m <= 48))
    ep.pl.violin(
        adata,
        keys=[metric],
        groupby="age_group_pediatric",
        ylabel="Stunden",
        xlabel="Altersgruppen",
    )
    plt.title(f"OP-Dauer (Violin) nach pädiatrischen Altersgruppen ({metric})")
    plt.gcf().savefig(
        os.path.join(OUT, f"AgeGroups_pediatric_violin_{metric}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# 6) NEU: Aggregiert – genau EIN Balken pro Altersgruppe (statt Histogramm mit vielen Bins)
order_labels = group_order  # gleiche Reihenfolge wie oben

counts_one_bar = (
    adata.obs["age_group_pediatric"].value_counts().reindex(order_labels, fill_value=0)
)

# --- Absolutzahlen (ein Block je Gruppe) ---
plt.figure(figsize=(8, 5))
bars = plt.bar(
    counts_one_bar.index.astype(str),
    counts_one_bar.values,
    color=["#87CEEB", "#FFA500", "#32CD32", "#FF69B4", "#9370DB", "#A0522D"],
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Anzahl Operationen")
plt.title("Altersverteilung pro pädiatrischer Gruppe (aggregiert)")

# Zahlen über die Balken
for b in bars:
    y = int(b.get_height())
    plt.text(
        b.get_x() + b.get_width() / 2,
        y + max(1, counts_one_bar.max() * 0.01),
        f"{y}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "Age_groups_bar_counts.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# --- Optional: Prozentanteile als Balken (kannst du weglassen, wenn nicht benötigt) ---
total = counts_one_bar.sum()
pct = (counts_one_bar / total * 100).round(1)

plt.figure(figsize=(8, 5))
bars = plt.bar(pct.index.astype(str), pct.values, color="#6aaed6")
plt.xticks(rotation=20, ha="right")
plt.ylabel("Anteil (%)")
plt.title("Anteil pro pädiatrischer Gruppe (in %)")
for b, v in zip(bars, pct.values):
    plt.text(
        b.get_x() + b.get_width() / 2,
        v + 0.5,
        f"{v}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylim(0, max(20, pct.max() + 3))
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "Age_groups_bar_percent.png"), dpi=300, bbox_inches="tight"
)
plt.close()

print("Fertig. Plots gespeichert in:", OUT)

# 7) (Dein zusätzlicher sauberer Violinplot ohne ehrapy; kann bleiben)
import seaborn as sns

df_plot = adata.obs[["age_group_pediatric", "duration_hours"]].copy()
df_plot = df_plot.dropna()
df_plot = df_plot[(df_plot["duration_hours"] >= 0) & (df_plot["duration_hours"] <= 48)]

plt.figure(figsize=(9, 6))
sns.violinplot(
    data=df_plot,
    x="age_group_pediatric",
    y="duration_hours",
    order=group_order,
    inner="box",
    cut=0,
)
plt.title("OP-Dauer nach pädiatrischen Altersgruppen")
plt.xlabel("Altersgruppen")
plt.ylabel("Dauer (Stunden)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "AgeGroups_pediatric_violin_clean.png"), dpi=300)
plt.close()
