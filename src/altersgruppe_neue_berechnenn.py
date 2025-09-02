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

ep.ad.infer_feature_types(adata)  # ehrapy erkennt auf Basis der Pandas-Dtypen

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
counts.plot(kind="bar")
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

# 6) Histogramm (ehrapy): Altersverteilung je Gruppe
# zeigt, dass die Gruppierung aus dem Alter abgeleitet ist
# Histogramm (Matplotlib): Altersverteilung nach pädiatrischen Gruppen
plt.figure(figsize=(8, 5))
for grp, sub in adata.obs.groupby("age_group_pediatric"):
    vals = pd.to_numeric(sub["age_years_at_first_op"], errors="coerce")
    vals = vals[(vals >= 0) & (vals <= 18)]
    if vals.empty:
        continue
    plt.hist(vals, bins=30, alpha=0.5, label=grp)

plt.title("Altersverteilung pro pädiatrischer Gruppe")
plt.xlabel("Alter (Jahre)")
plt.ylabel("Häufigkeit")
plt.legend(title="Altersgruppe")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "Age_years_at_first_op_hist_by_group.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


print("Fertig. Plots gespeichert in:", OUT)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# DataFrame direkt aus AnnData
df_plot = adata.obs[["age_group_pediatric", "duration_hours"]].copy()
df_plot = df_plot.dropna()
df_plot = df_plot[(df_plot["duration_hours"] >= 0) & (df_plot["duration_hours"] <= 48)]

plt.figure(figsize=(9, 6))
sns.violinplot(
    data=df_plot,
    x="age_group_pediatric",
    y="duration_hours",
    order=[
        "Neonates (0–28 T.)",
        "Infants (1–12 Mon.)",
        "Toddlers (1–3 J.)",
        "Preschool (3–5 J.)",
        "School-age (6–12 J.)",
        "Adolescents (13–18 J.)",
    ],
    inner="box",  # zeigt Median+Quartile statt viele Punkte
    cut=0,
)
plt.title("OP-Dauer nach pädiatrischen Altersgruppen")
plt.xlabel("Altersgruppen")
plt.ylabel("Dauer (Stunden)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "AgeGroups_pediatric_violin_clean.png"), dpi=300)
plt.close()
