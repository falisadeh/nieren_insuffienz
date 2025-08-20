#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_trajectories_analysis.py
Autor: du
Ziel: Vereinfachte Trajektorien-Analyse (OP-Ende → AKI-Onset 0–7 Tage)
      mit Subgruppen (Re-OP, OP-Dauer-Tertile) und optionalem Sankey-Plot.

Eingangsdaten (aus deinem Projekt):
  H5AD:  ~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad
  obs-Spalten erwartet: 
    - AKI_linked_0_7 (0/1)
    - days_to_AKI (float, Tage; NaN falls kein AKI)
    - duration_hours (float)
    - is_reop (0/1)

Ausgaben (PNG) nach:
  ~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme/
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot-Einstellungen
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

# -----------------------------
# Pfade
# -----------------------------
HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Library", "Mobile Documents", "com~apple~CloudDocs", "cs-transfer")
H5AD_PATH = os.path.join(BASE, "aki_ops_master_S1_survival.h5ad")
OUTDIR = os.path.join(BASE, "Diagramme")
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Daten laden
# -----------------------------
try:
    import anndata as ad
    adata = ad.read_h5ad(H5AD_PATH)
    df = adata.obs.copy()
except Exception as e:
    raise SystemExit(f"Fehler beim Laden der H5AD-Datei:\n{H5AD_PATH}\n{e}")

required_cols = ["AKI_linked_0_7", "days_to_AKI", "duration_hours", "is_reop"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Fehlende Spalten in adata.obs: {missing}\n"
                     f"Verfügbare Spalten: {list(df.columns)}")

# -----------------------------
# Helper: AKI-Onset-Kategorien
# -----------------------------
def classify_aki_onset(row) -> str:
    # Kein AKI im 0–7-d-Fenster
    if int(row["AKI_linked_0_7"]) == 0 or pd.isna(row["days_to_AKI"]):
        return "kein AKI (0–7 d)"
    # Mit AKI: Tage in Kategorien einteilen
    d = float(row["days_to_AKI"])
    if d <= 1:
        return "früh (0–24 h)"
    elif d <= 2:
        return "mittelfrüh (24–48 h)"
    else:
        return "spät (3–7 d)"

df["AKI_onset_group"] = df.apply(classify_aki_onset, axis=1)

# Feste Reihenfolge für Achsen/Legenden
ONSET_ORDER = ["früh (0–24 h)", "mittelfrüh (24–48 h)", "spät (3–7 d)", "kein AKI (0–7 d)"]

# -----------------------------
# OP-Dauer in Tertile
# -----------------------------
# Robuster Umgang mit Identischen Werten: rank(method="first") vermeidet Kantenfälle
valid_duration = df["duration_hours"].dropna()
if valid_duration.empty:
    raise SystemExit("duration_hours enthält keine gültigen Werte.")

# Labels für Tertile
TERTILE_LABELS = ["kurz", "mittel", "lang"]
df.loc[:, "duration_tertile"] = pd.qcut(
    df["duration_hours"].rank(method="first"), 
    q=3, labels=TERTILE_LABELS
)

# -----------------------------
# Plot 1: Gesamtverteilung der Onset-Trajektorien
# -----------------------------
counts = df["AKI_onset_group"].value_counts()
counts = counts.reindex(ONSET_ORDER).fillna(0).astype(int)

plt.figure(figsize=(7, 4.2))
plt.bar(counts.index, counts.values)
plt.ylabel("Anzahl Operationen")
plt.title("Trajektorien des AKI-Onsets (0–7 Tage nach OP-Ende)")
plt.xticks(rotation=15, ha="right")
for i, v in enumerate(counts.values):
    plt.text(i, v + max(counts.values)*0.01, str(v), ha="center", va="bottom", fontsize=9)
out1 = os.path.join(OUTDIR, "Traj_Onset_overall.png")
plt.savefig(out1); plt.close()

# -----------------------------
# Plot 2: Onset-Trajektorien nach Re-OP (0/1)
# -----------------------------
# Saubere Labels
df["is_reop_label"] = df["is_reop"].map({0: "Erst-OP", 1: "Re-OP"}).fillna("Erst-OP")

ct_reop = (df
           .groupby(["AKI_onset_group", "is_reop_label"])
           .size()
           .reset_index(name="n"))

# Pivot für gruppierte Balken
pv = ct_reop.pivot(index="AKI_onset_group", columns="is_reop_label", values="n").reindex(ONSET_ORDER)
pv = pv.fillna(0).astype(int)
pv = pv.reindex(columns=["Erst-OP", "Re-OP"])  # feste Reihenfolge

ax = pv.plot(kind="bar", figsize=(8, 4.5))
ax.set_xlabel("AKI-Onset-Kategorie")
ax.set_ylabel("Anzahl Operationen")
ax.set_title("Trajektorien des AKI-Onsets nach Re-OP-Status")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", padding=2, fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.legend(title="OP-Typ", frameon=False)
out2 = os.path.join(OUTDIR, "Traj_Onset_by_ReOP.png")
plt.savefig(out2); plt.close()

# -----------------------------
# Plot 3: Onset-Trajektorien nach OP-Dauer (Tertile)
# -----------------------------
ct_dur = (df
          .groupby(["AKI_onset_group", "duration_tertile"])
          .size()
          .reset_index(name="n"))

pv2 = ct_dur.pivot(index="AKI_onset_group", columns="duration_tertile", values="n").reindex(ONSET_ORDER)
pv2 = pv2.reindex(columns=TERTILE_LABELS).fillna(0).astype(int)

ax = pv2.plot(kind="bar", figsize=(8, 4.5))
ax.set_xlabel("AKI-Onset-Kategorie")
ax.set_ylabel("Anzahl Operationen")
ax.set_title("Trajektorien des AKI-Onsets nach OP-Dauer (Tertile)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", padding=2, fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.legend(title="OP-Dauer", frameon=False)
out3 = os.path.join(OUTDIR, "Traj_Onset_by_DurationTertile.png")
plt.savefig(out3); plt.close()

# -----------------------------
# (Optional) Plot 4: Sankey (OP-Ende → Onset-Kategorie)
# -----------------------------
# Ohne Zusatzabhängigkeiten lauffähig? Wir versuchen plotly, sonst überspringen.
try:
    import plotly.graph_objects as go

    labels = ["OP-Ende"] + ONSET_ORDER
    source = [0, 0, 0, 0]
    target = [1, 2, 3, 4]
    values = counts.values.tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=18, thickness=16),
        link=dict(source=source, target=target, value=values)
    )])
    fig.update_layout(title_text="Trajektorien: OP-Ende → AKI-Onset (0–7 d)", font_size=12)
    out4_html = os.path.join(OUTDIR, "Traj_Sankey_Onset.html")
    fig.write_html(out4_html)
    print(f"Sankey gespeichert: {out4_html}")
except Exception as e:
    print(f"Sankey übersprungen (plotly nicht verfügbar oder Fehler): {e}")

# -----------------------------
# Tabellarische Zusammenfassungen speichern
# -----------------------------
counts.to_csv(os.path.join(OUTDIR, "Traj_Onset_counts.csv"), header=["n"])
pv.to_csv(os.path.join(OUTDIR, "Traj_Onset_by_ReOP_table.csv"))
pv2.to_csv(os.path.join(OUTDIR, "Traj_Onset_by_DurationTertile_table.csv"))

print("Fertig.")
print(f"- {out1}")
print(f"- {out2}")
print(f"- {out3}")

