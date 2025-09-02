# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pfade
BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
CSV_CLEAN = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Daten/ehrapy_input_clean.csv"
os.makedirs(BASE, exist_ok=True)

# Daten laden
df = pd.read_csv(CSV_CLEAN)

# AKI_Stufe sauber numerisch (1..3 behalten)
# --- robuste Klassifikation: kein AKI vs. AKI 1-3 ---
df["AKI"] = pd.to_numeric(df.get("AKI"), errors="coerce")
df["AKI_Stufe"] = pd.to_numeric(df.get("AKI_Stufe"), errors="coerce")


def classify(row):
    s = row["AKI_Stufe"]
    a = row["AKI"]
    if s in [1, 2, 3]:
        return f"AKI {int(s)}"
    # kein AKI, wenn AKI==0 ODER (Stage fehlt und AKI fehlt/0)
    if (a == 0) or (pd.isna(s) and (pd.isna(a) or a == 0)):
        return "kein AKI"
    # alles andere: unklare Fälle (z. B. AKI=1 aber Stage fehlt)
    return "unbekannt"


df["AKI_cat"] = df.apply(classify, axis=1)

order = ["kein AKI", "AKI 1", "AKI 2", "AKI 3", "unbekannt"]
counts_all = df["AKI_cat"].value_counts().reindex(order, fill_value=0)

# Für die Grafik i. d. R. "unbekannt" ausblenden
counts = counts_all.drop("unbekannt")
labels = counts.index.tolist()

# Farben (Grau für kein AKI, dann Grün/Orange/Rot)
farben = ["#32CD32", "#FFD700", "#FFA500", "#FF0000"][: len(counts)]

plt.figure(figsize=(6, 6))
plt.pie(
    counts.values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=farben,
)
plt.title("Verteilung von AKI (kein AKI vs. AKI-Stadien 1–3)")
plt.tight_layout()
outpath = os.path.join(BASE, "AKI_stages_with_noAKI_pie.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close()

# Tabellen ausgeben + Sanity-Check
counts.to_csv(os.path.join(BASE, "AKI_stages_with_noAKI_counts.csv"))
print("Gesamtfälle:", len(df))
print(counts_all)  # inkl. 'unbekannt'
print("Geplottet (ohne 'unbekannt'):", counts.sum())
print("Gespeichert:", outpath)
