# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ehrapy as ep

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5 = f"{BASE}/h5ad/ops_with_patient_features.h5ad"
OUT = f"{BASE}/Diagramme"
os.makedirs(OUT, exist_ok=True)

adata = ep.io.read_h5ad(H5)
obs = adata.obs.copy()

# --- Alter in Jahren sicherstellen (falls versehentlich in Tagen) ---
age = pd.to_numeric(obs.get("age_years_at_first_op"), errors="coerce")
if age.dropna().max() and age.dropna().max() > 100:  # Heuristik: dann sind es Tage
    age = age / 365.25
obs["age_years_at_first_op"] = age

# --- patientenbasiert: pro PMID genau eine Zeile (Alter bei ERSTER OP) ---
pat = obs[["PMID", "age_years_at_first_op"]].dropna().drop_duplicates("PMID").copy()


# Einteilungsfunktionen (sauber inkl. Grenzen)
def group4(a):
    if pd.isna(a):
        return np.nan
    if a <= 1:
        return "0–1 Jahr"
    if a <= 5:
        return "1–5 Jahre"
    if a <= 12:
        return "6–12 Jahre"
    if a <= 18:
        return "13–18 Jahre"
    return np.nan


neonate_y = 28 / 365.25


def group6(a):
    if pd.isna(a):
        return np.nan
    if a <= neonate_y:
        return "Neonates (0–28 T.)"
    if a <= 1:
        return "Infants (1–12 Mon.)"
    if a <= 3:
        return "Toddlers (1–3 J.)"
    if a <= 5:
        return "Preschool (3–5 J.)"
    if a <= 12:
        return "School-age (6–12 J.)"
    if a <= 18:
        return "Adolescents (13–18 J.)"
    return np.nan


order4 = ["0–1 Jahr", "1–5 Jahre", "6–12 Jahre", "13–18 Jahre"]
order6 = [
    "Neonates (0–28 T.)",
    "Infants (1–12 Mon.)",
    "Toddlers (1–3 J.)",
    "Preschool (3–5 J.)",
    "School-age (6–12 J.)",
    "Adolescents (13–18 J.)",
]

# ---- Patientenbasiert (korrekt für Präsentation) ----
pat["age_group4"] = [group4(x) for x in pat["age_years_at_first_op"]]
pat["age_group6"] = [group6(x) for x in pat["age_years_at_first_op"]]

c4_pat = pat["age_group4"].value_counts().reindex(order4).fillna(0).astype(int)
p4_pat = (c4_pat / c4_pat.sum() * 100).round(1)

c6_pat = pat["age_group6"].value_counts().reindex(order6).fillna(0).astype(int)
p6_pat = (c6_pat / c6_pat.sum() * 100).round(1)

print("Patientenbasiert 4er:", p4_pat.to_dict())
print(
    "Patientenbasiert 6er:",
    p6_pat.to_dict(),
    "Summe ≤1 Jahr =",
    float(p6_pat["Neonates (0–28 T.)"] + p6_pat["Infants (1–12 Mon.)"]),
    "%",
)

# ---- Episodenbasiert (nur zur Kontrolle/Erklärung der Abweichung) ----
epi = obs[["PMID", "age_years_at_first_op"]].dropna().copy()
epi["age_group4"] = [group4(x) for x in epi["age_years_at_first_op"]]
c4_epi = epi["age_group4"].value_counts().reindex(order4).fillna(0).astype(int)
p4_epi = (c4_epi / c4_epi.sum() * 100).round(1)
print("Episodenbasiert 4er:", p4_epi.to_dict())

# ---- Plot: Patientenbasiert, 6-Gruppen (pädiatrisch) ----
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    c6_pat.values, labels=order6, autopct="%1.1f%%", pctdistance=0.8, startangle=90
)
ax.set_title("Altersgruppen (pädiatrisch) – Anteil (patientenbasiert)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "AgeGroups_pediatric_counts_pie_PATIENT.png"), dpi=300)
plt.close()

# ---- Plot: Patientenbasiert, 4-Gruppen (Vergleich zum alten Chart) ----
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(c4_pat.values, labels=order4, autopct="%1.1f%%", startangle=90)
ax.set_title("Anteil der Patienten pro Altersgruppe (patientenbasiert)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "AgeGroups_4groups_pie_PATIENT.png"), dpi=300)
plt.close()
