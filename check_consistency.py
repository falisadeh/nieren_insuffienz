#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_consistency.py
Vergleicht Mastertabelle (Audit/master_table_lab_vis.csv)
mit der Deskriptivtabelle (Audit/ehrapy_deskriptiv_median_iqr.csv).

Zeigt nur Abweichungen >0.01 an (relevante Unterschiede).
"""

import pandas as pd
from pathlib import Path

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Audit")

MASTER = BASE / "master_table_lab_vis.csv"
DESC   = BASE / "ehrapy_deskriptiv_median_iqr.csv"

# ---------------- Dateien laden ----------------
df_master = pd.read_csv(MASTER)
df_desc   = pd.read_csv(DESC)

# Einheitliche Namen
df_master["Variable"] = df_master["Variable"].str.strip()
df_desc["Variable"]   = df_desc["Variable"].str.strip()

# Nur die relevanten Spalten für Median-Vergleich
cols_master = ["Variable", "Median_AKI0", "Median_AKI1"]
cols_desc   = ["Variable", "Median_AKI0", "Median_AKI1"]

df_master_sub = df_master[cols_master].set_index("Variable")
df_desc_sub   = df_desc[cols_desc].set_index("Variable")

# Merge
merged = df_master_sub.join(df_desc_sub, lsuffix="_master", rsuffix="_desc", how="inner")

# Abweichungen berechnen
merged["diff_AKI0"] = (merged["Median_AKI0_master"] - merged["Median_AKI0_desc"]).abs()
merged["diff_AKI1"] = (merged["Median_AKI1_master"] - merged["Median_AKI1_desc"]).abs()

# Nur auffällige Zeilen (größer 0.01)
threshold = 0.01
diffs = merged[(merged["diff_AKI0"] > threshold) | (merged["diff_AKI1"] > threshold)]

print(f"✅ Gesamtvariablen: {len(merged)}")
if diffs.empty:
    print(f"Alle Medians stimmen überein (keine Abweichung > {threshold}) ✅")
else:
    print(f"⚠️ Abweichungen > {threshold} gefunden:")
    print(diffs[["Median_AKI0_master","Median_AKI0_desc","diff_AKI0",
                 "Median_AKI1_master","Median_AKI1_desc","diff_AKI1"]])

# Ergebnisse speichern
out_path = BASE / "consistency_check.csv"
merged.to_csv(out_path)
print(f"Detail-Ergebnisse gespeichert: {out_path}")
