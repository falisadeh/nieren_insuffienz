#%%
import pandas as pd
import ehrapy as ep
import os
print("CWD:", os.getcwd())
print("Dateien im Ordner:", os.listdir())
#%%


# 1. CSV-Dateien einlesen
df_aki = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/AKI Label.csv", sep=";")
print("df_aki geladen")
df_op = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/HLM Operationen.csv", sep=";")
print("df_op geladen")
df_patient = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv", sep=";")
print("df_patient geladen")
df_supp = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Procedure Supplement.csv", sep=";")  
print("df_supp geladen")

# erst wenn alles klappt:
# for df in (df_op, df_patient, df_aki, df_supp):
#     df.columns = df.columns.str.strip()

print("HLM Operationen:", df_op.shape)
print("Patientendaten:", df_patient.shape)
print("AKI Label:", df_aki.shape)
print("Supplement:", df_supp.shape)

# Datumsspalten in datetime konvertieren
df_op["Start of surgery"] = pd.to_datetime(df_op["Start of surgery"])
df_patient["DateOfBirth"] = pd.to_datetime(df_patient["DateOfBirth"])

# OP-Tabelle mit Patiententabelle mergen (über PMID)
df_merged = df_op.merge(df_patient[["PMID", "DateOfBirth", "Sex"]], on="PMID", how="left")

# Alter bei OP berechnen (in Tagen)
df_merged["Age_at_OP"] = (df_merged["Start of surgery"] - df_merged["DateOfBirth"]).dt.days

# Ausgabe zur Kontrolle
print(df_merged[["PMID", "Start of surgery", "DateOfBirth", "Age_at_OP"]].head())

import matplotlib.pyplot as plt

# Funktion zum Einteilen der Altersklassen
def alter_kategorie(tage):
    if tage < 7:
        return "< 1 Woche"
    elif tage < 31:
        return "1–4 Wochen"
    elif tage < 366:
        return "1–12 Monate"
    elif tage < 1826:
        return "1–5 Jahre"
    else:
        return "> 5 Jahre"

# Neue Spalte: AgeGroup
df_merged["AgeGroup"] = df_merged["Age_at_OP"].apply(alter_kategorie)

# Häufigkeiten berechnen
# Feste Reihenfolge für die Altersgruppen
kategorien = ["< 1 Woche", "1–4 Wochen", "1–12 Monate", "1–5 Jahre", "> 5 Jahre"]

# Zählen und sortieren
age_counts = df_merged["AgeGroup"].value_counts().reindex(kategorien)


# Plot
plt.figure(figsize=(8, 5))
age_counts.plot(kind="bar")
plt.title("Verteilung der Altersgruppen bei erster OP")
plt.xlabel("Altersgruppe")
plt.ylabel("Anzahl Operationen")
plt.tight_layout()
plt.show()


