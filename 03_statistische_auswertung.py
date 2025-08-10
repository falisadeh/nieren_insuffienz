#%%
import pandas as pd
import ehrapy as ep
import os
print("CWD:", os.getcwd())
print("Dateien im Ordner:", os.listdir())


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
print("OP-Tabelle:", df_op.columns.tolist())
print("AKI-Tabelle:", df_aki.columns.tolist())

print("HLM Operationen:", df_op.shape)
print("Patientendaten:", df_patient.shape)
print("AKI Label:", df_aki.shape)
print("Supplement:", df_supp.shape)
#%%
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

# Häufigkeit der Geschlechter zählen
gender_counts = df_merged["Sex"].value_counts()

# Plot
plt.figure(figsize=(5, 4))
gender_counts.plot(kind="bar")
plt.title("Geschlechterverteilung")
plt.xlabel("Geschlecht")
plt.ylabel("Anzahl Operationen")
plt.xticks(rotation=0)
plt.tight_layout()
# Balkenfarben: männlich = blau, weiblich = pink
farben = {"m": "blue", "f": "hotpink"}

# Plot mit eigenen Farben
gender_counts.plot(kind="bar", color=[farben.get(g, "gray") for g in gender_counts.index])

plt.show()

# Ausgabe im Terminal
print(gender_counts)
# OP-Anzahl pro Patient zählen
op_counts = df_merged["PMID"].value_counts().rename("HLM_OP_Anzahl")

# Als DataFrame
op_counts = op_counts.reset_index().rename(columns={"index": "PMID"})

# Ausgabe zur Kontrolle
print(op_counts.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Violinplot: Alter bei OP vs. Geschlecht
sns.violinplot(data=df_merged, x="Sex", y="Age_at_OP", palette={"m": "blue", "f": "hotpink"})
plt.title("Verteilung des Alters bei OP nach Geschlecht")
plt.xlabel("Geschlecht")
plt.ylabel("Alter bei OP (Tage)")
plt.tight_layout()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
# Schritt 1: AKI Label vorbereiten
df_aki["Start"] = pd.to_datetime(df_aki["Start"])
df_aki["End"] = pd.to_datetime(df_aki["End"])

# Schritt 2: AKI ja/nein und AKI-Stufe definieren
df_aki["AKI"] = df_aki["Decision"].notna().astype(int)

# AKI-Stufe extrahieren aus z. B. "AKI 2"
def extract_aki_stufe(decision):
    if pd.isna(decision):
        return None
    parts = decision.split()
    return int(parts[-1]) if parts[-1].isdigit() else None

df_aki["AKI_Stufe"] = df_aki["Decision"].apply(extract_aki_stufe)

# Schritt 3: Pro Patient den AKI-Status aggregieren (höchste Stufe zählt)
aki_status = df_aki.groupby("PMID").agg({
    "AKI": "max",
    "AKI_Stufe": "max"
}).reset_index()

# Schritt 4: AKI-Daten mit df_merged verbinden
df_merged = df_merged.merge(aki_status, on="PMID", how="left")

# Fehlende AKI-Werte auf 0 setzen (für Patienten ohne AKI)
df_merged["AKI"] = df_merged["AKI"].fillna(0).astype(int)
sns.violinplot(data=df_merged, x="AKI", y="Age_at_OP", palette="Set2")
plt.title("Alter bei OP nach AKI-Zustand")
plt.xlabel("AKI (0 = nein, 1 = ja)")
plt.ylabel("Alter bei OP (Tage)")
plt.tight_layout()
plt.show()

# Schritt 5: Altersgruppen in df_merged hinzufügen

from anndata import AnnData

# DataFrame in AnnData-Objekt umwandeln
adata = AnnData(df_merged)
import ehrapy as ep
import matplotlib.pyplot as plt

# Fehlende Werte pro Spalte zählen
missing_per_column = df_merged.isnull().sum()

# Nur Spalten mit fehlenden Werten anzeigen
missing_per_column = missing_per_column[missing_per_column > 0]

# Plot
missing_per_column.sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
plt.title("Fehlende Werte pro Merkmal (Spalte)")
plt.ylabel("Anzahl fehlender Werte")
plt.tight_layout()
plt.show()
# Fehlende Werte pro Patient/Zeile
missing_per_row = df_merged.isnull().sum(axis=1)
#Fehlende Werte pro Zeile (Beobachtung = „obs“):
# Histogramm
plt.figure(figsize=(8, 4))
plt.hist(missing_per_row, bins=30)
plt.title("Verteilung der fehlenden Werte pro Beobachtung")
plt.xlabel("Anzahl fehlender Werte pro Zeile")
plt.ylabel("Anzahl Zeilen")
plt.tight_layout()
plt.show()
total_cells = df_merged.size
missing_total = df_merged.isnull().sum().sum()
print(f"Gesamt fehlende Werte: {missing_total} ({missing_total / total_cells:.2%})")

# Python/Pandas zum Entfernen:
# Falls  die Spalte loswerden soll??????
#df = df.drop(columns=['Tx?'])

#Oder: prüfen , ob wirklich überall 0 steht
#print(df['Tx?'].value_counts())
#%%

import pandas as pd
import matplotlib.pyplot as plt

# 1. Einlesen & Spalten bereinigen
df_op  = pd.read_csv("HLM Operationen.csv", sep=";", parse_dates=['Start of surgery'])
df_aki = pd.read_csv("AKI Label.csv",     sep=";", parse_dates=['Start'])
for df in (df_op, df_aki):
    df.columns = df.columns.str.strip()

# 2. Erstes AKI-Datum pro Patient
df_aki_kurz = (
    df_aki[['PMID','Start']]
      .rename(columns={'Start':'AKI_Start'})
      .sort_values('AKI_Start')
      .groupby('PMID', as_index=False)
      .first()
)
print("1) Unique AKI-Patienten (frühestes Datum):", df_aki_kurz['PMID'].nunique())
# → sollte ~ 547 sein, genau wie dein Label

# 3. Erstes OP-Datum pro Patient
first_op = (
    df_op.sort_values('Start of surgery')
         .groupby('PMID', as_index=False)['Start of surgery']
         .first()
         .rename(columns={'Start of surgery':'First_OP'})
)
print("2) Unique OP-Patienten insgesamt:", first_op['PMID'].nunique())

# 4. Merge auf Patienten-Ebene
df_patient = pd.merge(first_op, df_aki_kurz, on='PMID', how='inner')
print("3) Nach Merge rows:", df_patient.shape[0])
print("   … unique Patienten:", df_patient['PMID'].nunique())
print("   … Duplikate (sollte 0 sein):", (df_patient.groupby('PMID').size()>1).sum())

# 5. days_to_AKI berechnen
df_patient['days_to_AKI'] = (df_patient['AKI_Start'] - df_patient['First_OP']).dt.days
### 1) arbeite auf einer Kopie
df_all = df_patient.copy()

# 2) wende den Filter auf eine neue Variable an
df_filtered = df_all[(df_all['days_to_AKI'] >= 0) & (df_all['days_to_AKI'] <= 30)]


# 6. Auf 0–30 Tage filtern
pre_filter = df_patient.shape[0]
df_patient = df_patient[(df_patient['days_to_AKI']>=0)&(df_patient['days_to_AKI']<=30)]
print(f"4) Vor Filter: {pre_filter} Zeilen, nach Filter: {df_patient.shape[0]} Zeilen")
print("   … unique Patienten nach Filter:", df_patient['PMID'].nunique())
print("Vor Filter:", df_all.shape[0], "Zeilen")
print("Nach Filter:", df_filtered.shape[0], "Zeilen")
#print("Outlier:", df_outliers.shape[0], "Zeilen")

# 7. Deskriptive Statistik & Histogramm
print(df_patient['days_to_AKI'].describe())

plt.figure(figsize=(6,4))
plt.hist(df_patient['days_to_AKI'], bins=range(0,31))
plt.xlim(0,30)
plt.xlabel('Tage bis AKI nach erster OP')
plt.ylabel('Anzahl der Patienten')
plt.title('Verteilung der Tage bis zum AKI-Beginn (0–30 Tage)')
plt.tight_layout()
plt.show()

# Zeige die Patienten, bei denen days_to_AKI < 0 oder > 30 (bzw. >7) ist:
ausreißer = df_patient[(df_patient['days_to_AKI'] < 0) | (df_patient['days_to_AKI'] > 7)]
print(ausreißer[['PMID','First_OP','AKI_Start','days_to_AKI']])



# %%
import pandas as pd

# 1. Einlesen mit deinem lokalen Pfad
df_op  = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/HLM Operationen.csv",
    sep=";", parse_dates=['Start of surgery']
)
df_aki = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/AKI Label.csv",
    sep=";", parse_dates=['Start']
)
for df in (df_op, df_aki):
    df.columns = df.columns.str.strip()

# 2. Erstes AKI und erste OP pro Patient
df_aki_kurz = (df_aki[['PMID','Start']]
               .rename(columns={'Start':'AKI_Start'})
               .sort_values('AKI_Start')
               .groupby('PMID', as_index=False)
               .first()
)
first_op = (df_op.sort_values('Start of surgery')
            .groupby('PMID', as_index=False)['Start of surgery']
            .first()
            .rename(columns={'Start of surgery':'First_OP'})
)

# 3. Merge und days_to_AKI
df_all = pd.merge(first_op, df_aki_kurz, on='PMID', how='inner')
df_all['days_to_AKI'] = (df_all['AKI_Start'] - df_all['First_OP']).dt.days

# 4. Outliers ermitteln (außerhalb 0–7 Tage)
mask = (df_all['days_to_AKI'] < 0) | (df_all['days_to_AKI'] > 7)
df_outliers = df_all[mask]

# 5. Anzeige in Jupyter/VS Code
#   a) Einfach mit print:
print(df_outliers[['PMID','First_OP','AKI_Start','days_to_AKI']])

#   b) als Table in Jupyter:
# Gibt alle Spalten und Zeilen formatiert als String aus
print(df_outliers.to_string(index=False))
# Wenn die Tabelle sehr groß ist, zeige  nur die ersten 10
print(df_outliers.head(10).to_string(index=False))
# Erzeugt eine Markdown-taugliche Tabelle, die etwa in Protokollen gut aussieht
print(df_outliers.to_markdown(index=False))
df_outliers.to_csv("outliers_days_to_AKI.csv", index=False, sep=";")
print("Outlier wurden in outliers_days_to_AKI.csv geschrieben.")
#Mit tabulate-Paket drucken:
from tabulate import tabulate
print(tabulate(df_outliers, headers="keys", tablefmt="psql", showindex=False))


# %%
