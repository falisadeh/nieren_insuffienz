#%%
import pandas as pd
file_path = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv'
df = pd.read_csv(file_path)

# Spaltennamen ausgeben
print("Spaltennamen (original):")
for i, col in enumerate(df.columns):
    print(f"{i+1}. '{col}'")
import pandas as pd

file_path = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv'

# Einlesen mit Semikolon als Trennzeichen
df = pd.read_csv(file_path, sep=";")

# Spalten bereinigen
df.columns = df.columns.str.strip()

# Vorschau
print(" Spalten:", df.columns.tolist())
print(" Anzahl eindeutiger Patienten:", df['PMID'].nunique())
print("\n Geschlechterverteilung:")
print(df['Sex'].value_counts())
#%%
import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
file_path = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv'
df = pd.read_csv(file_path, sep=';')
df.columns = df.columns.str.strip()

# Geschlechterverteilung zählen
gender_counts = df['Sex'].value_counts()

# Balkendiagramm erstellen
plt.figure(figsize=(6, 4))
gender_counts.plot(kind='bar')
plt.title('Geschlechterverteilung der Patienten')
plt.xlabel('Geschlecht')
plt.ylabel('Anzahl')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
#%%
# Stelle sicher, dass DateOfBirth als Datum erkannt wird
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')

# Geburtsjahr extrahieren
df['Geburtsjahr'] = df['DateOfBirth'].dt.year

# Histogramm
plt.figure(figsize=(8, 4))
df['Geburtsjahr'].value_counts().sort_index().plot(kind='bar')
plt.title('Verteilung der Geburtsjahre')
plt.xlabel('Geburtsjahr')
plt.ylabel('Anzahl Patienten')
plt.tight_layout()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Datei einlesen (falls nicht bereits geschehen)
file_path = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv'
df = pd.read_csv(file_path, sep=';')
df.columns = df.columns.str.strip()

# Datum in datetime-Format umwandeln
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')

# Geburtsjahr extrahieren
df['Geburtsjahr'] = df['DateOfBirth'].dt.year

# Nach Jahr sortieren und zählen
jahr_verteilung = df['Geburtsjahr'].value_counts().sort_index()

# Diagramm anzeigen
plt.figure(figsize=(10, 4))
jahr_verteilung.plot(kind='bar')
plt.title('Verteilung der Geburtsjahre')
plt.xlabel('Geburtsjahr')
plt.ylabel('Anzahl Patienten')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt
# 1. Einlesen beider Dateien
path_patient = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv'
path_op = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/HLM Operationen.csv'

df_patient = pd.read_csv(path_patient, sep=';')
df_op = pd.read_csv(path_op, sep=';')

# 2. Spaltennamen bereinigen
df_patient.columns = df_patient.columns.str.strip()
df_op.columns = df_op.columns.str.strip()

# 3. Datumsfelder konvertieren
df_patient['DateOfBirth'] = pd.to_datetime(df_patient['DateOfBirth'], errors='coerce', dayfirst=True)
df_op['Start of surgery'] = pd.to_datetime(df_op['Start of surgery'], errors='coerce')

# 4. Tabellen über PMID zusammenführen
df_merged = pd.merge(df_op, df_patient[['PMID', 'DateOfBirth']], on='PMID', how='left')

# 5. Alter in Tagen berechnen
df_merged['Alter_in_Tagen'] = (df_merged['Start of surgery'] - df_merged['DateOfBirth']).dt.days

# 6. Vorschau
print(df_merged[['PMID', 'Start of surgery', 'DateOfBirth', 'Alter_in_Tagen']].head())
plt.figure(figsize=(10, 5))
plt.hist(df_merged['Alter_in_Tagen'], bins=50, edgecolor='black')
plt.title('Altersverteilung der Patienten zum OP-Zeitpunkt (in Tagen)')
plt.xlabel('Alter in Tagen')
plt.ylabel('Anzahl Operationen')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np


# Alter in Monaten berechnen (wenn Alter_in_Tagen schon existiert!)
df_merged['Alter_in_Monaten'] = (df_merged['Alter_in_Tagen'] / 30.44).round().astype('Int64')
# Alter in Monaten
alter_monate = df_merged['Alter_in_Monaten'].dropna()

# Feste Klassen (alle 6 Monate bis 18 Jahre = 216 Monate)
bins = np.arange(0, 220, 6)

plt.figure(figsize=(12, 5))
plt.hist(alter_monate, bins=bins, edgecolor='black')
plt.title('Altersverteilung der Patienten zum OP-Zeitpunkt (in Monaten, 6-Monats-Gruppen)')
plt.xlabel('Alter in Monaten')
plt.ylabel('Anzahl Operationen')
plt.xticks(bins, rotation=45)
plt.tight_layout()
plt.show()
#%% Alter in Jahren berechnen
df_merged['Alter_in_Jahren'] = (df_merged['Alter_in_Tagen'] / 365.25).round().astype('Int64')

# Plot: Histogramm nach Alter in Jahren
bins = range(0, 20 + 1)  # 0 bis 20 Jahre
plt.figure(figsize=(10, 5))
plt.hist(df_merged['Alter_in_Jahren'].dropna(), bins=bins, edgecolor='black')
plt.title('Altersverteilung der Patienten zum OP-Zeitpunkt (in Jahren)')
plt.xlabel('Alter in Jahren')
plt.ylabel('Anzahl Operationen')
plt.xticks(bins)
plt.tight_layout()
plt.show()
#%% Altersgruppen zuweisen
def gruppiere_alter(jahre):
    if jahre < 1:
        return '0–1 Jahr'
    elif jahre < 6:
        return '1–5 Jahre'
    elif jahre < 13:
        return '6–12 Jahre'
    elif jahre < 19:
        return '13–18 Jahre'
    else:
        return '≥19 Jahre'

df_merged['Altersgruppe'] = df_merged['Alter_in_Jahren'].apply(gruppiere_alter)

# Verteilung zählen
verteilung = df_merged['Altersgruppe'].value_counts().sort_index()

# Kreisdiagramm plotten
plt.figure(figsize=(7, 7))
plt.pie(verteilung, labels=verteilung.index, autopct='%1.1f%%', startangle=90)
plt.title('Anteil der Patienten pro Altersgruppe')
plt.axis('equal')
plt.show()
# %%
df_merged['OP_Jahr'] = df_merged['Start of surgery'].dt.year
op_counts = df_merged['OP_Jahr'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
op_counts.plot(kind='bar')
plt.title('Anzahl Operationen pro Jahr')
plt.xlabel('Jahr')
plt.ylabel('Anzahl Operationen')
plt.tight_layout()
plt.show()
# Sicherstellen, dass 'Start of surgery' ein Datum ist
df_merged['Start of surgery'] = pd.to_datetime(df_merged['Start of surgery'], errors='coerce')

# Basisjahr (z. B. erstes Jahr der Studie)
start_jahr = df_merged['Start of surgery'].dt.year.min()

# Neues Feld 'Studienjahr' hinzufügen
df_merged['Studienjahr'] = df_merged['Start of surgery'].dt.year - start_jahr + 1
import matplotlib.pyplot as plt

# Operationen pro Studienjahr zählen
op_pro_jahr = df_merged['Studienjahr'].value_counts().sort_index()

# Balkendiagramm
plt.figure(figsize=(10, 5))
op_pro_jahr.plot(kind='bar')
plt.title('Anzahl Operationen pro Studienjahr')
plt.xlabel('Studienjahr')
plt.ylabel('Anzahl Operationen')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# Liniendiagramm
plt.figure(figsize=(10, 5))
op_pro_jahr.plot(kind='line', marker= 'o')
plt.title('Anzahl Operationen pro Studienjahr')
plt.xlabel('Studienjahr')
plt.ylabel('Anzahl Operationen')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
import pandas as pd

# Vorab sicherstellen: df_op und df_patient existieren bereits und sind korrekt geladen
# Datumsspalten in datetime umwandeln
df_op['Start of surgery'] = pd.to_datetime(df_op['Start of surgery'], errors='coerce')
df_patient['DateOfDie'] = pd.to_datetime(df_patient['DateOfDie'], errors='coerce')

# Relevante Spalten auswählen und mergen
df_sterbeanalyse = df_op[['PMID', 'Start of surgery']].merge(
    df_patient[['PMID', 'DateOfDie']],
    on='PMID',
    how='left'
)

# Berechnung der Differenz in Tagen
df_sterbeanalyse['Tage_bis_Tod'] = (df_sterbeanalyse['DateOfDie'] - df_sterbeanalyse['Start of surgery']).dt.days

# Nur Patienten mit Sterbedatum (also nicht NULL)
df_verstorbene = df_sterbeanalyse[df_sterbeanalyse['DateOfDie'].notna()]

# Überblick: wie viele Patienten überhaupt gestorben sind
print(" Gesamtzahl verstorbener Patienten:", df_verstorbene['PMID'].nunique())

# Wie viele innerhalb von 30 Tagen nach OP?
df_30tage = df_verstorbene[df_verstorbene['Tage_bis_Tod'].between(0, 30)]
print("Verstorbene innerhalb 30 Tage nach OP:", df_30tage.shape[0])

# Wie viele innerhalb von 7 Tagen?
df_7tage = df_verstorbene[df_verstorbene['Tage_bis_Tod'].between(0, 7)]
print(" Verstorbene innerhalb 7 Tage nach OP:", df_7tage.shape[0])

# Optional: Histogramm der Tage bis zum Tod
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
df_verstorbene['Tage_bis_Tod'].dropna().hist(bins=30)
plt.title("Tage zwischen OP und Tod")
plt.xlabel("Tage")
plt.ylabel("Anzahl Patienten")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt

pf_aki = '/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/AKI Label.csv'
df_aki = pd.read_csv(pf_aki, sep=';', encoding='utf-8')  
# Optional: Spalten anzeigen
print(df_aki.columns)
print(df_aki['Decision'].value_counts(dropna=False))
# AKI-Spalte umbenennen zur Klarheit
df_aki.rename(columns={"Decision": "AKI_Stadium"}, inplace=True)

# AKI-Daten an die Patienten mit Sterbedatum anhängen
df_verstorbene_aki = df_verstorbene.merge(df_aki[['PMID', 'AKI_Stadium']], on='PMID', how='left')

#Duplikate aus df_verstorbene entfernen (pro Patient nur eine Zeile)
df_verstorbene_unique = df_verstorbene.drop_duplicates(subset='PMID')

# Merge mit AKI-Daten
df_verstorbene_aki = df_verstorbene_unique.merge(df_aki[['PMID', 'AKI_Stadium']], on='PMID', how='left')

# Ergebnis anzeigen
print("AKI-Verteilung bei verstorbenen Patienten:")
print(df_verstorbene_aki['AKI_Stadium'].value_counts(dropna=False))
# AKI-Stadium als Schweregrad kodieren
aki_stufen = {'AKI 1': 1, 'AKI 2': 2, 'AKI 3': 3}
df_aki['AKI_Score'] = df_aki['AKI_Stadium'].map(aki_stufen)

# Maximaler AKI-Score pro Patient
df_aki_max = df_aki.groupby("PMID", as_index=False)['AKI_Score'].max()

# Merge mit eindeutigen verstorbenen Patienten
df_verstorbene_unique = df_verstorbene.drop_duplicates(subset='PMID')
df_verstorbene_aki = df_verstorbene_unique.merge(df_aki_max, on="PMID", how="left")

# Patienten mit dokumentiertem AKI
anzahl_aki = df_verstorbene_aki['AKI_Score'].notna().sum()
anzahl_verstorben = len(df_verstorbene_aki)

print(f"Verstorbene Patienten: {anzahl_verstorben}")
print(f"Davon mit AKI: {anzahl_aki} ({anzahl_aki / anzahl_verstorben:.0%})")
# AKI-Schweregrad zu Score (numerisch)
aki_score_map = {"AKI 1": 1, "AKI 2": 2, "AKI 3": 3}
df_aki['AKI_Score'] = df_aki['AKI_Stadium'].map(aki_score_map)

# Höchster Score pro Patient (PMID)
df_aki_max = df_aki.groupby("PMID", as_index=False)['AKI_Score'].max()

# Zurückmappen in Stadium
score_to_stage = {1: "AKI 1", 2: "AKI 2", 3: "AKI 3"}
df_aki_max["AKI_Max_Stadium"] = df_aki_max['AKI_Score'].map(score_to_stage)

# Ergebnis
print(df_aki_max.head())
print("Anzahl eindeutiger Patienten mit AKI:", df_aki_max['PMID'].nunique())
df_aki_max.sort_values('AKI_Score', ascending=False)
# Schritt: Häufigkeit der AKI-Maximalstadien
print(df_aki_max['AKI_Max_Stadium'].value_counts())
# Prüfen ob Pat dopellt gerechnet ist:
print("Anzahl eindeutiger PMIDs in df_aki_max:", df_aki_max['PMID'].nunique())
print("Gesamtsumme aller AKI-Stadium-Werte:", df_aki_max['AKI_Max_Stadium'].value_counts().sum())


# %%
import pandas as pd
import matplotlib.pyplot as plt

# AKI-Verteilung
aki_counts = pd.Series({
    "AKI 1": 334,
    "AKI 2": 140,
    "AKI 3": 73
})

# Balkendiagramm
plt.figure(figsize=(8, 5))
aki_counts.plot(kind='bar')
plt.title("Anzahl Patienten pro AKI-Maximalstadium")
plt.xlabel("AKI-Stadium")
plt.ylabel("Anzahl Patienten")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme")
plt.show()

# Kreisdiagramm
plt.figure(figsize=(6, 6))
aki_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title("Anteil der AKI-Stadien bei betroffenen Patienten")
plt.ylabel('')
plt.tight_layout()
plt.savefig("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme")


# %%
import pandas as pd
import matplotlib.pyplot as plt

# AKI-Tabelle einlesen 
df_aki = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/AKI Label.csv", sep=";")

# Nur Zeilen mit validem AKI-Stadium verwenden
df_aki = df_aki[df_aki["Decision"].notna()]
print("Spaltennamen in df_aki:", df_aki.columns.tolist())

# AKI-Stadium in numerischen Score umwandeln
aki_score_map = {"AKI 1": 1, "AKI 2": 2, "AKI 3": 3}
df_aki['AKI_Score'] = df_aki['Decision'].map(aki_score_map)

# Maximaler Score je Patient
df_aki_max = df_aki.groupby("PMID", as_index=False)['AKI_Score'].max()

# Score wieder zurück in Stadium umwandeln
score_to_stage = {1: "AKI 1", 2: "AKI 2", 3: "AKI 3"}
df_aki_max['AKI_Max_Stadium'] = df_aki_max['AKI_Score'].map(score_to_stage)

# Häufigkeit zählen
aki_counts = df_aki_max['AKI_Max_Stadium'].value_counts().sort_index()

# Neue Tabelle mit Prozenten
df_aki_statistik = aki_counts.reset_index()
df_aki_statistik.columns = ['AKI-Max_Stadium', 'Anzahl Patienten']
gesamt = df_aki_statistik["Anzahl Patienten"].sum()
df_aki_statistik["Prozent"] = (df_aki_statistik["Anzahl Patienten"] / gesamt * 100).round(1).astype(str) + "%"

# Ausgabe
print(df_aki_statistik)

# Diagrammpfad
diagramm_pfad = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"

# Balkendiagramm
plt.figure(figsize=(8, 5))
aki_counts.plot(kind='bar')
plt.title("Anzahl Patienten pro AKI-Maximalstadium")
plt.xlabel("AKI-Stadium")
plt.ylabel("Anzahl Patienten")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{diagramm_pfad}/AKI_Stadium_Balken.png")
plt.show()

# Kreisdiagramm
plt.figure(figsize=(6, 6))
aki_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title("Anteil der AKI-Stadien bei betroffenen Patienten")
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{diagramm_pfad}/AKI_Stadium_Kreis.png")
plt.show()



# %%
