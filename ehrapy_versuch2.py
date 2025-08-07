#%%
# %%
import pandas as pd
import ehrapy as ep
import matplotlib.pyplot as plt

# 1. CSV-Dateien einlesen
df_aki = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/AKI Label.csv", sep=";")
df_hlm = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/HLM Operationen.csv", sep=";")
df_patient = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv", sep=";")
df_proc = pd.read_csv("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Procedure Supplement.csv", sep=";")  

# 2. Spalten bereinigen & Datumsformate umwandeln
df_patient.columns = df_patient.columns.str.strip()
df_patient["DateOfBirth"] = pd.to_datetime(df_patient["DateOfBirth"], errors="coerce")
df_hlm["Start of surgery"] = pd.to_datetime(df_hlm["Start of surgery"], errors="coerce")
df_aki["Decision"] = df_aki["Decision"].astype(str).str.strip()

# 3. Alter bei erster OP berechnen
df_first_op = df_hlm.sort_values("Start of surgery").groupby("PMID").first().reset_index()
df_age = pd.merge(df_patient, df_first_op[["PMID", "Start of surgery"]], on="PMID", how="left")
df_age["Age_at_OP"] = (df_age["Start of surgery"] - df_age["DateOfBirth"]).dt.days / 365.25
df_age["Age_at_OP"] = df_age["Age_at_OP"].round(1)

# 4. Altersgruppen-Kategorisierung
def alter_kategorie(alter):
    if pd.isna(alter):
        return "Unbekannt"
    elif alter < 1:
        return "<1 Jahr"
    elif alter < 2:
        return "1–2 Jahre"
    elif alter < 5:
        return "2–5 Jahre"
    elif alter < 10:
        return "5–10 Jahre"
    elif alter < 18:
        return "10–18 Jahre"
    else:
        return "≥18 Jahre"

df_age["AgeGroup"] = df_age["Age_at_OP"].apply(alter_kategorie)
df_age["AgeGroup"] = df_age["AgeGroup"].astype("category")

# 5. HLM-OP-Anzahl berechnen
op_count = df_hlm.groupby("PMID").size().reset_index(name="HLM_OP_Anzahl")

# 6. AKI-Werte berechnen
df_aki["AKI"] = df_aki["Decision"].apply(lambda x: 0 if "no aki" in x.lower() else 1)
df_aki["AKI_Stufe"] = df_aki["Decision"].apply(
    lambda x: 3 if "3" in x else (2 if "2" in x else (1 if "1" in x else 0))
)

# 7. Zusammenführen
merged = df_age[["PMID", "Sex", "AgeGroup", "Age_at_OP"]].merge(op_count, on="PMID", how="left")
merged = merged.merge(df_aki[["PMID", "AKI", "AKI_Stufe"]], on="PMID", how="left")

# 8. Gruppieren auf eine Zeile pro Patient
final_df = merged.groupby("PMID").agg({
    "Sex": "first",
    "AgeGroup": "first",
    "Age_at_OP": "first",
    "HLM_OP_Anzahl": "sum",
    "AKI": "max",
    "AKI_Stufe": "max"
}).reset_index()

# ... (Dein Code bis zur Erstellung von final_df bleibt unverändert) ...

# ... (Dein Code bis zur Erstellung von final_df bleibt unverändert) ...

# ... (Dein Code bis zur Erstellung von final_df bleibt unverändert) ...

# 9. Fehlende Werte füllen
final_df["AKI"] = final_df["AKI"].fillna(0).astype(int)
final_df["AKI_Stufe"] = final_df["AKI_Stufe"].fillna(0).astype(int)

# 10. ID entfernen
final_df = final_df.drop(columns=["PMID"])

# ---

# **Wichtige Korrektur:** Manuelle Konvertierung in AnnData
import anndata as ad

# Definiere die Spalten
obs_cols = ['Sex', 'AgeGroup']
var_cols = ['Age_at_OP', 'HLM_OP_Anzahl', 'AKI', 'AKI_Stufe']

# Erstelle ein AnnData-Objekt
# Die numerischen Spalten (`var_cols`) bilden die .X Matrix
adata = ad.AnnData(
    X=final_df[var_cols].values,
    obs=final_df[obs_cols],
    var=pd.DataFrame(index=var_cols)
)

# 11. Plot anzeigen
print("Spalten in .obs nach der Konvertierung:", adata.obs.columns)

if 'AgeGroup' in adata.obs.columns:
    adata.obs["AgeGroup"].value_counts().plot(kind="bar")
    plt.title("Verteilung der Altersgruppen bei erster OP")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Anzahl Patienten")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Fehler: 'AgeGroup' ist nicht in adata.obs vorhanden.")
# ... (dein bestehender Code bis zur Plot-Funktion) ...

# 12. Plot anzeigen
import matplotlib.pyplot as plt

if 'AgeGroup' in adata.obs.columns:
    # ⚠️ Wichtige Korrektur: Reihenfolge der Kategorien festlegen
    altersgruppen_reihenfolge = [
        "<1 Jahr",
        "1–2 Jahre",
        "2–5 Jahre",
        "5–10 Jahre",
        "10–18 Jahre",
        "≥18 Jahre",
        "Unbekannt"
    ]
    
    # Konvertiere 'AgeGroup' in eine geordnete Kategorie
    adata.obs["AgeGroup"] = pd.Categorical(
        adata.obs["AgeGroup"],
        categories=altersgruppen_reihenfolge,
        ordered=True
    )

    # Plot erstellen
    adata.obs["AgeGroup"].value_counts(sort=False).plot(kind="bar")
    plt.title("Verteilung der Altersgruppen bei erster OP")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Anzahl Patienten")
    plt.grid(False) # ⬅️ Wichtige Korrektur: Gitterlinien entfernen
    plt.tight_layout()
    plt.show()
else:
    print("Fehler: 'AgeGroup' ist nicht in adata.obs vorhanden.")