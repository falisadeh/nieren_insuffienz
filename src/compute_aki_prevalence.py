# Kontrolle
# "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Original Daten/AKI Label.csv"
import pandas as pd

# CSV einlesen
df = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Original Daten/AKI Label.csv",
    sep=";",
    encoding="utf-8-sig",
)


# Spaltennamen säubern (falls Leerzeichen oder BOM drin sind)
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)

# Decision in AKI-Ja/Nein übersetzen
df["AKI_any"] = df["Decision"].str.contains("AKI", case=False, na=False).astype(int)


# pro Patient (PMID) höchste AKI-Stufe bestimmen
def extract_stage(x):
    if pd.isna(x):
        return 0
    s = str(x)
    if "3" in s:
        return 3
    if "2" in s:
        return 2
    if "1" in s:
        return 1
    return 0


df["AKI_stage"] = df["Decision"].map(extract_stage)

# Patient-Level aggregieren
per_patient = (
    df.groupby("PMID")
    .agg(
        AKI_any=("AKI_any", "max"),  # hatte Patient jemals AKI? (0/1)
        AKI_stage_max=("AKI_stage", "max"),  # höchste AKI-Stufe des Patienten
    )
    .reset_index()
)

# Ergebnisse
n_patients = per_patient.shape[0]
n_aki = per_patient["AKI_any"].sum()
prevalence = n_aki / n_patients * 100

print(f"Patienten insgesamt: {n_patients}")
print(f"Patienten mit AKI: {n_aki} ({prevalence:.1f}%)")
print("Verteilung höchste AKI-Stufe:")
print(per_patient["AKI_stage_max"].value_counts().sort_index())
