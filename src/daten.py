import ehrapy as ep
import pandas as pd

from paths import cs_transfer_path

ORIGINAL_DATA_DIR = cs_transfer_path("Orginal Daten")

# Laden der Daten aus den CSV-Dateien
lab_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "Laboratory_Kreatinin+CystatinC.csv",
    sep=";",
)
aki_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "AKI Label.csv",
    sep=";",
)
patient_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "Patient Master Data.csv",
    sep=";",
)

# IM MOMENT WERDEN DIESE DATEN NICHT EINGELESEN, PROBLEM LOESEN
vis_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "VIS.csv",
    sep=";",
)
procedure_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "Procedure Supplement.csv",
    sep=";",
)
hlm_df = pd.read_csv(
    ORIGINAL_DATA_DIR / "HLM Operationen.csv",
    sep=";",
)

# Zusammenführen der Daten (Beispiel: Laborwerte mit Patientendaten)
# Die Daten müssen zunächst auf Patientenebene aggregiert werden,
# da ehrapy AnnData-Objekte für Patienten (Zeilen) und Merkmale (Spalten) strukturiert.
# Hier konzentrieren wir uns auf die Labordaten als Hauptmerkmal.
merged_lab_patient_df = pd.merge(lab_df, patient_df, on="PMID", how="left")

# Erstellen des AnnData-Objekts aus den zusammengeführten Daten
# Wir müssen eine Tabelle erstellen, in der jede Zeile einem Patienten entspricht
# und die Spalten die Merkmale sind, die wir für die Regression verwenden wollen.
# Hier aggregieren wir die Labordaten nach PMID.
# Dies ist ein kritischer Schritt, da Rohdaten pro Messung vorliegen.
# Für eine einfache Regression benötigen wir aggregierte Werte pro Patient.
# Beispielsweise der Durchschnittswert von Kreatinin.
patient_avg_creatinin = (
    merged_lab_patient_df.groupby("PMID")["QuantitativeValue"].mean().reset_index()
)
patient_avg_creatinin.rename(
    columns={"QuantitativeValue": "Avg_Creatinin"}, inplace=True
)

# Jetzt können wir die Label und andere Patientendaten hinzufügen
final_df = pd.merge(patient_avg_creatinin, aki_df, on="PMID", how="left")
final_df = pd.merge(final_df, patient_df, on="PMID", how="left")

# Erstellen des AnnData-Objekts
# Die `ehrapy` Funktion `data.df_to_anndata()` ist hier nützlich.
# Wir benötigen numerische Spalten für `X` und die Zielvariable für `y`.
# Da unsere Zielvariable (`Decision`) kategorisch ist, müssen wir sie für eine lineare Regression umwandeln.
# Für eine Klassifikation wäre dies besser geeignet, aber die Aufgabe verlangt eine lineare Regression.
# Wir müssen die `Decision` Spalte in numerische Werte konvertieren (z.B. 'AKI 1' -> 1, 'AKI 2' -> 2, etc.)
final_df["AKI_Severity"] = final_df["Decision"].str.extract("(\d)").astype(float)
final_df = final_df.dropna(subset=["AKI_Severity"])

# Erstellen des AnnData-Objekts
# 'PMID' wird der Index, 'Avg_Creatinin' die Feature-Matrix
adata = ep.data.df_to_anndata(
    df=final_df,
    columns_obs_only=["PMID", "Decision", "Sex", "DateOfBirth", "DateOfDie"],
    index_column="PMID",
)
# Speichern der Zielvariablen im .obs-Attribut
adata.obs["AKI_Severity"] = final_df.set_index("PMID")["AKI_Severity"]

# Die Haupt-Feature-Matrix (X) sollte numerische Werte enthalten.
# Die Spalte 'Avg_Creatinin' wird zu X.
adata.X = final_df[["Avg_Creatinin"]].values

print(adata)
