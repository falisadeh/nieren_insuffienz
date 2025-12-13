from __future__ import annotations

import pandas as pd

from paths import ORIGINAL_DATA_DIR

try:
    import ehrapy as ep
except ImportError as exc:  # pragma: no cover - hints for missing dependency
    raise ImportError(
        "ehrapy muss installiert sein, um src/daten.py auszuführen."
    ) from exc


CSV_FILES = {
    "lab": "Laboratory_Kreatinin+CystatinC.csv",
    "aki": "AKI Label.csv",
    "patients": "Patient Master Data.csv",
    "vis": "VIS.csv",
    "procedure": "Procedure Supplement.csv",
    "hlm": "HLM Operationen.csv",
}


def read_original_csv(filename: str) -> pd.DataFrame:
    """Zentraler Loader für alle Originaldaten (Semikolon-getrennt)."""
    path = ORIGINAL_DATA_DIR / filename
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def compute_patient_level_features(
    lab_df: pd.DataFrame, patient_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregiert Laborwerte auf Patientenebene (hier: Mittelwert Kreatinin)."""
    merged = pd.merge(lab_df, patient_df, on="PMID", how="left")
    agg = merged.groupby("PMID")["QuantitativeValue"].mean().reset_index()
    agg = agg.rename(columns={"QuantitativeValue": "Avg_Creatinin"})
    return agg


def encode_aki_severity(decisions: pd.Series) -> pd.Series:
    """Extrahiert AKI-Stufen aus Strings wie 'AKI 1' → 1.0."""
    return decisions.str.extract(r"(\d)").astype(float)


def build_final_dataframe() -> pd.DataFrame:
    """Liest relevante CSVs und baut die patientenweise Tabelle."""
    lab_df = read_original_csv(CSV_FILES["lab"])
    aki_df = read_original_csv(CSV_FILES["aki"])
    patient_df = read_original_csv(CSV_FILES["patients"])

    # ggf. später genutzt – derzeit nur geladen, damit die Daten geprüft werden können
    _ = read_original_csv(CSV_FILES["vis"])
    _ = read_original_csv(CSV_FILES["procedure"])
    _ = read_original_csv(CSV_FILES["hlm"])

    features = compute_patient_level_features(lab_df, patient_df)
    combined = (
        features.merge(aki_df, on="PMID", how="left")
        .merge(patient_df, on="PMID", how="left")
        .drop_duplicates(subset=["PMID"])
    )
    combined["AKI_Severity"] = encode_aki_severity(combined["Decision"])
    combined = combined.dropna(subset=["AKI_Severity"])
    return combined


def build_anndata(df: pd.DataFrame):
    """Erstellt das AnnData-Objekt mit `Avg_Creatinin` als Feature und AKI-Label."""
    adata = ep.data.df_to_anndata(
        df=df,
        columns_obs_only=["PMID", "Decision", "Sex", "DateOfBirth", "DateOfDie"],
        index_column="PMID",
    )
    adata.obs["AKI_Severity"] = df.set_index("PMID")["AKI_Severity"]
    adata.X = df[["Avg_Creatinin"]].values
    return adata


def main():
    df = build_final_dataframe()
    adata = build_anndata(df)
    print(adata)


if __name__ == "__main__":
    main()
