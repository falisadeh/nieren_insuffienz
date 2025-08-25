#!/usr/bin/env python3
import os
import pandas as pd

BASE = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")

FILES = {
    "AKI_Latenz_alle": os.path.join(BASE, "AKI_Latenz_alle.csv"),
    "analytic_ops_master_ehrapy": os.path.join(BASE, "analytic_ops_master_ehrapy.csv"),
    "analytic_patient_summary": os.path.join(BASE, "analytic_patient_summary.csv"),
}

def load_semicolon_csv(path):
    return pd.read_csv(path, sep=";", dtype=str)  # erst als string einlesen, dann gezielt konvertieren

def quick_info(name, df):
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Head:")
    print(df.head(5))
    # Schlüsselkandidaten grob prüfen
    for key in ["Procedure_ID", "PMID", "SMID"]:
        if key in df.columns:
            nunique = df[key].nunique(dropna=True)
            print(f"Key '{key}': non-null={df[key].notna().sum()}, unique={nunique}, duplicated={df[key].duplicated().sum()}")

def main():
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"{name}: file not found -> {path}")
            continue
        try:
            df = load_semicolon_csv(path)
            quick_info(name, df)
        except Exception as e:
            print(f"{name}: error {e}")

if __name__ == "__main__":
    main()



