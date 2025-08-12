from anndata import read_h5ad
import pandas as pd
H5AD = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
adata = read_h5ad(H5AD)
cols = list(adata.obs.columns)
print(f"{len(cols)} obs-Spalten geladen.")

needed_any = {
  "AKI": ["AKI_linked_0_7","aki_linked_0_7","AKI0_7","aki_0_7"],
  "AKI_Start": ["AKI_Start","aki_start","AkiStart"],
  "Surgery_End": ["Surgery_End","End of surgery","surgery_end","End_of_surgery"],
  "Surgery_Start": ["Surgery_Start","Start of surgery","surgery_start","Start_of_surgery"],
  "days_to_AKI": ["days_to_AKI","days_to_aki","DaysToAKI"],
  "is_reop": ["is_reop","reop","ReOP","IsReOp"],
  "age_years_at_op": ["age_years_at_op","age_years","AgeYearsAtOP"],
  "age_days_at_op": ["age_days_at_op","age_days","AgeDaysAtOP"],
}

def pick(cands):
    for c in cands:
        if c in cols:
            return c
    return None

alias = {k: pick(v) for k,v in needed_any.items()}
for k,v in alias.items():
    print(f"{k:16} -> {v}")

missing_hours = [k for k in ["AKI_Start","Surgery_End","days_to_AKI"] if alias[k] is None]
if alias["days_to_AKI"] or (alias["AKI_Start"] and alias["Surgery_End"]):
    print("OK: Stunden-Plot kann erzeugt werden.")
else:
    print("FEHLT für Stunden-Plot:", missing_hours)

if alias["is_reop"]:
    print("OK: Re-OP-Stratifizierung möglich.")
else:
    print("Hinweis: 'is_reop' nicht gefunden – Stratifikationsplot wird übersprungen.")
