# 02_op_dauer_from_supp.py
import pandas as pd
import numpy as np
from pathlib import Path
# Optional: für AnnData/ehrapy
# from anndata import AnnData
# import ehrapy as ep

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
OUT  = Path(BASE); OUT.mkdir(exist_ok=True)

# --- 1) Supplement laden & säubern ---
df_supp = pd.read_csv(f"{BASE}/Procedure Supplement.csv", sep=";", parse_dates=["Timestamp"])
df_supp.columns = df_supp.columns.str.strip()
for c in ["PMID","SMID","Procedure_ID","TimestampName"]:
    df_supp[c] = df_supp[c].astype(str).str.strip()

# Nur Start/Ende der OP verwenden
mask = df_supp["TimestampName"].isin(["Start of surgery", "End of surgery"])
supp_se = df_supp.loc[mask, ["PMID","SMID","Procedure_ID","TimestampName","Timestamp"]].copy()

# --- 2) Pro (PMID,SMID,Procedure_ID) Start/End bestimmen ---
keys = ["PMID","SMID","Procedure_ID"]

starts = (supp_se[supp_se["TimestampName"]=="Start of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].min()
          .rename(columns={"Timestamp":"Start_supp"}))

ends   = (supp_se[supp_se["TimestampName"]=="End of surgery"]
          .groupby(keys, as_index=False)["Timestamp"].max()
          .rename(columns={"Timestamp":"End_supp"}))

op_times = starts.merge(ends, on=keys, how="outer")

# --- 3) Dauer berechnen & Plausibilitäten ---
op_times["duration_hours"]  = (op_times["End_supp"] - op_times["Start_supp"]).dt.total_seconds() / 3600
op_times["duration_minutes"]= (op_times["End_supp"] - op_times["Start_supp"]).dt.total_seconds() / 60

# Flag für merkwürdige Daten (z.B. Jahr in ferner Zukunft wie 2152)
op_times["flag_future_years"] = op_times[["Start_supp","End_supp"]].stack().dt.year.max(level=0) > 2035

# Filter: nur valide Dauern behalten (>= 0 h, z.B. <= 24 h; passe ggf. an)
valid = (op_times["duration_hours"] >= 0) & (op_times["duration_hours"] <= 24)
op_times_valid = op_times.loc[valid].copy()

# --- 4) Optionaler Cross-Check gegen HLM Operationen (falls vorhanden) ---
try:
    df_op = pd.read_csv(f"{BASE}/HLM Operationen.csv", sep=";", parse_dates=["Start of surgery","End of surgery"])
    df_op.columns = df_op.columns.str.strip()
    for c in ["PMID","SMID","Procedure_ID"]:
        if c in df_op.columns:
            df_op[c] = df_op[c].astype(str).str.strip()

    # Dauer aus HLM
    if {"Start of surgery","End of surgery"}.issubset(df_op.columns):
        op_dur_hlm = df_op.dropna(subset=["Start of surgery","End of surgery"]).copy()
        op_dur_hlm["hlm_hours"] = (op_dur_hlm["End of surgery"] - op_dur_hlm["Start of surgery"]).dt.total_seconds()/3600
        # Join auf gemeinsame Keys (falls Procedure_ID fehlt in HLM, nur PMID+SMID nehmen)
        join_keys = [k for k in keys if k in op_dur_hlm.columns]
        merged = op_times_valid.merge(op_dur_hlm[join_keys+["Start of surgery","End of surgery","hlm_hours"]],
                                      on=join_keys, how="left")
        merged["delta_hours_supp_vs_hlm"] = merged["duration_hours"] - merged["hlm_hours"]
        out_compare = OUT / "op_durations_compare_supp_vs_hlm.csv"
        merged.to_csv(out_compare, sep=";", index=False)
        print(f"Vergleich (Supplement vs. HLM) gespeichert: {out_compare}")
except Exception as e:
    print("HLM-Comparison übersprungen:", e)

# --- 5) Ergebnisse speichern ---
out_csv = OUT / "op_durations_from_supp.csv"
op_times_valid.to_csv(out_csv, sep=";", index=False)
print(f"OP-Dauern (Supplement-basiert) gespeichert: {out_csv}")

# Kurz-Statistik ausgeben
print("\n=== Kurzstatistik (gültige OPs) ===")
print(op_times_valid["duration_hours"].describe())

# --- 6) (Optional) In ehrapy/AnnData überführen ---
# op_times_valid["op_id"] = op_times_valid["PMID"] + "_" + op_times_valid["SMID"]
# adata = AnnData(op_times_valid.set_index("op_id"))
# adata.write_h5ad(OUT / "op_durations_from_supp.h5ad")
# print("AnnData geschrieben: op_durations_from_supp.h5ad")


