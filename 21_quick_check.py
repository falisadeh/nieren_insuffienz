from pathlib import Path
import pandas as pd
import ehrapy as ep

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad")
FILES = [
    "ops_with_crea_cysc_vis_features_with_AKI.h5ad",
    "ops_with_patient_features.h5ad",
    "analytic_patient_summary_v2.h5ad",
    "causal_dataset_op_level.h5ad",
]

def booleanize01(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num > 0).astype(int)
    return s.astype(str).str.lower().str.strip().isin(["1","true","wahr","ja","yes"]).astype(int)

for f in FILES:
    ad = ep.io.read_h5ad(str(BASE / f))
    df = ad.obs.copy()
    src = None
    if "AKI_linked_0_7_time" in df.columns and df["AKI_linked_0_7_time"].notna().any():
        s = booleanize01(df["AKI_linked_0_7_time"]); src = "AKI_linked_0_7_time"
    elif "AKI_linked_0_7" in df.columns:
        s = booleanize01(df["AKI_linked_0_7"]); src = "AKI_linked_0_7"
    elif {"AKI_Start","Surgery_End"}.issubset(df.columns):
        se = pd.to_datetime(df["Surgery_End"], errors="coerce")
        ak = pd.to_datetime(df["AKI_Start"],    errors="coerce")
        days = (ak - se).dt.total_seconds()/86400.0
        s = days.between(0,7).fillna(False).astype(int); src = "AKI_Start/Surgery_End"
    else:
        s = pd.Series([], dtype=int); src = "NA"

    n1 = int(s.sum()); n0 = int((s==0).sum()); nt = len(df)
    print(f"{f:45s}  | Quelle: {src:20s} | AKI=1: {n1:4d} | AKI=0: {n0:4d} | total: {nt}")
