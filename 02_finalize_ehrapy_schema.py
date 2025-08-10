from pathlib import Path
import pandas as pd
import numpy as np
from anndata import AnnData

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
P = Path(BASE)

src = P / "analytic_ops_master.csv"
dst_csv = P / "analytic_ops_master_ehrapy.csv"
dst_h5  = P / "aki_ops_master.h5ad"

assert src.exists(), f"Fehlt: {src}"

# --- 1) Laden ---
df = pd.read_csv(src, sep=";")
df.columns = df.columns.str.strip()

# --- 2) OP_ID sicherstellen ---
if "OP_ID" not in df.columns:
    df["OP_ID"] = (
        df["PMID"].astype(str) + "_" +
        df.get("SMID", "").astype(str) + "_" +
        df.get("Procedure_ID", "").astype(str)
    )

# --- 3) Helper ---
def to_iso_object(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    iso = s.dt.strftime("%Y-%m-%d %H:%M:%S")
    return iso.where(~iso.isna(), None).astype(object)  # object-Strings, kein pandas "string"

def to_object_string(s: pd.Series) -> pd.Series:
    out = s.astype("object")
    return out.where(out.notna(), None)

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def to_int8(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).round().astype("int8")

# --- 4) Schema anwenden ---
schema = {
    "OP_ID": "object",
    "PMID": "object",
    "SMID": "object",
    "Procedure_ID": "object",
    "Surgery_Start": "iso",
    "Surgery_End": "iso",
    "AKI_Start": "iso",
    "duration_hours": "float",
    "duration_minutes": "float",
    "days_to_AKI": "float",
    "AKI_linked": "int8",
    "AKI_linked_0_7": "int8",
    "Sex_norm": "object",
}

for col, kind in schema.items():
    if col not in df.columns:
        continue
    if kind == "iso":
        df[col] = to_iso_object(df[col])
    elif kind == "object":
        df[col] = to_object_string(df[col])
    elif kind == "float":
        df[col] = to_float(df[col])
    elif kind == "int8":
        df[col] = to_int8(df[col])

# eindeutiger Index
df = df.drop_duplicates("OP_ID", keep="first").set_index("OP_ID")

# --- 5) ehrapy-CSV schreiben ---
df.to_csv(dst_csv, sep=";")
print("CSV geschrieben:", dst_csv, "| Shape:", df.shape)

# --- 6) H5AD: X = duration_hours, obs = Rest ---
assert "duration_hours" in df.columns, "duration_hours fehlt nach Schema!"
X = df["duration_hours"].to_numpy(dtype=float).reshape(-1, 1)
obs = df.drop(columns=["duration_hours"]).copy()

# alle String-Spalten als object absichern
for c in obs.columns:
    if pd.api.types.is_string_dtype(obs[c].dtype):
        obs[c] = obs[c].astype(object)

adata = AnnData(X=X, obs=obs)
adata.var_names = ["duration_hours"]
adata.write_h5ad(dst_h5)
print("H5AD geschrieben:", dst_h5, "| AnnData:", adata.shape)
