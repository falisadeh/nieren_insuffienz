# 01_repair_outputs.py  (V2)
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import AnnData

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
P = Path(BASE)

# ---- Quelle wählen: ehrapy-CSV bevorzugen, sonst Master-CSV ----
candidates = ["analytic_ops_master_ehrapy.csv", "analytic_ops_master.csv"]
src = None
for name in candidates:
    p = P / name
    if p.exists():
        src = p
        break
assert src is not None, f"Keine Quelle gefunden: {candidates}"

print("Quelle:", src)

df = pd.read_csv(src, sep=";")
df.columns = df.columns.str.strip()

# OP_ID sicherstellen
if "OP_ID" not in df.columns:
    df["OP_ID"] = (
        df["PMID"].astype(str) + "_" +
        df.get("SMID","").astype(str) + "_" +
        df.get("Procedure_ID","").astype(str)
    )

# --- Datumsfelder tolerant parsen (nur wenn es keine ISO-Strings sind) ---
date_cols = [c for c in ["Surgery_Start","Surgery_End","AKI_Start"] if c in df.columns]
for c in date_cols:
    # falls schon ISO-Strings drin sind, macht das nichts kaputt
    df[c] = pd.to_datetime(df[c], errors="coerce")

# ---- A) prior 0–7 exportieren ----
keep = [
    "OP_ID","PMID","SMID","Procedure_ID",
    "Surgery_Start","Surgery_End",
    "duration_hours","AKI_Start","days_to_AKI","Sex_norm"
]
prior = df[df.get("AKI_linked_0_7", 0) == 1][[k for k in keep if k in df.columns]].copy()
prior.to_csv(P/"aki_prior_0_7.csv", sep=";", index=False)
print("geschrieben:", P/"aki_prior_0_7.csv", "| n =", len(prior))

# ---- B) H5AD bauen: X=duration_hours, obs=Rest (nur Strings & Zahlen) ----
df_idx = df.drop_duplicates("OP_ID", keep="first").set_index("OP_ID").copy()

# Datums → ISO-String (Python-Strings, keine pandas StringArray)
def to_iso_str_object(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    iso = s.dt.strftime("%Y-%m-%d %H:%M:%S")
    return iso.fillna("").astype(object)

for c in date_cols:
    if c in df_idx.columns:
        df_idx[c] = to_iso_str_object(df_idx[c])

# Numerik bereinigen
for c in ["duration_hours","duration_minutes","days_to_AKI"]:
    if c in df_idx.columns:
        df_idx[c] = pd.to_numeric(df_idx[c], errors="coerce")

for c in ["AKI_linked","AKI_linked_0_7"]:
    if c in df_idx.columns:
        df_idx[c] = pd.to_numeric(df_idx[c], errors="coerce").fillna(0).round().astype("int8")

# Textspalten zu echten Python-Strings (object) ohne None/nan
for c in ["PMID","SMID","Procedure_ID","Sex_norm"]:
    if c in df_idx.columns:
        df_idx[c] = df_idx[c].astype(str).replace({"nan":"", "NaT":"", "None":""}).astype(object)

# X vorbereiten
assert "duration_hours" in df_idx.columns, "duration_hours fehlt!"
X = df_idx["duration_hours"].to_numpy(dtype=float).reshape(-1,1)
obs = df_idx.drop(columns=["duration_hours"])

# Sicherheit: alle object-Spalten sind wirklich Strings (keine None)
for c in obs.columns:
    if obs[c].dtype == "O":  # object
        obs[c] = obs[c].apply(lambda x: "" if pd.isna(x) else str(x))

adata = AnnData(X=X, obs=obs)
adata.var_names = ["duration_hours"]
h5 = P/"aki_ops_master.h5ad"
adata.write_h5ad(h5)
print("geschrieben:", h5, "| AnnData:", adata.shape)
# ---- C) Analytic Patient Summary exportieren ----
from anndata import read_h5ad
import numpy as np, pandas as pd

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
adata = read_h5ad(f"{BASE}/aki_ops_master.h5ad")
print(adata)  # sollte: 1209 × 1, var_names=['duration_hours']

# Dauer holen & kurzer Check
dur = np.asarray(adata.X).ravel().astype(float)
print(pd.Series(dur).describe())

# AKI-Labels & 0–7 Fenster
obs = adata.obs.copy()
obs["AKI_linked"] = pd.to_numeric(obs["AKI_linked"], errors="coerce").astype("int8")
obs["AKI_linked_0_7"] = pd.to_numeric(obs["AKI_linked_0_7"], errors="coerce").astype("int8")
print("linked_0_7 =", int((obs["AKI_linked_0_7"]==1).sum()))  # erwartet: 533

