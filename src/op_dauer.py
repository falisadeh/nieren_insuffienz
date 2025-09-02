# === Finale Version (n = 1209) ===
# Datei: ops_with_patient_features.h5ad
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path

# Pfade anpassen falls nötig
BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5AD = f"{BASE}/h5ad/ops_with_patient_features.h5ad"  # n=1209 (gereinigt)
OUTD = Path(BASE) / "Diagramme"
OUTD.mkdir(parents=True, exist_ok=True)

# Laden
adata = ad.read_h5ad(H5AD)
assert adata.n_obs == 1209, f"Erwarte 1209 Zeilen, gefunden: {adata.n_obs}"

# AKI-Spalte festlegen und robust binär machen
aki_col = "AKI_linked_0_7"


def to_bin(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "true", "ja", "yes", "y"}:
        return 1
    if s in {"0", "false", "nein", "no", "n"}:
        return 0
    if "aki" in s and any(ch.isdigit() for ch in s):
        return 1  # "AKI 1/2/3"
    if "kein" in s or "none" in s:
        return 0  # "keine AKI"
    return pd.to_numeric(x, errors="coerce")


adata.obs["AKI_bin"] = pd.Series(adata.obs[aki_col]).map(to_bin)

# Dauer (Minuten) anlegen, falls nicht vorhanden
if "duration_minutes" not in adata.obs.columns:
    assert "duration_hours" in adata.obs.columns, "Keine OP-Dauer verfügbar."
    adata.obs["duration_minutes"] = adata.obs["duration_hours"] * 60

# Daten für Plot
d0 = adata.obs.loc[adata.obs["AKI_bin"] == 0, "duration_minutes"].dropna()
d1 = adata.obs.loc[adata.obs["AKI_bin"] == 1, "duration_minutes"].dropna()
n0, n1 = len(d0), len(d1)
med0, med1 = (float(np.median(d0)) if n0 > 0 else np.nan), (
    float(np.median(d1)) if n1 > 0 else np.nan
)

# Plot (klinische Labels, keine Variablennamen)
plt.figure(figsize=(12, 7))
plt.hist(d0, bins=30, alpha=0.6, label=f"Kein AKI 0–7 Tage (n={n0})")
plt.hist(d1, bins=30, alpha=0.6, label=f"AKI innerhalb 0–7 Tage (n={n1})")

if not np.isnan(med0):
    plt.axvline(
        med0, linestyle="--", linewidth=2, label=f"Median ohne AKI: {med0:.0f} Min."
    )
if not np.isnan(med1):
    plt.axvline(
        med1, linestyle=":", linewidth=2, label=f"Median mit AKI: {med1:.0f} Min."
    )

plt.title("Verteilung der Operationsdauer (nach AKI 0–7 Tagen stratifiziert)")
plt.xlabel("Operationsdauer (Minuten)")
plt.ylabel("Häufigkeit")
plt.legend()
plt.tight_layout()
fig_path = OUTD / "hist_operationsdauer_nach_AKI_1209.png"
plt.savefig(fig_path, dpi=200)
print(f"Bild gespeichert: {fig_path}")


# Kennzahlen-Tabelle (für Folie/Anhang)
def q(x, p):
    return float(np.percentile(x, p)) if len(x) > 0 else np.nan


summary = pd.DataFrame(
    {
        "Gruppe": ["Kein AKI 0–7", "AKI 0–7"],
        "n": [n0, n1],
        "Median_Min": [med0, med1],
        "Q1_Min": [q(d0, 25), q(d1, 25)],
        "Q3_Min": [q(d0, 75), q(d1, 75)],
        "Min": [
            float(np.min(d0)) if n0 > 0 else np.nan,
            float(np.min(d1)) if n1 > 0 else np.nan,
        ],
        "Max": [
            float(np.max(d0)) if n0 > 0 else np.nan,
            float(np.max(d1)) if n1 > 0 else np.nan,
        ],
    }
)
csv_path = OUTD / "hist_operationsdauer_nach_AKI_1209_summary.csv"
summary.to_csv(csv_path, index=False)
print("Kennzahlen:\n", summary.to_string(index=False))
print(f"CSV gespeichert: {csv_path}")
