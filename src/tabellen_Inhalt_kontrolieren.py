# %%
import anndata as ad

# Beispiel: eine Datei öffnen
adata = ad.read_h5ad(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/h5ad/ops_with_patient_features.h5ad"
)

print("Spalten in .obs:")
print(adata.obs.columns.tolist())
print("\nErste 5 Zeilen:")
print(adata.obs.head())

# # === 0) Setup ===
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

# Kandidaten (passe Pfade an, falls nötig)
candidates = [
    "ops_with_patient_features.h5ad",
    "ops_with_patient_features_ehrapy_enriched.h5ad",
    "analytic_patient_summary_v2.h5ad",
    "causal_dataset_op_level.h5ad",
]
base = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"  # dein Projektordner
candidates = [
    (
        str(Path("/mnt/data") / Path(x).name)
        if os.path.exists("/mnt/data")
        else str(Path(base) / "h5ad" / Path(x).name)
    )
    for x in candidates
]


# === 1) Funktion: Datei inspizieren ===
def inspect_h5ad(path):
    if not os.path.exists(path):
        return {"file": path, "exists": False}
    try:
        A = ad.read_h5ad(path)
    except Exception as e:
        return {"file": path, "exists": True, "error": str(e)}

    obs_cols = list(A.obs.columns)
    aki_candidates = [
        "AKI_linked_0_7",
        "AKI_0_7",
        "aki_0_7",
        "AKI",
        "AKI_flag",
        "aki_linked_0_7",
    ]
    aki_col = next((c for c in aki_candidates if c in obs_cols), None)

    has_dm = "duration_minutes" in obs_cols
    has_dh = "duration_hours" in obs_cols

    non_null_dm = int(A.obs["duration_minutes"].notna().sum()) if has_dm else 0
    non_null_aki = int(A.obs[aki_col].notna().sum()) if aki_col else 0

    # einfache „Score“-Heuristik zur Auswahl
    score = 0
    score += 2 if has_dm else 0
    score += 2 if aki_col else 0
    score += int(non_null_dm > 0)
    score += int(non_null_aki > 0)

    return {
        "file": path,
        "exists": True,
        "n_obs": int(A.n_obs),
        "n_vars": int(A.n_vars),
        "obs_cols_count": len(obs_cols),
        "has_duration_minutes": has_dm,
        "has_duration_hours": has_dh,
        "AKI_col": aki_col,
        "non_null_duration_minutes": non_null_dm,
        "non_null_AKI": non_null_aki,
        "score": score,
        "_adata": A,  # für spätere Nutzung
    }


# === 2) Alle inspizieren und beste wählen ===
reports = [inspect_h5ad(p) for p in candidates]
df_rep = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "_adata"} for r in reports]
).sort_values("score", ascending=False)
print("Übersicht:\n", df_rep.to_string(index=False))

# Beste Datei wählen
best = None
for r in reports:
    if r.get("exists") and not r.get("error") and r.get("score", 0) >= 5:
        best = r
        break
if best is None:
    # Fallback: die mit höchstem Score
    best = max(
        (r for r in reports if r.get("exists") and not r.get("error")),
        key=lambda x: x.get("score", 0),
    )

adata = best["_adata"]
aki_col = best["AKI_col"]
print(f"\n>>> Ausgewählt: {best['file']}\nAKI-Spalte: {aki_col}\n")


# Optional: AKI binär robust machen (0/1)
def to_bin(s):
    if pd.isna(s):
        return np.nan
    s2 = str(s).strip().lower()
    if s2 in {"1", "true", "ja", "yes", "y", "aki", "aki1", "aki2", "aki3"}:
        return 1
    if s2 in {"0", "false", "nein", "no", "n", "none", "kein", "keine"}:
        return 0
    # „AKI 1/2/3“ → 1, „keine AKI“ → 0
    if "aki" in s2 and any(ch.isdigit() for ch in s2):
        return 1
    if "kein" in s2 or "none" in s2:
        return 0
    return pd.to_numeric(s, errors="coerce")


adata.obs["AKI_bin"] = pd.Series(adata.obs[aki_col]).map(to_bin)

# === 3) Plot mit „sprechenden“ Labels erzeugen ===
df = adata.obs.copy()
assert (
    "duration_minutes" in df.columns
), "Spalte 'duration_minutes' fehlt in der gewählten Datei."

# Subsets
d0 = df.loc[df["AKI_bin"] == 0, "duration_minutes"].dropna()
d1 = df.loc[df["AKI_bin"] == 1, "duration_minutes"].dropna()

n0, n1 = len(d0), len(d1)
med0, med1 = (float(np.median(d0)) if n0 > 0 else np.nan), (
    float(np.median(d1)) if n1 > 0 else np.nan
)

plt.figure(figsize=(10, 6))
plt.hist(d0, bins=30, alpha=0.6, label=f"Kein AKI 0–7 Tage (n={n0})", density=False)
plt.hist(
    d1, bins=30, alpha=0.6, label=f"AKI innerhalb 0–7 Tage (n={n1})", density=False
)

# Median-Linien
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

out_dir = Path(base) / "Diagramme"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "hist_operationsdauer_nach_AKI.png"
plt.savefig(out_path, dpi=200)
print(f"\nBild gespeichert: {out_path}")
