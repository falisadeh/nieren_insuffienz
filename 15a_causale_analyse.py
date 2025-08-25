import ehrapy as ep
import pandas as pd
import numpy as np
from anndata import AnnData
from pathlib import Path
import networkx as nx

BASE = Path("/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer")
AD_CSLIM = BASE / "h5ad" / "causal_dataset_op_level.h5ad"

ad_in = ep.io.read_h5ad(str(AD_CSLIM))
obs = ad_in.obs.copy()

# Pflichtspalten
must = ["duration_hours","AKI_linked_0_7"]
for m in must:
    if m not in obs.columns:
        raise ValueError(f"Fehlt: {m}")

# numerisieren/aufräumen
for c in ["duration_hours","AKI_linked_0_7","age_years_at_first_op","Sex_norm","n_ops","is_reop"]:
    if c in obs.columns:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")

# Re-OP falls nötig
if "is_reop" not in obs.columns and "n_ops" in obs.columns:
    obs["is_reop"] = (obs["n_ops"] > 1).astype(int)

# Confounder mit Coverage ≥80%
conf_cands = [c for c in ["age_years_at_first_op","is_reop","Sex_norm"] if c in obs.columns]
coverage = {c: 1 - obs[c].isna().mean() for c in conf_cands}
good = [c for c in conf_cands if coverage[c] >= 0.80]
print("Confounder-Coverage:", {k: f"{v*100:.1f}%" for k,v in coverage.items()})

req = must + good
df = obs[req].dropna(subset=req).copy()
if df.empty:
    raise ValueError("Kein Datensatz nach Drop-NA. Coverage prüfen.")

# AnnData + DAG
ad = AnnData(X=df.to_numpy())
ad.obs_names = df.index.astype(str)
ad.var_names = req

G = nx.DiGraph()
for c in good:
    G.add_edge(c, "duration_hours")
    G.add_edge(c, "AKI_linked_0_7")
G.add_edge("duration_hours", "AKI_linked_0_7")

# Schätzung (ehrapy/dowhy)
est = ep.tl.causal_inference(
    adata=ad,
    graph=G,
    treatment="duration_hours",
    outcome="AKI_linked_0_7",
    estimation_method="backdoor.linear_regression",
    refute_methods=[],
    print_causal_estimate=True,
    print_summary=False,
    show_graph=False,
    show_refute_plots=False,
    return_as="estimate",
)

print("ATE (pro zusätzlicher OP-Stunde):", float(est.value))
ad = ep.io.read_h5ad("/…/ops_with_patient_features.h5ad")
num_cols = ["duration_hours","age_years_at_first_op","crea_delta_0_48","cysc_delta_0_48","vis_auc_0_24","Sex_norm","n_ops"]
num_cols = [c for c in num_cols if c in ad.obs.columns]
X = ad.obs[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
ad.X = np.nan_to_num(X)  # X wird numerisch
