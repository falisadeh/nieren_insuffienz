#%%!/usr/bin/env python3


"""
Kausale Schätzung AKI ~ long_op (OP-Dauer) – schnell & robust

Pipeline (in dieser Reihenfolge = von schnell → aufwendig):
  1) LPM  (backdoor.linear_regression)  → sehr schnell
  2) IPW  (backdoor.propensity_score_weighting) → schnell
# PSM entfernt

Kompakte Konsolen-Ausgabe: Effekt, Konfidenzintervall, p-Wert.
Refutation schlank: nur Placebo, 5 Simulationen.

Hinweise:
- Für Trockenläufe SUBSAMPLE_FRAC < 1.0 (z. B. 0.25)
- Für stärkeren Treatment-Kontrast USE_P75_TREATMENT = True (statt Median-Split)
"""

from __future__ import annotations
import numpy as np
import ehrapy as ep
from anndata import read_h5ad
DO_PSM = False
import os, sys
print("LADE DATEI:", __file__)
# ========================= Pfad anpassen =========================
H5AD_PATH = \
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master.h5ad"
# ================================================================

# ---------- Lauf-Parameter ----------
RANDOM_STATE = 42
SUBSAMPLE_FRAC = 1.0       # z. B. 0.25 für schnellen Testlauf
USE_P75_TREATMENT = False  # True → long_op ab 75. Perzentil statt Median
N_SIM_REFUTER = 5          # Placebo-Simulationen (klein halten)

# ---------- Daten laden ----------
adata = read_h5ad(H5AD_PATH)
print(adata)
print("Spaltennamen in .obs:", list(adata.obs.columns))

if SUBSAMPLE_FRAC < 1.0:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = adata.obs.sample(frac=SUBSAMPLE_FRAC, random_state=RANDOM_STATE).index
    adata = adata[idx].copy()
    print(f"Subsample: verwende {len(adata)} Beobachtungen ({SUBSAMPLE_FRAC*100:.0f}%)")

# ---------- Variablen vorbereiten ----------
if "duration_minutes" not in adata.obs.columns:
    raise ValueError("Spalte 'duration_minutes' fehlt in adata.obs.")

thr = float(np.nanpercentile(adata.obs["duration_minutes"], 75)) if USE_P75_TREATMENT \
      else float(np.nanmedian(adata.obs["duration_minutes"]))
adata.obs["long_op"] = (adata.obs["duration_minutes"] >= thr).astype(int)

if "AKI_linked_0_7" in adata.obs.columns:
    adata.obs["aki_bin"] = adata.obs["AKI_linked_0_7"].astype(int)
elif "AKI_linked" in adata.obs.columns:
    adata.obs["aki_bin"] = adata.obs["AKI_linked"].astype(int)
else:
    raise ValueError("Weder 'AKI_linked_0_7' noch 'AKI_linked' vorhanden.")

if "Sex_norm" not in adata.obs.columns:
    raise ValueError("Spalte 'Sex_norm' fehlt (erwartet 'm'/'f').")
adata.obs["sex_bin"] = adata.obs["Sex_norm"].map({"m": 1, "f": 0}).astype("Int64")

# Optional: Procedure_ID als Kategorie
# adata.obs["Procedure_ID"] = adata.obs["Procedure_ID"].astype("category")

# ---------- Kausalgraph ----------
causal_graph = """
digraph {
  sex_bin -> long_op;
  sex_bin -> aki_bin;

  Procedure_ID -> long_op;
  Procedure_ID -> aki_bin;

  long_op -> aki_bin;
}
"""

# ---------- Helper: kompakte Zusammenfassung ----------
def summarize_result(result: dict, title: str):
    est   = result.get("estimate", result.get("value"))
    ci    = result.get("confidence_intervals", result.get("ci"))
    p     = result.get("p_value", result.get("pval"))
    meth  = result.get("estimation_method", "?")

    print(f"[{title}]")
    print(f"  Methode: {meth}")
    if est is not None:
        try:
            print(f"  Effekt (ATE/ATT): {float(est):.4f}")
        except Exception:
            print(f"  Effekt: {est}")
    if ci is not None:
        print(f"  Konf.-Intervall: {ci}")
    if p is not None:
        print(f"  p-Wert: {p}")

    refuts = result.get("refutations", [])
    if refuts:
        from collections import defaultdict
        agg = defaultdict(list)
        for r in refuts:
            name = r.get("refuter_name", r.get("name", "refuter"))
            diff = r.get("estimated_effect", r.get("effect_difference"))
            if diff is not None:
                try:
                    agg[name].append(float(diff))
                except Exception:
                    pass
        if agg:
            print("  Refutations (Zusammenfassung):")
            for name, diffs in agg.items():
                diffs = np.asarray(diffs, dtype=float)
                sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
                print(f"    - {name}: n={len(diffs)}, Mittel={diffs.mean():.4f}, SD={sd:.4f}")

# ---------- Gemeinsame Kwargs ----------
IDENTIFY_KW = {"proceed_when_unidentifiable": True}
# Achtung: Kein 'random_state' in estimate_kwargs – DoWhy/ehrapy reicht das nicht an alle Schätzer durch
ESTIMATE_KW = {"test_significance": True, "confidence_intervals": True}
REFUTE_KW   = {"refuters": [{"name": "placebo_treatment_refuter", "placebo_type": "permute", "num_simulations": N_SIM_REFUTER}]}

# ============ 1) LPM ============
print("Starte kausale Schätzung (Lineare Regression, LPM)...")
res_lpm = ep.tl.causal_inference(
    adata=adata,
    graph=causal_graph,
    treatment="long_op",
    outcome="aki_bin",
    estimation_method="backdoor.linear_regression",
    identify_kwargs=IDENTIFY_KW,
    estimate_kwargs=ESTIMATE_KW,
)
summarize_result(res_lpm, "LPM")

# ============ 2) IPW ============
print("Starte kausale Schätzung (IPW)...")
res_ipw = ep.tl.causal_inference(
    adata=adata,
    graph=causal_graph,
    treatment="long_op",
    outcome="aki_bin",
    estimation_method="backdoor.propensity_score_weighting",
    identify_kwargs=IDENTIFY_KW,
    estimate_kwargs=ESTIMATE_KW,
    refute_kwargs=None,
)


summarize_result(res_ipw, "IPW")

# ============ 3) PSM ============
print("Starte kausale Schätzung (PSM)...")
#try:
# #     # PSM block removed
#except KeyboardInterrupt:
#    print("PSM abgebrochen – weiter mit bereits berechneten Ergebnissen.")

print("Fertig. Hinweise:- Für schnellere Läufe SUBSAMPLE_FRAC < 1 setzen oder N_SIM_REFUTER reduzieren.- Für klareren Treatment-Kontrast USE_P75_TREATMENT=True verwenden.- Wenn Refutations häufig 'failed': weitere Confounder ergänzen (Alter, Re-OP, CPB/Clamp, Komplexität).")
