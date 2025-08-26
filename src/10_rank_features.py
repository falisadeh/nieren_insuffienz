#!/usr/bin/env python3
"""
Univariates Feature-Ranking für deinen AKI-Binary-Outcome (AKI 0–7).

Priorität: ehrapy (falls vorhanden) -> scanpy -> manueller Welch-t-Test mit BH-FDR.
Robuste Extraktion aus `ad.uns[...]` (auch für recarrays/structured arrays).

Eingaben (automatisch gewählt):
  1) Daten/ops_ml_processed.h5ad (empfohlen; bereits transformiert/skaliert)
  2) Daten/ops_with_patient_features.h5ad (Fallback)

Ausgaben:
  - Diagramme/ranked_features_ep.png (wenn ehrapy/scanpy-Plot möglich)
  - Daten/ranked_features_table.csv (immer; Ranking-Tabelle)
"""
from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd

# Matplotlib: nicht-interaktives Backend (unterdrückt macOS-Dialoge)
import matplotlib as mpl

mpl.use("Agg")  # headless, kein macOS-Fenster
import matplotlib.pyplot as plt


def save_barplot_from_table(df, out_png, title, topn=15):
    import numpy as np
    import matplotlib.pyplot as plt

    d = df.copy()
    key = "qval" if "qval" in d.columns else ("pval" if "pval" in d.columns else None)
    if key is None or d.empty:
        return
    d = d.dropna(subset=[key]).sort_values(key).head(topn)
    x = -np.log10(d[key].values + 1e-300)
    plt.figure(figsize=(8, 5))
    plt.barh(d["feature"][::-1], x[::-1])
    plt.xlabel("-log10(q)" if key == "qval" else "-log10(p)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# ---- Imports & Verfügbarkeit prüfen
try:
    import anndata as ad
except Exception as e:
    raise RuntimeError(f"Bitte 'anndata' installieren/aktivieren: {e}")

_have_ep = False
_have_sc = False
try:
    import ehrapy as ep  # optional

    _have_ep = True
except Exception:
    _have_ep = False

try:
    import scanpy as sc  # optional

    _have_sc = True
except Exception:
    _have_sc = False

# ---- Pfade (an deine Struktur angepasst)
BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_CANDIDATES = [
    os.path.join(BASE, "Daten", "ops_ml_processed.h5ad"),
    os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad"),
]
OUT_DIR_PLOTS = os.path.join(BASE, "Diagramme")
OUT_DIR_DATA = os.path.join(BASE, "Daten")
os.makedirs(OUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUT_DIR_DATA, exist_ok=True)

# ---- Lade AnnData (erste vorhandene Datei)
H5_PATH = next((p for p in H5_CANDIDATES if os.path.exists(p)), None)
if H5_PATH is None:
    raise FileNotFoundError(
        "Keine H5AD gefunden. Erwartet: Daten/ops_ml_processed.h5ad oder Daten/ops_with_patient_features.h5ad"
    )
adata = ad.read_h5ad(H5_PATH)

# ---- Zielvariable bestimmen (bevorzugt had_aki/AKI_linked_0_7)
preferred = ["had_aki", "AKI_linked_0_7", "AKI", "aki", "aki_linked_0_7"]
group_col = next((c for c in preferred if c in adata.obs.columns), None)
if group_col is None:
    # Fallback: erste binäre obs-Spalte
    for c in adata.obs.columns:
        vals = adata.obs[c].dropna().unique()
        if len(vals) == 2:
            group_col = c
            break
if group_col is None:
    raise RuntimeError(
        "Kein binäres Zielmerkmal in adata.obs gefunden (z. B. 'had_aki' oder 'AKI_linked_0_7')."
    )

# Normiere auf Kategorien "0"/"1"
if not pd.api.types.is_categorical_dtype(adata.obs[group_col]):
    adata.obs[group_col] = adata.obs[group_col].astype("category")
adata.obs[group_col] = (
    adata.obs[group_col]
    .astype(str)
    .replace({"False": "0", "True": "1", "no": "0", "yes": "1"})
    .astype("category")
)

# ---- Feature-Matrix bestimmen
use_X = True
try:
    n_obs, n_vars = adata.X.shape
    if n_vars == 0:
        use_X = False
except Exception:
    use_X = False

id_like = {
    "PMID",
    "SMID",
    "Procedure_ID",
    "pmid",
    "smid",
    "procedure_id",
    "patient_id",
    "stay_id",
}

if use_X:
    feature_names = (
        list(adata.var_names)
        if (adata.var_names is not None and len(adata.var_names) > 0)
        else [f"f{i}" for i in range(adata.X.shape[1])]
    )
    X = np.asarray(adata.X)
else:
    numeric_cols = [
        c
        for c in adata.obs.columns
        if c != group_col
        and c not in id_like
        and pd.api.types.is_numeric_dtype(adata.obs[c])
    ]
    if not numeric_cols:
        raise RuntimeError("Keine numerischen Features gefunden.")
    feature_names = numeric_cols
    X = adata.obs[numeric_cols].to_numpy()

# ---- Helper: recarray/structured array -> 2D-Array (rows x groups)
import numpy as _np


def to_2d(rec_or_arr):
    arr = rec_or_arr
    # structured/recarray?
    if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
        group_names = list(arr.dtype.names)
        cols = [_np.asarray(arr[name]) for name in group_names]  # je Gruppe 1 Spalte
        M = _np.column_stack(cols)  # (top_n x n_groups)
        return M, group_names
    # normales Array/Listen
    A = _np.asarray(arr, dtype=object)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A, [str(i) for i in range(A.shape[1])]


# ---- Weg A: ehrapy/scanpy verwenden
saved_any = False

try:
    if _have_ep or _have_sc:
        # Arbeitskopie mit konsistenten var_names
        if not use_X:
            adata_ep = ad.AnnData(X=X.copy())
            adata_ep.obs = adata.obs.copy()
            adata_ep.var_names = pd.Index(feature_names)
        else:
            adata_ep = adata.copy()
            if (adata_ep.var_names is None) or (len(adata_ep.var_names) != X.shape[1]):
                adata_ep.var_names = pd.Index(feature_names)

        # --- ehrapy bevorzugt ---
        if _have_ep and hasattr(ep, "tl") and hasattr(ep.tl, "rank_features_groups"):
            ran = False
            try:
                # Ohne 'method' wegen Versionskonflikten; use_raw=False falls vorhanden
                ep.tl.rank_features_groups(adata_ep, groupby=group_col, use_raw=False)
                ran = True
            except TypeError:
                try:
                    ep.tl.rank_features_groups(adata_ep, groupby=group_col)
                    ran = True
                except Exception:
                    ran = False
            except Exception:
                ran = False

            if ran:
                # Ergebnis robust extrahieren (funktioniert auch mit recarrays/structured arrays)
                r = adata_ep.uns.get("rank_genes_groups") or adata_ep.uns.get(
                    "rank_features_groups"
                )
                if isinstance(r, dict):
                    # Key-Namen robust finden
                    names_key = next(
                        (k for k in ("names", "features", "vars") if k in r), None
                    )
                    q_key = next(
                        (
                            k
                            for k in (
                                "pvals_adj",
                                "pvals_adj_mean",
                                "pvals_corrected",
                                "pvals",
                            )
                            if k in r
                        ),
                        None,
                    )
                    score_key = next(
                        (k for k in ("scores", "tvals", "statistics") if k in r), None
                    )
                    if names_key is not None and q_key is not None:
                        M_names, _ = to_2d(r[names_key])
                        M_q, _ = to_2d(r[q_key])
                        # Form anpassen
                        n_rows = min(M_names.shape[0], M_q.shape[0])
                        n_cols = min(M_names.shape[1], M_q.shape[1])
                        M_names = M_names[:n_rows, :n_cols]
                        M_q = M_q[:n_rows, :n_cols]
                        rows = []
                        for j in range(n_cols):
                            for i in range(n_rows):
                                nval = M_names[i, j]
                                qval = M_q[i, j]
                                try:
                                    qf = float(qval)
                                except Exception:
                                    qf = float("nan")
                                rows.append({"feature": str(nval), "qval": qf})
                        df_ep = (
                            pd.DataFrame(rows)
                            .groupby("feature", as_index=False)["qval"]
                            .min()
                            .sort_values("qval", na_position="last")
                        )
                        # optional: Score (falls vorhanden) aus erster Gruppe ziehen
                        if score_key is not None:
                            M_s, _ = to_2d(r[score_key])
                            s = {}
                            for i in range(min(M_s.shape[0], M_names.shape[0])):
                                try:
                                    s[str(M_names[i, 0])] = float(M_s[i, 0])
                                except Exception:
                                    pass
                            df_ep["score"] = df_ep["feature"].map(s)
                        df_ep.to_csv(
                            os.path.join(OUT_DIR_DATA, "ranked_features_table.csv"),
                            index=False,
                        )
                        saved_any = True

        # --- scanpy als Fallback ---
        if (not saved_any) and _have_sc:
            try:
                sc.tl.rank_genes_groups(
                    adata_ep,
                    groupby=group_col,
                    method="t-test",
                    corr_method="benjamini-hochberg",
                )

                rg = adata_ep.uns.get("rank_genes_groups", None)
                if rg is not None:
                    # scanpy speichert meist nach Gruppennamen dicts; wir nehmen alle Gruppen zusammen -> min q je Feature
                    names = rg.get("names")
                    pvals_adj = rg.get("pvals_adj")
                    if names is not None and pvals_adj is not None:
                        M_names, _ = to_2d(names)
                        M_q, _ = to_2d(pvals_adj)
                        n_rows = min(M_names.shape[0], M_q.shape[0])
                        n_cols = min(M_names.shape[1], M_q.shape[1])
                        rows = []
                        for j in range(n_cols):
                            for i in range(n_rows):
                                rows.append(
                                    {
                                        "feature": str(M_names[i, j]),
                                        "qval": float(M_q[i, j]),
                                    }
                                )
                        df_sc = (
                            pd.DataFrame(rows)
                            .groupby("feature", as_index=False)["qval"]
                            .min()
                            .sort_values("qval", na_position="last")
                        )
                        out_png = os.path.join(OUT_DIR_PLOTS, "ranked_features_ep.png")
                        save_barplot_from_table(
                            df_sc, out_png, "Univariates Ranking (scanpy)"
                        )

                        df_sc.to_csv(
                            os.path.join(OUT_DIR_DATA, "ranked_features_table.csv"),
                            index=False,
                        )
                        saved_any = True
            except Exception as e:
                print(f"[Warnung] scanpy-Ranking fehlgeschlagen → Fallback. Grund: {e}")
except Exception as e:
    print(f"[Warnung] ehrapy/scanpy-Ranking fehlgeschlagen → Fallback. Grund: {e}")


# ---- Weg B: Manuell (Welch t-test + BH-FDR) — falls nötig
if not saved_any:
    y = adata.obs[group_col].astype(str).values
    cats = sorted(pd.unique(y).tolist())
    if len(cats) != 2:
        raise RuntimeError(f"Erwarte zwei Gruppen, gefunden: {cats}")
    g1, g2 = cats[0], cats[1]
    idx1, idx2 = (y == g1), (y == g2)

    def welch_t(a, b):
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        n1, n2 = len(a), len(b)
        if n1 < 3 or n2 < 3:
            return np.nan, np.nan
        m1, m2 = a.mean(), b.mean()
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        den = math.sqrt(v1 / n1 + v2 / n2)
        if den == 0 or not np.isfinite(den):
            return np.nan, np.nan
        t = (m1 - m2) / den
        # df nach Welch-Satterthwaite
        df_num = (v1 / n1 + v2 / n2) ** 2
        df_den = (v1 * v1) / ((n1 * n1) * (n1 - 1)) + (v2 * v2) / ((n2 * n2) * (n2 - 1))
        df = df_num / df_den if df_den != 0 else (n1 + n2 - 2)
        try:
            from scipy.stats import t as tdist

            p = 2 * tdist.sf(abs(t), df)
        except Exception:
            # Normal-Approximation als Fallback
            from math import erf, sqrt

            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return t, p

    t_stats, pvals = [], []
    for j, name in enumerate(feature_names):
        t, p = welch_t(X[idx1, j], X[idx2, j])
        t_stats.append(t)
        pvals.append(p)

    df = pd.DataFrame(
        {"feature": feature_names, "t_stat": t_stats, "pval": pvals}
    ).copy()
    # BH-FDR
    m = df["pval"].notna().sum()
    order = np.argsort(df["pval"].fillna(1.0).values)
    p_sorted = df["pval"].fillna(1.0).values[order]
    q = np.empty_like(p_sorted)
    prev = 1.0
    for i in range(len(p_sorted) - 1, -1, -1):
        val = p_sorted[i] * m / (i + 1)
        prev = min(prev, val)
        q[i] = prev
    q_full = np.empty_like(q)
    q_full[order] = q
    df["qval"] = q_full
    df = df.sort_values(["qval", "pval"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT_DIR_DATA, "ranked_features_table.csv"), index=False)

    # Top-15 Plot
    topn = 15
    df_top = df.head(topn).copy()
    plt.figure(figsize=(8, 5))
    plt.barh(df_top["feature"][::-1], np.abs(df_top["t_stat"][::-1]))
    plt.xlabel("|t|-Statistik")
    plt.ylabel("Feature")
    plt.title(f"Top-{topn} Merkmale (Welch t-test): {g1} vs. {g2}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR_PLOTS, "ranked_features_manual.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

print("Fertig. Dateien in:", OUT_DIR_PLOTS, "und", OUT_DIR_DATA)
