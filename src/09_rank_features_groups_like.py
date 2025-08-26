#!/usr/bin/env python3
"""
Univariates Feature-Ranking für binäre Gruppen (AKI 0–7 = 0/1).

Versuch 1: ehrapy -> ep.tl.rank_features_groups(ad, groupby=…)
   (ohne 'method'-Argument, da manche Versionen das intern schon setzen)
   Ergebnis aus ad.uns[...] extrahieren.

Fallback: Welch-t-Test + Effektgrößen (Cohen's d, Hedges' g) + BH-FDR (q)
   -> Daten/rank_features_ttest.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from anndata import read_h5ad

FORCE_FALLBACK_ONLY = True

# Optional: ehrapy & scipy
try:
    import ehrapy as ep  # type: ignore
except Exception:
    ep = None  # type: ignore

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None  # type: ignore

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
H5_MAIN = os.path.join(BASE, "Daten", "ops_ml_processed.h5ad")
H5_FALL = os.path.join(BASE, "Daten", "ops_with_patient_features.h5ad")
OUT_CSV = os.path.join(BASE, "Daten", "rank_features_ttest.csv")

GROUP_COLS = ["had_aki", "AKI_linked_0_7"]  # erste vorhandene wird genutzt


def get_adata():
    path = H5_MAIN if os.path.exists(H5_MAIN) else H5_FALL
    return read_h5ad(path)


def pick_group_col(obs: pd.DataFrame) -> str:
    for c in GROUP_COLS:
        if c in obs.columns:
            return c
    raise KeyError("Gruppierungsvariable 'had_aki' oder 'AKI_linked_0_7' fehlt in .obs")


def welch_t_and_effects(x0: np.ndarray, x1: np.ndarray):
    """Welch's t (two-sided) + Cohen's d + Hedges' g."""
    if stats is not None:
        t_stat, p = stats.ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
    else:
        t_stat, p = np.nan, np.nan
    n0, n1 = x0.size, x1.size
    m0, m1 = float(np.nanmean(x0)), float(np.nanmean(x1))
    s0, s1 = float(np.nanstd(x0, ddof=1)), float(np.nanstd(x1, ddof=1))
    sp_num = (n0 - 1) * (s0**2) + (n1 - 1) * (s1**2)
    sp_den = n0 + n1 - 2
    sp = np.sqrt(sp_num / sp_den) if sp_den > 0 and sp_num >= 0 else np.nan
    d = (m1 - m0) / sp if (np.isfinite(sp) and sp > 0) else np.nan
    J = 1.0 - (3.0 / (4.0 * (n0 + n1) - 9.0)) if (n0 + n1) > 2 else 1.0
    g = d * J if np.isfinite(d) else np.nan
    return (
        float(t_stat) if np.isfinite(t_stat) else np.nan,
        float(p) if np.isfinite(p) else np.nan,
        float(d) if np.isfinite(d) else np.nan,
        float(g) if np.isfinite(g) else np.nan,
    )


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = q
    return out


def try_rank_with_ehrapy(ad, gcol: str) -> pd.DataFrame | None:
    """Versucht ep.tl.rank_features_groups und gibt DataFrame (variable, qval) zurück – oder None."""
    if not (
        ep is not None and hasattr(ep, "tl") and hasattr(ep.tl, "rank_features_groups")
    ):
        return None

    # Aufruf OHNE 'method' (manche Versionen setzen das intern)
    ran = False
    try:
        ep.tl.rank_features_groups(ad, groupby=gcol, use_raw=False)
        ran = True
    except TypeError:
        try:
            ep.tl.rank_features_groups(ad, groupby=gcol)
            ran = True
        except Exception:
            ran = False
    except Exception:
        ran = False
    if not ran:
        return None

    r = ad.uns.get("rank_genes_groups") or ad.uns.get("rank_features_groups")
    if not isinstance(r, dict):
        return None

    # ---- Helper: recarray/structured array -> 2D-Array ----
    import numpy as _np

    def to_2d(rec_or_arr):
        """Gibt (M, group_names) zurück: M = 2D-Array (rows x groups)"""
        arr = rec_or_arr
        # structured/recarray?
        if hasattr(arr, "dtype") and arr.dtype.names:
            group_names = list(arr.dtype.names)
            cols = [
                _np.asarray(arr[name]) for name in group_names
            ]  # je Gruppe 1 Spalte
            M = _np.column_stack(cols)  # (top_n x n_groups)
            return M, group_names
        # normales Array/Listen
        A = _np.asarray(arr, dtype=object)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        # Gruppenbezeichner unbekannt -> 0..n-1
        return A, [str(i) for i in range(A.shape[1])]

    # Schlüsselfelder robust bestimmen
    names_key = None
    for k in ("names", "features", "vars"):
        if k in r:
            names_key = k
            break
    q_key = None
    for k in ("pvals_adj", "pvals_adj_mean", "pvals_corrected", "pvals"):
        if k in r:
            q_key = k
            break
    if names_key is None or q_key is None:
        return None

    names_2d, _ = to_2d(r[names_key])  # Strings
    q_2d, _ = to_2d(r[q_key])  # Zahlen, aber evtl. als object

    # Form kompatibel machen
    n_rows = min(names_2d.shape[0], q_2d.shape[0])
    n_cols = min(names_2d.shape[1], q_2d.shape[1])
    names_2d = names_2d[:n_rows, :n_cols]
    q_2d = q_2d[:n_rows, :n_cols]

    # In float umwandeln (Element-weise), dabei Fehler abfangen
    rows = []
    for j in range(n_cols):
        for i in range(n_rows):
            n = names_2d[i, j]
            qv = q_2d[i, j]
            try:
                qf = float(qv)
            except Exception:
                qf = _np.nan
            rows.append({"variable": str(n), "qval": qf})

    if not rows:
        return None

    df = (
        pd.DataFrame(rows)
        .groupby("variable", as_index=False)["qval"]
        .min()  # minimales q je Feature über Gruppen
        .sort_values("qval", na_position="last")
    )
    return df


def main():
    ad = get_adata()

    # Gruppenspalte vorbereiten
    gcol = pick_group_col(ad.obs)
    g = (
        ad.obs[gcol]
        .astype(str)
        .replace({"False": "0", "True": "1", "no": "0", "yes": "1"})
    )
    idx0 = np.where(g.values == "0")[0]
    idx1 = np.where(g.values == "1")[0]
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError(
            f"Eine Gruppe ist leer: counts -> 0:{idx0.size}, 1:{idx1.size}"
        )

    var_names = list(map(str, ad.var_names))
    df_ep = None if FORCE_FALLBACK_ONLY else try_rank_with_ehrapy(ad, gcol)
    if df_ep is not None and not df_ep.empty:
        df_ep.to_csv(OUT_CSV, index=False)
        print("✔ gespeichert (ehrapy):", OUT_CSV)
        return

    # 1) ehrapy versuchen
    df_ep = try_rank_with_ehrapy(ad, gcol)
    if df_ep is not None and not df_ep.empty:
        df_ep.to_csv(OUT_CSV, index=False)
        print("✔ gespeichert (ehrapy):", OUT_CSV)
        return  # <<< return NUR innerhalb main()

    # 2) Fallback: Welch t-Test + Effektgrößen + BH-FDR
    X = np.asarray(ad.X, dtype=float)
    rows = []
    for j, v in enumerate(var_names):
        x0 = X[idx0, j]
        x1 = X[idx1, j]
        x0 = x0[np.isfinite(x0)]
        x1 = x1[np.isfinite(x1)]
        if x0.size < 3 or x1.size < 3:
            t = p = d = g_eff = np.nan
            m0 = m1 = s0 = s1 = np.nan
        else:
            t, p, d, g_eff = welch_t_and_effects(x0, x1)
            m0, m1 = float(np.mean(x0)), float(np.mean(x1))
            s0, s1 = float(np.std(x0, ddof=1)), float(np.std(x1, ddof=1))
        rows.append(
            {
                "variable": v,
                "n0": int(x0.size),
                "mean0": m0,
                "sd0": s0,
                "n1": int(x1.size),
                "mean1": m1,
                "sd1": s1,
                "t": t,
                "p": p,
                "cohens_d": d,
                "hedges_g": g_eff,
                "mean_diff_1_minus_0": (
                    (m1 - m0) if (np.isfinite(m0) and np.isfinite(m1)) else np.nan
                ),
            }
        )

    df = pd.DataFrame(rows)
    df["q"] = bh_fdr(df["p"].values)
    df = df.sort_values(["q", "p"], na_position="last").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print("✔ gespeichert:", OUT_CSV)


if __name__ == "__main__":
    main()
