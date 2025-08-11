import os, math
import numpy as np
import pandas as pd
from anndata import read_h5ad

H5 = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
OUT_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
CSV_OUT = os.path.join(OUT_DIR, "Table1_OP_level_with_stats.csv")
DOCX_OUT = os.path.join(OUT_DIR, "Table1_OP_level_with_stats.docx")

try:
    from scipy import stats
    SCIPY = True
except Exception:
    SCIPY = False

def r(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.{nd}f}"

def binary_effects(ctab_2x2: pd.DataFrame):
    a,b = int(ctab_2x2.iloc[0,0]), int(ctab_2x2.iloc[0,1])
    c,d = int(ctab_2x2.iloc[1,0]), int(ctab_2x2.iloc[1,1])
    aa,bb,cc,dd = a or 0.5, b or 0.5, c or 0.5, d or 0.5
    OR = (aa*dd)/(bb*cc)
    se = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    ci_lo, ci_hi = math.exp(math.log(OR)-1.96*se), math.exp(math.log(OR)+1.96*se)
    p1 = a/(a+c) if (a+c)>0 else np.nan
    p0 = b/(b+d) if (b+d)>0 else np.nan
    RD = p1 - p0 if not (np.isnan(p1) or np.isnan(p0)) else np.nan
    if SCIPY:
        try:
            _, p = stats.fisher_exact(ctab_2x2.values)
            test = "Fisher"
        except Exception:
            chi2, p, _, _ = stats.chi2_contingency(ctab_2x2.values, correction=False)
            test = "Chi²"
    else:
        p, test = None, "—"
    return OR, (ci_lo, ci_hi), RD, p, test

def kgt2_effect(ctab: pd.DataFrame):
    if SCIPY:
        chi2, p, _, _ = stats.chi2_contingency(ctab.values, correction=False)
        n = ctab.values.sum()
        k = min(ctab.shape)-1
        V = math.sqrt(chi2/(n*k)) if n>0 and k>0 else np.nan
        return V, p, "Chi²"
    return np.nan, None, "—"

def build_table1(adata) -> pd.DataFrame:
    need = ["event_idx","duration_hours","Sex_norm","is_reop","duration_tertile"]
    for c in need:
        if c not in adata.obs.columns:
            adata.obs[c] = np.nan

    surv = adata.obs[need].copy()
    g1, g0 = "AKI-Index-OP", "Keine AKI-Index-OP"
    surv["Gruppe"] = np.where(surv["event_idx"].astype(int)==1, g1, g0)

    rows = []

    # Dauer (Stunden): MWU + Welch + Hedges g
    x1 = pd.to_numeric(surv.loc[surv["Gruppe"]==g1, "duration_hours"], errors="coerce").dropna()
    x0 = pd.to_numeric(surv.loc[surv["Gruppe"]==g0, "duration_hours"], errors="coerce").dropna()
    if SCIPY and x1.size>1 and x0.size>1:
        U, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        t, p_welch = stats.ttest_ind(x1, x0, equal_var=False)
    else:
        p_mw, p_welch = None, None

    g = np.nan
    if x1.size>1 and x0.size>1:
        m1, m0 = x1.mean(), x0.mean()
        s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
        n1, n0 = x1.size, x0.size
        denom = (n1+n0-2)
        s_pooled = math.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/denom) if denom>0 else np.nan
        if s_pooled and not np.isnan(s_pooled) and s_pooled>0:
            d = (m1-m0)/s_pooled
            J = 1 - (3/(4*(n1+n0)-9)) if (n1+n0)>3 else 1.0
            g = J*d

    rows.append({
        "Variable": "Dauer (Stunden)",
        g1: f"{r(x1.median())} [{r(x1.quantile(.25))}; {r(x1.quantile(.75))}] (n={x1.size})",
        g0: f"{r(x0.median())} [{r(x0.quantile(.25))}; {r(x0.quantile(.75))}] (n={x0.size})",
        "Hedges g": r(g,2),
        "p (Mann-Whitney)": r(p_mw,3) if p_mw is not None else "",
        "p (Welch)": r(p_welch,3) if p_welch is not None else "",
        "Hinweis": "Median[IQR]; MW primär; Welch als Zusatz"
    })

    def cat_block(col, label):
        out = []
        ser = surv[col]
        if ser.isna().all(): 
            return out
        ctab = pd.crosstab(ser, surv["Gruppe"])
        if ctab.shape[0]==2:
            OR, ci, RD, p, test = binary_effects(ctab)
            for cat, row in ctab.iterrows():
                n1, n0 = int(row.get(g1,0)), int(row.get(g0,0))
                tot1 = int(ctab[g1].sum()); tot0 = int(ctab[g0].sum())
                out.append({
                    "Variable": f"{label}: {cat}",
                    g1: f"{n1} ({r(100*n1/tot1,1)}%)" if tot1 else f"{n1}",
                    g0: f"{n0} ({r(100*n0/tot0,1)}%)" if tot0 else f"{n0}",
                    "Effekt": f"OR {r(OR,2)} [{r(ci[0],2)}; {r(ci[1],2)}]",
                    "p-Wert": r(p,3) if p is not None else "",
                    "Test": test
                })
            out.append({
                "Variable": f"{label}: Risikodifferenz (G1−G0, erste Kategorie)",
                g1: f"n={int(ctab[g1].sum())}",
                g0: f"n={int(ctab[g0].sum())}",
                "Effekt": f"RD {r(RD,3)}",
                "p-Wert": r(p,3) if p is not None else "",
                "Test": test
            })
        else:
            V, p, test = kgt2_effect(ctab)
            for cat, row in ctab.iterrows():
                n1, n0 = int(row.get(g1,0)), int(row.get(g0,0))
                tot1 = int(ctab[g1].sum()); tot0 = int(ctab[g0].sum())
                out.append({
                    "Variable": f"{label}: {cat}",
                    g1: f"{n1} ({r(100*n1/tot1,1)}%)" if tot1 else f"{n1}",
                    g0: f"{n0} ({r(100*n0/tot0,1)}%)" if tot0 else f"{n0}",
                    "Effekt": f"Cramérs V {r(V,2)}",
                    "p-Wert": r(p,3) if p is not None else "",
                    "Test": test
                })
        return out

    rows += cat_block("Sex_norm", "Geschlecht")
    rows += cat_block("is_reop", "Re-OP (ja=1, nein=0)")
    rows += cat_block("duration_tertile", "OP-Dauer (Tertile)")

    return pd.DataFrame(rows)

def maybe_write_docx(table: pd.DataFrame, path: str):
    try:
        from docx import Document
    except Exception:
        return False
    doc = Document()
    doc.add_heading('Table 1 (OP-Level): Basischarakteristika mit Tests', level=1)
    doc.add_paragraph('Gruppen: AKI-Index-OP vs. Keine AKI-Index-OP. Prozentwerte sind gruppenintern.')
    doc.add_paragraph('Dauer: Median [IQR]; Mann-Whitney p; Welch p als Zusatz.')
    doc.add_paragraph('Kategorial: Fisher/Chi²; Effekte: OR (95%-KI) bzw. Cramérs V; RD zusätzlich.')
    t = doc.add_table(rows=1, cols=len(table.columns))
    for j, c in enumerate(table.columns):
        t.rows[0].cells[j].text = str(c)
    for _, row in table.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(table.columns):
            cells[j].text = "" if pd.isna(row[c]) else str(row[c])
    doc.save(path)
    return True

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    adata = read_h5ad(H5)
    table = build_table1(adata)
    table.to_csv(CSV_OUT, index=False)
    print("CSV gespeichert:", CSV_OUT)
    adata.uns["table1"] = {
        "groups": ["AKI-Index-OP", "Keine AKI-Index-OP"],
        "notes": {
            "duration_tests": "Mann-Whitney (primär), Welch t (Zusatz)",
            "categorical_tests": "Fisher/Chi²; Effekte: OR (2x2) bzw. Cramérs V (>2 Kat.), RD zusätzlich"
        },
        "table": table.to_dict(orient="records")
    }
    adata.write_h5ad(H5)
    print("uns['table1'] geschrieben und H5AD gespeichert.")
    if maybe_write_docx(table, DOCX_OUT):
        print("DOCX gespeichert:", DOCX_OUT)
    else:
        print("DOCX übersprungen (python-docx fehlt oder Fehler).")

    # kurze Kontrolle
    ad = read_h5ad(H5)
    print("uns keys:", list(ad.uns.keys()))
    print("table1 rows:", len(ad.uns.get("table1", {}).get("table", [])))

if __name__ == "__main__":
    main()
