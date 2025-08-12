#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
from anndata import read_h5ad, AnnData

SEC_PER_YEAR = 365.25 * 24 * 3600

def _calc_age_years_from_dob(op, dob):
    op_dt  = pd.to_datetime(op,  errors="coerce")
    dob_dt = pd.to_datetime(dob, errors="coerce")
    return (op_dt - dob_dt).dt.total_seconds() / SEC_PER_YEAR

def _sanitize_peds(age):
    age = pd.to_numeric(age, errors="coerce")
    age[(age < 0) | (age > 21)] = np.nan
    return age

def _make_age_cats(age):
    bins   = [-np.inf, 1, 5, 12, 18, np.inf]
    labels = ["<1 J", "1–4 J", "5–11 J", "12–17 J", "≥18 J"]
    return pd.Categorical(pd.cut(age, bins=bins, labels=labels, right=False))

def add_age_from_dob(adata: AnnData, dob_col: str, op_col="Surgery_Start", aki_col="AKI_Start") -> int:
    if dob_col not in adata.obs.columns or op_col not in adata.obs.columns:
        print(f"[WARN] Spalte fehlt (dob={dob_col!r} oder op={op_col!r}).")
        return 0
    adata.obs["age_years_at_op"] = _sanitize_peds(_calc_age_years_from_dob(adata.obs[op_col], adata.obs[dob_col]))
    if aki_col in adata.obs.columns:
        adata.obs["age_years_at_aki"] = _sanitize_peds(_calc_age_years_from_dob(adata.obs[aki_col], adata.obs[dob_col]))
    adata.obs["age_cat_op"] = _make_age_cats(adata.obs["age_years_at_op"])
    return int(adata.obs["age_years_at_op"].notna().sum())

def add_age_from_existing_fields(adata: AnnData, years=None, months=None, days=None) -> int:
    cols = list(map(str, adata.obs.columns))
    if years and years in cols:
        age = pd.to_numeric(adata.obs[years], errors="coerce")
    elif months and months in cols:
        age = pd.to_numeric(adata.obs[months], errors="coerce") / 12.0
    elif days and days in cols:
        age = pd.to_numeric(adata.obs[days], errors="coerce") / 365.25
    else:
        print("[WARN] Keine passenden age_* Felder gefunden.")
        return 0
    age = _sanitize_peds(age)
    adata.obs["age_years_at_op"] = age
    adata.obs["age_cat_op"] = _make_age_cats(adata.obs["age_years_at_op"])
    return int(age.notna().sum())

def merge_age_from_csv(adata: AnnData, csv_path: str, id_col="PMID",
                       dob_col: str=None, age_col: str=None, op_col: str="Surgery_Start") -> int:
    """
    Robust: liest CSV (Semikolon/Komma), säubert Spaltennamen (BOM/Spaces),
    mappt per ID ohne DataFrame-Merge.
    """
    import os, pandas as pd, numpy as np
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV nicht gefunden: {csv_path}")
        return 0
    if "PMID" not in adata.obs.columns:
        print("[WARN] 'PMID' fehlt in adata.obs.")
        return 0

    # CSV lesen, Separator automatisch erkennen, BOM/Whitespace aus Spalten entfernen
    meta = pd.read_csv(csv_path, sep=None, engine="python") if csv_path.lower().endswith(".csv") else pd.read_excel(csv_path)
    meta.columns = [str(c).strip().replace("\ufeff", "") for c in meta.columns]

    if id_col not in meta.columns:
        print(f"[WARN] ID '{id_col}' nicht in CSV. Spalten = {list(meta.columns)[:12]}")
        return 0

    # IDs als String angleichen (vermeidet Typ-Mismatches)
    obs_id = adata.obs["PMID"].astype(str)
    meta_id = meta[id_col].astype(str)

    age = None
    if dob_col and dob_col in meta.columns:
        dob_map = pd.Series(meta[dob_col].values, index=meta_id)
        dob_series = obs_id.map(dob_map)  # DOB je Zeile in adata.obs
        if op_col not in adata.obs.columns:
            print(f"[WARN] OP-Datum '{op_col}' nicht in adata.obs.")
            return 0
        age = _calc_age_years_from_dob(adata.obs[op_col], dob_series)
    elif age_col and age_col in meta.columns:
        age_map = pd.Series(meta[age_col].values, index=meta_id)
        age = pd.to_numeric(obs_id.map(age_map), errors="coerce")
    else:
        print(f"[WARN] Weder dob_col ({dob_col}) noch age_col ({age_col}) in CSV nutzbar.")
        return 0

    # Pädiatrische Plausibilität & speichern
    age = _sanitize_peds(age)
    adata.obs["age_years_at_op"] = age.values
    adata.obs["age_cat_op"] = _make_age_cats(adata.obs["age_years_at_op"])
    return int(adata.obs["age_years_at_op"].notna().sum())
def main():
    ap = argparse.ArgumentParser(description="ehrapy-first Altersfeatures in H5AD ergänzen")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--save-inplace", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dob-col")
    ap.add_argument("--op-col", default="Surgery_Start")
    ap.add_argument("--aki-col", default="AKI_Start")
    ap.add_argument("--age-years-col")
    ap.add_argument("--age-months-col")
    ap.add_argument("--age-days-col")
    ap.add_argument("--csv")
    ap.add_argument("--csv-id-col", default="PMID")
    ap.add_argument("--csv-dob-col")
    ap.add_argument("--csv-age-col")
    args = ap.parse_args()

    ad = read_h5ad(args.h5)

    if args.inspect:
        cols = list(map(str, ad.obs.columns))
        cands = [c for c in cols if any(k in c.lower() for k in ["dob","birth","geb","datum","age","alter"])]
        print("OBS-Spalten (erste 60):", cols[:60])
        print("Kandidaten:", cands)
        return

    n = 0; method = None
    if args.dob_col:
        n = add_age_from_dob(ad, args.dob_col, op_col=args.op_col, aki_col=args.aki_col); method=f"dob:{args.dob_col}"
    elif any([args.age_years_col, args.age_months_col, args.age_days_col]):
        n = add_age_from_existing_fields(ad, args.age_years_col, args.age_months_col, args.age_days_col); method="existing"
    elif args.csv:
        n = merge_age_from_csv(ad, args.csv, id_col=args.csv_id_col, dob_col=args.csv_dob_col, age_col=args.csv_age_col, op_col=args.op_col); method="csv"
    else:
        print("[INFO] Keine Methode gewählt. Nutze --inspect oder eine der Optionen.")
        return

    ad.uns.setdefault("age_features", {})
    ad.uns["age_features"].update({"method": method, "n_valid_age_op": int(n), "op_col": args.op_col, "aki_col": args.aki_col})
    out = args.h5 if args.save_inplace else (args.out or args.h5.replace(".h5ad","_with_age.h5ad"))
    ad.write_h5ad(out)
    print(f"[OK] age_years_at_op in {n} Zeilen gesetzt. Gespeichert: {out}")

if __name__ == "__main__":
    main()