#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
import pandas as pd

# ---------- Deterministische Datumsparser ----------
DEFAULT_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
]

def _parse_with_formats(s: pd.Series, formats, *, dayfirst=False) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.mask(s.eq(""), other=pd.NA)
    out = None
    for f in formats:
        try:
            t = pd.to_datetime(s, format=f, errors="coerce", dayfirst=dayfirst)
            out = t if out is None else out.fillna(t)
        except Exception:
            pass
    if out is None:
        out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return pd.to_datetime(out, errors="coerce")  # ns-normalisiert

def enforce_datetime_columns(df: pd.DataFrame, fmt_map: dict, *, dayfirst=False) -> pd.DataFrame:
    df = df.copy()
    for col, fmts in fmt_map.items():
        if col in df.columns:
            df[col] = _parse_with_formats(df[col], list(fmts), dayfirst=dayfirst)
    return df

# ---------- Schema-Check ----------
def require_columns(df: pd.DataFrame, required: list[str], table_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        cols_preview = ", ".join(map(str, list(df.columns)[:20]))
        raise SystemExit(
            f"[ERROR] {table_name}: Pflichtspalten fehlen: {missing}\n"
            f"Vorhandene Spalten (Auszug): {cols_preview}"
        )

def dtype_str(s):  # hübsche dtype-Ausgabe
    try: return str(getattr(s, "dtype", s))
    except: return str(s)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Validate OPS/PATIENTS/AKI CSVs: Schema, Datums-Parsing, Feature-Keys, X-Features.")
    ap.add_argument("--ops", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--aki")
    ap.add_argument("--features", nargs="*", default=[])
    ap.add_argument("--x-features", nargs="*", default=[])
    ap.add_argument("--dayfirst", action="store_true")
    args = ap.parse_args()

    print("=== VALIDATION START ===")
    print(f"OPS       : {args.ops}")
    print(f"PATIENTS  : {args.patients}")
    print(f"AKI       : {args.aki or 'None'}")
    print(f"FEATURES  : {args.features or '[]'}")
    print(f"X-FEATURES: {args.x_features or '[]'}\n")

    # OPS
    ops = pd.read_csv(args.ops, sep=None, engine="python")
    ops.columns = [str(c).strip() for c in ops.columns]
    print(f"[OPS] shape={ops.shape}")
    require_columns(ops, ["PMID","SMID","Procedure_ID","Surgery_Start","Surgery_End"], "OPS")
    ops = enforce_datetime_columns(ops, {"Surgery_Start": DEFAULT_DT_FORMATS, "Surgery_End": DEFAULT_DT_FORMATS},
                                   dayfirst=args.dayfirst)
    if "duration_minutes" not in ops.columns:
        dur = (ops["Surgery_End"] - ops["Surgery_Start"]).dt.total_seconds() / 60.0
        ops["duration_minutes"] = dur
        print("[OPS] duration_minutes berechnet.")
    for c in ["Surgery_Start","Surgery_End"]:
        nonna = ops[c].notna().sum()
        print(f"[OPS] {c}: parsed={nonna}/{len(ops)}  min={ops[c].min()}  max={ops[c].max()}")

    # PATIENTS
    pat = pd.read_csv(args.patients, sep=None, engine="python")
    pat.columns = [str(c).strip() for c in pat.columns]
    print(f"[PAT] shape={pat.shape}")
    require_columns(pat, ["PMID"], "PATIENTS")

    # AKI
    if args.aki:
        aki = pd.read_csv(args.aki, sep=None, engine="python")
        aki.columns = [str(c).strip() for c in aki.columns]
        print(f"[AKI] shape={aki.shape}")
        if ("AKI_Start" not in aki.columns) and ("Start" in aki.columns):
            aki = aki.rename(columns={"Start":"AKI_Start"})
        require_columns(aki, ["PMID","AKI_Start"], "AKI")
        aki = enforce_datetime_columns(aki, {"AKI_Start": DEFAULT_DT_FORMATS}, dayfirst=args.dayfirst)
        nonna = aki["AKI_Start"].notna().sum()
        print(f"[AKI] AKI_Start parsed={nonna}/{len(aki)}  min={aki['AKI_Start'].min()}  max={aki['AKI_Start'].max()}")

    # Feature-Dateien (Schlüssel & dtypes)
    base_keys = ["PMID","SMID","Procedure_ID"]
    have_cols = set(map(str, ops.columns))
    for fp in args.features:
        p = Path(fp)
        if not p.exists():
            print(f"[WARN] Feature-Datei fehlt: {p}")
            continue
        f = pd.read_csv(p, sep=None, engine="python", nrows=1000)
        f.columns = [str(c).strip() for c in f.columns]
        print(f"[FEAT] {p.name} shape={f.shape}  first_cols={list(f.columns)[:8]}")
        common = [k for k in base_keys if k in f.columns and k in ops.columns]
        if not common and ("uniq_key" in f.columns and "uniq_key" in ops.columns):
            common = ["uniq_key"]
        if not common:
            raise SystemExit(f"[ERROR] {p.name}: keine gemeinsamen Schlüssel in {base_keys} bzw. uniq_key.")
        for k in common:
            print(f"[FEAT] Key '{k}' dtypes -> OPS:{dtype_str(ops[k])}  FEAT:{dtype_str(f[k])}")
            if str(ops[k].dtype) != str(f[k].dtype):
                print(f"[HINT] {p.name}: Cast keys to str beim Merge empfohlen.")
        have_cols |= set(map(str, f.columns))

    # X-Features-Check
    if args.x-features if False else False: pass  # Platzhalter, verhindert Pylint-Fehlalarme :)
    if args.x_features:
        missing = [x for x in args.x_features if x not in have_cols]
        if missing:
            print(f"[WARN] X-Features fehlen in Eingaben: {missing}")
        else:
            print("[OK] Alle X-Features in OPS/Features vorhanden.")

    print("\n=== VALIDATION OK (keine harten Fehler) ===")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
