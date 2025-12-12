#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Daten prüfen (Standard-Workflow):
#   ./scripts/1_check_data.sh
#
# Direkt (ohne Skript):
#   conda activate ehrapy_ml
#   python src/validate_dates_and_schema.py \
#     --ops "HLM Operationen.csv" \
#     --patients "Patient Master Data.csv" \
#     --aki "AKI Label.csv" \
#     --features "Daten/ops_with_crea_cysc_vis_features.csv" \
#     --x-features crea_delta_0_48 crea_rate_0_48 vis_auc_0_24 vis_auc_0_48 duration_minutes
# -----------------------------------------------------------------------------

import sys, argparse
from pathlib import Path
import pandas as pd
import re

# -------- Spalten-Aliase & Cleaner --------
OPS_ALIASES = {
    "Start of surgery": "Surgery_Start",
    "End of surgery": "Surgery_End",
    "start of surgery": "Surgery_Start",
    "end of surgery": "Surgery_End",
}


def clean_columns(df, aliases=None):
    """Trim, sonderbare Leerzeichen entfernen, Aliase anwenden."""
    trans = {
        ord("\ufeff"): None,
        ord("\u00a0"): " ",
        ord("\u2007"): " ",
        ord("\u202f"): " ",
    }
    cols = []
    for c in df.columns:
        c = str(c).translate(trans)
        c = re.sub(r"\s+", " ", c).strip()
        cols.append(c)
    df.columns = cols
    if aliases:
        df = df.rename(columns=aliases)
    return df


# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Validate OPS/PAT/AKI CSV & Feature-Schema"
    )
    ap.add_argument("--ops", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--aki", default=None)
    ap.add_argument("--features", nargs="*", default=[])
    ap.add_argument("--x-features", nargs="*", default=[])
    ap.add_argument(
        "--dayfirst", action="store_true", help="Europäische Datumserkennung bevorzugen"
    )
    return ap.parse_args()


def head_cols(df, n=6):
    return ", ".join(list(map(str, df.columns[:n])))


DERIVED_OK = {"duration_minutes", "duration_hours"}


def main():
    args = parse_args()
    print("=== VALIDATION START ===")
    print("OPS       :", args.ops)
    print("PATIENTS  :", args.patients)
    print("AKI       :", args.aki)
    print("FEATURES  :", args.features if args.features else None)
    print("X-FEATURES:", args.x_features if args.x_features else None)
    print()

    # ----- OPS laden & normalisieren -----
    ops = pd.read_csv(args.ops, sep=None, engine="python")
    ops = clean_columns(ops, aliases=OPS_ALIASES)
    print(f"[OPS] shape={ops.shape}")

    # Pflichtspalten in OPS
    need_ops = {"PMID", "Surgery_Start", "Surgery_End"}
    miss_ops = [c for c in need_ops if c not in ops.columns]
    if miss_ops:
        print(f"[ERROR] OPS: Pflichtspalten fehlen: {miss_ops}")
        print("Vorhandene Spalten (Auszug):", head_cols(ops))
        return 1

    # ----- PAT nur minimal prüfen -----
    try:
        pat = pd.read_csv(args.patients, sep=None, engine="python")
        pat = clean_columns(pat)
        print(f"[PAT] shape={pat.shape}")
        if "PMID" not in pat.columns:
            print("[WARN] PATIENTS: 'PMID' fehlt – Join wird ggf. eingeschränkt.")
    except Exception as e:
        print(f"[ERROR] PATIENTS nicht lesbar: {e}")
        return 1

    # ----- AKI optional prüfen -----
    aki_cols_union = set()
    if args.aki:
        try:
            aki = pd.read_csv(args.aki, sep=None, engine="python")
            aki = clean_columns(aki)
            aki_cols_union = set(aki.columns)
            print(f"[AKI] shape={aki.shape}")
            if not (("AKI_Start" in aki.columns) or ("Start" in aki.columns)):
                print(
                    "[WARN] AKI: Weder 'AKI_Start' noch 'Start' gefunden – Linking evtl. ohne Effekt."
                )
        except Exception as e:
            print(f"[WARN] AKI nicht lesbar: {e}")

    # ----- Feature-Dateien (optional) prüfen & X-Features validieren -----
    union_cols = set(ops.columns)
    for f in args.features or []:
        try:
            df = pd.read_csv(f, nrows=1, sep=None, engine="python")
            df = clean_columns(df)
            union_cols.update(df.columns)
        except Exception as e:
            print(f"[WARN] Features-Datei nicht lesbar ({f}): {e}")

    missing_x = [
        x
        for x in (args.x_features or [])
        if (x not in union_cols and x not in DERIVED_OK)
    ]
    if missing_x:
        print(f"[WARN] X-Features fehlen in OPS/Features: {missing_x}")
    else:
        if args.x_features:
            print("[OK] Alle X-Features vorhanden.")

    print("\n=== VALIDATION OK (keine harten Fehler) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
