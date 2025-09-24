#!/usr/bin/env python3
"""
Build or update a clinician‑friendly AnnData (.h5ad) from hospital CSVs.

Made for: pediatric cardiac surgery cohort (OP‑Ereignisse, Stammdaten, AKI‑Labels, optionale Feature‑Tabellen)
Author: (your name)

USAGE (examples)
----------------
# 1) Erstellen (Build) – Minimal (nur 3 Pflicht‑Tabellen)
python src/00_build_anndata_cli.py \
  --ops "/Users/fa/.../HLM Operationen.csv" \
  --patients "/Users/fa/.../Patient Master Data.csv" \
  --aki "/Users/fa/.../AKI Label.csv" \
  --out "/Users/fa/.../h5ad/ops_with_age_groups.h5ad"

# 2) Mit optionalen Feature‑Tabellen (werden per Schlüssel gemerged)
python src/00_build_anndata_cli.py \
  --ops ".../HLM Operationen.csv" \
  --patients ".../Patient Master Data.csv" \
  --aki ".../AKI Label.csv" \
  --features ".../lab_features_by_op.csv" ".../vis_features_by_op.csv" \
  --x-features "crea_delta_0_48" "crea_rate_0_48" "vis_auc_0_24" "vis_auc_0_48" "duration_minutes" \
  --out ".../h5ad/ops_ml_processed.h5ad"

# 3) Update – bestehendes .h5ad inkrementell erweitern (neue OPs/Patienten)
python src/00_build_anndata_cli.py \
  --ops ".../HLM Operationen_NEW.csv" \
  --patients ".../Patient Master Data_NEW.csv" \
  --aki ".../AKI Label_NEW.csv" \
  --update "/Users/fa/.../h5ad/ops_with_age_groups.h5ad" \
  --out "/Users/fa/.../h5ad/ops_with_age_groups.h5ad"

HINWEISE
--------
• Für Kliniker:innen gedacht: robuste Einleseroutine, einfache Schlüssel, saubere .obs. 
• Datumswerte werden in ISO‑Strings konvertiert (HDF5 verträgt keine datetime64 in .obs).
• .X wird nur gefüllt, wenn --x-features angegeben sind. Andernfalls bleibt .X=None (lesefreundlich).
• Deduplizierung über zusammengesetzten Schlüssel (PMID|SMID|Procedure_ID|Surgery_Start_ISO).

Benötigte Pakete: pandas, numpy, anndata, python-dateutil
pip install pandas numpy anndata python-dateutil
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from dateutil import tz
from anndata import AnnData
import anndata as ad

# ------------------------------
# Utils
# ------------------------------

def _read_csv_robust(path: str | Path, dtype_map: Optional[dict] = None) -> pd.DataFrame:
    """Robust CSV reader: auto‑detect sep, handle UTF‑8‑SIG, keep IDs as strings.
    Falls sep nicht erkannt wird, probiere Standardtrennzeichen.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    try:
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig", dtype=dtype_map)
    except Exception:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(p, sep=sep, encoding="utf-8-sig", dtype=dtype_map)
                break
            except Exception:
                df = None  # type: ignore
        if df is None:
            raise
    # Spaltennamen säubern
    df.columns = df.columns.str.strip()
    # Strings trimmen
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def _to_dt(series: pd.Series) -> pd.Series:
    """Parse datetime robust (errors='coerce'), entferne TZ (naiv), gebe pd.Series zurück."""
    s = pd.to_datetime(series, errors="coerce")
    # In naive Zeit umwandeln (falls Zeitzone vorhanden war)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)  # type: ignore
    return s


def _iso_or_empty(series: pd.Series) -> pd.Series:
    """Datetime‑Serie → ISO‑String (YYYY-MM-DD HH:MM:SS), fehlend = '' (leer)."""
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S").where(dt.notna(), "")


def _normalize_sex(x: str | float | int) -> str:
    s = ("" if pd.isna(x) else str(x)).strip().lower()
    if s in {"w", "f", "female", "frau"}:  # weiblich
        return "f"
    if s in {"m", "male", "mann"}:  # männlich
        return "m"
    return "u"  # unbekannt


def _age_years(start: pd.Series, dob: pd.Series) -> pd.Series:
    days = (start - dob).dt.days
    return (days / 365.25).round(3)


def _age_group(years: pd.Series) -> pd.Categorical:
    # Kategorien (Beipiel, an BA angepasst)
    bins = [-np.inf, 0.083, 1, 3, 6, 12, 18, np.inf]  # ~ 1 Monat, 1J, 3J, 6J, 12J, 18J
    labels = [
        "Neonates",
        "Infants",
        "Toddlers",
        "Preschool",
        "School-age",
        "Adolescents",
        "Unbekannt/außerhalb",
    ]
    cat = pd.cut(years.fillna(-1), bins=bins, labels=labels, right=True, include_lowest=True)
    cat = cat.astype("category")
    cat = cat.cat.reorder_categories(labels, ordered=True)
    return cat


def _make_unique_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["PMID", "SMID", "Procedure_ID", "Surgery_Start"]:
        if col in df.columns:
            parts.append(df[col].astype(str).fillna(""))
        else:
            parts.append(pd.Series([""] * len(df)))
    key = parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3]
    return key


# ------------------------------
# Readers & transformers
# ------------------------------

def read_ops_table(path_ops: str | Path) -> pd.DataFrame:
    """Read operations table and standardize columns."""
    dtype_map = {"PMID": str, "SMID": str, "Procedure_ID": str}
    df = _read_csv_robust(path_ops, dtype_map=dtype_map)
    # Erwartete Spaltenvarianten mappen
    rename_map = {
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "start_of_surgery": "Surgery_Start",
        "end_of_surgery": "Surgery_End",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    # Pflichtspalten prüfen
    needed = {"PMID", "Surgery_Start", "Surgery_End"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"In OP‑Tabelle fehlen Spalten: {missing}")
    # Datumswerte
    df["Surgery_Start"] = _to_dt(df["Surgery_Start"])  # type: ignore
    df["Surgery_End"] = _to_dt(df["Surgery_End"])  # type: ignore
    # Dauer
    df["duration_minutes"] = (df["Surgery_End"] - df["Surgery_Start"]).dt.total_seconds() / 60.0
    df["duration_hours"] = (df["duration_minutes"]) / 60.0
    # Sortierung & OP‑Index je Patient
    df = df.sort_values(["PMID", "Surgery_Start", "Surgery_End"]).reset_index(drop=True)
    df["op_index"] = df.groupby("PMID").cumcount() + 1
    # String‑IDs garantieren
    for c in ["PMID", "SMID", "Procedure_ID"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def read_patient_table(path_pat: str | Path) -> pd.DataFrame:
    dtype_map = {"PMID": str}
    df = _read_csv_robust(path_pat, dtype_map=dtype_map)
    # Pflicht
    needed_any = {"PMID", "Sex", "DateOfBirth"}
    missing = needed_any - set(df.columns)
    if missing:
        raise ValueError(f"In Patiententabelle fehlen Spalten: {missing}")
    df["DateOfBirth"] = _to_dt(df["DateOfBirth"])  # type: ignore
    if "DateOfDie" in df.columns:
        df["DateOfDie"] = _to_dt(df["DateOfDie"])  # type: ignore
    df["Sex_norm"] = df["Sex"].apply(_normalize_sex)
    return df[["PMID", "Sex_norm", "DateOfBirth"] + (["DateOfDie"] if "DateOfDie" in df.columns else [])]


def read_aki_table(path_aki: str | Path) -> pd.DataFrame:
    dtype_map = {"PMID": str}
    df = _read_csv_robust(path_aki, dtype_map=dtype_map)
    # Spalten mappen
    rename_map = {
        "Start": "AKI_Start",
        "End": "AKI_End",
        "Duartion": "Duration",  # häufiger Tippfehler
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if "Decision" in df.columns:
        # Textuelle Entscheidungen → normalisierte Klassen
        dec = df["Decision"].astype(str).str.strip().str.lower()
        df["AKI_class"] = dec.replace({
            "aki 1": "AKI 1",
            "aki 2": "AKI 2",
            "aki 3": "AKI 3",
            "keine aki": "Keine AKI",
            "nein": "Keine AKI",
            "ja": "AKI",
            "tx": "Tx",
            "0": "Keine AKI",
            "1": "AKI",
        })
    # Datumsfelder
    for c in ["AKI_Start", "AKI_End"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])  # type: ignore
    return df


def link_aki_to_ops(df_ops: pd.DataFrame, df_aki: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Zeitbasiertes Linking ohne merge_asof.
    Robuste Implementierung rein über int64‑Nanosekunden (keine NumPy‑Unit‑Konflikte).
    Bei Fehlern wird ein Warnhinweis ausgegeben und die AKI‑Felder leer gesetzt, statt den Build zu stoppen.
    """
    ops = df_ops.reset_index(drop=True).copy()
    try:
        # Basis-Typen harmonisieren
        ops["PMID"] = ops["PMID"].astype(str)
        ops["Surgery_End"] = pd.to_datetime(ops["Surgery_End"], errors="coerce")
        ops["_orig_i"] = np.arange(len(ops))

        if df_aki is None or df_aki.empty:
            raise RuntimeError("AKI-Tabelle leer oder fehlt")

        aki = df_aki.copy()
        if "AKI_Start" not in aki.columns and "Start" in aki.columns:
            aki = aki.rename(columns={"Start": "AKI_Start"})
        if "PMID" not in aki.columns or "AKI_Start" not in aki.columns:
            raise RuntimeError("AKI-Tabelle ohne benötigte Spalten (PMID/AKI_Start)")

        aki["PMID"] = aki["PMID"].astype(str)
        aki["AKI_Start"] = pd.to_datetime(aki["AKI_Start"], errors="coerce")
        aki = aki.dropna(subset=["AKI_Start"]).sort_values(["PMID", "AKI_Start"]).reset_index(drop=True)

        # int64-Repräsentationen (ns seit Epoch)
        NA_I64 = np.iinfo(np.int64).min
        ops_sorted = ops.sort_values(["PMID", "Surgery_End", "_orig_i"], kind="mergesort").reset_index(drop=True)
        op_end_i64 = ops_sorted["Surgery_End"].values.astype("datetime64[ns]").astype("int64")

        aki_groups_i64: dict[str, np.ndarray] = {}
        for pmid, g in aki.groupby("PMID", sort=False):
            aki_groups_i64[pmid] = g["AKI_Start"].values.astype("datetime64[ns]").astype("int64")

        matched_i64 = np.full(len(ops_sorted), NA_I64, dtype="int64")
        start_idx = 0
        for pmid, g in ops_sorted.groupby("PMID", sort=False):
            sl = slice(start_idx, start_idx + len(g)); start_idx += len(g)
            ts = aki_groups_i64.get(pmid)
            if ts is None or ts.size == 0:
                continue
            pos = np.searchsorted(ts, op_end_i64[sl], side="left")
            take = pos < ts.size
            if np.any(take):
                tmp = np.full(len(g), NA_I64, dtype="int64")
                tmp[take] = ts[pos[take]]
                matched_i64[sl] = tmp

        # zurück nach datetime + Delta als int64‑Differenz (Tage)
        ops_sorted["AKI_Start"] = pd.to_datetime(matched_i64, unit="ns", errors="coerce")
        end_i64 = ops_sorted["Surgery_End"].values.astype("datetime64[ns]").astype("int64")
        aki_i64 = ops_sorted["AKI_Start"].values.astype("datetime64[ns]").astype("int64")
        delta_days = np.full(len(ops_sorted), np.nan, dtype=float)
        mask = (aki_i64 != NA_I64) & (end_i64 != NA_I64)
        delta_days[mask] = (aki_i64[mask] - end_i64[mask]) / 86400_000_000_000.0
        ops_sorted["days_to_AKI"] = delta_days
        ops_sorted["AKI_linked_0_7"] = ((delta_days >= 0) & (delta_days <= 7)).astype("Int64")

        ops_final = ops_sorted.sort_values("_orig_i", kind="mergesort").drop(columns=["_orig_i"])  # type: ignore
        return ops_final
    except Exception as e:
        print(f"[WARN] AKI-Linking deaktiviert: {e}")
        ops["AKI_Start"] = pd.NaT
        ops["days_to_AKI"] = np.nan
        ops["AKI_linked_0_7"] = pd.Series(pd.NA, dtype="Int64")
        return ops


def merge_patients(df_ops: pd.DataFrame, df_pat: pd.DataFrame) -> pd.DataFrame:(df_ops: pd.DataFrame, df_pat: pd.DataFrame) -> pd.DataFrame:(df_ops: pd.DataFrame, df_pat: pd.DataFrame) -> pd.DataFrame:
    df = df_ops.merge(df_pat, on="PMID", how="left")
    df["age_years_at_op"] = _age_years(df["Surgery_Start"], df["DateOfBirth"])  # type: ignore
    df["age_days_at_op"] = (df["Surgery_Start"] - df["DateOfBirth"]).dt.days
    df["age_group"] = _age_group(df["age_years_at_op"])  # type: ignore
    return df


def merge_features(df_ops: pd.DataFrame, feature_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Beliebige Feature‑CSV(s) per Schlüssel joinen. Erwartet mindestens PMID + Surgery_Start/Procedure_ID.
    Die Funktion versucht pragmatisch mehrere Schlüsselvarianten.
    """
    df = df_ops.copy()
    for fp in feature_paths or []:
        fdf = _read_csv_robust(fp)
        # Schlüssel‑Heuristik
        join_keys = None
        candidates = [
            ["PMID", "SMID", "Procedure_ID", "Surgery_Start"],
            ["PMID", "Procedure_ID", "Surgery_Start"],
            ["PMID", "Surgery_Start"],
            ["PMID", "op_index"],
        ]
        for keys in candidates:
            if all(k in fdf.columns for k in keys) and all(k in df.columns for k in keys):
                join_keys = keys
                break
        if join_keys is None:
            print(f"[WARN] Feature‑Datei ohne passende Schlüssel übersprungen: {fp}")
            continue
        df = df.merge(fdf, on=join_keys, how="left")
    return df


def sanitize_for_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert problematische Typen für .obs:
    datetime→ISO-String ('' bei fehlend), category→str, object→str (oder ISO), bool→Int64."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        # 1) echte datetime64
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = _iso_or_empty(s)
            continue
        # 2) object: Datumsähnliches erkennen; sonst String
        if s.dtype == object:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().any():
                out[c] = _iso_or_empty(dt)
            else:
                out[c] = s.where(pd.notna(s), "").astype(str)
            continue
        # 3) Kategorie -> String
        if isinstance(getattr(s, "dtype", None), pd.CategoricalDtype):
            out[c] = s.astype(str).fillna("")
            continue
        # 4) Bool -> Int64
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("Int64")
            continue
    return out


def build_anndata(
    path_ops: str | Path,
    path_pat: str | Path,
    path_aki: Optional[str | Path] = None,
    feature_paths: Optional[Iterable[str | Path]] = None,
    x_features: Optional[Iterable[str]] = None,
) -> AnnData:
    # 1) Basisdaten
    ops = read_ops_table(path_ops)
    pat = read_patient_table(path_pat)
    aki = read_aki_table(path_aki) if path_aki else None

    # 2) Linking & Merges
    ops = link_aki_to_ops(ops, aki)
    ops = merge_patients(ops, pat)
    if feature_paths:
        ops = merge_features(ops, feature_paths)

    # 3) Schlüssel & obs_names
    ops["Surgery_Start_ISO"] = _iso_or_empty(ops["Surgery_Start"])  # type: ignore
    ops["uniq_key"] = _make_unique_key(ops)
    obs = sanitize_for_obs(ops)
    # Zusätzliche Absicherung: Datums-/Zeitspalten als Strings ("NaT" -> "")
    for _c in ["AKI_Start", "Surgery_Start", "Surgery_End", "DateOfBirth", "DateOfDie"]:
        if _c in obs.columns:
            obs[_c] = obs[_c].astype(str).replace({"NaT": ""})
    obs.index = obs["uniq_key"].astype(str)
    # 4) .X (optional)
    X = None
    var = None
    if x_features:
        xf = [c for c in x_features if c in ops.columns]
        if not xf:
            print("[INFO] Keine der angegebenen --x-features gefunden. .X bleibt leer.")
        else:
            X = ops[xf].to_numpy(dtype=float)
            var = pd.DataFrame(index=xf)

    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns["build_info"] = {
        "source_files": {
            "ops": str(path_ops),
            "patients": str(path_pat),
            "aki": str(path_aki) if path_aki else None,
            "features": [str(p) for p in (feature_paths or [])],
        },
        "x_features": list(x_features) if x_features else [],
        "n_ops": int(len(obs)),
        "n_features": 0 if X is None else int(X.shape[1]),
    }
    return adata


def update_anndata(existing_h5ad: str | Path, new_adata: AnnData) -> AnnData:
    old = AnnData.read_h5ad(existing_h5ad)
    # Align Spalten in .obs
    for c in new_adata.obs.columns:
        if c not in old.obs.columns:
            old.obs[c] = pd.NA
    for c in old.obs.columns:
        if c not in new_adata.obs.columns:
            new_adata.obs[c] = pd.NA
    # Concat (obs‑Axis)
    combined = old.concatenate(new_adata, join="outer", index_unique=None)
    # Deduplizierung nach uniq_key
    if "uniq_key" in combined.obs.columns:
        combined = combined[~combined.obs.index.duplicated(keep="first")].copy()
    # X angleichen (einfacher Fall: kein Mischen verschiedener X‑Schemata)
    if (old.X is None) != (new_adata.X is None):
        print("[WARN] Unterschiedliches .X‑Schema (alt vs. neu). .X wird verworfen (None).")
        combined.X = None
        combined.var = None
    return combined


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/Update AnnData from CSV tables (clinician‑friendly)")
    p.add_argument("--ops", required=True, help="Pfad zur OP‑Tabelle (HLM Operationen.csv)")
    p.add_argument("--patients", required=True, help="Pfad zur Patiententabelle (Patient Master Data.csv)")
    p.add_argument("--aki", required=False, default=None, help="Pfad zur AKI‑Tabelle (AKI Label.csv)")
    p.add_argument("--features", nargs="*", default=None, help="Optionale Feature‑CSV(s), die per Schlüssel gemerged werden")
    p.add_argument("--x-features", nargs="*", default=None, help="Spaltennamen, die als .X übernommen werden (numerisch)")
    p.add_argument("--out", required=True, help="Ziel‑.h5ad Pfad")
    p.add_argument("--update", default=None, help="Bestehendes .h5ad, das inkrementell erweitert werden soll")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print("=== Build AnnData ===")
    print(f"OPS      : {args.ops}")
    print(f"PATIENTS : {args.patients}")
    print(f"AKI      : {args.aki}")
    print(f"FEATURES : {args.features}")
    print(f"X‑FEATURES: {args.__dict__.get('x_features')}")
    try:
        adata_new = build_anndata(
            path_ops=args.ops,
            path_pat=args.patients,
            path_aki=args.aki,
            feature_paths=args.features,
            x_features=args.__dict__.get("x_features"),
        )
    except Exception as e:
        print(f"[ERROR] Build fehlgeschlagen: {e}")
        return 2

    # Update‑Pfad?
    if args.update:
        try:
            print(f"=== Update bestehendes H5AD: {args.update} ===")
            adata_combined = update_anndata(args.update, adata_new)
            adata_combined.write_h5ad(args.out)
            print(f"[OK] Updated geschrieben: {args.out}")
        except Exception as e:
            print(f"[ERROR] Update fehlgeschlagen: {e}")
            return 3
    else:
        adata_new.write_h5ad(args.out)
        print(f"[OK] Geschrieben: {args.out}")

    # Kurzreport
    a = ad.read_h5ad(args.out)
    print("--- SUMMARY ---")
    print(f"n_obs: {a.n_obs}")
    print(f".X  : {'None' if a.X is None else a.X.shape}")
    print("obs columns (Auszug):", list(a.obs.columns)[:12], "...")
    if a.uns.get("build_info"):
        print("build_info:", a.uns["build_info"]) 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


