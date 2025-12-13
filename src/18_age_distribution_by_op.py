#!/usr/bin/env python3
"""
18_age_distribution_by_op.py

Ziel:
- Operationsdatensatz mit Patientengeburtsdaten zusammenführen
- OP-Index (1., 2., 3. …) je Patient bestimmen
- Alter an OP berechnen und in pädiatrische Altersgruppen einteilen (wie Präsentationsfolie)
- Ergebnisse als AnnData/H5AD, CSV-Summary und Balkendiagramme speichern

Voraussetzungen:
- Dateien im Projektordner:
  Daten/HLM Operationen.csv
  Daten/Patient Master Data.csv
  (optional) Daten/AKI Label.csv für AKI-Linking

Laufbeispiel:
  conda activate ehrapy_env
  python src/18_age_distribution_by_op.py
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_integer_dtype,
    is_timedelta64_dtype,
)

from paths import cs_transfer_path

# ---------------------------
# Pfade (robust, findet CSV auch in "Original Daten")
# ---------------------------
BASE = cs_transfer_path()
DATA_DIR = BASE / "Daten"
H5AD_DIR = BASE / "h5ad"
DIAG_DIR = BASE / "Diagramme"
for p in (DATA_DIR, H5AD_DIR, DIAG_DIR):
    p.mkdir(parents=True, exist_ok=True)


def resolve_csv(preferred_name: str) -> Path:
    """Suche CSV zuerst direkt in Daten/, dann in Unterordnern (z. B. 'Original Daten')."""
    candidates = [
        DATA_DIR / preferred_name,
        DATA_DIR / "Original Daten" / preferred_name,
        DATA_DIR / "Original_Daten" / preferred_name,
        BASE / "Original Daten" / preferred_name,
        BASE / preferred_name,
    ]
    for c in candidates:
        if c.exists():
            print(f"Verwende: {c}")
            return c
    # Fuzzy-Suche (falls Dateiname leicht abweicht)
    for c in DATA_DIR.rglob("*.csv"):
        if all(
            tok in c.name.lower()
            for tok in preferred_name.lower().replace(".csv", "").split()
        ):
            print(f"Gefunden (fuzzy): {c}")
            return c
    raise FileNotFoundError(
        f"Konnte '{preferred_name}' nicht finden. "
        f"Bitte Pfad prüfen. Durchsucht: {DATA_DIR}"
    )


OPS_CSV = resolve_csv("HLM Operationen.csv")
PAT_CSV = resolve_csv("Patient Master Data.csv")
AKI_CSV = resolve_csv("AKI Label.csv")  # optional


# ----------------------------------------
# Hilfsfunktionen: robustes Einlesen & Co.
# ----------------------------------------
def read_csv_robust(
    path: Path, parse_date_cols: list[str] | None = None
) -> pd.DataFrame:
    """CSV robust einlesen: erkennt Komma/Semikolon, entfernt BOM, trimmt Spaltennamen."""
    if parse_date_cols is None:
        parse_date_cols = []
    df = pd.read_csv(
        path,
        sep=None,  # Autodetektor für , ; \t
        engine="python",
        encoding="utf-8-sig",
        dtype=str,  # erstmal als String, wir parsen gezielt
    )
    df.columns = df.columns.str.strip()
    for c in parse_date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def normalize_op_columns(df_ops: pd.DataFrame) -> pd.DataFrame:
    """Spalten der OP-Tabelle vereinheitlichen → Surgery_Start / Surgery_End."""
    # mögliche Originalnamen
    rename_map = {
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        "Start": "Surgery_Start",
        "End": "Surgery_End",
        "Tx?": "Tx_flag",
    }
    for k, v in rename_map.items():
        if k in df_ops.columns and v not in df_ops.columns:
            df_ops = df_ops.rename(columns={k: v})

    # Strings → Datetime
    for col in ["Surgery_Start", "Surgery_End"]:
        if col in df_ops.columns:
            df_ops[col] = pd.to_datetime(df_ops[col], errors="coerce")

    # sicherstellen: zentrale Schlüssel existieren
    required = ["PMID", "Surgery_Start", "Surgery_End"]
    missing = [c for c in required if c not in df_ops.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in OP-Datei: {missing}")

    # Duplikate vermeiden (konservativ): PMID + Surgery_Start + Surgery_End + optional Procedure_ID
    key_cols = ["PMID", "Surgery_Start", "Surgery_End"]
    if "Procedure_ID" in df_ops.columns:
        key_cols.append("Procedure_ID")
    df_ops = df_ops.sort_values(key_cols).drop_duplicates(key_cols)

    return df_ops


def compute_op_index(df_ops: pd.DataFrame) -> pd.DataFrame:
    """OP-Index pro Patient (1,2,3,...) anhand Surgery_Start."""
    df_ops = df_ops.sort_values(["PMID", "Surgery_Start"]).copy()
    df_ops["op_index"] = df_ops.groupby("PMID").cumcount() + 1  # 1-basierte Zählung
    return df_ops


def compute_age_at_op(df_ops_pat: pd.DataFrame) -> pd.DataFrame:
    """Alter an OP (Tage/Jahre) berechnen."""
    if "DateOfBirth" not in df_ops_pat.columns:
        raise ValueError("Spalte 'DateOfBirth' fehlt in Patientendaten.")
    df = df_ops_pat.copy()
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")
    df["age_days_at_op"] = (
        df["Surgery_Start"] - df["DateOfBirth"]
    ).dt.total_seconds() / (24 * 3600)
    df["age_years_at_op"] = df["age_days_at_op"] / 365.25
    return df


# Pädiatrische Altersgruppen genau wie auf der Folie
AGE_CATEGORIES = [
    "Neonates (0–28 T.)",
    "Infants (1–12 Mon.)",
    "Toddlers (1–3 J.)",
    "Preschool (3–5 J.)",
    "School-age (6–12 J.)",
    "Adolescents (13–18 J.)",
    "Unbekannt/außerhalb",
]


def assign_pediatric_age_group(age_days: float | int | None) -> str:
    """Mappt Alter in Tagen auf pädiatrische Altersgruppen (inkl. Fallback)."""
    if pd.isna(age_days):
        return "Unbekannt/außerhalb"
    d = float(age_days)
    y = d / 365.25

    if 0 <= d <= 28:
        return "Neonates (0–28 T.)"
    elif 28 < d < 365.25:  # 1–12 Monate
        return "Infants (1–12 Mon.)"
    elif 1 <= y < 3:
        return "Toddlers (1–3 J.)"
    elif 3 <= y < 6:
        return "Preschool (3–5 J.)"
    elif 6 <= y < 13:
        return "School-age (6–12 J.)"
    elif 13 <= y < 18.01:  # 18.0 ±
        return "Adolescents (13–18 J.)"
    else:
        return "Unbekannt/außerhalb"


def make_age_group_categorical(series: pd.Series) -> pd.Series:
    """Sorgt für feste Reihenfolge in Tabellen/Plots."""
    cat = pd.Categorical(series, categories=AGE_CATEGORIES, ordered=True)
    return cat


def summary_counts_by_opindex(df: pd.DataFrame) -> pd.DataFrame:
    """Zählung pro OP-Index × Altersgruppe."""
    counts = (
        df.groupby(["op_index", "age_group"], dropna=False)
        .size()
        .reset_index(name="n_ops")
    )
    # Sicherstellen, dass alle Gruppen sichtbar sind
    counts["age_group"] = make_age_group_categorical(counts["age_group"])
    counts = counts.sort_values(["op_index", "age_group"])
    return counts


def plot_age_distribution_for_op(
    df_op: pd.DataFrame, op_index: int, out_png: Path
) -> None:
    """Erstellt EIN Balkendiagramm wie auf der Folie – für einen OP-Index."""
    # vollständige Reihenfolge erzwingen (auch Gruppen mit 0 zählen)
    all_groups = pd.Index(AGE_CATEGORIES, name="age_group")
    counts = df_op["age_group"].value_counts().reindex(all_groups, fill_value=0)

    plt.figure(figsize=(11, 6))
    ax = counts.plot(kind="bar")
    ax.set_title(f"Altersverteilung – OP {op_index}")
    ax.set_xlabel("")
    ax.set_ylabel("Anzahl Operationen")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

    # Werte über Balken
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(
            f"{height}",
            (p.get_x() + p.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------------
# (Optional) AKI
# ---------------
def _to_naive_datetime(s: pd.Series) -> pd.Series:
    """Nach datetime konvertieren und ggf. Zeitzone entfernen."""
    s = pd.to_datetime(s, errors="coerce")
    if is_datetime64tz_dtype(s.dtype):
        s = s.dt.tz_convert(None)  # tz-aware -> naive
    return s


def link_aki_0_7_days(df_ops: pd.DataFrame, aki_csv: Path) -> pd.DataFrame:
    """
    Verlinkt pro OP das erste AKI-Ereignis NACH OP-Ende.
    Arbeitet patientenweise (gruppiert nach PMID), damit Sortierfehler ausgeschlossen sind.
    Erzeugt: AKI_Start, days_to_AKI, AKI_linked_0_7 (Int64 0/1)
    """
    if not aki_csv.exists():
        return df_ops.copy()

    # AKI-Datei lesen & normalisieren
    df_aki_raw = read_csv_robust(aki_csv)
    ren = {}
    if "Start" in df_aki_raw.columns:
        ren["Start"] = "AKI_Start"
    if "End" in df_aki_raw.columns:
        ren["End"] = "AKI_End"
    if "Duartion" in df_aki_raw.columns:
        ren["Duartion"] = "Duration"
    df_aki = df_aki_raw.rename(columns=ren)

    # nur benötigte Spalten & saubere Datetimes
    if "PMID" not in df_aki.columns:
        return df_ops.copy()  # kein valider AKI-Datensatz

    if "AKI_Start" not in df_aki.columns:
        # ohne Zeitstempel kein zeitbasiertes Linking möglich
        out = df_ops.copy()
        out["AKI_Start"] = pd.NaT
        out["days_to_AKI"] = np.nan
        out["AKI_linked_0_7"] = pd.Series([0] * len(out), dtype="Int64")
        return out

    df_aki = df_aki[["PMID", "AKI_Start"]].copy()
    df_aki["AKI_Start"] = _to_naive_datetime(df_aki["AKI_Start"])
    df_aki = df_aki.dropna(subset=["PMID", "AKI_Start"]).sort_values(
        ["PMID", "AKI_Start"]
    )

    # linke Seite vorbereiten
    ops = df_ops.copy()
    ops["Surgery_End"] = _to_naive_datetime(ops["Surgery_End"])
    ops = ops.dropna(subset=["PMID", "Surgery_End"]).sort_values(
        ["PMID", "Surgery_End"]
    )

    # patientenweiser asof-Join
    merged_parts = []
    for pid, left in ops.groupby("PMID", sort=False):
        right = df_aki.loc[df_aki["PMID"] == pid, ["AKI_Start"]].copy()
        left_sorted = left.sort_values("Surgery_End")
        right_sorted = right.sort_values("AKI_Start")

        if right_sorted.empty:
            tmp = left_sorted.copy()
            tmp["AKI_Start"] = pd.NaT
        else:
            tmp = pd.merge_asof(
                left_sorted,
                right_sorted,
                left_on="Surgery_End",
                right_on="AKI_Start",
                direction="forward",
                allow_exact_matches=False,
            )
        merged_parts.append(tmp)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Kennzahlen berechnen
    merged["days_to_AKI"] = (
        merged["AKI_Start"] - merged["Surgery_End"]
    ).dt.total_seconds() / (24 * 3600)
    merged["AKI_linked_0_7"] = (
        ((merged["days_to_AKI"] >= 0) & (merged["days_to_AKI"] <= 7))
        .astype("Int64")
        .fillna(0)
    )

    return merged


def sanitize_obs_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macht .obs für AnnData/H5AD speicherbar:
    - Datetime → ISO-String
    - Timedelta → Sekunden (float64)
    - Bool → int8 (0/1)
    - Nullable-Integer (Int64) → float64
    - object-Mix → String
    - Kategorien bleiben Kategorien (geordnet)
    """
    out = df.copy()

    # IDs sicher als String
    for id_col in ["PMID", "SMID", "Procedure_ID"]:
        if id_col in out.columns:
            out[id_col] = out[id_col].astype(str)

    for c in out.columns:
        s = out[c]
        if is_datetime64_any_dtype(s):
            out[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        elif is_timedelta64_dtype(s):
            out[c] = s.dt.total_seconds().astype("float64")
        elif is_bool_dtype(s):
            out[c] = s.astype("int8")
        elif is_integer_dtype(s):
            # deckt auch pandas 'Int64' (nullable) ab
            out[c] = s.astype("float64")
        elif is_categorical_dtype(s):
            # Kategorien sind ok; optional als String speichern:
            # out[c] = s.astype(str)
            pass
        else:
            # object/alles andere robust zu String
            if out[c].dtype == "object":
                out[c] = out[c].astype(str)

    # keine NaN/None in Strings
    out = out.replace({"<NA>": "", "nan": "", "None": ""})
    return out


# ------- NEU: Multi-Panel-Plot je OP-Index (OP 1–4 nebeneinander)---
def make_multipanel_age_plots(
    df_ops: pd.DataFrame, upto: int = 4, same_ylim: bool = True
) -> None:
    """
    Erstellt eine 1xupto-Übersicht: Altersverteilung OP1..OP_upto.
    same_ylim=True setzt überall die gleiche y-Achse (bessere Vergleichbarkeit).
    """
    import matplotlib.pyplot as plt

    op_indices = [i for i in range(1, int(df_ops["op_index"].max()) + 1) if i <= upto]
    if not op_indices:
        return

    # Counts für gemeinsame y-Achse bestimmen
    max_count = 0
    per_op_counts = {}
    for i in op_indices:
        subset = df_ops.loc[df_ops["op_index"] == i]
        counts = (
            subset["age_group"].value_counts().reindex(AGE_CATEGORIES, fill_value=0)
        )
        per_op_counts[i] = counts
        max_count = max(max_count, counts.max())

    fig, axes = plt.subplots(
        1, len(op_indices), figsize=(4.8 * len(op_indices), 5), sharey=same_ylim
    )

    if len(op_indices) == 1:
        axes = [axes]

    for ax, i in zip(axes, op_indices):
        counts = per_op_counts[i]
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"OP {i} (n={int(counts.sum())})")
        ax.set_xlabel("")
        ax.set_ylabel("Anzahl Operationen" if ax is axes[0] else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        # Werte über Balken
        for p in ax.patches:
            h = int(p.get_height())
            ax.annotate(
                f"{h}",
                (p.get_x() + p.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if same_ylim:
            ax.set_ylim(0, max_count * 1.10)

    fig.suptitle("Altersverteilung nach OP-Index")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = DIAG_DIR / "age_distribution_multipanel_op1_4.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Gespeichert: {out}")


# ---  kompakte Summary-Tabellen .csv---
def write_age_summary_tables(df_ops: pd.DataFrame) -> None:
    """
    Schreibt zwei Tabellen:
    1) counts:  OP-Index x Altersgruppe (Anzahl)
    2) props:   OP-Index x Altersgruppe (Prozente je OP-Index)
    """
    counts = (
        df_ops.groupby(["op_index", "age_group"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    # Pivot-Form
    counts_piv = counts.pivot(
        index="op_index", columns="age_group", values="n"
    ).reindex(columns=AGE_CATEGORIES)
    counts_piv = counts_piv.fillna(0).astype(int)

    # Prozente pro OP-Index
    props_piv = counts_piv.div(counts_piv.sum(axis=1), axis=0) * 100
    props_piv = props_piv.round(1)

    out_counts = DATA_DIR / "age_counts_by_opindex_pivot.csv"
    out_props = DATA_DIR / "age_percent_by_opindex_pivot.csv"
    counts_piv.to_csv(out_counts)
    props_piv.to_csv(out_props)
    print(f"Gespeichert: {out_counts}")
    print(f"Gespeichert: {out_props}")
    # --- Gestapelte Balken je Altersgruppe (AKI 0–7 vs. kein AKI) ---


def plot_age_stacked_by_aki(df_ops: pd.DataFrame, op_index: int) -> None:
    if "AKI_linked_0_7" not in df_ops.columns:
        print("AKI_linked_0_7 nicht vorhanden – übersprungen.")
        return

    sub = df_ops.loc[df_ops["op_index"] == op_index].copy()
    if sub.empty:
        return
    # Zählungen
    aki0 = (
        sub.loc[sub["AKI_linked_0_7"].fillna(0).astype(int) == 0, "age_group"]
        .value_counts()
        .reindex(AGE_CATEGORIES, fill_value=0)
    )
    aki1 = (
        sub.loc[sub["AKI_linked_0_7"].fillna(0).astype(int) == 1, "age_group"]
        .value_counts()
        .reindex(AGE_CATEGORIES, fill_value=0)
    )

    idx = np.arange(len(AGE_CATEGORIES))
    width = 0.75

    fig, ax = plt.subplots(figsize=(10.5, 5))
    b0 = ax.bar(idx, aki0.values, width, label="kein AKI 0–7")
    b1 = ax.bar(idx, aki1.values, width, bottom=aki0.values, label="AKI 0–7")

    ax.set_title(f"Altersverteilung – OP {op_index} (gestapelt: AKI 0–7)")
    ax.set_ylabel("Anzahl Operationen")
    ax.set_xticks(idx, AGE_CATEGORIES, rotation=25, ha="right")
    ax.legend(loc="upper right")

    # Totals über Stapel
    totals = aki0.values + aki1.values
    for x, h in zip(idx, totals):
        ax.annotate(
            f"{int(h)}",
            (x, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out = DIAG_DIR / f"age_stacked_aki_op{op_index}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Gespeichert: {out}")


# -------------
# Hauptablauf
# -------------
def main():
    # 1) Daten laden
    df_ops_raw = read_csv_robust(OPS_CSV)
    df_pat = read_csv_robust(PAT_CSV, parse_date_cols=["DateOfBirth"])

    # 2) OP-Spalten normalisieren & OP-Index
    df_ops = normalize_op_columns(df_ops_raw)
    df_ops = compute_op_index(df_ops)

    # 3) Patienteninfos mergen
    df_ops = df_ops.merge(df_pat[["PMID", "DateOfBirth", "Sex"]], on="PMID", how="left")

    # 4) Alter berechnen
    df_ops = compute_age_at_op(df_ops)

    # 5) Altersgruppen (geordnet)
    df_ops["age_group"] = df_ops["age_days_at_op"].apply(assign_pediatric_age_group)
    df_ops["age_group"] = make_age_group_categorical(df_ops["age_group"])
    make_multipanel_age_plots(df_ops, upto=4, same_ylim=True)
    write_age_summary_tables(df_ops)
    # 6) (Optional) AKI-Linking 0–7 Tage
    df_ops = link_aki_0_7_days(df_ops, AKI_CSV)

    # 7) Summary-CSV: Counts pro OP-Index × Altersgruppe
    counts = summary_counts_by_opindex(df_ops)
    out_counts_csv = DATA_DIR / "age_counts_by_opindex.csv"
    counts.to_csv(out_counts_csv, index=False)

    # 8) Plots wie im Foto – je OP-Index eine Datei
    max_idx = int(df_ops["op_index"].max()) if not df_ops.empty else 0
    for i in range(1, max_idx + 1):
        subset = df_ops.loc[df_ops["op_index"] == i]
        if subset.empty:
            continue
        out_png = DIAG_DIR / f"age_distribution_op{i}.png"
        plot_age_distribution_for_op(subset, i, out_png)

    # 9) AnnData schreiben (alles in .obs)
    # adata = AnnData(df_ops)  # <-- das war falsch (landet in .X)

    obs = sanitize_obs_for_h5ad(df_ops)
    adata = AnnData(X=None, obs=obs)  # nur Metadaten in .obs, keine Matrix
    out_h5ad = H5AD_DIR / "ops_with_age_groups.h5ad"
    adata.write(out_h5ad)

    # 10) Konsolenhinweis
    print("Fertig.")
    print(f"- Summary CSV: {out_counts_csv}")
    print(f"- Plots je OP-Index: {DIAG_DIR}/age_distribution_op#.png")
    print(f"- H5AD: {out_h5ad}")


if __name__ == "__main__":
    main()
