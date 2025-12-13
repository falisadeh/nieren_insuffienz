from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import ORIGINAL_DATA_DIR, cs_transfer_path

CHECKS = True

BASE = cs_transfer_path()
ORIG_DIR = ORIGINAL_DATA_DIR
OPS_CSV = ORIG_DIR / "HLM Operationen.csv"
PAT_CSV = ORIG_DIR / "Patient Master Data.csv"
AKI_CSV = ORIG_DIR / "AKI Label.csv"
OUT_DIR = BASE / "Daten"
DIAG_DIR = BASE / "Diagramme"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def check(name: str, cond: bool, msg_if_fail: str) -> None:
    if CHECKS and not cond:
        raise AssertionError(f"CHECK FAIL – {name}: {msg_if_fail}")
    if CHECKS:
        print(f"CHECK OK – {name}")


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Robustes CSV-Lesen mit Fallbacks (erkennt BOM und unterschiedliche Separatoren)."""
    for kwargs in (
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )
    return df


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def map_decision_to_flag(x: pd.Series) -> pd.Series:
    def _coerce(value):
        if pd.isna(value):
            return pd.NA
        s = str(value).lower().strip()
        neg = any(k in s for k in ["keine", "nein", "no", "0"])
        pos = any(k in s for k in ["aki 1", "aki1", "aki 2", "aki2", "aki 3", "aki3"])
        if pos and not neg:
            return 1
        if neg and not pos:
            return 0
        return pd.NA

    return x.map(_coerce)


def ensure_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} – fehlende Spalten: {missing}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ops = read_csv_auto(OPS_CSV)
    pat = read_csv_auto(PAT_CSV)
    aki = read_csv_auto(AKI_CSV)
    print("OPS-Spalten:", list(ops.columns))
    print("PAT-Spalten:", list(pat.columns))
    print("AKI-Spalten:", list(aki.columns))
    return ops, pat, aki


def prepare_ops(ops: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    ops = ops.rename(
        columns={"Start of surgery": "Surgery_Start", "End of surgery": "Surgery_End"}
    )
    ensure_columns(ops, ["PMID", "Surgery_Start", "Surgery_End"], "HLM Operationen")
    ensure_columns(pat, ["PMID", "DateOfBirth"], "Patient Master Data")

    ops["PMID"] = ops["PMID"].astype(str).str.strip()
    pat["PMID"] = pat["PMID"].astype(str).str.strip()
    ops["Surgery_Start"] = to_dt(ops["Surgery_Start"])
    ops["Surgery_End"] = to_dt(ops["Surgery_End"])
    pat["DateOfBirth"] = to_dt(pat["DateOfBirth"])

    dedup_cols = (
        ["PMID", "Procedure_ID", "Surgery_Start", "Surgery_End"]
        if "Procedure_ID" in ops.columns
        else ["PMID", "Surgery_Start", "Surgery_End"]
    )
    before = len(ops)
    ops = (
        ops.drop_duplicates(subset=dedup_cols)
        .sort_values(["PMID", "Surgery_Start", "Surgery_End"])
        .reset_index(drop=True)
    )
    print(f"OP-Zeilen vor/nach Deduplizierung: {before}/{len(ops)}")

    ops["op_idx"] = ops.groupby("PMID").cumcount() + 1
    check(
        "Patientenzahl = Anzahl erster OPs",
        ops["PMID"].nunique() == int((ops["op_idx"] == 1).sum()),
        "Erste OPs ≠ Patientenzahl (Duplikate?).",
    )

    ops = ops.merge(pat[["PMID", "DateOfBirth"]], on="PMID", how="left")
    ops["Age_years_at_op"] = (
        ops["Surgery_Start"] - ops["DateOfBirth"]
    ).dt.total_seconds() / (365.25 * 24 * 3600)
    return ops


def prepare_aki(aki: pd.DataFrame) -> pd.DataFrame:
    aki = aki.rename(
        columns={
            "Duartion": "Duration",
            "Start": "AKI_Start",
            "End": "AKI_End",
            "Decision": "AKI_Decision",
        }
    )
    if "PMID" not in aki.columns:
        raise ValueError("AKI Label.csv muss eine Spalte 'PMID' enthalten.")
    aki["PMID"] = aki["PMID"].astype(str).str.strip()
    if "AKI_Start" in aki.columns:
        aki["AKI_Start"] = to_dt(aki["AKI_Start"]).dt.tz_localize(None)
    if "AKI_End" in aki.columns:
        aki["AKI_End"] = to_dt(aki["AKI_End"]).dt.tz_localize(None)
    if "AKI_Decision" in aki.columns:
        aki["AKI_patient_flag"] = map_decision_to_flag(aki["AKI_Decision"])
    return aki


def link_ops_to_aki(
    ops: pd.DataFrame, aki: pd.DataFrame, tolerance_days: int = 7
) -> pd.DataFrame:
    if "AKI_Start" not in aki.columns:
        out = ops.copy()
        out["AKI_Start"] = pd.NaT
        out["AKI_End"] = pd.NaT
        out["Duration"] = pd.NA
        out["AKI_linked_0_7"] = 0
        out["days_to_AKI"] = np.nan
        return out

    merged_parts = []
    tol = pd.Timedelta(days=tolerance_days)
    for pid, op_grp in ops.groupby("PMID", sort=False):
        right = aki.loc[aki["PMID"] == pid].drop(columns=["PMID"], errors="ignore")
        left = op_grp.sort_values("Surgery_End")
        if right.empty:
            tmp = left.copy()
            tmp["AKI_Start"] = pd.NaT
            tmp["AKI_End"] = pd.NaT
            tmp["Duration"] = pd.NA
            merged_parts.append(tmp)
            continue
        merged = pd.merge_asof(
            left,
            right.sort_values("AKI_Start"),
            left_on="Surgery_End",
            right_on="AKI_Start",
            direction="forward",
            tolerance=tol,
            suffixes=("", "_aki"),
        )
        merged_parts.append(merged)

    linked = pd.concat(merged_parts, ignore_index=True)
    linked["days_to_AKI"] = (
        linked["AKI_Start"] - linked["Surgery_End"]
    ).dt.total_seconds() / (24 * 3600)
    linked["AKI_linked_0_7"] = linked["AKI_Start"].notna().astype("int8")

    check(
        "AKI_linked_0_7 ist binär",
        set(linked["AKI_linked_0_7"].unique()) <= {0, 1},
        "AKI_linked_0_7 enthält andere Werte als 0/1.",
    )
    return linked


def build_patient_summary(ops: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ops.groupby("PMID")["op_idx"]
        .max()
        .rename("n_ops")
        .reset_index()
        .sort_values("PMID")
    )
    first_aki = (
        linked.loc[linked["AKI_linked_0_7"] == 1, ["PMID", "op_idx"]]
        .sort_values(["PMID", "op_idx"])
        .drop_duplicates("PMID")
        .rename(columns={"op_idx": "first_aki_op_idx"})
    )
    summary = summary.merge(first_aki, on="PMID", how="left")
    summary["any_aki"] = summary["first_aki_op_idx"].notna().astype(int)
    return summary


def summarize_age_by_op_idx(linked: pd.DataFrame) -> pd.DataFrame:
    return (
        linked.dropna(subset=["Age_years_at_op"])
        .groupby("op_idx")["Age_years_at_op"]
        .agg(
            n="count",
            median_years=lambda s: float(np.median(s)),
            q1_years=lambda s: float(np.percentile(s, 25)),
            q3_years=lambda s: float(np.percentile(s, 75)),
            min_years=lambda s: float(np.min(s)),
            max_years=lambda s: float(np.max(s)),
        )
        .reset_index()
    )


def summarize_aki_by_op_idx(linked: pd.DataFrame) -> pd.DataFrame:
    ops_by_idx = linked.groupby("op_idx").size().rename("n_ops_at_idx").reset_index()
    aki_cases = (
        linked[linked["AKI_linked_0_7"] == 1]
        .groupby("op_idx")
        .size()
        .rename("AKI_cases")
        .reset_index()
    )
    merged = ops_by_idx.merge(aki_cases, on="op_idx", how="left")
    merged["AKI_cases"] = merged["AKI_cases"].fillna(0).astype(int)
    merged["AKI_rate_%"] = (
        merged["AKI_cases"] / merged["n_ops_at_idx"].replace(0, np.nan) * 100
    ).fillna(0.0)
    check(
        "Summe AKI-Events passt zur OP-Index-Tabelle",
        int(linked["AKI_linked_0_7"].sum())
        == int(merged["AKI_cases"].sum()),
        "linked-Summe ≠ Summe AKI_cases je OP-Index.",
    )
    return merged


def export_tables(
    ops: pd.DataFrame,
    linked: pd.DataFrame,
    patient_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    aki_rate_by_idx: pd.DataFrame,
) -> None:
    patient_summary.to_csv(OUT_DIR / "patient_summary_ops_aki.csv", index=False)
    ops.groupby("PMID")["op_idx"].max().rename("n_ops").reset_index().to_csv(
        OUT_DIR / "ops_per_patient.csv", index=False
    )
    age_summary.to_csv(OUT_DIR / "age_by_op_index.csv", index=False)
    aki_rate_by_idx.to_csv(OUT_DIR / "aki_by_op_index_0_7.csv", index=False)
    linked[
        [
            "PMID",
            "SMID",
            "Procedure_ID",
            "Surgery_Start",
            "Surgery_End",
            "op_idx",
            "Age_years_at_op",
            "AKI_Start",
            "AKI_linked_0_7",
        ]
    ].to_csv(OUT_DIR / "ops_linked_with_aki_0_7.csv", index=False)


def plot_aki_rate(aki_rate_by_idx: pd.DataFrame, mode: str = "dim") -> None:
    df = aki_rate_by_idx.copy()
    n = df["n_ops_at_idx"].astype(int).to_numpy()
    k = df["AKI_cases"].astype(int).to_numpy()
    x = df["op_idx"].astype(int).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / np.where(n == 0, 1, n)

    z = 1.96
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    lower = np.clip(center - half, 0, 1)
    upper = np.clip(center + half, 0, 1)

    heights = p * 100
    yerr = np.vstack(((p - lower) * 100, (upper - p) * 100))
    if mode == "dim":
        mask_ci = n >= 5
        yerr[:, ~mask_ci] = 0.0

    fig, ax = plt.subplots()
    bars = ax.bar(x, heights, yerr=yerr, capsize=4, linewidth=1)
    if mode == "dim":
        for b, size in zip(bars, n):
            if size < 5:
                b.set_alpha(0.35)

    for xi, hi, ki, ni in zip(x, heights, k, n):
        ax.text(xi, hi + 1.2, f"{hi:.1f}% ({ki}/{ni})", ha="center", va="bottom")

    ax.set_xticks(x, [f"{xi}\n(n={ni})" for xi, ni in zip(x, n)])
    ax.set_xlabel("OP-Index")
    ax.set_ylabel("AKI-Rate (%)")
    ax.set_title("AKI (0–7 Tage) nach OP-Index")
    ymax = np.nanmax(heights + yerr[1]) if len(heights) else 5
    ax.set_ylim(0, math.ceil(ymax * 1.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    subtitle = {
        "dim": "Fehlerbalken: 95%-Wilson; Kategorien mit n<5 ohne CI und abgeblendet.",
        "hide": "Fehlerbalken: 95%-Wilson; Kategorien mit n<5 nicht dargestellt.",
        "show": "Fehlerbalken: 95%-Wilson; alle Kategorien mit CI.",
    }[mode]
    fig.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "AKI_rate_by_opindex.png", dpi=300)
    plt.close(fig)


def plot_age_box(linked: pd.DataFrame) -> None:
    groups, labels = [], []
    for idx in sorted(linked["op_idx"].dropna().unique().astype(int)):
        values = linked.loc[linked["op_idx"] == idx, "Age_years_at_op"].dropna().values
        if len(values) == 0:
            continue
        groups.append(values)
        labels.append(f"{idx}\n(n={len(values)})")
    if not groups:
        return

    fig, ax = plt.subplots()
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_xlabel("OP-Index")
    ax.set_ylabel("Alter (Jahre)")
    ax.set_title("Alter bei OP nach OP-Index")
    ax.set_ylim(0, max(v.max() for v in groups) * 1.05)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "age_box_by_opindex.png", dpi=300)
    plt.close(fig)


def report_summary(
    ops: pd.DataFrame,
    patient_summary: pd.DataFrame,
    aki_rate_by_idx: pd.DataFrame,
    age_summary: pd.DataFrame,
) -> None:
    total_patients = int(ops["PMID"].nunique())
    patients_ge2 = int((patient_summary["n_ops"] >= 2).sum())
    patients_with_aki = int(patient_summary["any_aki"].sum())
    any_aki_rate = 100 * patients_with_aki / total_patients if total_patients else 0.0

    pct_ge2 = 100 * patients_ge2 / total_patients if total_patients else 0.0

    print("\n===== KOHORTEN-REPORT =====")
    print(f"Patient:innen gesamt: {total_patients}")
    print(f"Patient:innen mit ≥2 OPs: {patients_ge2} ({pct_ge2:.1f}%)")
    print(
        f"Patient:innen mit AKI (0–7) bei ≥1 OP: {patients_with_aki} ({any_aki_rate:.1f}%)"
    )
    first_op = int((patient_summary["first_aki_op_idx"] == 1).sum())
    later_op = int(
        (patient_summary["first_aki_op_idx"].fillna(0).astype(int) > 1).sum()
    )
    print(f"  davon erstes AKI bei 1. OP: {first_op}")
    print(f"  davon erstes AKI bei OP >1: {later_op}")

    print("\nAKI (0–7) nach OP-Index (zeilenbasiert):")
    print(aki_rate_by_idx.to_string(index=False))
    print("\nAlter nach OP-Index (Jahre):")
    print(age_summary.to_string(index=False))


def main() -> None:
    ops_raw, pat_raw, aki_raw = load_tables()
    ops = prepare_ops(ops_raw, pat_raw)
    aki = prepare_aki(aki_raw)
    linked = link_ops_to_aki(ops, aki)
    patient_summary = build_patient_summary(ops, linked)
    aki_rate_by_idx = summarize_aki_by_op_idx(linked)
    age_summary = summarize_age_by_op_idx(linked)
    export_tables(ops, linked, patient_summary, age_summary, aki_rate_by_idx)
    plot_aki_rate(aki_rate_by_idx)
    plot_age_box(linked)
    report_summary(ops, patient_summary, aki_rate_by_idx, age_summary)


if __name__ == "__main__":
    main()
