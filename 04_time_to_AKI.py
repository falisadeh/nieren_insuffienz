#%%
"""
04_time_to_AKI_ehrapy.py
=========================
Fragestellung: "Wie effektiv unterstützt das Framework ehrapy die Identifikation von Risikofaktoren für AKI
bei Kindern nach Herzoperationen anhand eines angereicherten Routinedatensatzes?"

Ansatz (ehrapy-first):
- **ehrapy/AnnData** ist die **zentrale Datenstruktur**. Alle abgeleiteten Variablen und Metadaten
  werden in `adata.obs` bzw. `adata.uns` abgelegt (Provenance!).
- Visualisierungen/Modelle verwenden stabile Ökosystem-Pakete (`lifelines`, `matplotlib`, `statsmodels/sklearn`).
- Ergebnis: Reproduzierbare Pipeline, die methodisch in ehrapy eingebettet bleibt.

Dieser Schritt (S1): Zeit-zu-Ereignis (Kaplan–Meier) für AKI ≤ 7 Tage
- Definiert **Index-OP** je AKI (letzte OP vor AKI im 0–7-Tage-Fenster), um Doppelzählungen zu vermeiden.
- Erzeugt `obs`-Variablen: `time_0_7`, `event_idx`.
- Schreibt Parameter/Provenance nach `adata.uns['survival_0_7']`.
- Exportiert CSV + 2 Plots; speichert Version der AnnData-Datei (H5AD) als S1.

Voraussetzungen
---------------
- Conda-Env: ehrapy_env (Python ≥ 3.10)
- Pakete: ehrapy, anndata, pandas, numpy, lifelines, matplotlib

Installation (falls nötig):
  conda install -c conda-forge ehrapy anndata pandas numpy lifelines matplotlib

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ehrapy/AnnData laden (robust mit Fallback)
try:
    import ehrapy as ep  # type: ignore
except Exception:
    ep = None  # Fallback erlaubt, Kern bleibt AnnData

from anndata import read_h5ad, AnnData

try:
    from lifelines import KaplanMeierFitter
except Exception as e:
    raise SystemExit(
        "lifelines fehlt. Bitte installieren:\n"
        "  conda install -c conda-forge lifelines\n\n"
        f"Technischer Hinweis: {e}"
    )


# ---------------------------------
# 1) KONFIGURATION
# ---------------------------------
@dataclass
class Config:
    PATH_H5AD: str = (
        "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master.h5ad"
    )
    SAVE_DIR: str = (
        "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
    )
    OUT_H5AD: str = (
        "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/aki_ops_master_S1_survival.h5ad"
    )
    WINDOW_DAYS: int = 7

CFG = Config()


# ---------------------------------
# 2) HILFSFUNKTIONEN (ehrapy-first)
# ---------------------------------

def load_adata() -> AnnData:
    """Lädt die AnnData-Struktur. Bevorzugt `ep.io.read_h5ad`, sonst Fallback `anndata.read_h5ad`.

    *Warum so?* Damit bleibt die Pipeline **ehrapy-first**, ist aber robust, falls einzelne I/O-Funktionen
    in Ihrer ehrapy-Version fehlen.
    """
    if ep is not None and hasattr(ep, "io") and hasattr(ep.io, "read_h5ad"):
        adata = ep.io.read_h5ad(CFG.PATH_H5AD)  # type: ignore[attr-defined]
    else:
        adata = read_h5ad(CFG.PATH_H5AD)
    return adata


def _ensure_datetime(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Robuste Datums-Konvertierung mit `errors='coerce'` (fehlerhafte Einträge -> NaT)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def annotate_provenance(adata: AnnData, key: str, meta: Dict) -> None:
    """Speichert Parameter/Provenance in `adata.uns[key]`.

    Beispiele: Fenstergröße, Versionshinweise, Anzahl Ereignisse.
    """
    if adata.uns is None:
        adata.uns = {}
    adata.uns[key] = meta


def define_event_index_on_adata(adata: AnnData) -> None:
    """Definiert den **Index-OP** je AKI-Episode und erzeugt `event_idx` in `obs`.

    Logik:
    - Kandidaten = OPs mit `AKI_linked_0_7 == 1` und gültigem `AKI_Start`.
    - Auswahl = letzte OP mit `Surgery_End` ≤ `AKI_Start` innerhalb der Gruppe (`PMID`, `AKI_Start`).
    - Ergebnis: `obs['event_idx']` ∈ {0,1}.
    """
    df = adata.obs.copy()

    # duration_hours ggf. aus X holen (falls nur 1 Var in var vorhanden)
    if "duration_hours" not in df.columns and "duration_hours" in adata.var_names:
        df = pd.concat([df, adata.to_df()[["duration_hours"]]], axis=1)

    # Typen/Zeiten vereinheitlichen
    df = _ensure_datetime(df, ("Surgery_Start", "Surgery_End", "AKI_Start"))
    if "days_to_AKI" in df.columns:
        df["days_to_AKI"] = pd.to_numeric(df["days_to_AKI"], errors="coerce")

    # Binärfelder harmonisieren
    for b in ("AKI_linked", "AKI_linked_0_7"):
        if b in df.columns:
            df[b] = (
                df[b].astype(str).str.strip().replace({"True": 1, "False": 0, "nan": np.nan, "None": np.nan})
            )
            df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)

    # Offensichtliche Datenfehler entfernen (negatives Intervall nicht zulassen)
    if "days_to_AKI" in df.columns:
        df = df[df["days_to_AKI"].isna() | (df["days_to_AKI"] >= 0)].copy()

    # Ereignis-Index initialisieren
    df["event_idx"] = 0

    if "AKI_linked_0_7" in df.columns and df["AKI_linked_0_7"].sum() > 0:
        cand = df[(df["AKI_linked_0_7"] == 1) & df["AKI_Start"].notna()].copy()
        cand = cand[cand["Surgery_End"].notna()]
        cand = cand[cand["Surgery_End"] <= cand["AKI_Start"]]

        if not cand.empty:
            idx_rows = (
                cand.sort_values(["PMID", "AKI_Start", "Surgery_End"]).groupby(["PMID", "AKI_Start"], observed=True, as_index=False).tail(1).index
            )
            df.loc[idx_rows, "event_idx"] = 1

            multi_counts = cand.groupby(["PMID", "AKI_Start"], observed=True).size().rename("n_ops_in_window").reset_index()
            n_multi = int((multi_counts["n_ops_in_window"] > 1).sum())
        else:
            n_multi = 0
    else:
        n_multi = 0

    # Ergebnisse zurück nach adata.obs (Index beibehalten)
    adata.obs.loc[df.index, "event_idx"] = df["event_idx"].astype(int)

    # Provenance
    annotate_provenance(
        adata,
        key="survival_0_7_index",
        meta={
            "window_days": CFG.WINDOW_DAYS,
            "note": "Index-OP = letzte OP vor AKI im 0–7-Tage-Fenster",
            "n_events_marked": int(df["event_idx"].sum()),
            "n_total": int(df.shape[0]),
            "n_multi_ops_per_aki": int(n_multi),
        },
    )


def build_survival_on_adata(adata: AnnData) -> pd.DataFrame:
    """Erzeugt `time_0_7` (Tage) und nutzt `event_idx` aus `obs`.

    - Für Index-OPs: `time_0_7 = min(days_to_AKI, 7)`.
    - Für andere gelinkte OPs: `time_0_7 = min(days_to_AKI, 7)`, aber `event_idx = 0`.
    - Für unverbundene OPs: `time_0_7 = 7`, `event_idx = 0`.

    Speichert `time_0_7` dauerhaft in `adata.obs` und gibt den kompakten **Survival-DataFrame** zurück.
    """
    df = adata.obs.copy()

    if "days_to_AKI" not in df.columns:
        raise KeyError("'days_to_AKI' fehlt in adata.obs – bitte beim Preprocessing erzeugen.")

    if "event_idx" not in df.columns:
        raise KeyError("'event_idx' fehlt. Bitte zuerst define_event_index_on_adata() ausführen.")

    # Basis: zensiert bei WINDOW_DAYS
    df["time_0_7"] = float(CFG.WINDOW_DAYS)

    mask_evt = df["event_idx"] == 1
    df.loc[mask_evt, "time_0_7"] = np.clip(df.loc[mask_evt, "days_to_AKI"], 0, CFG.WINDOW_DAYS)

    mask_non_index_linked = (df.get("AKI_linked_0_7", 0) == 1) & (~mask_evt)
    df.loc[mask_non_index_linked, "time_0_7"] = np.clip(df.loc[mask_non_index_linked, "days_to_AKI"], 0, CFG.WINDOW_DAYS)

    # negative Zeiten vermeiden
    df.loc[df["time_0_7"] < 0, "time_0_7"] = 0.0

    # zurückschreiben
    adata.obs.loc[df.index, "time_0_7"] = df["time_0_7"].astype(float)

    # kompakten Survival-DF bilden
    keep = [c for c in ["time_0_7", "event_idx", "days_to_AKI", "AKI_linked_0_7", "Surgery_End", "AKI_Start", "PMID", "SMID", "Procedure_ID", "duration_hours", "Sex_norm"] if c in df.columns]
    surv = df[keep].copy()

    # Provenance
    annotate_provenance(
        adata,
        key="survival_0_7",
        meta={
            "window_days": CFG.WINDOW_DAYS,
            "n_rows": int(surv.shape[0]),
            "n_events": int((surv["event_idx"] == 1).sum()),
            "time_summary": {
                "min": float(surv["time_0_7"].min()),
                "median": float(surv["time_0_7"].median()),
                "max": float(surv["time_0_7"].max()),
            },
        },
    )

    return surv


def plot_km_and_cuminc(surv: pd.DataFrame, save_dir: str) -> Tuple[str, str]:
    """Speichert zwei Plots: KM (Überleben ohne AKI) und kumulative Inzidenz (1−S(t))."""
    os.makedirs(save_dir, exist_ok=True)

    # KM
    kmf = KaplanMeierFitter()
    T = surv["time_0_7"].astype(float)
    E = surv["event_idx"].astype(int)
    kmf.fit(T, event_observed=E, label="AKI ≤ 7 Tage")

    plt.figure(figsize=(7, 5))
    ax = kmf.plot(ci_show=True)
    ax.set_title("Kaplan–Meier: Zeit bis AKI ≤ 7 Tage ab OP-Ende")
    ax.set_xlabel("Tage seit OP-Ende")
    ax.set_ylabel("Überlebenswahrscheinlichkeit ohne AKI")
    ax.grid(True, alpha=0.3)
    km_path = os.path.join(save_dir, "KM_0_7_overall.png")
    plt.tight_layout(); plt.savefig(km_path, dpi=150); plt.close()

    # kumulative Inzidenz
    cuminc = 1.0 - kmf.survival_function_
    plt.figure(figsize=(7, 5))
    plt.step(cuminc.index.values, cuminc.values.flatten(), where="post")
    plt.title("Kumulative Inzidenz: AKI ≤ 7 Tage ab OP-Ende")
    plt.xlabel("Tage seit OP-Ende")
    plt.ylabel("Anteil mit AKI (1 − S(t))")
    plt.grid(True, alpha=0.3)
    ci_path = os.path.join(save_dir, "KM_0_7_cuminc.png")
    plt.tight_layout(); plt.savefig(ci_path, dpi=150); plt.close()

    return km_path, ci_path
# nachdem 'adata' in S1/S2 fertig konstruiert ist, vor adata.write_h5ad(...):
try:
    from anndata import read_h5ad
    old_h5 = CFG.OUT_H5AD if hasattr(CFG, "OUT_H5AD") else CFG.PATH_H5AD
    if os.path.exists(old_h5):
        ad_old = read_h5ad(old_h5)
        for col in ["age_years_at_op", "age_years_at_aki", "age_cat_op"]:
            if col in getattr(ad_old, "obs", pd.DataFrame()).columns and col not in adata.obs.columns:
                adata.obs[col] = ad_old.obs.reindex(adata.obs.index)[col]
                print(f"[INFO] Spalte aus altem H5 übernommen: {col}")
except Exception as e:
    print(f"[WARN] Konnte alte Alters-Spalten nicht übernehmen: {e}")


# ---------------------------------
# 3) SUBGRUPPEN & BASISBESCHREIBUNG (S2)
# ---------------------------------

def compute_reop_and_duration_features(adata: AnnData) -> None:
    """Berechnet zusätzliche Variablen in `adata.obs`:
    - `is_reop` : 1 = Re-Operation (Patient hatte bereits eine frühere OP), sonst 0.
    - `duration_tertile` : kategorial (1/2/3) basierend auf `duration_hours`.

    Begründung:
    - Re-OPs sind klinisch relevant (höheres Risiko?).
    - Tertile der OP-Dauer erlauben eine robuste, nicht-parametrische Subgruppenbildung.
    """
    df = adata.obs.copy()

    # Sicherstellen, dass Zeitvariable für Sortierung existiert
    if "Surgery_End" in df.columns:
        df = df.copy()
        df["Surgery_End"] = pd.to_datetime(df["Surgery_End"], errors="coerce")
        # Reihenfolge je Patient
        df = df.sort_values(["PMID", "Surgery_End"]).copy()
        # Laufender Zähler pro Patient, beginnend bei 1
        df["op_seq"] = df.groupby("PMID", observed=True).cumcount() + 1
        df["is_reop"] = (df["op_seq"] > 1).astype(int)
        # zurückschreiben
        adata.obs.loc[df.index, "is_reop"] = df["is_reop"].astype(int)
    else:
        adata.obs["is_reop"] = 0  # konservativer Fallback

    # Dauer (in Stunden) prüfen
    if "duration_hours" not in adata.obs.columns and "duration_hours" in adata.var_names:
        # falls noch nicht in obs
        adata.obs = pd.concat([adata.obs, adata.to_df()[["duration_hours"]]], axis=1)

    if "duration_hours" in adata.obs.columns:
        # Tertile nur über valide Werte bilden
        dh = adata.obs["duration_hours"].astype(float)
        valid_mask = dh.notna()
        try:
            q = dh[valid_mask].quantile([1/3, 2/3]).values
            # 1: kurz, 2: mittel, 3: lang
            tertile = pd.Series(np.nan, index=adata.obs.index, dtype=float)
            tertile.loc[valid_mask & (dh <= q[0])] = 1
            tertile.loc[valid_mask & (dh > q[0]) & (dh <= q[1])] = 2
            tertile.loc[valid_mask & (dh > q[1])] = 3
            adata.obs["duration_tertile"] = tertile.astype("Int64")
        except Exception:
            adata.obs["duration_tertile"] = pd.Series(np.nan, index=adata.obs.index, dtype="Int64")
    else:
        adata.obs["duration_tertile"] = pd.Series(np.nan, index=adata.obs.index, dtype="Int64")


def km_by_group(surv: pd.DataFrame, group_col: str, save_dir: str, fname: str, order: list | None = None, label_map: dict | None = None) -> str:
    """Zeichnet KM-Kurven **stratifiziert** nach `group_col`.

    - `order` legt die Reihenfolge der Gruppen fest (optional).
    - `label_map` erlaubt schönere Legenden-Texte.
    """
    from lifelines import KaplanMeierFitter

    os.makedirs(save_dir, exist_ok=True)

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(7, 5))

    groups = surv[group_col].dropna().unique().tolist()
    if order is not None:
        groups = [g for g in order if g in groups]

    for g in groups:
        sub = surv[surv[group_col] == g]
        if sub.empty:
            continue
        T = sub["time_0_7"].astype(float)
        E = sub["event_idx"].astype(int)
        label = label_map.get(g, str(g)) if label_map else str(g)
        kmf.fit(T, event_observed=E, label=label)
        kmf.plot(ci_show=False)

    plt.title(f"Kaplan–Meier (0–7 Tage) nach {group_col}")
    plt.xlabel("Tage seit OP-Ende")
    plt.ylabel("Überlebenswahrscheinlichkeit ohne AKI")
    plt.grid(True, alpha=0.3)
    plt.legend(title=group_col)

    out_path = os.path.join(save_dir, fname)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

    return out_path


def make_table1_op_level(surv: pd.DataFrame, save_dir: str) -> str:
    """Erstellt eine einfache **Table 1** auf OP-Level, stratifiziert nach `event_idx` (0/1).

    Hinweis: `event_idx==1` bezeichnet **Index-OPs mit AKI**; `0` sind alle übrigen OPs (inkl. zensiert).
    Für die Bachelorarbeit ist das als *OP-basierte* Baseline sinnvoll, die *Patienten-basierte* Variante
    (erste OP pro Patient) können wir als Sensitivitätsanalyse ergänzen.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _summary_num(x: pd.Series) -> dict:
        x = pd.to_numeric(x, errors="coerce")
        return {
            "n": int(x.notna().sum()),
            "mean": float(x.mean()),
            "sd": float(x.std()),
            "median": float(x.median()),
            "q1": float(x.quantile(0.25)),
            "q3": float(x.quantile(0.75)),
        }

    rows = []

    for grp, gdf in surv.groupby("event_idx", observed=True):
        label = "AKI-Index-OP" if grp == 1 else "Keine AKI-Index-OP"

        # Dauer
        if "duration_hours" in gdf.columns:
            rows.append({"Gruppe": label, "Variable": "duration_hours", **_summary_num(gdf["duration_hours"])})

        # Kategoriale: Geschlecht
        if "Sex_norm" in gdf.columns:
            vc = gdf["Sex_norm"].value_counts(dropna=False)
            n = int(len(gdf))
            for k, v in vc.items():
                rows.append({"Gruppe": label, "Variable": f"Sex_norm={k}", "n": int(v), "pct": float(100.0*v/n)})

        # Kategoriale: Re-OP
        if "is_reop" in gdf.columns:
            vc = gdf["is_reop"].value_counts(dropna=False)
            n = int(len(gdf))
            for k, v in vc.items():
                rows.append({"Gruppe": label, "Variable": f"is_reop={k}", "n": int(v), "pct": float(100.0*v/n)})

        # Kategoriale: Dauer-Tertile
        if "duration_tertile" in gdf.columns:
            vc = gdf["duration_tertile"].value_counts(dropna=False)
            n = int(len(gdf))
            for k, v in vc.items():
                rows.append({"Gruppe": label, "Variable": f"duration_tertile={k}", "n": int(v), "pct": float(100.0*v/n)})

    out = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "table1_op_level.csv")
    out.to_csv(csv_path, index=False)
    return csv_path
#-----------------------------
def carry_over_age_cols(adata, old_h5_path, id_col="PMID",
                        cols=("age_years_at_op", "age_years_at_aki", "age_cat_op")) -> int:
    """
    Übernimmt Alters-Spalten aus einem bereits existierenden H5AD.
    - Bevorzugt Join über PMID; fällt sonst auf Index-Reindex zurück.
    - Ergänzt fehlende Spalten/Values in adata.obs.
    """
    import os
    import pandas as pd
    from anndata import read_h5ad

    if not os.path.exists(old_h5_path):
        return 0

    ad_old = read_h5ad(old_h5_path)
    n_set = 0

    # 1) Falls beide Daten PMID haben -> Merge über PMID
    if (id_col in adata.obs.columns) and (id_col in ad_old.obs.columns):
        left  = adata.obs[[id_col]].copy()
        right = ad_old.obs[[id_col] + [c for c in cols if c in ad_old.obs.columns]].drop_duplicates(id_col)
        merged = left.merge(right, how="left", on=id_col)
        merged.index = adata.obs.index  # gleiche Reihenfolge wie adata

        for c in cols:
            if c in merged.columns:
                if c not in adata.obs.columns:
                    adata.obs[c] = merged[c]
                else:
                    adata.obs[c] = adata.obs[c].combine_first(merged[c])
                n_set += int(adata.obs[c].notna().sum())
        return n_set

    # 2) Sonst: Versuch per Index-Reindex
    for c in cols:
        if c in ad_old.obs.columns:
            if c not in adata.obs.columns:
                adata.obs[c] = ad_old.obs.reindex(adata.obs.index)[c]
            else:
                adata.obs[c] = adata.obs[c].combine_first(ad_old.obs.reindex(adata.obs.index)[c])
            n_set += int(adata.obs[c].notna().sum())

    return n_set


# ---------------------------------
# 3) HAUPTABLAUF (S1)
# ---------------------------------

def main():
    # A) Laden (ehrapy-first, Fallback vorhanden)
    adata = load_adata()

    # B) Index-OPs definieren
    define_event_index_on_adata(adata)

    # C) Survival-Variablen aufbauen und zurück in adata.obs schreiben
    surv = build_survival_on_adata(adata)

    # D) CSV-Export (für externe Re-Use und QA)
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(CFG.SAVE_DIR, "survival_dataset_0_7.csv")
    surv.to_csv(csv_path, index=False)
    print(f"Survival-CSV gespeichert: {csv_path}")

    # E) Plots
    km_path, ci_path = plot_km_and_cuminc(surv, CFG.SAVE_DIR)
    print(f"Plots gespeichert: {km_path} | {ci_path}")
# REMOVED_TOPLEVEL_CARRYOVER: # nachdem 'adata' in S1/S2 fertig konstruiert ist, vor adata.write_h5ad(...):
# REMOVED_TOPLEVEL_CARRYOVER: try:
# REMOVED_TOPLEVEL_CARRYOVER:     from anndata import read_h5ad
# REMOVED_TOPLEVEL_CARRYOVER:     old_h5 = CFG.OUT_H5AD if hasattr(CFG, "OUT_H5AD") else CFG.PATH_H5AD
# REMOVED_TOPLEVEL_CARRYOVER:     if os.path.exists(old_h5):
# REMOVED_TOPLEVEL_CARRYOVER:         ad_old = read_h5ad(old_h5)
# REMOVED_TOPLEVEL_CARRYOVER:         for col in ["age_years_at_op", "age_years_at_aki", "age_cat_op"]:
# REMOVED_TOPLEVEL_CARRYOVER:             if col in getattr(ad_old, "obs", pd.DataFrame()).columns and col not in adata.obs.columns:
# REMOVED_TOPLEVEL_CARRYOVER:                 adata.obs[col] = ad_old.obs.reindex(adata.obs.index)[col]
# REMOVED_TOPLEVEL_CARRYOVER:                 print(f"[INFO] Spalte aus altem H5 übernommen: {col}")
# REMOVED_TOPLEVEL_CARRYOVER: except Exception as e:
# REMOVED_TOPLEVEL_CARRYOVER:     print(f"[WARN] Konnte alte Alters-Spalten nicht übernehmen: {e}")





    # F) Versionierte H5AD-Datei als S1 persistieren (ehrapy/AnnData)
    adata.write_h5ad(CFG.OUT_H5AD)
    print(f"AnnData (S1) gespeichert: {CFG.OUT_H5AD}")

    print("\nFERTIG (S1): ehrapy-first Survival-Setup abgeschlossen.\n")


def main_s2():
    """S2: Subgruppenplots + Table 1 (ehrapy-integriert).

    Lädt, falls vorhanden, die S1-Datei (`CFG.OUT_H5AD`), ergänzt Subgruppen-Features und erzeugt
    stratifizierte KM-Kurven sowie eine einfache Table 1.
    """
    # Laden: bevorzugt die versionierte S1-Datei
    adata = read_h5ad(CFG.OUT_H5AD) if os.path.exists(CFG.OUT_H5AD) else load_adata()

    # Subgruppen-Features berechnen
    compute_reop_and_duration_features(adata)

    # Survival-DF aus obs zusammenstellen
    needed = ["time_0_7", "event_idx", "duration_hours", "Sex_norm", "is_reop", "duration_tertile"]
    surv_cols = [c for c in needed if c in adata.obs.columns]
    surv = adata.obs[surv_cols].copy()

    # KM nach Geschlecht
    if "Sex_norm" in surv.columns:
        _ = km_by_group(
            surv=surv,
            group_col="Sex_norm",
            save_dir=CFG.SAVE_DIR,
            fname="KM_0_7_by_sex.png",
            order=None,
            label_map=None,
        )

    # KM Erst-OP vs Re-OP
    if "is_reop" in surv.columns:
        _ = km_by_group(
            surv=surv,
            group_col="is_reop",
            save_dir=CFG.SAVE_DIR,
            fname="KM_0_7_by_reop.png",
            order=[0, 1],
            label_map={0: "Erst-OP", 1: "Re-OP"},
        )

    # KM nach Dauer-Tertilen
    if "duration_tertile" in surv.columns:
        _ = km_by_group(
            surv=surv,
            group_col="duration_tertile",
            save_dir=CFG.SAVE_DIR,
            fname="KM_0_7_by_duration_tertile.png",
            order=[1, 2, 3],
            label_map={1: "kurz", 2: "mittel", 3: "lang"},
        )

    # Table 1
    t1_path = make_table1_op_level(surv, CFG.SAVE_DIR)
    print(f"Table 1 gespeichert: {t1_path}")

    # Speichern (gleicher OUT-Pfad, aktualisierte obs)
    adata.write_h5ad(CFG.OUT_H5AD)
    print(f"AnnData (S1+S2) gespeichert: {CFG.OUT_H5AD}")
# --------------------------------- Alter Features (S3) ---------------------------------
def add_age_features(adata, dob_candidates=("Birth_Date", "DOB", "Geburtsdatum")):
    """
    Berechnet:
      - age_years_at_op  : Alter (Jahre) zum OP-Zeitpunkt
      - age_years_at_aki : Alter (Jahre) zum AKI-Zeitpunkt (falls AKI_Start vorhanden)
      - age_cat_op       : klinisch sinnvolle Alterskategorien zum OP-Zeitpunkt
    Schreibt die Felder in adata.obs (ehrapy-first).
    """
    import numpy as np, pandas as pd

    # 1) Spalten finden
    dob_col = None
    for c in dob_candidates:
        if c in adata.obs.columns:
            dob_col = c
            break
    if dob_col is None:
        print("WARN: Kein Geburtsdatum gefunden – überspringe age-Features.")
        return adata

    # 2) Datumsfelder zu Timestamps machen (Anonymisierung egal -> es zählt die Differenz)
    for c in [dob_col, "Surgery_Start", "AKI_Start"]:
        if c in adata.obs.columns:
            adata.obs[c] = pd.to_datetime(adata.obs[c], errors="coerce")

    # 3) Alter in Jahren berechnen (OP & AKI)
    sec_per_year = 365.25 * 24 * 3600
    adata.obs["age_years_at_op"]  = (adata.obs["Surgery_Start"] - adata.obs[dob_col]).dt.total_seconds() / sec_per_year
    if "AKI_Start" in adata.obs:
        adata.obs["age_years_at_aki"] = (adata.obs["AKI_Start"] - adata.obs[dob_col]).dt.total_seconds() / sec_per_year

    # 4) Plausibilitätscheck für Pädiatrie (negativ oder > 21 Jahre -> auf NaN setzen)
    mask_bad = (adata.obs["age_years_at_op"] < 0) | (adata.obs["age_years_at_op"] > 21)
    adata.obs.loc[mask_bad, "age_years_at_op"] = np.nan
    if "age_years_at_aki" in adata.obs:
        mask_bad2 = (adata.obs["age_years_at_aki"] < 0) | (adata.obs["age_years_at_aki"] > 21)
        adata.obs.loc[mask_bad2, "age_years_at_aki"] = np.nan

    # 5) Klinische Kategorien (Beispiel; gern anpassen)
    bins   = [-np.inf, 1, 5, 12, 18, np.inf]
    labels = ["<1 J", "1–4 J", "5–11 J", "12–17 J", "≥18 J"]
    adata.obs["age_cat_op"] = pd.Categorical(pd.cut(adata.obs["age_years_at_op"], bins=bins, labels=labels, right=False))

    # 6) Kurz-Report
    print("AGE: n_valid_op =", int(adata.obs["age_years_at_op"].notna().sum()),
          "| median =", round(float(adata.obs["age_years_at_op"].median(skipna=True)), 2))

    # 7) Doku ins uns (nur einfache Typen)
    meta = {
        "dob_col": dob_col,
        "age_years_at_op_desc": "Alter in Jahren zur Operation (aus DOB & Surgery_Start)",
        "age_years_at_aki_desc": "Alter in Jahren zum AKI-Beginn (falls AKI_Start vorhanden)",
        "age_cat_op_bins": bins,
        "age_cat_op_labels": labels,
    }
    adata.uns["age_features"] = meta
    return adata



#%%
# ---------------------------------
# S4) PRÄDIKTION & INFERENZ – Logistische Regression (ehrapy-first)
#     - GLM (Binomial, Logit) mit **cluster-robusten SE** (Cluster = PMID)
#     - Cross-Validation mit **GroupKFold (PMID)** und scikit-learn-Pipeline
#     - Outputs: OR-Tabelle (CSV), ROC/PR/Calibration-Plots + Metriken
#     - Ergebnisse in adata.uns['S4_glm'] und adata.uns['S4_cv'] (nur einfache Typen)
# ---------------------------------

# ============================
# S4 – Logit + GroupKFold CV
# ============================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def _prepare_model_df(adata):
    import numpy as np, pandas as pd
    base = ["PMID","event_idx","duration_hours","is_reop","Sex_norm"]
    extra = []
    if "age_years_at_op" in adata.obs.columns:
        extra.append("age_years_at_op")

    cols = [c for c in (base + extra) if c in adata.obs.columns]
    df = adata.obs[cols].copy()

    # Typen
    df["event_idx"] = pd.to_numeric(df["event_idx"], errors="coerce").astype("Int64")
    df = df[df["event_idx"].notna() & df["PMID"].notna()].copy()
    df["event_idx"] = df["event_idx"].astype(int)

    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["is_reop"] = pd.to_numeric(df["is_reop"], errors="coerce").fillna(0).astype(int)
    df["Sex_norm"] = df["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan})

    if "age_years_at_op" in df.columns:
        df["age_years_at_op"] = pd.to_numeric(df["age_years_at_op"], errors="coerce")

    return df
def _fit_glm_clustered(df, save_dir):
    """GLM (Binomial Logit) + cluster-robuste SE (Cluster=PMID). Speichert OR-Tabelle."""
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Zielvariable
    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values

    # Features dynamisch (Alter nur, wenn vorhanden)
    feats = ["duration_hours", "is_reop", "Sex_norm"]
    if "age_years_at_op" in df.columns:
        feats.append("age_years_at_op")

    X = df[feats].copy()
    X["duration_hours"] = X["duration_hours"].fillna(X["duration_hours"].median())
    X["is_reop"] = X["is_reop"].fillna(0).astype(int)
    X["Sex_norm"] = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")
    if "age_years_at_op" in X.columns:
        X["age_years_at_op"] = pd.to_numeric(X["age_years_at_op"], errors="coerce")
        X["age_years_at_op"] = X["age_years_at_op"].fillna(X["age_years_at_op"].median())

    # One-Hot für Sex_norm
    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = (X_dm
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float64))

    # Konstante hinzufügen
    X_dm = sm.add_constant(X_dm, has_constant="add")

    # Cluster-Codes (PMID)
    groups = df["PMID"].astype("category").cat.codes.to_numpy()

    # GLM fitten – direkt mit cluster-robuster Kovarianz
    model = sm.GLM(y, X_dm.values, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    # OR-Tabelle
    params = res.params
    se     = res.bse
    z      = res.tvalues
    p      = res.pvalues
    ci_lo  = params - 1.96 * se
    ci_hi  = params + 1.96 * se

    terms = list(X_dm.columns)
    out = pd.DataFrame({
        "Term": terms,
        "Coef": params,
        "OR": np.exp(params),
        "CI_low": np.exp(ci_lo),
        "CI_high": np.exp(ci_hi),
        "z": z,
        "p": p,
    })

    # Intercept nach oben sortieren
    intercept = out[out["Term"] == "const"]
    body = out[out["Term"] != "const"].sort_values("Term")
    out = pd.concat([intercept, body], axis=0) if not intercept.empty else body

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "S4_glm_cluster_or.csv")
    out.to_csv(path, index=False)
    print("GLM (cluster-robust) OR-Tabelle gespeichert:", path)
    return path
def _run_groupkfold_cv(df, save_dir, n_splits=5):
    """GroupKFold (Gruppen=PMID) – ROC/PR/Kalibration + Metriken."""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve

    y = df["event_idx"].values
    groups = df["PMID"].values

    feats = ["duration_hours", "is_reop", "Sex_norm"]
    if "age_years_at_op" in df.columns:
        feats.append("age_years_at_op")
    X = df[feats].copy()

    num_feats = ["duration_hours", "is_reop"]
    if "age_years_at_op" in X.columns:
        num_feats.append("age_years_at_op")
    cat_feats = ["Sex_norm"]  # ← wichtig

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_feats),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_feats)
    ])

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs"))
    ])

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    proba = np.empty_like(y, dtype=float)

    for i, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        clf.fit(X.iloc[tr], y[tr])
        proba[te] = clf.predict_proba(X.iloc[te])[:, 1]
        print(f"Fold {i} train={len(tr)} test={len(te)}")

    # Kennzahlen
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)

    # Plots
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(6,5)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("S4 ROC (GroupKFold)"); plt.tight_layout()
    roc_path = os.path.join(save_dir, "S4_ROC.png"); plt.savefig(roc_path, dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(y, proba)
    plt.figure(figsize=(6,5)); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("S4 Precision-Recall (GroupKFold)"); plt.tight_layout()
    pr_path = os.path.join(save_dir, "S4_PR.png"); plt.savefig(pr_path, dpi=150); plt.close()

    prob_true, prob_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5)); plt.plot(prob_pred, prob_true); plt.plot([0,1],[0,1],'--'); plt.xlabel("Vorhergesagte Wahrscheinlichkeit"); plt.ylabel("Beobachteter Anteil"); plt.title("S4 Kalibration (GroupKFold)"); plt.tight_layout()
    cal_path = os.path.join(save_dir, "S4_Calibration.png"); plt.savefig(cal_path, dpi=150); plt.close()

    metrics = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier": float(brier),
               "roc_path": roc_path, "pr_path": pr_path, "calibration_path": cal_path}
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "S4_cv_metrics.csv"), index=False)
    print(f"CV-Metriken: ROC_AUC={roc_auc:.3f}, PR_AUC={pr_auc:.3f}, Brier={brier:.3f}")
    return metrics

# ---------------------------------
def _fit_glm_clustered_interaction(df, save_dir):
    """
    GLM mit Interaktion: duration_hours × is_reop (cluster-robuste SE, Cluster=PMID).
    Speichert:
      - S4_glm_interaction_or.csv      (OR & 95%-KI für alle Terme)
      - S4_glm_interaction_slopes_by_group.csv  (Steigung der Dauer je Gruppe als OR/h)
    """
    import os, numpy as np, pandas as pd, statsmodels.api as sm

    # Ziel
    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values

    # Basisfeatures
    feats = ["duration_hours", "is_reop", "Sex_norm"]
    if "age_years_at_op" in df.columns:
        feats.append("age_years_at_op")

    X = df[feats].copy()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce").fillna(X["duration_hours"].median())
    X["is_reop"]        = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    X["Sex_norm"]       = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")
    if "age_years_at_op" in X.columns:
        X["age_years_at_op"] = pd.to_numeric(X["age_years_at_op"], errors="coerce").fillna(X["age_years_at_op"].median())

    # Interaktion
    X["duration_x_reop"] = X["duration_hours"] * X["is_reop"]

    # Dummies für Sex
    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = (X_dm.apply(pd.to_numeric, errors="coerce")
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0).astype(np.float64))
    X_dm = sm.add_constant(X_dm, has_constant="add")

    # Cluster-GLM
    groups = df["PMID"].astype("category").cat.codes.to_numpy()
    model  = sm.GLM(y, X_dm.values, family=sm.families.Binomial())
    res    = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    # OR-Tabelle
    params = res.params; se = res.bse; z = res.tvalues; p = res.pvalues
    ci_lo  = params - 1.96*se; ci_hi = params + 1.96*se
    terms  = list(X_dm.columns)
    or_df = pd.DataFrame({
        "Term": terms, "Coef": params, "OR": np.exp(params),
        "CI_low": np.exp(ci_lo), "CI_high": np.exp(ci_hi), "z": z, "p": p
    })
    inter_csv = os.path.join(save_dir, "S4_glm_interaction_or.csv")
    or_df.to_csv(inter_csv, index=False)

    # Marginaler Dauer-Effekt (Steigung) je Gruppe (Erst-OP vs. Re-OP)
    # slope(is_reop=g) = beta_duration + g * beta_interaction;  Var nach Delta-Methode
    b = or_df.set_index("Term")["Coef"]
    cov = res.cov_params()
    def slope_and_ci(g=0):
        beta_d = b.get("duration_hours", 0.0)
        beta_i = b.get("duration_x_reop", 0.0)
        slope  = beta_d + g*beta_i
        # Var = var(d) + g^2 var(i) + 2g cov(d,i)
        try:
            idx_d = or_df.index[or_df["Term"]=="duration_hours"][0]
            idx_i = or_df.index[or_df["Term"]=="duration_x_reop"][0]
            var = cov[idx_d, idx_d] + (g**2)*cov[idx_i, idx_i] + 2*g*cov[idx_d, idx_i]
        except Exception:
            var = 0.0
        se = np.sqrt(max(var, 0.0))
        lo, hi = slope - 1.96*se, slope + 1.96*se
        return slope, lo, hi

    rows = []
    for g, label in [(0, "Erst-OP"), (1, "Re-OP")]:
        s, lo, hi = slope_and_ci(g)
        rows.append({
            "group": label,
            "logit_slope_per_hour": s,
            "OR_per_hour": float(np.exp(s)),
            "OR_CI_low": float(np.exp(lo)),
            "OR_CI_high": float(np.exp(hi)),
        })
    slopes_df = pd.DataFrame(rows)
    slopes_csv = os.path.join(save_dir, "S4_glm_interaction_slopes_by_group.csv")
    slopes_df.to_csv(slopes_csv, index=False)

    print("Interaktion-ORs gespeichert:", inter_csv)
    print("Steigungen (OR/h) gespeichert:", slopes_csv)
    return {"or_path": inter_csv, "slopes_path": slopes_csv, "columns": terms, "params": params, "cov": cov}
#s4_plot_margins_duration_by_reop_interaction
def s4_plot_margins_duration_by_reop(df: pd.DataFrame, save_dir: str) -> str:
    """Marginaleffekte p(AKI) ~ OP-Dauer für Erst-OP vs. Re-OP (95%-KI), ohne Interaktion im Fit."""
    import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt

    # Features wie im GLM
    feats = ["duration_hours", "is_reop", "Sex_norm"]
    if "age_years_at_op" in df.columns:
        feats.append("age_years_at_op")
    X = df[feats].copy()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce").fillna(X["duration_hours"].median())
    X["is_reop"]        = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    X["Sex_norm"]       = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")
    if "age_years_at_op" in X.columns:
        X["age_years_at_op"] = pd.to_numeric(X["age_years_at_op"], errors="coerce").fillna(X["age_years_at_op"].median())

    # Designmatrix wie im GLM
    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = (X_dm.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64))
    X_dm = sm.add_constant(X_dm, has_constant="add")

    # Fit (cluster-robust)
    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values
    groups = df["PMID"].astype("category").cat.codes.to_numpy()
    model  = sm.GLM(y, X_dm.values, family=sm.families.Binomial())
    res    = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    cols = list(X_dm.columns)
    beta = res.params
    cov  = res.cov_params()

    def row_for(duration, reop_flag):
        d = {c: 0.0 for c in cols}
        d["const"] = 1.0
        d["duration_hours"] = float(duration)
        d["is_reop"] = float(reop_flag)
        if "age_years_at_op" in cols:
            d["age_years_at_op"] = float(X["age_years_at_op"].median())
        # Sex-Dummies bleiben 0 => Basis-Kategorie
        return np.array([d[c] for c in cols], dtype=float)

    def invlogit(z):
        z = np.clip(z, -50, 50); ez = np.exp(z); return ez/(1+ez)

    grid = np.linspace(np.percentile(X["duration_hours"], 2),
                       np.percentile(X["duration_hours"], 98), 80)

    plt.figure(figsize=(7.5, 5.0))
    for g, label in [(0, "Erst-OP"), (1, "Re-OP")]:
        eta, se = [], []
        for x in grid:
            r = row_for(x, g)
            eta.append(float(r @ beta))
            se.append(float(np.sqrt(max(r @ cov @ r, 0.0))))
        eta = np.array(eta); se = np.array(se)
        p   = invlogit(eta)
        lo  = invlogit(eta - 1.96*se)
        hi  = invlogit(eta + 1.96*se)
        plt.plot(grid, p, label=label)
        plt.fill_between(grid, lo, hi, alpha=0.15)

    plt.xlabel("OP-Dauer (Stunden)")
    plt.ylabel("Prädizierte AKI-Wahrscheinlichkeit (0–7 Tage)")
    plt.title("S4 – Marginaleffekte der OP-Dauer nach OP-Typ")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "S4_margins_duration_by_reop.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Margins (Erst-OP vs. Re-OP) gespeichert:", out)
    return out

####
# ========= S4: Interaktion Dauer × Re-OP =========
def _fit_glm_clustered_interaction(df: pd.DataFrame, save_dir: str) -> dict:
    """GLM (Logit) mit Interaktion duration_hours * is_reop; cluster-robuste SE (Cluster=PMID)."""
    import statsmodels.api as sm
    os.makedirs(save_dir, exist_ok=True)

    # Daten
    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values
    groups = df["PMID"].astype("category").cat.codes.to_numpy()

    X = df[["duration_hours","is_reop","Sex_norm"]].copy()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce").fillna(X["duration_hours"].median())
    X["is_reop"] = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    X["Sex_norm"] = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")

    # Dummies + Interaktion
    dm = pd.get_dummies(X[["Sex_norm"]], drop_first=True).astype(float)
    Xdm = pd.concat([X[["duration_hours","is_reop"]].astype(float), dm], axis=1)
    Xdm["duration_hours:is_reop"] = Xdm["duration_hours"] * Xdm["is_reop"]
    Xdm = Xdm.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float64)
    Xdm = sm.add_constant(Xdm, has_constant="add")

    model = sm.GLM(y, Xdm.values, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    # OR-Tabelle
    terms = list(Xdm.columns)
    b, se, z, p = res.params, res.bse, res.tvalues, res.pvalues
    ci_lo, ci_hi = b - 1.96*se, b + 1.96*se
    or_tab = pd.DataFrame({
        "Term": terms, "Coef": b, "OR": np.exp(b),
        "CI_low": np.exp(ci_lo), "CI_high": np.exp(ci_hi),
        "z": z, "p": p
    })
    intercept = or_tab[or_tab["Term"]=="const"]; body = or_tab[or_tab["Term"]!="const"].sort_values("Term")
    or_tab = pd.concat([intercept, body], axis=0) if not intercept.empty else body
    or_csv = os.path.join(save_dir, "S4_glm_interaction_or.csv")
    or_tab.to_csv(or_csv, index=False)

    # Abgeleitete Steigungen (Delta-Methode)
    cov = res.cov_params()
    idx_d = terms.index("duration_hours")
    idx_i = terms.index("duration_hours:is_reop")
    beta_d = float(b[idx_d]); var_d = float(cov[idx_d, idx_d])
    beta_i = float(b[idx_i]); var_i = float(cov[idx_i, idx_i])
    cov_di = float(cov[idx_d, idx_i])

    # Erst-OP
    or_erst = np.exp(beta_d)
    se_erst = np.sqrt(max(var_d, 0.0))
    ci_erst = (np.exp(beta_d - 1.96*se_erst), np.exp(beta_d + 1.96*se_erst))
    # Re-OP
    beta_sum = beta_d + beta_i
    var_sum  = max(var_d + var_i + 2.0*cov_di, 0.0)
    se_sum   = np.sqrt(var_sum)
    or_reop  = np.exp(beta_sum)
    ci_reop  = (np.exp(beta_sum - 1.96*se_sum), np.exp(beta_sum + 1.96*se_sum))
    slopes = pd.DataFrame([
        {"Gruppe":"Erst-OP", "OR_pro_Stunde": or_erst, "CI_low": ci_erst[0], "CI_high": ci_erst[1]},
        {"Gruppe":"Re-OP",   "OR_pro_Stunde": or_reop, "CI_low": ci_reop[0], "CI_high": ci_reop[1]},
    ])
    slopes_csv = os.path.join(save_dir, "S4_glm_interaction_slopes_by_group.csv")
    slopes.to_csv(slopes_csv, index=False)

    out = {
        "or_table_path": or_csv,
        "slopes_path": slopes_csv,
        "p_interaction": float(p[terms.index("duration_hours:is_reop")])
    }
    print("Interaktions-GLM gespeichert:", or_csv, "| slopes:", slopes_csv, "| p_interaction=", out["p_interaction"])
    return out


def s4_plot_margins_duration_by_reop_interaction(df: pd.DataFrame, save_dir: str) -> str:
    """Plot p(AKI) ~ Dauer für Erst-OP vs. Re-OP aus dem Interaktionsmodell."""
    import statsmodels.api as sm
    os.makedirs(save_dir, exist_ok=True)

    # Daten
    X = df[["duration_hours","is_reop","Sex_norm"]].copy()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce").fillna(X["duration_hours"].median())
    X["is_reop"] = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    X["Sex_norm"] = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")

    dm = pd.get_dummies(X[["Sex_norm"]], drop_first=True).astype(float)
    Xdm = pd.concat([X[["duration_hours","is_reop"]].astype(float), dm], axis=1)
    Xdm["duration_hours:is_reop"] = Xdm["duration_hours"] * Xdm["is_reop"]
    Xdm = Xdm.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float64)
    Xdm = sm.add_constant(Xdm, has_constant="add")

    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values
    groups = df["PMID"].astype("category").cat.codes.to_numpy()

    model = sm.GLM(y, Xdm.values, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    params = res.params; cov = res.cov_params()

    # Grid
    q = X["duration_hours"].quantile([0.05,0.95]).values
    grid = np.linspace(q[0], q[1], 80)
    cols = list(Xdm.columns)

    def pred_curve(is_reop_val: int):
        Xg = pd.DataFrame(0.0, index=np.arange(len(grid)), columns=cols, dtype=np.float64)
        Xg["const"] = 1.0
        Xg["duration_hours"] = grid
        Xg["is_reop"] = float(is_reop_val)
        Xg["duration_hours:is_reop"] = grid * float(is_reop_val)
        M = Xg.values
        eta = M @ params
        var = np.einsum("ij,jk,ik->i", M, cov, M)
        se = np.sqrt(np.clip(var, 0, np.inf))
        inv = lambda z: 1/(1+np.exp(-z))
        return inv(eta), inv(eta-1.96*se), inv(eta+1.96*se)

    p0, l0, h0 = pred_curve(0)
    p1, l1, h1 = pred_curve(1)

    plt.figure(figsize=(7,5))
    plt.plot(grid, p0, label="Erst-OP"); plt.fill_between(grid, l0, h0, alpha=0.15)
    plt.plot(grid, p1, label="Re-OP");  plt.fill_between(grid, l1, h1, alpha=0.15)
    plt.xlabel("OP-Dauer (Stunden)"); plt.ylabel("Prädizierte AKI-Wahrscheinlichkeit (0–7 Tage)")
    plt.title("S4 – Marginaleffekte (Interaktionsmodell)")
    plt.legend(); plt.tight_layout()
    out = os.path.join(save_dir, "S4_margins_duration_by_reop_INTERACTION.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Interaktions-Margins gespeichert:", out)
    return out

#  Zusatzgrafik, die die beiden Steigungen (OR/h) mit 95 %-KI nebeneinander als Forest-Mini-Plot zeig
def s4_plot_interaction_slopes_forest(save_dir: str) -> str:
    """
    Liest Diagramme/S4_glm_interaction_slopes_by_group.csv und zeichnet
    einen kompakten Forest-Plot (OR/h mit 95%-KI) für Erst-OP vs. Re-OP.
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

    csv_path = os.path.join(save_dir, "S4_glm_interaction_slopes_by_group.csv")
    if not os.path.exists(csv_path):
        print(f"WARN: {csv_path} nicht gefunden – überspringe Forest (Slopes).")
        return ""

    df = pd.read_csv(csv_path)
    # Erwartete Spalten: group / OR_per_hour / OR_CI_low / OR_CI_high
    # (deine CSV hat evtl. deutsche Namen – abfangen)
    if {"group","OR_per_hour","OR_CI_low","OR_CI_high"}.issubset(df.columns):
        g   = df["group"].astype(str).tolist()
        OR  = df["OR_per_hour"].astype(float).values
        lo  = df["OR_CI_low"].astype(float).values
        hi  = df["OR_CI_high"].astype(float).values
    else:
        # Deutsche Varianten aus deiner Ausgabe:
        g   = df.iloc[:,0].astype(str).tolist()                  # Gruppe
        OR  = df.iloc[:,1].astype(float).values                  # OR_pro_Stunde
        lo  = df.iloc[:,2].astype(float).values                  # CI_low
        hi  = df.iloc[:,3].astype(float).values                  # CI_high

    # Reihenfolge: Erst-OP oben, Re-OP unten (wie in deiner Tabelle)
    y = np.arange(len(g))[::-1]

    plt.figure(figsize=(6.5, 2.8))
    # Konfidenzintervalle
    plt.hlines(y, lo, hi)
    # Punkt-OR
    plt.plot(OR, y, "o")
    # Referenzlinie bei OR=1
    plt.vlines(1.0, -1, len(g), linestyles="--")

    plt.yticks(y, g)
    plt.xscale("log")
    plt.xlabel("OR pro Stunde (log-Skala)")
    plt.title("Steigung der AKI-Odds pro Stunde – Erst-OP vs. Re-OP")
    plt.tight_layout()

    out = os.path.join(save_dir, "S4_interaction_slopes_forest.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Mini-Forest (Slopes) gespeichert:", out)
    return out
def s4_plot_margins_age(df: pd.DataFrame, save_dir: str) -> str:
    """
    Marginaleffekt: pr(AKI) über Alter (Jahre) – Basis: Erst-OP, Referenz-Geschlecht,
    Dauer fix auf Median. Nutzt das gleiche GLM-Setup (cluster-robust).
    """
    import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt

    if "age_years_at_op" not in df.columns or df["age_years_at_op"].notna().sum() == 0:
        print("WARN: Keine Altersvariable vorhanden – Alter-Plot übersprungen.")
        return ""

    # === Daten wie im GLM ===
    feats = ["duration_hours", "is_reop", "Sex_norm", "age_years_at_op"]
    X = df[feats].copy()

    # Fixwerte: Dauer = Median, Re-OP = 0 (Erst-OP), Sex = häufigste Kategorie
    dur_med  = pd.to_numeric(X["duration_hours"], errors="coerce").median()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce").fillna(dur_med)
    X["is_reop"]        = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    sex_mode = (X["Sex_norm"].astype(str)
                .replace({"nan": np.nan, "None": np.nan})
                .fillna("Missing")).mode().iat[0]
    X["Sex_norm"] = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna(sex_mode)
    X["age_years_at_op"] = pd.to_numeric(X["age_years_at_op"], errors="coerce")

    # Designmatrix wie im GLM
    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = (X_dm.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0).astype(np.float64))
    X_dm = sm.add_constant(X_dm, has_constant="add")

    # Fit
    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values
    groups = df["PMID"].astype("category").cat.codes.to_numpy()
    model  = sm.GLM(y, X_dm.values, family=sm.families.Binomial())
    res    = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    cols = list(X_dm.columns)
    beta = res.params
    cov  = res.cov_params()

    # Hilfsvektor für Vorhersage: Erst-OP, Dauer=Median, Sex=Referenz (alle Sex-Dummies=0)
    def row_for(age_years: float):
        d = {c: 0.0 for c in cols}
        d["const"] = 1.0
        d["duration_hours"]  = float(dur_med)
        d["is_reop"]         = 0.0
        d["age_years_at_op"] = float(age_years)
        # Sex-Dummyspalten bleiben 0 → Referenzkategorie
        return np.array([d[c] for c in cols], dtype=float)

    def invlogit(z):
        z = np.clip(z, -50, 50); ez = np.exp(z); return ez / (1.0 + ez)

    s = df["age_years_at_op"].dropna().astype(float)
    lo_q, hi_q = np.percentile(s, [2, 98])
    grid = np.linspace(lo_q, hi_q, 80)

    eta, se = [], []
    for a in grid:
        r = row_for(a)
        eta.append(float(r @ beta))
        se.append(float(np.sqrt(max(r @ cov @ r, 0.0))))
    eta = np.array(eta); se = np.array(se)

    p   = invlogit(eta)
    lo  = invlogit(eta - 1.96*se)
    hi  = invlogit(eta + 1.96*se)

    plt.figure(figsize=(7.5, 5.5))
    plt.plot(grid, p, label="geschätzt")
    plt.fill_between(grid, lo, hi, alpha=0.15, label="95%-KI")
    plt.xlabel("Alter zum OP-Zeitpunkt (Jahre)")
    plt.ylabel("Prädizierte AKI-Wahrscheinlichkeit (0–7 Tage)")
    plt.title("S4 – Marginaleffekt des Alters (Basis: Erst-OP, Dauer=Median)")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(save_dir, "S4_margins_age.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Marginaleffekt (Alter) gespeichert:", out)
    return out




def main_s4():
    """S4 ausführen und Ergebnisse in adata.uns ablegen (nur einfache Typen)."""
    from anndata import read_h5ad
    import pandas as pd

    h5 = CFG.OUT_H5AD if hasattr(CFG, "OUT_H5AD") else CFG.PATH_H5AD
    adata = read_h5ad(h5)

    # Basis-DF
    df = _prepare_model_df(adata)

    # GLM (ohne Interaktion), CV, Standard-Plots
    or_csv = _fit_glm_clustered(df, CFG.SAVE_DIR)
    metrics = _run_groupkfold_cv(df, CFG.SAVE_DIR, n_splits=5)
    s4_plot_forest(or_csv, CFG.SAVE_DIR)
    s4_plot_margins_duration(df, CFG.SAVE_DIR)
    s4_plot_margins_duration_by_reop(df, CFG.SAVE_DIR)
    # Alter-Plot nur, wenn vorhanden
    if "age_years_at_op" in df.columns and df["age_years_at_op"].notna().any():
        s4_plot_margins_age(df, CFG.SAVE_DIR)
    # Interaktionsanalyse + Plot
    #inter = _fit_glm_clustered_interaction(df, CFG.SAVE_DIR)
    #inter_png = s4_plot_margins_duration_by_reop_interaction(df, CFG.SAVE_DIR)
    s4_plot_interaction_slopes_forest(CFG.SAVE_DIR)

    # in uns referenzieren
    try:
        k_terms = int(len(pd.read_csv(or_csv)))
    except Exception:
        k_terms = 0
    adata.uns["S4_glm"] = {
        "or_table_path": or_csv,
        "forest_path": os.path.join(CFG.SAVE_DIR, "S4_forest_or.png"),
        "margins_duration_path": os.path.join(CFG.SAVE_DIR, "S4_margins_duration.png"),
        "margins_by_reop_path": os.path.join(CFG.SAVE_DIR, "S4_margins_duration_by_reop.png"),
        "n": int(df.shape[0]),
        "k": k_terms
    }
    #adata.uns["S4_interaction"] = {**inter, "margins_path": inter_png}
    adata.uns["S4_cv"] = metrics

    adata.write_h5ad(h5)
    print(f"AnnData (S4) gespeichert: {h5}")

        
    # ===== S4: Visualisierungen =====
import matplotlib.pyplot as plt

def s4_plot_forest(or_csv_path: str, save_dir: str) -> str:
    """Forest-Plot für ORs (exkl. Intercept)."""
    import pandas as pd, numpy as np, os
    df = pd.read_csv(or_csv_path)
    df = df[df["Term"] != "const"].copy()
    if df.empty:
        return ""
    # Sortierung: erst binär/kategorial, dann kontinuierlich (oder einfach alphabetisch)
    df = df.sort_values("Term")
    terms = df["Term"].tolist()
    OR = df["OR"].values
    lo = df["CI_low"].values
    hi = df["CI_high"].values

    y = np.arange(len(terms))[::-1]
    plt.figure(figsize=(7, 0.6*len(terms) + 1))
    plt.hlines(y, lo, hi)
    plt.plot(OR, y, "o")
    plt.vlines(1.0, -1, len(terms), linestyles="--")
    plt.yticks(y, terms)
    plt.xlabel("Odds Ratio (log-Skala)")
    plt.xscale("log")
    plt.title("S4 – Odds Ratios (95%-KI)")
    plt.tight_layout()
    out = os.path.join(save_dir, "S4_forest_or.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Forest-Plot gespeichert:", out)
    return out


#Plot

def s4_plot_margins_duration(df: pd.DataFrame, save_dir: str) -> str:
    """Marginaleffekt: p(AKI) über OP-Dauer (Baseline: Erst-OP, Referenz-Geschlecht)."""
    import statsmodels.api as sm, numpy as np, pandas as pd, os, math

    # === Daten & Dummy-Matrix wie im GLM ===
    X = df[["duration_hours", "is_reop", "Sex_norm"]].copy()
    X["duration_hours"] = pd.to_numeric(X["duration_hours"], errors="coerce")
    X["duration_hours"] = X["duration_hours"].fillna(X["duration_hours"].median())
    X["is_reop"] = pd.to_numeric(X["is_reop"], errors="coerce").fillna(0).astype(int)
    X["Sex_norm"] = X["Sex_norm"].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna("Missing")

    X_dm = pd.get_dummies(X, columns=["Sex_norm"], drop_first=True)
    X_dm = X_dm.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float64)
    X_dm_const = sm.add_constant(X_dm, has_constant="add")

    y = pd.to_numeric(df["event_idx"], errors="coerce").astype(int).values
    groups = df["PMID"].astype("category").cat.codes.to_numpy()

    # === GLM mit cluster-robusten SE direkt beim Fit ===
    model = sm.GLM(y, X_dm_const.values, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    params = res.params
    cov = res.cov_params()

    # === Grid für Dauer (5.–95. Perzentil), Baseline: Erst-OP (is_reop=0), Referenz-Geschlecht ===
    q = X["duration_hours"].quantile([0.05, 0.95]).values
    grid = np.linspace(q[0], q[1], 60)

    # Designmatrix für Grid mit exakt gleichen Spalten wie X_dm_const
    cols = list(X_dm_const.columns)
    Xg = pd.DataFrame(0.0, index=np.arange(len(grid)), columns=cols, dtype=np.float64)
    Xg["const"] = 1.0
    if "duration_hours" in Xg.columns:
        Xg["duration_hours"] = grid
    if "is_reop" in Xg.columns:
        Xg["is_reop"] = 0.0  # Baseline: Erst-OP
    # alle Sex-Dummies bleiben 0 → Referenzkategorie
    
    # Vorhersage auf Link-Skala + Delta-Methode für 95%-KI, dann inverse Logit
    Xg_mat = Xg.values
    eta = Xg_mat @ params
    var = np.einsum("ij,jk,ik->i", Xg_mat, cov, Xg_mat)
    se = np.sqrt(np.clip(var, 0, np.inf))

    lo_eta = eta - 1.96*se
    hi_eta = eta + 1.96*se
    invlogit = lambda z: 1.0/(1.0+np.exp(-z))
    p = invlogit(eta)
    p_lo = invlogit(lo_eta)
    p_hi = invlogit(hi_eta)

    # Plotten
    plt.figure(figsize=(7,5))
    plt.plot(grid, p, label="geschätzt")
    plt.fill_between(grid, p_lo, p_hi, alpha=0.2, label="95%-KI")
    plt.xlabel("OP-Dauer (Stunden)")
    plt.ylabel("Prädizierte AKI-Wahrscheinlichkeit (0–7 Tage)")
    plt.title("S4 – Marginaleffekt der OP-Dauer (Baseline: Erst-OP, Referenz-Geschlecht)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "S4_margins_duration.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Marginaleffekte gespeichert:", out)
    return out

    

if __name__ == "__main__":
    #main()     # S1
   # main_s2()  # S2
    main_s4()  # S4  
    
    
    

# %%



