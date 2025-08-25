import pandas as pd

def load_ops_hlm(path: str) -> pd.DataFrame:
    # 1) Erster Versuch: Auto-Delimiter
    try:
        ops = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        # Fallback, falls Auto-Erkennung scheitert
        ops = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # 2) Prüfen: Wurde nur 1 Spalte eingelesen (verräterisch für falsches sep)?
    if ops.shape[1] == 1:
        # harter Fallback auf Semikolon
        ops = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # 3) Spalten säubern + umbenennen
    ops.columns = ops.columns.str.strip()
    rename_map = {
        "Start of surgery": "Surgery_Start",
        "End of surgery": "Surgery_End",
        # falls schon vereinheitlicht, passiert nichts
    }
    ops = ops.rename(columns=rename_map)

    # 4) Pflichtspalten prüfen (nach dem Umbenennen!)
    required = ["PMID", "SMID", "Procedure_ID", "Surgery_Start", "Surgery_End"]
    missing = [c for c in required if c not in ops.columns]
    if missing:
        raise ValueError(
            f"Fehlende Spalten in HLM-Datei: {missing}. "
            f"Aktuell vorhanden: {ops.columns.tolist()}"
        )

    # 5) Datumsfelder parsen + Dauer berechnen
    ops["Surgery_Start"] = pd.to_datetime(ops["Surgery_Start"], errors="coerce")
    ops["Surgery_End"]   = pd.to_datetime(ops["Surgery_End"], errors="coerce")
    ops = ops.dropna(subset=["Surgery_Start", "Surgery_End"]).copy()
    ops["op_hours"] = (ops["Surgery_End"] - ops["Surgery_Start"]).dt.total_seconds() / 3600.0
    # Plausible Grenzen (anpassbar)
    ops = ops[(ops["op_hours"] >= 0.1) & (ops["op_hours"] <= 24)].copy()

    # 6) Unnötige Spalte raus
    ops = ops.drop(columns=["Tx?"], errors="ignore")

    return ops


# ======= Nutzung =======
path = "HLM Operationen.csv"  # ggf. absoluter Pfad
ops = load_ops_hlm(path)
print("Spalten OK:", ops.columns.tolist())
print("n_OP-Zeilen:", len(ops))
print(ops.head(3))
import re
import numpy as np

# ---------- Patientendaten laden (robust) ----------
def load_patients(path: str) -> pd.DataFrame:
    try:
        pat = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        pat = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if pat.shape[1] == 1 and ";" in pat.columns[0]:
        pat = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    pat.columns = pat.columns.str.strip()
    # Pflichtspalten prüfen (mindestens diese drei)
    req = {"PMID", "Sex", "DateOfBirth"}
    missing = [c for c in req if c not in pat.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in Patientendatei: {missing}. Vorhanden: {pat.columns.tolist()}")
    # Datumsfelder parsen
    pat["DateOfBirth"] = pd.to_datetime(pat["DateOfBirth"], errors="coerce")
    if "DateOfDie" in pat.columns:
        pat["DateOfDie"] = pd.to_datetime(pat["DateOfDie"], errors="coerce")
    # Sex normalisieren, falls noch nicht vorhanden
    if "Sex_norm" not in pat.columns:
        pat["Sex_norm"] = (
            pat["Sex"].astype(str).str.strip().str.lower()
              .map({"m":"m","male":"m","w":"w","f":"w","female":"w"})
              .fillna(pat["Sex"].astype(str))
        )
    return pat[["PMID", "Sex_norm", "DateOfBirth"]].copy()

# ---------- AKI laden & harmonisieren ----------
def map_aki_stage(s):
    if pd.isna(s): return 0
    s = str(s).strip().lower()
    if re.search(r"\baki\s*3\b", s): return 3
    if re.search(r"\baki\s*2\b", s): return 2
    if re.search(r"\baki\s*1\b", s): return 1
    if s in {"0","keine","kein aki","keine aki","none","no","nein"}: return 0
    # konservativer Fallback
    return 0

def load_aki(path: str) -> pd.DataFrame:
    try:
        aki = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        aki = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if aki.shape[1] == 1 and ";" in aki.columns[0]:
        aki = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    aki.columns = aki.columns.str.strip()

    # mögliche Tippfehler/Varianten harmonisieren
    rename_map = {
        "Duartion": "Duration",
        "Start": "AKI_Start",
        "End": "AKI_End",
        "Decision": "Decision",
    }
    aki = aki.rename(columns=rename_map)
    # Pflichtfelder
    req = {"PMID", "AKI_Start", "AKI_End", "Decision"}
    missing = [c for c in req if c not in aki.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in AKI-Datei: {missing}. Vorhanden: {aki.columns.tolist()}")

    aki["AKI_Start"] = pd.to_datetime(aki["AKI_Start"], errors="coerce")
    aki["AKI_End"]   = pd.to_datetime(aki["AKI_End"], errors="coerce")
    aki["aki_stage"] = aki["Decision"].apply(map_aki_stage).astype("int8")
    # Nur Zeilen mit Patient & irgendeinem Datum behalten
    aki = aki.dropna(subset=["PMID"]).copy()
    return aki[["PMID", "AKI_Start", "AKI_End", "aki_stage"]].sort_values(["PMID","AKI_Start"])

# ---------- Index-OP & OP-Aggregationen ----------
ops_sorted = ops.sort_values(["PMID", "Surgery_Start"]).copy()

idxop = (ops_sorted.groupby("PMID").first()
         [["Surgery_Start","Surgery_End","op_hours"]]
         .rename(columns={"Surgery_Start":"first_op_date",
                          "Surgery_End":"first_op_end",
                          "op_hours":"first_op_hours"})
         .reset_index())

agg = (ops_sorted.groupby("PMID").agg(
            n_ops=("Procedure_ID","count"),
            total_op_hours=("op_hours","sum"),
            mean_op_hours=("op_hours","mean"),
            max_op_hours=("op_hours","max"),
        ).reset_index())

print("Index-OP Datensätze:", len(idxop), "| Aggregat-Zeilen:", len(agg))

# ---------- Patienten & Alter mergen ----------
pat = load_patients("Patient Master Data.csv")  # Pfad anpassen falls nötig
base = pat.merge(idxop, on="PMID", how="inner")
base["age_years_at_first_op"] = (base["first_op_date"] - base["DateOfBirth"]).dt.days / 365.2425

# ---------- AKI zeitbasiert nach erster OP verknüpfen ----------
aki = load_aki("AKI Label.csv")  # Pfad anpassen
match = []
# schneller Zugriff per GroupBy
aki_g = {pid: g for pid, g in aki.groupby("PMID")}
for _, row in base.iterrows():
    pid = row["PMID"]
    op_end = row["first_op_end"]
    # AKI-Ereignisse dieses Patienten
    g = aki_g.get(pid)
    if g is None or g["AKI_Start"].isna().all():
        match.append({"PMID": pid, "aki_start_date": pd.NaT, "aki_stage": 0})
        continue
    g = g.dropna(subset=["AKI_Start"]).sort_values("AKI_Start")
    g = g[g["AKI_Start"] >= op_end]
    if len(g) == 0:
        match.append({"PMID": pid, "aki_start_date": pd.NaT, "aki_stage": 0})
    else:
        first = g.iloc[0]
        match.append({"PMID": pid,
                      "aki_start_date": first["AKI_Start"],
                      "aki_stage": int(first["aki_stage"])})
aki_match = pd.DataFrame(match)

# ---------- Zusammenführen & Fenster 0–7 Tage ----------
out = (base.merge(agg, on="PMID", how="left")
           .merge(aki_match, on="PMID", how="left"))

out["days_to_AKI"] = (out["aki_start_date"] - out["first_op_end"]).dt.total_seconds() / 86400.0
out["AKI_any_0_7"] = np.where(
    (out["days_to_AKI"] >= 0) & (out["days_to_AKI"] <= 7) & (out["aki_stage"] > 0),
    1, 0
).astype("int8")

out["AKI1_0_7"] = ((out["AKI_any_0_7"]==1) & (out["aki_stage"]==1)).astype("int8")
out["AKI2_0_7"] = ((out["AKI_any_0_7"]==1) & (out["aki_stage"]==2)).astype("int8")
out["AKI3_0_7"] = ((out["AKI_any_0_7"]==1) & (out["aki_stage"]==3)).astype("int8")
out["highest_AKI_stage_0_7"] = np.select(
    [out["AKI3_0_7"]==1, out["AKI2_0_7"]==1, out["AKI1_0_7"]==1],
    [3, 2, 1],
    default=0
).astype("int8")

# ---------- Finale Spalten & Export ----------
cols = [
    "PMID", "Sex_norm",
    "first_op_date", "first_op_end", "first_op_hours",
    "age_years_at_first_op",
    "n_ops", "total_op_hours", "mean_op_hours", "max_op_hours",
    "aki_start_date", "days_to_AKI",
    "AKI_any_0_7", "AKI1_0_7", "AKI2_0_7", "AKI3_0_7", "highest_AKI_stage_0_7",
]
# nur vorhandene Spalten nehmen (falls Sex_norm anders heißt)
cols = [c for c in cols if c in out.columns]
out = out[cols].sort_values("PMID").reset_index(drop=True)

out.to_csv("analytic_patient_summary_v2.csv", index=False)
print("✅ geschrieben:", "analytic_patient_summary_v2.csv")
print("AKI 0–7 Tage (any):", int(out["AKI_any_0_7"].sum()),
      "/", len(out), f"({out['AKI_any_0_7'].mean():.1%})")
print(out["highest_AKI_stage_0_7"].value_counts())
out["days_to_AKI"].hist(bins=14, range=(0,7))
print(out.groupby("AKI_any_0_7")["first_op_hours"].describe())
print("Gesamtzahl Patienten im Datensatz:", out["PMID"].nunique())
print("Mit AKI 0–7 Tage:", out["AKI_any_0_7"].sum())
print("Ohne AKI 0–7 Tage:", (out["AKI_any_0_7"]==0).sum())
print("Summe:", out.shape[0])
missing_ids = set(pat["PMID"]) - set(out["PMID"])
print("Patienten die fehlen:", len(missing_ids))
print(list(missing_ids)[:20])  # nur ein paar anzeigen
import pandas as pd

# 1) Laden
pat = pd.read_csv("analytic_patient_summary_v2.csv")
ops = pd.read_csv("ops_with_crea_cysc_vis_features_with_AKI.csv")

# 2) Spaltennamen säubern
pat.columns = pat.columns.str.strip()
ops.columns = ops.columns.str.strip()

# 3) Merge (Patientendaten an jede OP hängen)
merged = ops.merge(
    pat[["PMID", "Sex_norm", "age_years_at_first_op", "n_ops", "highest_AKI_stage_0_7"]],
    on="PMID",
    how="left"
)

# 4) Dauer in Stunden sicherstellen
merged["duration_hours"] = (pd.to_datetime(merged["Surgery_End"]) -
                            pd.to_datetime(merged["Surgery_Start"])).dt.total_seconds() / 3600.0

# 5) Export
merged.to_csv("ops_with_patient_features.csv", index=False)
print("✅ geschrieben: ops_with_patient_features.csv")
#Plausibilität Check
import pandas as pd
import numpy as np

df = pd.read_csv("ops_with_patient_features.csv")
df.columns = df.columns.str.strip()

# Pflichtspalten prüfen
required = [
    "PMID","SMID","Procedure_ID","Surgery_Start","Surgery_End",
    "AKI_linked_0_7","duration_hours","Sex_norm","age_years_at_first_op","n_ops"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}")

# Datumsfelder parsen (falls noch nicht)
for c in ["Surgery_Start","Surgery_End","AKI_Start"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Dauer ggf. neu berechnen (Sicherheitsnetz)
if df["duration_hours"].isna().any():
    dur = (df["Surgery_End"] - df["Surgery_Start"]).dt.total_seconds() / 3600.0
    df["duration_hours"] = df["duration_hours"].fillna(dur)

# Realistische Dauergrenzen (optional, nicht droppen – nur Flag für Prüfung)
df["duration_flag"] = np.where((df["duration_hours"]<0.1)|(df["duration_hours"]>24), 1, 0)
print("Unplausible Dauern:", int(df["duration_flag"].sum()), "von", len(df))

# Zielvariable prüfen
print("AKI 0–7 (OP-Ebene):", df["AKI_linked_0_7"].value_counts(dropna=False))

# Fehlende Werte grob
na_counts = df.isna().sum().sort_values(ascending=False)
print("Top NA-Spalten:\n", na_counts.head(10))



