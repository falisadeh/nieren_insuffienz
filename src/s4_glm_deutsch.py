import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
CSV = f"{BASE}/Daten/S4_glm_cluster_or.csv"
OUT = f"{BASE}/Diagramme/Forest_OR_clean.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
#'/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme/S4_glm_cluster_or.csv'
#'/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme/S4_forest_or.png'
#'/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/S4_glm_cluster_or.csv'
df = pd.read_csv(CSV)
df = df[df["Term"] != "const"].copy()  # Intercept raus
df = df[["Term", "OR", "CI_low", "CI_high"]].dropna()
df = df[(df["OR"] > 0) & (df["CI_low"] > 0) & (df["CI_high"] > 0)]

label_map = {
    "Sex_norm_m": "Männlich",
    "age_years_at_op": "Alter bei OP",
    "duration_hours": "OP-Dauer (Stunden)",
    "is_reop": "Re-Operation",
}
df["Label"] = [label_map.get(t, t.replace("_", " ")) for t in df["Term"]]

y = np.arange(len(df))[::-1]
xerr = np.vstack([df["OR"] - df["CI_low"], df["CI_high"] - df["OR"]])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(
    df["OR"],
    y,
    xerr=xerr,
    fmt="o",
    color="tab:blue",
    ecolor="tab:blue",
    capsize=4,
    lw=2,
)
ax.axvline(1.0, color="tab:blue", ls="--", alpha=0.6)

ax.set_yticks(y)
ax.set_yticklabels(df["Label"])
ax.set_xscale("log")
ax.set_xlabel("Odds Ratio (log-Skala)")
ax.set_title("S4 – Odds Ratios (95%-KI)")
ax.set_xlim(min(df["CI_low"]) * 0.9, max(df["CI_high"]) * 1.1)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print("Gespeichert:", OUT)
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

BASE = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
CSV = f"{BASE}/S4_glm_cluster_or.csv"  # ggf. anpassen
OUT = f"{BASE}/Diagramme/Forest_OR_clean_with_ref.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# --- Daten laden ---
df = pd.read_csv(CSV)
df = df[df["Term"] != "const"].copy()  # Intercept raus
df = df[["Term", "OR", "CI_low", "CI_high"]].dropna()
df = df[(df["OR"] > 0) & (df["CI_low"] > 0) & (df["CI_high"] > 0)]

# Sprechende Labels
label_map = {
    "Sex_norm_m": "Männlich",
    "age_years_at_op": "Alter bei OP",
    "duration_hours": "OP-Dauer (Stunden)",
    "is_reop": "Re-Operation",
}
df["Label"] = [label_map.get(t, t.replace("_", " ")) for t in df["Term"]]

# --- Referenzzeile für weiblich einfügen (nur wenn 'Männlich' vorhanden) ---
if "Sex_norm_m" in df["Term"].values:
    ref_row = pd.DataFrame(
        [
            {
                "Term": "Sex_norm_f_ref",
                "OR": 1.0,
                "CI_low": 1.0,
                "CI_high": 1.0,
                "Label": "Weiblich (Referenz)",
            }
        ]
    )
    # Reihenfolge: Weiblich (Ref.) oberhalb von Männlich anzeigen
    i_male = df.index[df["Term"] == "Sex_norm_m"][0]
    df = pd.concat([df.iloc[:i_male], ref_row, df.iloc[i_male:]], ignore_index=True)

# --- Plot vorbereiten ---
y = np.arange(len(df))[::-1]  # oben = erste Zeile
xerr = np.vstack([df["OR"] - df["CI_low"], df["CI_high"] - df["OR"]])

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.errorbar(
    df["OR"],
    y,
    xerr=xerr,
    fmt="o",
    color="tab:blue",
    ecolor="tab:blue",
    capsize=4,
    lw=2,
)

# Referenzlinie OR=1
ax.axvline(1.0, color="tab:blue", ls="--", alpha=0.6)

# Labels & Achsen
ax.set_yticks(y)
ax.set_yticklabels(df["Label"])
ax.set_xscale("log")
ax.set_xlabel("Odds Ratio (log-Skala)")
ax.set_title("S4 – Odds Ratios (95%-KI)")

# Grenzen schön setzen
xmin = min(df["CI_low"].min(), 0.5)
xmax = max(df["CI_high"].max(), 2.0)
ax.set_xlim(xmin * 0.9, xmax * 1.1)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print("Gespeichert:", OUT)
