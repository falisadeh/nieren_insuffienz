#%%
import pandas as pd

# CSV einlesen (mit Semikolon als Trennzeichen)
df_ops = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/HLM Operationen.csv",
    sep=";"
)

# Spaltennamen bereinigen
df_ops.columns = df_ops.columns.str.strip()

# Gruppieren nach Patient → Anzahl der OPs und Liste der OP-Typen (Procedure_IDs)
op_anzahl = df_ops.groupby("PMID").agg(
    Anzahl_OPs=("SMID", "count"),
    Procedure_IDs=("Procedure_ID", lambda x: ', '.join(x.dropna().astype(str)))
).reset_index()

# CSV exportieren (ohne Index-Spalte)
export_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/op_anzahl.csv"
op_anzahl.to_csv(export_path, index=False)

print("Export erfolgreich gespeichert unter:")
print(export_path)

import matplotlib.pyplot as plt

# Histogramm: Wie viele Patienten hatten wie viele OPs?
plt.figure(figsize=(8, 5))
plt.hist(op_anzahl["Anzahl_OPs"], bins=range(1, op_anzahl["Anzahl_OPs"].max() + 2), edgecolor='black')
plt.title("Histogramm der OP-Anzahl pro Patient")
plt.xlabel("Anzahl der Operationen")
plt.ylabel("Anzahl der Patienten")
plt.xticks(range(1, op_anzahl["Anzahl_OPs"].max() + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Bild speichern
hist_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/hist_op_anzahl.png"
plt.savefig(hist_path)
plt.close()

print(" Histogramm gespeichert unter:")
print(hist_path)

# Anzahl OPs je Patient als Häufigkeitstabelle mit Kreisdiagra
#op_counts = op_anzahl["Anzahl_OPs"].value_counts().sort_index()

# Kreisdiagramm erzeugen
#plt.figure(figsize=(6, 6))
#plt.pie(op_counts, labels=[f"{i} OP" for i in op_counts.index],
       # autopct='%1.1f%%', startangle=90, counterclock=False)
#plt.title("Verteilung der OP-Anzahl pro Patient (Kreisdiagramm)")
#plt.tight_layout()

# Speichern
#pie_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/pie_op_anzahl.png"
#plt.savefig(pie_path)
#plt.close()

#print(" Kreisdiagramm gespeichert unter:")
#print(pie_path)

import pandas as pd
import matplotlib.pyplot as plt

# Gruppiere OP-Anzahl: alles über 2 als "≥3 OPs"
def gruppiere_op_anzahl(n):
    if n == 1:
        return "1 OP"
    elif n == 2:
        return "2 OP"
    else:
        return "≥3 OPs"

# Neue Gruppenspalte erstellen
op_anzahl["OP_Kategorie"] = op_anzahl["Anzahl_OPs"].apply(gruppiere_op_anzahl)

# Häufigkeit zählen
op_kat_counts = op_anzahl["OP_Kategorie"].value_counts().sort_index()
labels = op_kat_counts.index
sizes = op_kat_counts.values

# Plot Kreisdiagramm
plt.figure(figsize=(7, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    pctdistance=0.8
)
plt.title("OP-Anzahl pro Patient (zusammengefasst)")
plt.tight_layout()

# Speichern
pie_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/pie_op_anzahl_klar.png"
plt.savefig(pie_path)
plt.close()

print("Zusammengefasstes Kreisdiagramm gespeichert unter:")
print(pie_path)

# Zähle, wie oft jeder Patient in der OP-Tabelle vorkommt
op_anzahl_raw = df_ops.groupby("PMID").size().reset_index(name="OP_Anzahl")

# Filter: Nur Patienten mit mehr als 1 OP
mehrfach_ops = op_anzahl_raw[op_anzahl_raw["OP_Anzahl"] > 1]

# Optional: Sortieren
mehrfach_ops = mehrfach_ops.sort_values("OP_Anzahl", ascending=False)

# Ausgabe
print(mehrfach_ops)
export_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/patienten_mehrfach_op.csv"
mehrfach_ops.to_csv(export_path, index=False)
print(" Exportiert nach:", export_path)
# Test
pmid_test = 300974  # Beispiel mit 5 OPs
# Zeige alle OP-Zeilen dieses Patienten
df_ops[df_ops["PMID"] == pmid_test]
op_häufigkeit = mehrfach_ops["OP_Anzahl"].value_counts().sort_index()
print(op_häufigkeit)
import matplotlib.pyplot as plt

# Verteilung (hast du schon berechnet):
# op_häufigkeit = mehrfach_ops["OP_Anzahl"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
op_häufigkeit.plot(kind="bar", color="coral", edgecolor="black")
plt.title("Verteilung der HLM-Operationen bei mehrfach operierten Patienten")
plt.xlabel("Anzahl der HLM-Operationen pro Patient")
plt.ylabel("Anzahl der Patienten")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Speichern
plot_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/hist_mehrfach_ops_klar.png"
plt.savefig(plot_path)
plt.close()

print(" Diagramm gespeichert unter:", plot_path)

# Lade die Patient Master Data.csv (mit DateOfDie)
df_pat = pd.read_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Patient Master Data.csv",
    sep=";"  # Achtung: auch hier wahrscheinlich Semikolon!
)
df_pat.columns = df_pat.columns.str.strip()
# Markiere „verstorben“ (wenn DateOfDie nicht leer ist)
df_pat["verstorben"] = df_pat["DateOfDie"].notna()
# Verknüpfen: Patienten + OP-Anzahl
df_pat_op = pd.merge(df_pat, op_anzahl_raw, on="PMID", how="left")

# Falls Patienten keine OP hatten → OP_Anzahl auf 0 setzen
df_pat_op["OP_Anzahl"] = df_pat_op["OP_Anzahl"].fillna(0).astype(int)
# Analysieren: Sterblichkeit nach OP-Anzahl, Z. B. Gruppen bilden: 0–1 OP (keine oder nur 1 OP)≥2 OPs (mehrfach operiert)
df_pat_op["OP_Gruppe"] = df_pat_op["OP_Anzahl"].apply(lambda x: "≥2 OPs" if x >= 2 else "0–1 OP")

# Gruppenweise Sterblichkeitsrate berechnen
sterbequote = df_pat_op.groupby("OP_Gruppe")["verstorben"].mean().round(3) * 100
anzahl_pat = df_pat_op.groupby("OP_Gruppe")["verstorben"].count()

# Ausgabe kombinieren
vergleich = pd.DataFrame({
    "Anzahl Patienten": anzahl_pat,
    "Sterblichkeitsrate (%)": sterbequote
})

print(vergleich)
vergleich.reset_index().to_csv(
    "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/sterblichkeit_op_gruppen.csv",
    index=False
)
print("Tabelle gespeichert.")

import matplotlib.pyplot as plt

vergleich["Sterblichkeitsrate (%)"].plot(kind="bar", color="crimson", edgecolor="black")
plt.title("Sterblichkeitsrate nach Anzahl HLM-Operationen")
plt.xlabel("OP-Gruppe")
plt.ylabel("Sterblichkeitsrate (%)")
plt.xticks(rotation=0)
#plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Speichern
plot_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/sterblichkeit_nach_op_anzahl.png"
plt.savefig(plot_path)
plt.close()

print("Balkendiagramm gespeichert unter:", plot_path)
# % auch eingeben.
import matplotlib.pyplot as plt

# Neue Plot-Figur
fig, ax = plt.subplots(figsize=(6, 4))

# Balkendiagramm zeichnen
bars = ax.bar(
    vergleich.index,
    vergleich["Sterblichkeitsrate (%)"],
    color="crimson",
    edgecolor="black"
)

# Achsen & Titel
ax.set_title("Sterblichkeitsrate nach Anzahl HLM-Operationen")
ax.set_xlabel("OP-Gruppe")
ax.set_ylabel("Sterblichkeitsrate (%)")
ax.set_ylim(0, vergleich["Sterblichkeitsrate (%)"].max() + 2)  # mehr Platz oben
ax.tick_params(axis='x', labelrotation=0)

# 📌 Prozentwerte über Balken schreiben
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,  # etwas oberhalb
        f"{height:.1f} %",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()

# Speichern
plot_path = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/sterblichkeit_nach_op_anzahl_beschriftet.png"
plt.savefig(plot_path)
plt.close()

print("Diagramm mit Prozentwerten gespeichert unter:")
print(plot_path)
