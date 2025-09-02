# 6) Aggregiert: genau 1 Balken pro pädiatrischer Altersgruppe
# (ersetzt das bisherige Histogramm mit vielen Bins)

# saubere Reihenfolge sicherstellen
order_labels = [
    "Neonates (0–28 T.)",
    "Infants (1–12 Mon.)",
    "Toddlers (1–3 J.)",
    "Preschool (3–5 J.)",
    "School-age (6–12 J.)",
    "Adolescents (13–18 J.)",
]

counts = (
    adata.obs["age_group_pediatric"].value_counts().reindex(order_labels, fill_value=0)
)

# --- Absolutzahlen (ein Balken pro Gruppe) ---
plt.figure(figsize=(8, 5))
bars = plt.bar(
    counts.index.astype(str),
    counts.values,
    color=["#87CEEB", "#FFA500", "#32CD32", "#FF69B4", "#9370DB", "#A0522D"],
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Anzahl Operationen")
plt.title("Altersverteilung pro pädiatrischer Gruppe (aggregiert)")

# Werte über die Balken schreiben
for b in bars:
    y = int(b.get_height())
    plt.text(
        b.get_x() + b.get_width() / 2,
        y + max(counts.values) * 0.01 + 0.2,
        f"{y}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "Age_groups_bar_counts.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# --- Optional: Prozentanteile als Balken ---
total = counts.sum()
pct = (counts / total * 100).round(1)

plt.figure(figsize=(8, 5))
bars = plt.bar(pct.index.astype(str), pct.values)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Anteil (%)")
plt.title("Anteil pro pädiatrischer Gruppe (in %)")
for b, v in zip(bars, pct.values):
    plt.text(
        b.get_x() + b.get_width() / 2,
        v + 0.5,
        f"{v}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylim(0, max(20, pct.max() + 3))  # etwas Luft nach oben
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "Age_groups_bar_percent.png"), dpi=300, bbox_inches="tight"
)
plt.close()
