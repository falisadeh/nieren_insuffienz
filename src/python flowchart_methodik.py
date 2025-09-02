# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------
# Ausgaben (dein iCloud-Ordner)
# ----------------------------------
OUT_DIR = "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer/Diagramme"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "ehrapy_method_flowchart.png")
OUT_PDF = os.path.join(OUT_DIR, "ehrapy_method_flowchart.pdf")

# ----------------------------------
# Schritte (Flow)
# ----------------------------------
steps = [
    "Explorative Datenanalyse mit ehrapy",
    "Deskriptive Beschreibung anhand AKI-Endpunkt",
    "Identifikation möglicher Risikofaktoren\n(statistisch / ML)",
    "Bestätigung der Faktoren",
    "Evaluation von ehrapy für Routinedaten",
]

# Farben (gelb → orange)
box_colors = ["#FFE08A", "#FFCD70", "#FFB957", "#FFA53E", "#FF9026"]

# ----------------------------------
# Figur & Achsen (normierte Koordinaten 0..1)
# ----------------------------------
fig, ax = plt.subplots(figsize=(7, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Layout-Parameter
x_center = 0.5
box_width = 0.78
box_height = 0.12  # Box-Höhe
top_margin = 0.93
bottom_margin = 0.08  # -> genug Platz unten
offset = 0.015  # Pfeil-Überstand
n = len(steps)

# Gleichmäßige Y-Positionen zwischen oben/unten
ys = np.linspace(top_margin, bottom_margin, n)

# Boxen & Texte
for i, (step, y) in enumerate(zip(steps, ys)):
    rect = FancyBboxPatch(
        (x_center - box_width / 2, y - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#555555",
        facecolor=box_colors[i % len(box_colors)],
    )
    ax.add_patch(rect)
    ax.text(x_center, y, step, ha="center", va="center", fontsize=12)

# Pfeile zwischen den Boxen
for y1, y2 in zip(ys[:-1], ys[1:]):
    arrow = FancyArrowPatch(
        (x_center, y1 - box_height / 2 - offset),
        (x_center, y2 + box_height / 2 + offset),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.2,
        color="#555555",
    )
    ax.add_patch(arrow)

# Titel
ax.set_title("Methodik – ehrapy-Workflow (AKI-Analyse)", fontsize=14, pad=20)

# Speichern
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print("Gespeichert:", OUT_PNG, "und", OUT_PDF)
