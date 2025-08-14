# 00_run.py — einfacher Orchestrator ohne Umzug/Imports
import subprocess, sys

SCRIPTS = [
    "04_time_to_AKI.py",
    #"06_S4_run_glm_cv.py",
    "05_table1_stats_ehrapy.py",
    "07_age_trends_splines.py",   # wenn vorhanden
    "09_sex_effect_adjusted.py"  # wenn vorhanden
]

for s in SCRIPTS:
    print(f"\n>>> Running: {s}")
    subprocess.run([sys.executable, s], check=True)
print("\nFertig. Ergebnisse im Ordner erscheinen (z. B. PNG/CSV).")