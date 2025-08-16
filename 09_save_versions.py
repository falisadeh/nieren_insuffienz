#!/usr/bin/env python3
"""
Speichert die Versionsinformationen von ehrapy und seinen Abhängigkeiten.

Erzeugt automatisch:
- ehrapy_versions.txt (Textformat)
- ehrapy_versions.csv (Tabellenformat, gut für Anhang)

Robuste Fallback-Logik: Wenn `ep.print_versions()` wegen Paketkonflikten
(nicht kompatible scikit-learn/imbalanced-learn) fehlschlägt, werden die
Versionen der wichtigsten Pakete direkt abgefragt (ohne ehrapy zu importieren).

Nutzung:
  python 09_save_versions.py
"""

import sys
import io
import importlib
import platform
import pandas as pd

# --- Konfiguration der zu protokollierenden Pakete ---
CORE_PACKAGES = [
    "ehrapy",
    "anndata",
    "scanpy",
    "pandas",
    "numpy",
    "scikit-learn",
    "imbalanced-learn",
    "matplotlib",
]

TXT_PATH = "ehrapy_versions.txt"
CSV_PATH = "ehrapy_versions.csv"


def try_ep_print_versions_to_txt() -> bool:
    """Versucht, ep.print_versions() in eine Textdatei zu schreiben.
    Gibt True zurück, wenn erfolgreich, sonst False.
    """
    try:
        import ehrapy as ep  # kann aufgrund von imblearn/sklearn Konflikten fehlschlagen
        with open(TXT_PATH, "w") as f:
            # Neuere ehrapy-Versionen unterstützen 'file=' Argument
            try:
                ep.print_versions(file=f)
            except TypeError:
                # Fallback: stdout umleiten
                sys_stdout = sys.stdout
                sys.stdout = f
                ep.print_versions()
                sys.stdout = sys_stdout
        return True
    except Exception as e:
        # Schreib wenigstens die Python-Version in die TXT, damit ein Nachweis existiert
        with open(TXT_PATH, "w") as f:
            f.write("ep.print_versions() fehlgeschlagen. Grund: " + repr(e) + "")
            f.write(f"Python: {platform.python_version()}")
        return False


def collect_versions_fallback() -> pd.DataFrame:
    """Fragt Paketversionen direkt ab, ohne ehrapy zu importieren."""
    rows = [("Python", platform.python_version())]
    for pkg in CORE_PACKAGES:
        try:
            mod = importlib.import_module(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "unbekannt")
        except Exception:
            ver = "nicht installiert"
        rows.append((pkg, ver))
    df = pd.DataFrame(rows, columns=["Paket", "Version"])
    return df


def main():
    ok = try_ep_print_versions_to_txt()

    # Erzeuge immer auch eine CSV-Tabelle (entweder aus ep oder per Fallback)
    if ok:
        # Auch wenn ep geklappt hat, füllen wir die CSV tabellarisch per Fallback (saubere Tabelle)
        df = collect_versions_fallback()
    else:
        # ep fehlgeschlagen → CSV vollständig per Fallback
        df = collect_versions_fallback()

    df.to_csv(CSV_PATH, index=False)

    print(f"Versionen gespeichert in {TXT_PATH}")
    print(f"Versionen gespeichert in {CSV_PATH}")
    print("Gefundene Versionen:")
    print(df)
    



if __name__ == "__main__":
    main()

