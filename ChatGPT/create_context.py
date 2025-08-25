import os
import sys
import textwrap


def erstelle_kontext_datei(
    *,
    kontext_kopf: str,
    erinnerung: str,
    quell_verzeichnis: str,
    ausgabe_dateiname: str = "ChatGPT/kontext.txt",
    einzuschliessende_elemente: list[str] | None = None,
    auszuschliessende_elemente: list[str] | None = None,
) -> None:
    """
    Kopiert den Inhalt aller Dateien in einem Verzeichnis (und seinen
    Unterverzeichnissen) in eine einzige Ausgabedatei, wobei bestimmte
    Dateien oder Ordner ausgeschlossen werden.

    Args:
        quell_verzeichnis (str): Der Pfad zu dem zu durchlaufenden Verzeichnis.
        ausgabe_dateiname (str): Der Name der Datei, in die der Inhalt
                                 geschrieben werden soll.
                                 Standardmäßig 'kontext.txt'.
        auszuschliessende_elemente (list[str]): Eine Liste von Dateinamen oder
                                                Ordnernamen (Basisnamen), die
                                                ausgeschlossen werden sollen.
                                                Standardmäßig None.
    """
    if auszuschliessende_elemente is None:
        auszuschliessende_elemente = []

    # Stelle sicher, dass das Quellverzeichnis existiert
    if not os.path.isdir(quell_verzeichnis):
        print(
            f"Fehler: Quellverzeichnis '{quell_verzeichnis}' existiert nicht.",
            file=sys.stderr,
        )
        return

    # Füge den Ausgabedateinamen selbst zur Ausschlussliste hinzu, um
    # eine Selbsteinbindung zu verhindern
    if ausgabe_dateiname not in auszuschliessende_elemente:
        auszuschliessende_elemente.append(ausgabe_dateiname)

    print(
        f"Starte die Erstellung von '{ausgabe_dateiname}' aus '{quell_verzeichnis}'..."
    )
    print(f"Ausgeschlossene Elemente: {auszuschliessende_elemente}")
    if einzuschliessende_elemente is not None:
        print(
            f"Es werden nur die folgenden Elemente eingeschlossen: {einzuschliessende_elemente}"
        )

    try:
        with open(
            ausgabe_dateiname, "w", encoding="utf-8", errors="ignore"
        ) as ausgabe_datei:
            ausgabe_datei.write(kontext_kopf.strip() + "\n\n")
            for root, dirs, files in os.walk(quell_verzeichnis):
                # Filtert ausgeschlossene Verzeichnisse heraus
                dirs[:] = [d for d in dirs if d not in auszuschliessende_elemente]

                for dateiname in files:
                    ist_enthalten = (
                        not (einzuschliessende_elemente is None)
                        and dateiname in einzuschliessende_elemente
                    )
                    if (
                        ist_enthalten
                        and dateiname not in auszuschliessende_elemente
                        and dateiname.endswith(".py")
                    ):
                        dateipfad = os.path.join(root, dateiname)
                        # Überspringe, wenn es keine reguläre Datei ist
                        if not os.path.isfile(dateipfad):
                            continue

                        try:
                            # Füge einen Trennstrich und den Dateipfad als Header hinzu
                            ausgabe_datei.write(f"\n--- DATEI: {dateipfad} ---\n\n")
                            with open(
                                dateipfad, "r", encoding="utf-8", errors="ignore"
                            ) as eingabe_datei:
                                ausgabe_datei.write(eingabe_datei.read())
                            ausgabe_datei.write(
                                "\n"
                            )  # Füge nach dem Dateiinhalt eine neue Zeile hinzu
                            print(f"  - Hinzugefügt: {dateipfad}")
                        except Exception as e:
                            print(
                                f"  - Übersprungen (Fehler beim Lesen): {dateipfad} - {e}",
                                file=sys.stderr,
                            )
            ausgabe_datei.write(erinnerung.strip() + "\n\n")
            print(f"\nErfolgreich '{ausgabe_dateiname}' erstellt.")
    except Exception as e:
        print(
            f"Fehler beim Erstellen der Ausgabedatei '{ausgabe_dateiname}': {e}",
            file=sys.stderr,
        )


# --- Beispielverwendung ---
if __name__ == "__main__":
    # Definiere das Verzeichnis, das verarbeitet werden soll
    # (z. B. das aktuelle Verzeichnis).
    # Du kannst '.' durch einen anderen Pfad wie 'mein_projekt_ordner' ändern.
    ziel_verzeichnis = (
        "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
    )

    kontext_kopf = textwrap.dedent(
        """
        Ich soll so viel wie möglich die Funktionen von ehrapy benutzen: 
        https://github.com/theislab/ehrapy. Tutorials üeber ehrapy findest du hier:
        https://github.com/theislab/ehrapy-tutorials. Mache dich damit vertraut.

        Meine Bachelorarbeit handelt von Niereninsuffiezienz in Kindern.
        Meine Daten findest du hier:
        {dateiköpfe}

        {anweisung}
        """
    )
    anweisung = "Nutze lineare Regression."
    erinnerung = textwrap.dedent(
        """
        Bitte antworte nur auf Deutsch.
        Antworte nur auf Basis der Informationen in diesem Kontext.
        """
    )
    nutze_alten_code = False

    dateikoepfe = ""
    for datei in os.listdir(ziel_verzeichnis + "/Orginal Daten"):
        dateipfad = os.path.join(ziel_verzeichnis + "/Orginal Daten", datei)
        if dateipfad.endswith(".csv"):
            # Lese die ersten 5 Zeilen der CSV-Datei
            daten_kopf = ""
            with open(
                dateipfad,
                "r",
                errors="ignore",
            ) as eingabe_datei:
                for _ in range(5):
                    zeile = eingabe_datei.readline()
                    if not zeile:
                        break
                    daten_kopf += zeile

            dateikoepfe += f"\n--- DATEI: {dateipfad} ---\n\n"
            dateikoepfe += daten_kopf + "\n"

    # Definiere Dateien oder Ordner, die ausgeschlossen werden sollen
    # 'kontext.txt' wird automatisch ausgeschlossen, um eine Selbsteinbindung
    # zu verhindern.
    # Füge andere Dateien/Ordner hinzu, wie 'venv', '__pycache__', '.git'
    if nutze_alten_code:
        einzuschliessende_elemente = None
        auszuschliessende_elemente = [
            "__pycache__",
            ".git",
            ".DS_Store",
            ".gitignore",
            "ChatGPT",
            "Archiv",
            "Audit",
            "Daten",
            "Diagramme",
            "h5ad",
            "Word\ und\ Text",
            "Original Daten",
        ]
    else:
        einzuschliessende_elemente = []
        auszuschliessende_elemente = []

    erstelle_kontext_datei(
        quell_verzeichnis=ziel_verzeichnis,
        kontext_kopf=kontext_kopf.format(dateiköpfe=dateikoepfe, anweisung=anweisung),
        erinnerung=erinnerung,
        auszuschliessende_elemente=auszuschliessende_elemente,
        einzuschliessende_elemente=einzuschliessende_elemente,
    )
