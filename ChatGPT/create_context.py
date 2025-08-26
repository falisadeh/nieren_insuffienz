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
    nutze_alten_code = False

    # --- CSV-Kopfzeilen aus dem Unterordner "Original Daten" einsammeln ---
    original_daten_pfad = os.path.join(ziel_verzeichnis, "Original Daten")
    dateikoepfe = ""
    if os.path.isdir(original_daten_pfad):
        for datei in os.listdir(original_daten_pfad):
            dateipfad = os.path.join(original_daten_pfad, datei)
            if os.path.isfile(dateipfad) and dateipfad.lower().endswith(".csv"):
                daten_kopf = ""
                try:
                    with open(dateipfad, "r", errors="ignore") as f:
                        for _ in range(5):
                            zeile = f.readline()
                            if not zeile:
                                break
                            daten_kopf += zeile
                except Exception as e:
                    print(
                        f"Warnung: Konnte Kopf von {dateipfad} nicht lesen: {e}",
                        file=sys.stderr,
                    )
                    continue
                dateikoepfe += f"\n--- DATEI: {dateipfad} ---\n\n{daten_kopf}\n"
    else:
        print(f"Hinweis: Unterordner fehlt: {original_daten_pfad}", file=sys.stderr)

    # Definiere Dateien oder Ordner, die ausgeschlossen werden sollen
    # 'kontext.txt' wird automatisch ausgeschlossen, um eine Selbsteinbindung
    # zu verhindern.
    # Füge andere Dateien/Ordner hinzu, wie 'venv', '__pycache__', '.git'
    if nutze_alten_code:
        einzuschliessende_elemente = None  # Alle .py Dateien einschließen
    else:
        # MAMA CODE HINZUFUEGEN
        einzuschliessende_elemente = [
            # "daten.py",
        ]  # Nur diese Dateien einschließen

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

    # MAMA AENDERN
    kontext_kopf = textwrap.dedent(
        """
        Ich soll so viel wie möglich die Funktionen von ehrapy benutzen: 
        https://github.com/theislab/ehrapy. Tutorials üeber ehrapy findest du hier:
        https://github.com/theislab/ehrapy-tutorials. Mache dich damit vertraut.

        Meine Bachelorarbeit handelt von Wie effektiv unterstützt das Framework ehrapy die Identifikation
        von Risikofaktoren für akutes Nierenversagen bei Kindern 
        nach Herzoperationen anhand eines angereicherten Routinedatensatzes?
        {dateiköpfe}
        1- LOINC:Medizinische Messwerte lassen sich theoretisch standardisieren. Um das zu dokumentieren, wird LOINC (https://loinc.org/) verwendet.
        2- QuantitativeValue ist der tatsächliche Messwert für diesen Laborwert. Die Specimen ID identifiziert die Probe aus der der Messwert genommen wurde.
        alle Werte, die die selbe SpecimenID haben, kommen bspw. aus derselben Blutprobe. 
        Wenn einem Patienten Proben entnommen werden, bspw. Blut mit einer eigenen SpecimenID, dann Speichel mit einer anderen SpecimenID und dann noch mal ein venöses Blut mit einer dritten SpecimenID, 
        können die aber zeitgleich in einem sogenannten Panel untersucht werden. Um dieses Panel zu identifizieren, gibt es die LaborPanelID.
        3- CollectionTimestamp: Das entspricht dem Zeitpunkt der Entnahme.Laborproben zeitlich VOR und NACH der HLM OP.
        Diese Messwerte könnte man als „Baseline“ also als Ausgangswert/Referenzwert vor der OP sehen
        4- Vis berechnet Vasoactive Inotropic Score
        5- PMID: Das ist der PatientMasterIndex
        6- Duration: Das ist die Dauer der AKI Episode in Millisekunden
        7- Start: Das ist der Startzeitpunkt der jeweiligen AKI Episode
        8- End: Das Ende der jeweiligen AKI Episode
        9- Decision: Das ist die jeweilige Information, welches Stadium / Grading des AKI hier gelabelt wurde. Also: Von Startzeitpunkt bis Endzeitpunkt besteht das AKI der Decision-Spalte. 
        Außerhalb dieses Zeitraum besteht kein AKI. Patienten, die in dieser Tabelle nicht aufgeführt sind, haben KEIN AKI. Also, wenn es einen Patienten gibt, der zwar in den anderen Datentabellen enthalten ist, aber hier nicht in dieser Datei, dann muss das so sein, weil dieser Patient kein AKI hat.
        10- HLM Operationen.csv: a) PMID: PatientMasterIndx. b) SMID: StayMasterIndex. Die SMID ist immer eine 9-stellige Zahl: Die ersten 6 Stellen ist die PMID des Patienten + eine fortlaufende 3-stellige Zahl die, die Aufenthalte aufzählt. Also Wenn Patient 300001 den ersten Aufenthalt hat, wird die PMID 300001 um 001 ergänzt = 300001001. Der zweite Aufenthalt wäre dann PMID + 002: 300001002 undso weiter. Das heißt, wenn du mal eine SMID findest, 
        die auf 009 endet, weißt du, dass das der 9. Aufenthalt des Patienten im Datensatz ist. (Aber Achtung, nicht alle diese Fälle enthalten eine HLM-OP).
        Ein Patient hat mindestens 1 Aufenthalt (=SMID), kann aber theoretisch beliebig viele Aufenthalte / SMIDs haben. Wenn ein Patient mehrere OPs hat, kann man über die SMID die jeweiligen Daten der anderen Tabellen später dem korrekten Fall zuordnen.
        11- Procedure_ID: Das ist eine eindeutige ID für eine HLM-Operation in der Kohorte. Damit können weitere Informationen zB aus der Datei Procedure Supplement.csv der Operation zugeordnet werden. Grundsätzlich könnte man diese Information auch über die PMID oder SMID zuordnen. Aber es könnte ja sein, dass ein Patient innerhalb eines Aufenthaltes zwei oder mehrere HLM-OPs hat, dann kann man das einfacher über die Procedure_ID zuordnen.
        12- Start of surgery: Startzeitpunkt der OP. 
        13- End of surgery: Endzeitpunkt der OP 
        14- Tx?: Das ist eine Information, ob die Operation eine Herz- oder Lungentransplantation war. Wenn „Tx“, dann war die Operation eine Transplantation, wenn „NULL“ nicht. Ich denke diese Spalte ist erst mal nicht super wichtig für dich, aber wenn interessant kann man berichten wie viele Transplantationen enthalten sind.
        15- Sex: Das Geschlecht des Patienten: f=female, m=male
        16- DateofBirth: Geburtsdatum des Patienten
        17- DateofDie: Sterbedatum des Patienten, wenn bekannt. Wenn NULL, gehen wir davon aus, dass der Patient noch lebt bzw. nicht im Krankenhaus verstorben ist.
        18- Procedure Supplement.csv: Weitere zeitliche Informationen zu den OPs. Ich weiß, hier sind einige seltsame Inhalte in der Spalte TimestampName wie „AZPFLVORV“. Solche Dinge sagen den Experten zwar etwas, für uns sind die erst mal egal. J Für die ersten Schritte ist diese Datei erst mal nicht super wichtig – ich würde diese Datei als letztes betrachten.
        19- TimestampName: Die Information was zu dem jeweiligen Zeitpunkt passiert ist. Also „Patient in OP aufgenommen“ oder „Start Anesthesia“
        20- Timestamp: Zeitpunkt zu dem das jeweils passiert ist.
        21- Procedure_ID: Das ist eine eindeutige ID für eine HLM-Operation in der Kohor

        
        {anweisung}
        """
    )
    anweisung = "Erstelle eine Datei mit allen Daten in ehrapy und imputiere fehlende Werte. Speicher im Ordner Daten als h5ad Datei."
    erinnerung = textwrap.dedent(
        """
        Bitte antworte nur auf Deutsch.
        Antworte nur auf Basis der Informationen in diesem Kontext.
        """
    )

    erstelle_kontext_datei(
        quell_verzeichnis=ziel_verzeichnis,
        kontext_kopf=kontext_kopf.format(dateiköpfe=dateikoepfe, anweisung=anweisung),
        erinnerung=erinnerung,
        auszuschliessende_elemente=auszuschliessende_elemente,
        einzuschliessende_elemente=einzuschliessende_elemente,
    )
