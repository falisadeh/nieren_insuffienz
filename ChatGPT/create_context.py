import os
import sys
import textwrap


def create_context_file(
    *,
    context_head: str,
    reminder: str,
    source_directory: str,
    output_filename: str = "ChatGPT/context.txt",
    include_items: list[str] | None = None,
    exclude_items: list[str] | None = None,
) -> None:
    """
    Copies the content of all files in a directory (and its subdirectories)
    into a single output file, excluding specified files or folders.

    Args:
        source_directory (str): The path to the directory to traverse.
        output_filename (str): The name of the file to write the content to.
                               Defaults to 'context.txt'.
        exclude_items (list[str]): A list of file or folder names (base names)
                                   to exclude. Defaults to None.
    """
    if exclude_items is None:
        exclude_items = []

    # Ensure the source directory exists
    if not os.path.isdir(source_directory):
        print(
            f"Error: Source directory '{source_directory}' does not exist.",
            file=sys.stderr,
        )
        return

    # Add the output filename itself to the exclude list to prevent self-inclusion
    if output_filename not in exclude_items:
        exclude_items.append(output_filename)

    print(f"Starting to create '{output_filename}' from '{source_directory}'...")
    print(f"Excluding items: {exclude_items}")
    if include_items is not None:
        print(f"Including only items: {include_items}")

    try:
        with open(output_filename, "w", encoding="utf-8", errors="ignore") as outfile:
            outfile.write(context_head.strip() + "\n\n")
            for root, dirs, files in os.walk(source_directory):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_items]

                for filename in files:
                    is_included = (
                        not (include_items is None) and filename in include_items
                    )
                    if (
                        is_included
                        and filename not in exclude_items
                        and filename.endswith(".py")
                    ):
                        filepath = os.path.join(root, filename)
                        # Skip if it's not a regular file (e.g., a symbolic link)
                        if not os.path.isfile(filepath):
                            continue

                        try:
                            # Add a separator and the file path as a header
                            outfile.write(f"\n--- FILE: {filepath} ---\n\n")
                            with open(
                                filepath, "r", encoding="utf-8", errors="ignore"
                            ) as infile:
                                outfile.write(infile.read())
                            outfile.write("\n")  # Add a newline after file content
                            print(f"  - Added: {filepath}")
                        except Exception as e:
                            print(
                                f"  - Skipped (error reading): {filepath} - {e}",
                                file=sys.stderr,
                            )
            outfile.write(reminder.strip() + "\n\n")
            print(f"\nSuccessfully created '{output_filename}'.")
    except Exception as e:
        print(f"Error creating output file '{output_filename}': {e}", file=sys.stderr)


# --- Example Usage ---
if __name__ == "__main__":
    # Define the directory you want to process (e.g., the current directory)
    # You can change '.' to any other path like 'my_project_folder'
    target_directory = (
        "/Users/fa/Library/Mobile Documents/com~apple~CloudDocs/cs-transfer"
    )

    context_head = textwrap.dedent(
        """
        Ich soll so viel wie möglich die Funktionen von ehrapy benutzen: 
        https://github.com/theislab/ehrapy. Tutorials üeber ehrapy findest du hier:
        https://github.com/theislab/ehrapy-tutorials. Mache dich damit vertraut.

        Meine Bachelorarbeit handelt von Niereninsuffiezienz in Kindern.
        Meine Daten findest du hier:
        {file_heads}

        {Anweisung}
        """
    )
    Anweisung = "Nutze lineare Regression."
    reminder = textwrap.dedent(
        """
        Bitte antworte nur auf Deutsch.
        Antworte nur auf Basis der Informationen in diesem Kontext.
        """
    )
    alten_code_nutzen = False

    file_heads = ""
    for file in os.listdir(target_directory + "/Orginal Daten"):
        filepath = os.path.join(target_directory + "/Orginal Daten", file)
        if filepath.endswith(".csv"):
            # read first 5 lines of the csv file
            data_head = ""
            with open(
                filepath,
                "r",
                errors="ignore",
            ) as infile:
                for _ in range(5):
                    line = infile.readline()
                    if not line:
                        break
                    data_head += line

            file_heads += f"\n--- FILE: {filepath} ---\n\n"
            file_heads += data_head + "\n"

    # Define files or folders to exclude
    # 'context.txt' is automatically excluded to prevent self-inclusion
    # Add other files/folders like 'venv', '__pycache__', '.git', 'my_secret_file.txt'
    if alten_code_nutzen:
        included_items = None
        excluded_items = [
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
        included_items = []
        excluded_items = []

    create_context_file(
        source_directory=target_directory,
        context_head=context_head.format(file_heads=file_heads, Anweisung=Anweisung),
        reminder=reminder,
        exclude_items=excluded_items,
        include_items=included_items,
    )
