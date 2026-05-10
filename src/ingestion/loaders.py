from pathlib import Path
import json


SUPPORTED_EXTENSIONS = [".txt", ".md", ".json"]


def load_document(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    try:
        if suffix in [".txt", ".md"]:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        elif suffix == ".json":
            data = json.loads(
                file_path.read_text(encoding="utf-8")
            )

            text = json.dumps(data)

        return {
            "source": str(file_path),
            "filename": file_path.name,
            "text": text.strip()
        }

    except Exception as e:
        print(f"Failed loading {file_path}: {e}")
        return None


def load_documents(folder_path: Path):
    docs = []

    for file in folder_path.rglob("*"):
        if file.is_file():
            doc = load_document(file)

            if doc:
                docs.append(doc)

    return docs