from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".py"}


def load_codebase(repo_path: str) -> list[Document]:
    """
    Load Python source files from a codebase into LangChain Documents.
    """

    documents = []

    repo = Path(repo_path)

    for file_path in repo.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            relative_path = file_path.relative_to(repo)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "file_path": str(relative_path),
                        "file_name": file_path.name,
                    },
                )
            )

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    return documents