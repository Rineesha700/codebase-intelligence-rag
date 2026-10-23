from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_code(documents: list[Document]) -> list[Document]:
    """
    Split source code into chunks while preserving
    source file and line-range metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=[
            "\nclass ",
            "\ndef ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        file_path = chunk.metadata["file_path"]

        # Find the original source document.
        original_document = next(
            document
            for document in documents
            if document.metadata["file_path"] == file_path
        )

        original_lines = original_document.page_content.splitlines()

        # Locate the chunk in the original source.
        chunk_text = chunk.page_content

        start_position = original_document.page_content.find(chunk_text)

        if start_position == -1:
            start_line = 1
        else:
            start_line = (
                original_document.page_content[:start_position].count("\n")
                + 1
            )

        end_line = start_line + len(chunk_text.splitlines()) - 1

        chunk.metadata["start_line"] = start_line
        chunk.metadata["end_line"] = end_line

        chunk.metadata["chunk_id"] = (
            f"{file_path}::{index}"
        )

    return chunks