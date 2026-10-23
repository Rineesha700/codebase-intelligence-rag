import re
from langchain_core.documents import Document


CITATION_PATTERN = re.compile(
    r"\[([^\]]+\.py):(\d+)-(\d+)\]"
)


def validate_citations(
    answer: str,
    documents: list[Document],
) -> tuple[str, list[str]]:
    """
    Validate citations against retrieved source chunks.
    """

    valid_ranges = {
        (
            document.metadata["file_path"],
            document.metadata["start_line"],
            document.metadata["end_line"],
        )
        for document in documents
    }

    citations = CITATION_PATTERN.findall(answer)

    validated = []

    for file_path, start, end in citations:
        start_line = int(start)
        end_line = int(end)

        is_valid = any(
            path == file_path
            and start_line >= chunk_start
            and end_line <= chunk_end
            for path, chunk_start, chunk_end in valid_ranges
        )

        citation = f"{file_path}:{start}-{end}"

        if is_valid:
            validated.append(citation)

    return answer, validated