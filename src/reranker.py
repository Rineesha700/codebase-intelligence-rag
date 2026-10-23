from langchain_core.documents import Document


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Lightweight lexical reranking of hybrid retrieval candidates.

    Scores documents based on:
    - query term matches in the code
    - query term matches in metadata
    - exact phrase matches
    """

    query_lower = query.lower()

    # Remove common question words that don't help identify code.
    stop_words = {
        "how", "does", "do", "the", "a", "an",
        "is", "are", "what", "where", "why",
        "rich",
    }

    query_terms = [
        term.strip(".,?!():")
        for term in query_lower.split()
        if term.strip(".,?!():") not in stop_words
    ]

    scored_documents = []

    for document in documents:
        content = document.page_content.lower()
        file_path = document.metadata.get("file_path", "").lower()
        symbol = document.metadata.get("symbol", "").lower()

        score = 0.0

        # Exact query phrase.
        if query_lower in content:
            score += 5.0

        for term in query_terms:
            # Strong signal if the term appears in the filename.
            if term in file_path:
                score += 3.0

            # Strong signal if the term appears in the symbol name.
            if term in symbol:
                score += 3.0

            # General content match.
            score += min(content.count(term), 5) * 0.5

        scored_documents.append((score, document))

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored_documents[:top_k]
    ]