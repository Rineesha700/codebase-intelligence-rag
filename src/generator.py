from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


def format_context(documents: list[Document]) -> str:
    """Format retrieved code chunks for the LLM."""

    context_parts = []

    for document in documents:
        file_path = document.metadata["file_path"]
        start_line = document.metadata["start_line"]
        end_line = document.metadata["end_line"]

        context_parts.append(
            f"FILE: {file_path}\n"
            f"LINES: {start_line}-{end_line}\n"
            f"CODE:\n{document.page_content}"
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(context_parts)


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """Generate a grounded answer using retrieved source code."""

    context = format_context(documents)

    prompt = f"""
You are a codebase assistant answering questions about a Python repository.

Answer the user's question using ONLY the provided source code.

Rules:
1. Do not invent implementation details.
2. If the retrieved code does not contain enough information,
   say that the available context is insufficient.
3. Explain the implementation clearly and concisely.
4. Every important implementation claim must include a source citation.
5. Citations must use this exact format:
   [path/to/file.py:START-END]
6. Do not cite files that are not present in the provided context.

SOURCE CODE:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
    )

    response = llm.invoke(prompt)

    return response.content