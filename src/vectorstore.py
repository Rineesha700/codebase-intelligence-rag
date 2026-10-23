from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_vectorstore(
    documents: list[Document],
    persist_directory: str = "chroma_db",
) -> Chroma:
    """
    Create a new Chroma vector store from code chunks.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore


def load_vectorstore(
    persist_directory: str = "chroma_db",
) -> Chroma:
    """
    Load an existing Chroma vector store.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    return vectorstore