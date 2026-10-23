import streamlit as st
from dotenv import load_dotenv

from src.ingestion import load_codebase
from src.chunking import chunk_code
from src.vectorstore import load_vectorstore
from src.hybrid_retrieval import HybridRetriever
from src.reranker import rerank_documents
from src.generator import generate_answer
from src.citation import validate_citations


load_dotenv()

REPOSITORY_PATH = "data/repository/rich/rich"


@st.cache_resource
def load_pipeline():
    documents = load_codebase(REPOSITORY_PATH)
    chunks = chunk_code(documents)

    vectorstore = load_vectorstore()

    retriever = HybridRetriever(
        documents=chunks,
        vectorstore=vectorstore,
        dense_k=20,
        bm25_k=20,
        final_k=20,
    )

    return retriever


st.set_page_config(
    page_title="Codebase Intelligence Assistant",
    page_icon="💻",
    layout="wide",
)

st.title("💻 Codebase Intelligence Assistant")

st.write(
    "Ask questions about the Rich Python codebase."
)

question = st.text_input(
    "Ask a question",
    placeholder="How does Rich render text objects?",
)

if st.button("Ask") and question:

    with st.spinner("Searching the codebase..."):

        retriever = load_pipeline()

        candidates = retriever.retrieve(question)

        results = rerank_documents(
            question,
            candidates,
            top_k=5,
        )

        answer = generate_answer(
            question,
            results,
        )

        answer, validated_citations = validate_citations(
            answer,
            results,
        )

    st.subheader("Answer")

    st.markdown(answer)

    st.subheader("Sources")

    if validated_citations:
        for citation in validated_citations:
            st.code(citation)
    else:
        st.write("No valid citations found.")