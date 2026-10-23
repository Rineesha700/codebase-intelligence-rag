import json

from dotenv import load_dotenv

from src.ingestion import load_codebase
from src.chunking import chunk_code
from src.vectorstore import load_vectorstore
from src.hybrid_retrieval import HybridRetriever


load_dotenv()

REPOSITORY_PATH = "data/repository/rich/rich"
TOP_K = 5


def evaluate_retrieval():

    documents = load_codebase(REPOSITORY_PATH)
    chunks = chunk_code(documents)

    vectorstore = load_vectorstore()

    retriever = HybridRetriever(
        documents=chunks,
        vectorstore=vectorstore,
        dense_k=10,
        bm25_k=10,
        final_k=TOP_K,
    )

    with open(
        "evaluation/benchmark.json",
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    hits = 0

    for item in benchmark:

        question = item["question"]
        expected_files = set(item["expected_files"])

        results = retriever.retrieve(question)

        retrieved_files = {
            result.metadata["file_path"]
            for result in results
        }

        success = bool(
            expected_files & retrieved_files
        )

        if success:
            hits += 1

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Expected: {sorted(expected_files)}")
        print(f"Retrieved: {sorted(retrieved_files)}")
        print(f"Hit@{TOP_K}: {success}")

    recall = hits / len(benchmark)

    print("\n" + "=" * 70)
    print(f"Hybrid Recall@{TOP_K}: {recall:.2%}")
    print(f"Correct: {hits}/{len(benchmark)}")


if __name__ == "__main__":
    evaluate_retrieval()