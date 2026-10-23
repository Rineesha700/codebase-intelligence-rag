import json

from dotenv import load_dotenv
from src.vectorstore import load_vectorstore

load_dotenv()


TOP_K = 5


def evaluate_retrieval():
    with open("evaluation/benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    vectorstore = load_vectorstore()

    hits = 0

    for item in benchmark:
        question = item["question"]
        expected_files = set(item["expected_files"])

        results = vectorstore.similarity_search(
            question,
            k=TOP_K
        )

        retrieved_files = {
            result.metadata["file_path"]
            for result in results
        }

        success = bool(expected_files & retrieved_files)

        if success:
            hits += 1

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Expected: {sorted(expected_files)}")
        print(f"Retrieved: {sorted(retrieved_files)}")
        print(f"Hit@{TOP_K}: {success}")

    recall_at_k = hits / len(benchmark)

    print("\n" + "=" * 70)
    print(f"Baseline Recall@{TOP_K}: {recall_at_k:.2%}")
    print(f"Correct: {hits}/{len(benchmark)}")


if __name__ == "__main__":
    evaluate_retrieval()