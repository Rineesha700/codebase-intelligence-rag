from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """
    Combines dense semantic retrieval with BM25 keyword retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        documents: list[Document],
        vectorstore,
        dense_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
        rrf_k: int = 60,
    ):
        self.documents = documents
        self.vectorstore = vectorstore

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

        self.document_lookup = {
            document.metadata["chunk_id"]: document
            for document in documents
        }

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve using dense search + BM25,
        then combine rankings using RRF.
        """

        dense_results = self.vectorstore.similarity_search(
            query,
            k=self.dense_k,
        )

        query_tokens = query.lower().split()

        bm25_results = self.bm25.get_top_n(
            query_tokens,
            self.documents,
            n=self.bm25_k,
        )

        scores = {}

        # Dense ranking
        for rank, document in enumerate(dense_results):
            chunk_id = document.metadata["chunk_id"]

            scores.setdefault(chunk_id, 0.0)

            scores[chunk_id] += (
                1 / (self.rrf_k + rank + 1)
            )

        # BM25 ranking
        for rank, document in enumerate(bm25_results):
            chunk_id = document.metadata["chunk_id"]

            scores.setdefault(chunk_id, 0.0)

            scores[chunk_id] += (
                1 / (self.rrf_k + rank + 1)
            )

        # Sort by fused score
        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        # Convert IDs back to original chunks.
        results = [
            self.document_lookup[chunk_id]
            for chunk_id in ranked_ids[:self.final_k]
        ]

        return results