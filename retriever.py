import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer


class FinancialRetriever:
    def __init__(self, data_path, top_k=1):
        self.data_path = data_path
        self.top_k = top_k

        index_path = os.path.join(
            data_path,
            "finqa_tatqa_faiss_index.bin"
        )

        documents_path = os.path.join(
            data_path,
            "finqa_tatqa_documents.pkl"
        )

        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, question, top_k=None):
        k = top_k if top_k is not None else self.top_k

        query_embedding = self.encoder.encode(
            [question],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            results.append({
                "document": self.documents[idx],
                "distance": float(distance)
            })

        return results
