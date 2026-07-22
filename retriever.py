
import faiss
import pickle
from sentence_transformers import SentenceTransformer


class FinancialRetriever:

    def __init__(self, data_path):

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = faiss.read_index(
            f"{data_path}/faiss_index.bin"
        )

        with open(
            f"{data_path}/documents.pkl",
            "rb"
        ) as f:

            self.documents = pickle.load(f)

    def retrieve(self, question, top_k=3):

        embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        _, indices = self.index.search(
            embedding,
            top_k
        )

        docs = []

        for i in indices[0]:

            doc = self.documents[i]

            # Keep the question and shorten the answer
            if "Answer:" in doc:
                question_part, answer_part = doc.split("Answer:", 1)

                # Keep only the first ~150 words of the answer
                shortened_answer = " ".join(answer_part.split()[:150])

                doc = f"{question_part}Answer: {shortened_answer}"

            docs.append(doc)

        return docs
