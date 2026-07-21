
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class FinancialGenerator:

    def __init__(self, model_id):

        print("Loading FLAN-T5 model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id
        )

        self.model.to(self.device)

        self.model.eval()

        print("Generator ready!")

    def build_prompt(self, question, retrieved_docs):

        evidence = ""

        for i, doc in enumerate(retrieved_docs, 1):
            evidence += f"""
    Document {i}

    {doc}

    """

        prompt = f"""
    You are an expert financial question answering assistant.

    You must answer the user's question ONLY using the retrieved evidence below.

    Instructions:

    - Read ALL retrieved documents carefully.
    - Combine information from multiple documents whenever possible.
    - Write a clear and concise answer in your own words.
    - Do NOT copy entire sentences directly from the evidence.
    - Do NOT use outside knowledge.
    - Do NOT make up information.
    - If the retrieved evidence does not contain enough information to answer the question, respond exactly with:

    "I cannot answer this question based on the retrieved evidence."

    Retrieved Evidence:

    {evidence}

    User Question:
    {question}

    Answer:
    """

        return prompt

    def generate(self, question, retrieved_docs):

        prompt = self.build_prompt(
            question,
            retrieved_docs
        )

        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

            truncation=True,

            max_length=512

        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=128,

                num_beams=4,

                early_stopping=True

            )

        answer = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        return answer
