
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

        answers = []

        for doc in retrieved_docs:

            if "Answer:" in doc:

                answer = doc.split(
                    "Answer:",
                    1
                )[1].strip()

                answers.append(answer)

        evidence = "\n\n".join(

            [
                f"Evidence {i+1}:\n{ans}"
                for i, ans in enumerate(answers)
            ]

        )

        prompt = f"""
You are an expert financial question answering assistant.

Use ONLY the evidence below to answer the user's question.

Financial Evidence:

{evidence}

Question:
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
