import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class FinancialGenerator:

    def __init__(self, model_id):

        print(
            "Loading fine-tuned FLAN-T5-Large model..."
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_id
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Generator ready!"
        )

        print(
            "Device:",
            self.device
        )


    def build_prompt(
        self,
        question,
        retrieved_docs
    ):

        documents = []

        for item in retrieved_docs:

            if isinstance(item, dict):

                documents.append(
                    item["document"]
                )

            else:

                documents.append(
                    str(item)
                )

        context = "\n\n".join(
            documents
        )

        prompt = f"""Use the following financial information to answer the question.

Financial Information:
{context}

Question:
{question}

Answer:
"""

        return prompt


    def generate(
        self,
        question,
        retrieved_docs
    ):

        prompt = self.build_prompt(
            question,
            retrieved_docs
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()
