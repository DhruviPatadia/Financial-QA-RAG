
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

        context = "\n\n".join(retrieved_docs)

        prompt = f"""
    Context:
    {context}

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
                max_new_tokens=150,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True

            )

        answer = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        return answer
