# Financial Question Answering using FLAN-T5-Large + Standard RAG

This project implements a financial Question Answering system using:

- Fine-tuned FLAN-T5-Large
- Sentence Transformers
- all-MiniLM-L6-v2 embeddings
- FAISS vector search
- Retrieval-Augmented Generation (RAG)
- Streamlit

The application retrieves relevant financial information from the
19,371-document financial QA training corpus and uses the retrieved
information together with the user's question to generate an answer.

The deployed system represents the Standard RAG configuration
developed and evaluated as part of the MSc research project.

The retrieval corpus contains Context and Question fields and excludes
reference answers.

The Enhanced RAG configuration is not deployed.
