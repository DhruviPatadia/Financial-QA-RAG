
import streamlit as st
import os

from retriever import FinancialRetriever
from generator import FinancialGenerator
from utils import (
    display_retrieved_documents,
    display_answer
)

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Financial QA using FLAN-T5 + RAG",
    page_icon="💰",
    layout="wide"
)

# ============================================
# Project Paths
# ============================================

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

# Hugging Face Model Repository
MODEL_ID = "DP037/financial-qa-flan-t5-base"

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "deployment",
    "data"
)

# ============================================
# Load Models (Cached)
# ============================================

@st.cache_resource
def load_models():

    retriever = FinancialRetriever(DATA_PATH)

    generator = FinancialGenerator(MODEL_ID)

    return retriever, generator


retriever, generator = load_models()

# ============================================
# Sidebar
# ============================================

st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Model

- Fine-tuned FLAN-T5 Base

### Retrieval

- Sentence Transformers
- FAISS Vector Database

### Generation

- Retrieval-Augmented Generation (RAG)

### Dataset

Financial Questions Dataset
""")

# ============================================
# Main Title
# ============================================

st.title("💰 Financial Question Answering")

st.markdown(
"""
This application answers financial questions using a
fine-tuned **FLAN-T5 Base** model together with
**Retrieval-Augmented Generation (RAG)**.
"""
)

question = st.text_area(
    "Enter your financial question",
    height=120
)

# ============================================
# Generate Button
# ============================================

if st.button("Generate Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving relevant documents..."):

            retrieved_docs = retriever.retrieve(
                question,
                top_k=3
            )

        display_retrieved_documents(
            retrieved_docs
        )

        with st.spinner("Generating answer..."):

            answer = generator.generate(
                question,
                retrieved_docs
            )

        display_answer(answer)

st.markdown("---")

st.caption(
    "Developed as part of the MSc Research Project "
    "on Financial Question Answering using LLMs and Retrieval-Augmented Generation (RAG)."
)
