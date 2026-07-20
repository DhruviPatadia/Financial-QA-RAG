
import streamlit as st


def display_retrieved_documents(documents):

    st.subheader("Retrieved Evidence")

    for i, doc in enumerate(documents, start=1):

        with st.expander(f"Evidence {i}"):

            st.write(doc)


def display_answer(answer):

    st.subheader("Generated Answer")

    st.success(answer)
