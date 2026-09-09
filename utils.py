import streamlit as st


def display_retrieved_documents(
    documents
):

    st.subheader(
        "Retrieved Evidence"
    )

    if not documents:

        st.warning(
            "No relevant financial evidence was retrieved."
        )

        return

    for i, doc in enumerate(
        documents,
        start=1
    ):

        with st.expander(
            f"Evidence {i}"
        ):

            st.write(doc)


def display_answer(
    answer
):

    st.subheader(
        "Generated Answer"
    )

    if answer.strip():

        st.success(
            answer
        )

    else:

        st.warning(
            "The model did not generate an answer."
        )
