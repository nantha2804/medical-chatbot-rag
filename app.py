
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from medical_rag import (
    answer_question,
    documents_from_bytes,
    save_documents,
    split_documents,
)


PROJECT_DIR = Path(__file__).resolve().parent
STARTER_DOCUMENT = PROJECT_DIR / "diabetes.pdf"
VECTOR_DIRECTORY = PROJECT_DIR / "vector_db"

load_dotenv(PROJECT_DIR / ".env")


st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide",
)


st.title("🩺 Medical RAG Chatbot")

st.markdown(
    "<p style='text-align: center;'>"
    "The chatbot retrieves relevant passages from your documents "
    "before generating an answer."
    "</p>",
    unsafe_allow_html=True,
)


def get_api_key() -> str:
    """Get the Groq API key from the .env file."""
    return os.getenv("GROQ_API_KEY", "").strip()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "➕ Add Documents",
        use_container_width=True,
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF or TXT file."
            )

        else:

            added = 0

            for uploaded_file in uploaded_files:

                try:

                    content = uploaded_file.getvalue()

                    documents = documents_from_bytes(
                        uploaded_file.name,
                        content,
                    )

                    chunks = split_documents(
                        documents
                    )

                    added += save_documents(
                        VECTOR_DIRECTORY,
                        chunks,
                    )

                except Exception as exc:

                    st.error(
                        f"Could not add {uploaded_file.name}: {exc}"
                    )

            if added:

                st.success(
                    f"Added {added} new document chunks."
                )

            else:

                st.info(
                    "No new document chunks were added."
                )


# ---------------------------------------------------------
# Prepare starter document
# ---------------------------------------------------------

if not VECTOR_DIRECTORY.exists():

    try:

        documents = documents_from_bytes(
            STARTER_DOCUMENT.name,
            STARTER_DOCUMENT.read_bytes(),
        )

        chunks = split_documents(
            documents
        )

        save_documents(
            VECTOR_DIRECTORY,
            chunks,
        )

    except Exception as exc:

        st.error(
            f"Could not prepare the starter document: {exc}"
        )


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    source_name = source.get(
                        "source",
                        "Unknown",
                    )

                    page = source.get(
                        "page",
                        "",
                    )

                    if page:

                        st.write(
                            f"- {source_name}, page {page}"
                        )

                    else:

                        st.write(
                            f"- {source_name}"
                        )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your medical documents..."
)


if question:

    api_key = get_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Please add it to the .env file."
        )

        st.stop()


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, sources = answer_question(
                    VECTOR_DIRECTORY,
                    api_key,
                    question,
                )

                st.markdown(answer)


                if sources:

                    with st.expander(
                        "📚 Sources"
                    ):

                        seen = set()

                        for source in sources:

                            source_name = source.get(
                                "source",
                                "Unknown",
                            )

                            page = source.get(
                                "page",
                                "",
                            )

                            source_key = (
                                source_name,
                                page,
                            )

                            if source_key in seen:

                                continue

                            seen.add(source_key)

                            if page:

                                st.write(
                                    f"- {source_name}, page {page}"
                                )

                            else:

                                st.write(
                                    f"- {source_name}"
                                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )


            except Exception as exc:

                st.error(
                    f"Error: {exc}"
                )

