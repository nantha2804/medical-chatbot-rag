from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVAL_COUNT = 4

VECTOR_FILE = "vectors.npy"
DOCUMENT_FILE = "documents.npy"


SYSTEM_PROMPT = """You are a careful medical information assistant.

Answer using only the supplied document context.

If the context does not support an answer, say that you could not find the
answer in the supplied documents.

Do not diagnose, prescribe, or present uncertain information as fact.

For urgent or emergency symptoms, tell the user to contact local emergency services.

Cite factual claims using the source labels provided in the context,
such as [diabetes.pdf, p. 2].

Document context:
{context}
"""


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def documents_from_bytes(filename: str, content: bytes) -> list[Document]:
    """Extract documents from PDF or TXT bytes."""

    source = Path(filename).name
    suffix = Path(source).suffix.lower()

    source_id = sha256(content).hexdigest()

    metadata = {
        "source": source,
        "source_id": source_id,
    }

    if suffix == ".pdf":
        documents = [
            Document(
                page_content=text,
                metadata={
                    **metadata,
                    "page": page_number,
                },
            )
            for page_number, page in enumerate(
                PdfReader(BytesIO(content)).pages,
                start=1,
            )
            if (text := (page.extract_text() or "").strip())
        ]

    elif suffix == ".txt":
        text = content.decode("utf-8-sig").strip()

        documents = (
            [
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            ]
            if text
            else []
        )

    else:
        raise ValueError("Only PDF and TXT documents are supported.")

    if not documents:
        raise ValueError(
            f"No extractable text was found in {source}."
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def create_embeddings(texts: list[str]) -> np.ndarray:
    """Create normalized FastEmbed embeddings."""

    from langchain_community.embeddings import FastEmbedEmbeddings

    embeddings = FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL,
        max_length=256,
        batch_size=32,
    )

    vectors = embeddings.embed_documents(texts)

    array = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(array, axis=1, keepdims=True)

    array = array / np.maximum(norms, 1e-12)

    return array


def save_documents(
    vector_directory: Path,
    documents: list[Document],
) -> int:
    """Add document chunks to the local NumPy vector store."""

    vector_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_documents: list[dict] = []

    metadata_file = vector_directory / "metadata.npy"

    if metadata_file.exists():
        existing_documents = np.load(
            metadata_file,
            allow_pickle=True,
        ).tolist()

    existing_ids = {
        item["chunk_id"]
        for item in existing_documents
    }

    new_documents = []

    for index, document in enumerate(documents):
        source_id = document.metadata["source_id"]

        page = document.metadata.get("page", "")

        chunk_id = sha256(
            f"{source_id}:{page}:{index}:{document.page_content}".encode(
                "utf-8"
            )
        ).hexdigest()

        if chunk_id in existing_ids:
            continue

        new_documents.append(
            {
                "chunk_id": chunk_id,
                "text": document.page_content,
                "source": document.metadata.get(
                    "source",
                    "Unknown source",
                ),
                "page": page,
            }
        )

    if not new_documents:
        return 0

    new_vectors = create_embeddings(
        [item["text"] for item in new_documents]
    )

    vectors_file = vector_directory / VECTOR_FILE

    if vectors_file.exists():
        old_vectors = np.load(vectors_file)

        vectors = np.vstack(
            [
                old_vectors,
                new_vectors,
            ]
        )

    else:
        vectors = new_vectors

    all_documents = (
        existing_documents + new_documents
    )

    np.save(
        vectors_file,
        vectors,
    )

    np.save(
        vector_directory / DOCUMENT_FILE,
        np.asarray(
            [item["text"] for item in all_documents],
            dtype=object,
        ),
    )

    np.save(
        metadata_file,
        np.asarray(
            all_documents,
            dtype=object,
        ),
    )

    return len(new_documents)


def search_documents(
    vector_directory: Path,
    question: str,
    k: int = RETRIEVAL_COUNT,
) -> list[dict]:
    """Find the most relevant document chunks using cosine similarity."""

    vectors_file = vector_directory / VECTOR_FILE
    metadata_file = vector_directory / "metadata.npy"

    if not vectors_file.exists() or not metadata_file.exists():
        return []

    vectors = np.load(vectors_file)

    metadata = np.load(
        metadata_file,
        allow_pickle=True,
    ).tolist()

    if len(vectors) == 0:
        return []

    query_vector = create_embeddings(
        [question]
    )[0]

    similarities = vectors @ query_vector

    top_indices = np.argsort(
        similarities
    )[::-1][:k]

    results = []

    for index in top_indices:
        item = metadata[int(index)].copy()

        item["score"] = float(
            similarities[int(index)]
        )

        results.append(item)

    return results


def answer_question(
    vector_directory: Path,
    api_key: str,
    question: str,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and generate an answer with Groq."""

    from langchain_groq import ChatGroq

    matches = search_documents(
        vector_directory,
        question,
        RETRIEVAL_COUNT,
    )

    context_parts = []
    sources = []

    for item in matches:
        source = item.get(
            "source",
            "Unknown source",
        )

        page = item.get(
            "page",
            "",
        )

        if page:
            label = f"{source}, p. {page}"
        else:
            label = source

        context_parts.append(
            f"[{label}]\n{item['text']}"
        )

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    context = "\n\n".join(context_parts)

    if not context:
        context = "No matching passages were retrieved."

    llm = ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=700,
        api_key=api_key,
    )

    response = llm.invoke(
        ANSWER_PROMPT.format_messages(
            question=question,
            context=context,
        )
    )

    return str(response.content), sources