import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from medical_rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    create_vector_store,
    documents_from_bytes,
    index_source,
)


class FakeVectorStore:
    def __init__(self):
        self.source_ids = set()
        self.added_documents = []

    def get(self, where, include):
        if include != ["metadatas"]:
            raise AssertionError("Index checks should fetch metadata only.")
        source_id = where["source_id"]
        return {"ids": [source_id] if source_id in self.source_ids else []}

    def add_documents(self, documents, ids):
        self.added_documents.extend(zip(ids, documents))
        self.source_ids.update(
            document.metadata["source_id"] for document in documents
        )


class DocumentParsingTests(unittest.TestCase):
    def test_text_document_has_source_metadata(self):
        documents = documents_from_bytes("notes.txt", b"Diabetes care notes")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Diabetes care notes")
        self.assertEqual(documents[0].metadata["source"], "notes.txt")
        self.assertIn("source_id", documents[0].metadata)

    def test_rejects_unsupported_file_types(self):
        with self.assertRaisesRegex(ValueError, "Only PDF and TXT"):
            documents_from_bytes("notes.docx", b"content")

    def test_rejects_empty_text_documents(self):
        with self.assertRaisesRegex(ValueError, "No extractable text"):
            documents_from_bytes("empty.txt", b"  ")

    def test_starter_pdf_contains_extractable_text(self):
        starter_pdf = Path(__file__).resolve().parents[1] / "diabetes.pdf"
        documents = documents_from_bytes(starter_pdf.name, starter_pdf.read_bytes())

        self.assertTrue(documents)
        self.assertTrue(all(document.metadata.get("page") for document in documents))


class IndexingTests(unittest.TestCase):
    def test_source_is_indexed_once(self):
        vector_store = FakeVectorStore()
        content = b"Medical information " * 100

        first_count = index_source(vector_store, "guide.txt", content)
        second_count = index_source(vector_store, "guide.txt", content)

        self.assertGreater(first_count, 0)
        self.assertEqual(second_count, 0)
        self.assertEqual(len(vector_store.added_documents), first_count)


class VectorStoreTests(unittest.TestCase):
    def test_vector_store_uses_local_fastembed_model(self):
        chroma = Mock(return_value=object())
        embeddings = Mock(return_value=object())
        chroma_module = ModuleType("langchain_chroma")
        chroma_module.Chroma = chroma
        embeddings_module = ModuleType("langchain_community.embeddings")
        embeddings_module.FastEmbedEmbeddings = embeddings

        with patch.dict(
            "sys.modules",
            {
                "langchain_chroma": chroma_module,
                "langchain_community.embeddings": embeddings_module,
            },
        ):
            create_vector_store(Path("chroma_db"))

        embeddings.assert_called_once_with(
            model_name=EMBEDDING_MODEL,
            max_length=256,
            batch_size=32,
        )
        chroma.assert_called_once_with(
            collection_name=COLLECTION_NAME,
            persist_directory="chroma_db",
            embedding_function=embeddings.return_value,
        )


if __name__ == "__main__":
    unittest.main()
