# Medical Chatbot using RAG

A Streamlit chatbot that retrieves passages from medical documents before answering.
The starter knowledge base is `diabetes.pdf`; add more PDF or TXT sources from the sidebar.
The unrelated technology-news files in `new_articles/` are intentionally not indexed.

## Features

- Local ONNX embeddings (`all-MiniLM-L6-v2`) through FastEmbed; PyTorch is not required.
- Persistent Chroma vector storage in `chroma_db/`.
- Groq chat completion with answers grounded in retrieved passages.
- Source references shown with each answer.
- Upload additional PDF and TXT documents from the app.

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/).
- A Groq API key.

## Run locally

```powershell
uv sync
Copy-Item .env.example .env
```

Add your key to `.env`:

```text
GROQ_API_KEY=your_groq_api_key
```

The app reads the key from `.env` locally or from `.streamlit/secrets.toml` in a
Streamlit deployment. It never displays the key in the app. Replace the example
value with your own key, and do not commit `.env` or `secrets.toml`.

Start the app:

```powershell
uv run streamlit run app.py
```

On Windows, you can also start it by double-clicking `run_app.bat` after running
`uv sync`.

If you change `.env` while Streamlit is running, stop the app and start it again
so the new key is loaded.

On the first question or document upload, the local embedding model is downloaded and the
starter PDF is indexed. The Chroma database is then reused between runs. Uploaded files are
indexed into the same local database; only upload medical documents you are authorized to
process.

To use the original notebooks as well:

```powershell
uv sync --group notebook
```

## Medical information notice

This tool provides educational information from the documents you supply. It is not a
diagnosis or a substitute for professional medical advice. Do not use it for emergencies;
contact local emergency services when immediate help is needed.
