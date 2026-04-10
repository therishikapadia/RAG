# RAG Application Details

This is a full-stack RAG (Retrieval-Augmented Generation) application with a FastAPI backend and a Streamlit frontend.
It allows you to upload documents (PDF, DOCX, TXT) and query them using an LLM.

## Setup & Running

A convenient setup script is provided. Simply run:

```bash
python setup_and_run.py
```

This script will create necessary directories, download NLTK data, check your Ollama installation, and can launch the FastAPI server for you.

To run the application manually, you need two terminal windows:

**1. Start the FastAPI Backend:**

```bash
# From within your virtual environment
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Start the Streamlit Frontend:**

```bash
# In a new terminal
streamlit run app.py
```

## Models Used

By default, the application uses:

- **Embeddings:** `BAAI/bge-m3` via the `sentence-transformers` library (runs locally).
- **LLM / Generation:** `llama3.2:latest` via **Ollama**.

## Configuration

You can customize the models and other settings in a `.env` file:

```env
OLLAMA_URL=http://localhost:11434
LLM_MODEL=llama3.2:latest
BGE_MODEL=BAAI/bge-m3
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K=5
```

## System Architecture & Data Flow

This project implements a modern completely-local Retrieval-Augmented Generation (RAG) architecture:

### 1. Frontend (Streamlit)
- Provides a clean, sleek user interface (`app.py`).
- Handles file uploads (`.pdf`, `.docx`, `.txt`) and passes them securely to the backend API.
- Renders the chat interface and seamlessly streams real-time LLM responses to the user.

### 2. Backend Engine (FastAPI)
- Acts as the central orchestrator (`main.py`).
- **Data Parsing:** Unpacks uploaded documents into raw text using lightweight parsing libraries like `PyMuPDF` (PDFs) and `python-docx` (Word).
- **Semantic Chunking:** Slices massive documents into smaller, overlapping context chunks using NLTK to retain contextual meaning (`doc_utils.py`).

### 3. Vector Storage (FAISS + SentenceTransformers)
- **Embedding:** Document chunks are mathematically converted into dense high-dimensional vectors locally using `sentence-transformers` (automatically sized to your `.env` model).
- **Indexing:** These vectors are saved into a high-performance in-memory local FAISS index (`faiss.index`) mapped to a JSON document store (`storage.py`), enabling instantaneous similarity searches.

### 4. Generation (Ollama)
- When you ask a question, the backend converts your query into a vector and searches FAISS for the Top-K most relevant document chunks.
- These highly relevant snippets are packaged together with your original question as "Context".
- The backend queries your **Ollama** LLM (like `gpt-oss` or `llama3.2`), forcing the AI to strictly answer your query based *only* on the provided context retrieved from your files!
