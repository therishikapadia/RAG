import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import httpx
import torch
import fitz  # for PDFs
from docx import Document

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import our local modules
from storage import FaissStore
from doc_utils import sentence_chunk_text, make_doc_id, make_metadata

load_dotenv()

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss.index")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "./doc_store.json")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "./id_map.json")
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BGE_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-m3")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:latest")

# Embedding dimension for bge-m3
EMBED_DIM = 1024

app = FastAPI(title="Streaming RAG - Ollama + bge-m3")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize store
store = FaissStore(
    dim=EMBED_DIM,
    index_path=FAISS_INDEX_PATH,
    doc_store_path=DOC_STORE_PATH,
    id_map_path=ID_MAP_PATH
)

# Setup embedding model
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using embedding device: {device}")
embedder = SentenceTransformer(BGE_MODEL, device=device)

# Pydantic models
class QueryModel(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class ResponseModel(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    hits: Optional[List[Dict[str, Any]]] = None

class UploadResponse(BaseModel):
    status: str
    num_chunks: int

# Utility functions
def load_file_content(file_path: str) -> str:
    """Load content from various file formats."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        text = ""
        pdf_doc = fitz.open(file_path)
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()
        return text

    elif ext in [".doc", ".docx"]:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def normalize_vectors(arr: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def ollama_embeddings(texts: List[str], model: str = BGE_MODEL) -> List[List[float]]:
    """Generate embeddings using sentence-transformers."""
    arr = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    arr = arr.astype("float32")
    arr = normalize_vectors(arr)
    return arr.tolist()

async def ollama_stream_chat(prompt: str, model: str = LLAMA_MODEL, temperature: float = 0.2):
    """Stream chat responses from Ollama."""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                
                try:
                    # Parse JSON response from Ollama
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        if content:
                            yield content
                    elif "response" in data:
                        content = data["response"]
                        if content:
                            yield content
                except json.JSONDecodeError:
                    # Fallback to raw line if not JSON
                    if line.strip():
                        yield line

# API Endpoints
@app.post("/api/upload_file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file temporarily
    temp_path = f"./temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and process file content
        text = load_file_content(temp_path)
        chunks = sentence_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        if not chunks:
            return UploadResponse(status="empty", num_chunks=0)

        ids = [make_doc_id(file.filename, i) for i in range(len(chunks))]
        metas = [make_metadata(file.filename, i) for i in range(len(chunks))]

        # Generate embeddings in batches
        batch_size = 16
        vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = ollama_embeddings(batch)
            vectors.extend(vecs)

        arr = np.array(vectors, dtype="float32")
        store.add(ids, arr, chunks, metas)
        
        return UploadResponse(status="ok", num_chunks=len(chunks))
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/upload_text", response_model=UploadResponse)
async def upload_text(source: str = Form(...), text: str = Form(...)):
    """Upload plain text document."""
    chunks = sentence_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        return UploadResponse(status="empty", num_chunks=0)

    ids = [make_doc_id(source, i) for i in range(len(chunks))]
    metas = [make_metadata(source, i) for i in range(len(chunks))]

    # Generate embeddings in batches
    batch_size = 16
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vecs = ollama_embeddings(batch)
        vectors.extend(vecs)

    arr = np.array(vectors, dtype="float32")
    store.add(ids, arr, chunks, metas)
    return UploadResponse(status="ok", num_chunks=len(chunks))

@app.post("/api/query", response_model=ResponseModel)
async def query(q: QueryModel):
    """Non-streaming query endpoint."""
    try:
        # Generate query embedding
        emb = ollama_embeddings([q.query])[0]
        vec = np.array(emb, dtype="float32")
        
        # Search for similar documents
        results = store.search(vec, top_k=q.top_k)
        context_texts = []
        sources = []
        hits = []
        
        for doc_id, score, meta in results:
            info = store.doc_store.get(doc_id)
            if info:
                context_texts.append(info["text"])
                sources.append(meta.get("source", "unknown"))
                hits.append({"id": doc_id, "score": float(score), "meta": meta})

        context = "\n\n---\n\n".join(context_texts)
        
        prompt = f"""You are a helpful assistant. Use the following document snippets to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{q.query}

Answer concisely and cite your sources.
"""

        # Call Ollama for answer generation
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{OLLAMA_URL}/api/chat"
            payload = {
                "model": LLAMA_MODEL, 
                "messages": [{"role": "user", "content": prompt}], 
                "stream": False
            }
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract answer from response
            if "message" in data and "content" in data["message"]:
                answer = data["message"]["content"]
            elif "response" in data:
                answer = data["response"]
            else:
                answer = "Sorry, I couldn't generate a response."

        return ResponseModel(answer=answer, sources=list(set(sources)), hits=hits)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/api/stream_chat")
async def stream_chat(request: Request):
    """Streaming chat endpoint."""
    try:
        body = await request.json()
        query_text = body.get("query", "")
        top_k = int(body.get("top_k", TOP_K))

        # Generate query embedding and search
        emb = ollama_embeddings([query_text])[0]
        vec = np.array(emb, dtype="float32")
        results = store.search(vec, top_k=top_k)
        
        context_texts = []
        sources = []
        hits = []
        
        for doc_id, score, meta in results:
            info = store.doc_store.get(doc_id)
            if info:
                context_texts.append(info["text"])
                sources.append(meta.get("source", "unknown"))
                hits.append({"id": doc_id, "score": float(score), "meta": meta})

        context = "\n\n---\n\n".join(context_texts)
        
        prompt = f"""You are a helpful assistant. Use the following document snippets to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query_text}

Answer concisely and cite your sources.
"""

        async def event_generator():
            try:
                # Stream tokens from Ollama
                async for chunk in ollama_stream_chat(prompt, model=LLAMA_MODEL):
                    if chunk:
                        data = chunk.replace("\n", "\\n")
                        yield f"data: {data}\n\n"
                
                # Send completion event with metadata
                meta_payload = json.dumps({
                    "event": "complete", 
                    "sources": list(set(sources)), 
                    "hits": hits
                })
                yield f"data: {meta_payload}\n\n"
                
            except Exception as e:
                err_payload = json.dumps({"event": "error", "error": str(e)})
                yield f"data: {err_payload}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream chat failed: {str(e)}")

@app.post("/api/clear_store")
def clear_store():
    store.clear()
    return {"status": "cleared"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "store_size": len(store.doc_store)}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API is running!", "endpoints": ["/docs", "/api/query", "/api/stream_chat", "/api/upload_file", "/api/upload_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
