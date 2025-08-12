# RAG Streamlit Interface

This is a Streamlit interface for the RAG (Retrieval-Augmented Generation) application that allows you to:
- Upload text documents
- Query those documents using AI-powered search and generation

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure your RAG API is running (typically on port 8000)

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Configuration

The app reads configuration from environment variables. You can set these in a `.env` file:

```env
API_BASE_URL=http://localhost:8000/api
OLLAMA_URL=http://localhost:11434
```

## Usage

1. Upload text files using the sidebar
2. Enter questions in the main text area
3. Adjust the number of context chunks and model as needed
4. View answers and sources in the response area