#!/usr/bin/env python3
"""
Setup and run script for RAG application
"""

import os
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = ["data", "temp"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False
    return True


def check_ollama():
    """Check if Ollama is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/version", timeout=5.0)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
        else:
            print("✗ Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"✗ Ollama is not running or accessible: {e}")
        print("Please start Ollama and make sure llama3.2:latest model is available")
        return False


def create_env_file():
    """Create .env file if it doesn't exist."""
    if not Path(".env").exists():
        env_content = """# Ollama Configuration
OLLAMA_URL=http://localhost:11434
LLAMA_MODEL=llama3.2:latest

# BGE Model Configuration  
BGE_MODEL=BAAI/bge-m3

# Storage Paths
FAISS_INDEX_PATH=./data/faiss.index
DOC_STORE_PATH=./data/doc_store.json
ID_MAP_PATH=./data/id_map.json

# Chunking Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K=5
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✓ Created .env file")
    else:
        print("✓ .env file already exists")


def run_server():
    """Run the FastAPI server."""
    print("\nStarting RAG server...")
    print("Server will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("\n✓ Server stopped")


def main():
    """Main setup and run function."""
    print("🤖 RAG Application Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed. Please install dependencies manually.")
        return
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed, but continuing...")
    
    # Check Ollama
    if not check_ollama():
        print("\n❌ Setup incomplete. Please:")
        print("1. Install and start Ollama (https://ollama.ai)")
        print("2. Pull the llama3.2:latest model: ollama pull llama3.2:latest")
        print("3. Run this script again")
        return
    
    print("\n✅ Setup completed successfully!")
    
    # Ask if user wants to run the server
    choice = input("\nDo you want to start the server now? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        run_server()
    else:
        print("\nTo start the server manually, run:")
        print("python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nTo test the system, run:")
        print("python test_rag.py")


if __name__ == "__main__":
    main()