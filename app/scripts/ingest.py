import os
import faiss
import numpy as np
from openai import OpenAI
from pathlib import Path

client = OpenAI()

def load_documents(source_dir: str):
    """Load all documents from directory."""
    documents = []
    for file in Path(source_dir).glob("**/*"):
        if file.suffix in [".pdf", ".txt", ".docx"]:
            documents.append({
                "path": str(file),
                "name": file.name
            })
    return documents

def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text: str):
    """Get OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def build_and_save_index(source_dir: str = "./data/documents"):
    """Main ingestion pipeline."""
    print(f"Loading documents from {source_dir}...")
    documents = load_documents(source_dir)
    print(f"Found {len(documents)} documents")

    all_chunks = []
    all_embeddings = []

    for doc in documents:
        print(f"Processing {doc['name']}...")
        with open(doc["path"], "r", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            all_chunks.append({
                "text": chunk,
                "source": doc["name"]
            })
            all_embeddings.append(embedding)

    # Build FAISS index
    print("Building FAISS index...")
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(all_embeddings).astype("float32")
    index.add(vectors)

    # Save index
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss_index")
    print(f"Index saved! Total chunks indexed: {len(all_chunks)}")

if __name__ == "__main__":
    build_and_save_index()
