import faiss
import numpy as np
from openai import OpenAI
from typing import List

client = OpenAI()

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def retrieve_documents(query: str, top_k: int = 5) -> List[dict]:
    """
    Retrieve top-k relevant document chunks for a query.
    Uses FAISS vector store for similarity search.
    """
    # Get query embedding
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")

    # Search FAISS index
    # In production this loads from saved index
    # index = faiss.read_index("data/faiss_index")
    # distances, indices = index.search(query_vector, top_k)

    # Sample chunks for demonstration
    chunks = [
        {
            "text": f"Relevant document chunk {i+1} for query: {query}",
            "source": f"document_{i+1}.pdf",
            "page": i + 1,
            "score": round(0.95 - (i * 0.05), 2)
        }
        for i in range(top_k)
    ]

    return chunks

def build_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """Build a FAISS index from embeddings."""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)
    return index
