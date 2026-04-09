from openai import OpenAI
from typing import List, Tuple

client = OpenAI()

def build_context(chunks: List[dict]) -> str:
    """Build context string from retrieved chunks."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']} - Page {chunk['page']}]\n"
        context += chunk["text"] + "\n"
    return context

def generate_answer(
    question: str,
    chunks: List[dict]
) -> Tuple[str, List[str]]:
    """
    Generate answer using GPT-4 based on retrieved chunks.
    Returns answer and list of sources used.
    """
    # Build context from chunks
    context = build_context(chunks)

    # Build prompt
    prompt = f"""You are an intelligent document assistant.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I could not find this information in the documents."

Context:
{context}

Question: {question}

Answer:"""

    # Call GPT-4
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a precise document Q&A assistant. Only answer based on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Extract sources
    sources = [
        f"{chunk['source']} (Page {chunk['page']})"
        for chunk in chunks
    ]

    return answer, sources
