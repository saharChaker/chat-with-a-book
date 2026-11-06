"""The RAG pipeline: retrieve relevant chunks, then generate an answer."""
import ollama
from embedder import embed
from load_book import load_book
from chunker import chunk_text
from vector_store import VectorStore

LLM_MODEL = "llama3.2"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the book Frankenstein.
Answer using ONLY the context provided below. If the context doesn't contain the answer,
say "I don't know based on the book." Keep answers concise.

Context from the book:
{context}

Question: {question}
Answer:"""


def answer_question(store: VectorStore, question: str, k: int = 6) -> str:
    """Retrieve top-k chunks, stuff them into a prompt, and ask the LLM."""
    # 1. RETRIEVE (the R in RAG) — find relevant passages
    results = store.search(question, k=k)
    context = "\n\n---\n\n".join(chunk for chunk, _score in results)

    # 2. AUGMENT (the A) — build the prompt with the book's text inside
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 3. GENERATE (the G) — the LLM answers using only that context
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print("Loading index...")
    chunks = chunk_text(load_book("book/frankenstein.pdf"))
    store = VectorStore()
    store.load(chunks)

    q = "Who is the monster's creator and why does the monster want revenge?"
    print(f"\nQ: {q}\n")
    print(answer_question(store, q))