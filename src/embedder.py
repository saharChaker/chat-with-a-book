"""Turn text into vectors using a local embedding model (Ollama)."""
import numpy as np
import ollama

MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns a 2D numpy array: one vector per text."""
    response = ollama.embed(model=MODEL, input=texts)
    return np.array(response["embeddings"])


if __name__ == "__main__":
    vectors = embed([
        "The dog chased the cat",
        "Puppies and kittens are playing together",
        "Quantum entanglement defies classical intuition",
    ])
    print(f"Shape: {vectors.shape}")  # (3, 768) = 3 sentences, 768 numbers each
    print(f"First 5 numbers of sentence 1: {vectors[0][:5]}")