"""Store vectors and search them by cosine similarity."""
import numpy as np
from src.embedder import embed       

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: -1 (opposite) to 1 (identical)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

if __name__ == "__main__":

    sentences = [
        "The dog chased the cat",
        "Puppies and kittens are playing together",
        "Quantum entanglement defies classical intuition",
    ]
    vectors = embed(sentences)

    print(f"'Dog' vs 'Puppies': {cosine_similarity(vectors[0], vectors[1]):.3f}")
    print(f"'Dog' vs 'Quantum':  {cosine_similarity(vectors[0], vectors[2]):.3f}")
    
class VectorStore:
    """Holds chunk texts and their vectors; searches by cosine similarity."""

    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None

    def build(self, chunks: list[str]) -> None:
        """Embed all chunks and save to disk (run once)."""
        self.chunks = chunks
        print(f"Embedding {len(chunks)} chunks... (this takes a few minutes)")
        self.vectors = embed(chunks)
        np.save("vectors.npy", self.vectors)
        print("Saved to vectors.npy")

    def load(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.vectors = np.load("vectors.npy")
        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                f"Index mismatch: {len(self.chunks)} chunks but "
                f"{len(self.vectors)} vectors. The chunker changed since the "
                f"index was built — rerun the build."
        )

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Find the k chunks most similar to the query."""
        q = embed([query])[0]  # one vector for the question
        # Vectorized cosine similarity: dot each stored vector with q, normalized
        scores = self.vectors @ q / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q))
        top_k = np.argsort(scores)[::-1][:k]  # indices of highest scores
        return [(self.chunks[i], float(scores[i])) for i in top_k]
    
if __name__ == "__main__":
    from src.load_book import load_book
    from src.chunker import chunk_text

    chunks = chunk_text(load_book("book/frankenstein.pdf"))
    store = VectorStore()
    store.build(chunks)   # embeds + saves; run ONCE
    #store.load(chunks)  # use this on later runs