"""Split a long text into overlapping chunks."""


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of `size` characters with `overlap` between them."""
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        chunks.append(chunk)
        if start + size >= len(text):
            break  # reached the end, stop
    return chunks


if __name__ == "__main__":
    from src.load_book import load_book  

    text = load_book("book/frankenstein.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print("\n---- Chunk #0 (first 300 chars) ----")
    print(chunks[0][:300])
    print("\n---- Chunk #42 (first 300 chars) ----")
    print(chunks[42][:300])
    print("\n---- Last chunk ----")
    print(chunks[-1][:300])