# Chat with a Book 📚

A tool that lets you ask questions to any book in plain English and get answers grounded in the actual text using RAG (Retrieval-Augmented Generation), embeddings, and cosine similarity  — built from scratch in Python, with **no RAG frameworks**.

I built this to understand Retrieval-Augmented Generation from first principles:
not by calling `vectorstore.similarity_search()` from a library, but by writing the chunking, the embeddings, and the cosine similarity myself. Demo book: *Frankenstein* (public domain, Project Gutenberg).

## Demo

<img width="680" height="183" alt="demo" src="https://github.com/user-attachments/assets/4e7504b5-eb06-4232-a1d5-105394179546" />

## How it works

```
┌──────────┐   ┌──────────┐   ![Uploading Screenshot 2026-11-06 120112.png…]()
┌─────────────┐   ┌────────────────┐
│ book.pdf │──▶│ chunker  │──▶│  embedder   │──▶│  vector store  │
│ (raw)    │   │ 910 chun-│   │ nomic-embed │   │ numpy + cosine │
│          │   │ ks, 500ch│   │ -text       │   │ similarity     │
└──────────┘   └──────────┘   └─────────────┘   └────────────────┘
                                    (one-time, saved to vectors.npy)
                                                          │
   ┌──────────┐   ┌──────────┐                     ┌──────▼───────┐
   │  You ask │──▶│ embed    │────────────────────▶│ top-k chunks │
   │ question │   │ question │                     │ by similarity│
   └──────────┘   └──────────┘                     └──────┬───────┘
                                                          │
                                                   ┌──────▼──────────────┐
                                                   │ llama3.2 prompt:    │
                                                   │ context + question  │
                                                   └──────┬──────────────┘
                                                          │
                                                   ┌──────▼───────┐
                                                   │    Answer    │
                                                   └──────────────┘
```

1. **Chunk** — the book is split into ~500-character chunks with 50-character overlap,
   so no sentence gets sliced in half at a boundary.
2. **Embed** — each chunk is converted into a 768-dimension vector by
   `nomic-embed-text` (running locally via Ollama). Similar meaning → similar numbers.
3. **Store** — vectors are saved once to `vectors.npy`; no re-processing on later runs.
4. **Retrieve** — your question is embedded the same way, and cosine similarity finds
   the k chunks whose vectors point in the most similar direction.
5. **Generate** — `llama3.2` answers using *only* the retrieved chunks as context,
   with instructions to say "I don't know based on the book" when the answer isn't there.

## Run it yourself

```bash
# 1. Install Ollama (ollama.com), then:
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Set up
git clone https://github.com/saharChaker/chat-with-a-book.git
cd chat-with-a-book
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Put any PDF in book/ and build the index (a few minutes, once):
python src/vector_store.py     # with store.build(chunks) uncommented

# 4. Chat
python src/main.py
```

## The code

| File | What it does |
|---|---|
| `src/load_book.py` | Extracts raw text from the PDF with pypdf |
| `src/chunker.py` | Splits text into 500-char chunks, 50-char overlap |
| `src/embedder.py` | Wraps Ollama's `nomic-embed-text` to turn text into 768-d vectors |
| `src/vector_store.py` | Stores vectors, computes cosine similarity, returns top-k chunks |
| `src/rag.py` | Retrieve → Augment → Generate: builds the prompt and calls llama3.2 |
| `src/main.py` | Interactive chat loop |

## What I learned (the honest version)

- **Embeddings turn meaning into geometry.** "Dog chased cat" and "puppies and
  kittens playing" score 0.64 cosine similarity; "quantum entanglement" scores 0.33
  against the same sentence. I verified this myself before building anything else.
- **Cosine similarity is just 3 lines of NumPy** once you understand it:
  dot product divided by the lengths. Writing it myself instead of importing it
  made the whole pipeline stop being magic.
- **RAG answers are only as good as what you retrieve.** My favorite bug: asking
  *"Why did Victor create the monster?"* returns revenge-themed passages, so the LLM
  confidently explains Victor's revenge — even though his real motive was scientific
  ambition at university. Increasing k from 4 to 6 didn't help, because the problem
  was *which* chunks ranked top, not how many. The question's wording matches the
  revenge scene's emotional language better than the creation scene's. This is called
  query-context mismatch, and fixing it (query rewriting) is on my roadmap.
- **Small engineering lessons that stuck:** relative paths resolve from the terminal's
  working directory, not the script's location; imports inside `if __name__ == "__main__"`
  don't exist when another file imports the module (a `NameError` taught me this);
  and `.gitignore` mistakes are how secrets and big files leak onto GitHub — always
  read `git status` before committing.
- **Everything runs locally.** No API keys, no cloud, no cost. Privacy and
  independence by design.

## Known limitations

- Retrieval quality depends on how the question is phrased (see above).
- Chunks include table-of-contents pages and page headers — I left this noise in
  deliberately; embeddings are robust to it, and cleaning is a future exercise.
- The vector store is a NumPy array scanned linearly — perfect for learning, and
  I'll swap in ChromaDB to understand what real vector DBs add at scale.

## Roadmap

- [ ] Chunk-size experiment (200 vs 500 vs 1000) with documented results
- [ ] Streamlit web UI
- [ ] ChromaDB swap for the vector store
- [ ] Chat memory (follow-up questions)
- [ ] Query rewriting to fix the retrieval mismatch

## Tech stack

Python · NumPy · pypdf · Ollama (llama3.2 + nomic-embed-text) — all local, all free.
