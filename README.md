# chat-with-a-book
An AI tool that lets you chat with any book using RAG (Retrieval-Augmented Generation), embeddings, and cosine similarity — built from scratch with no frameworks

## Project Plan
- [X] Phase 0: Git & GitHub
- [X] Phase 1: Project setup
- [X] Phase 2: Load & chunk the book
- [X] Phase 3: Embeddings & vector search
- [X] Phase 4: RAG pipeline & chat
- [ ] Phase 5: Polish (in progress)

## How It Works

┌─────────┐   ┌──────────┐   ┌────────────┐   ┌──────────────┐
│ book.pdf │──▶│ chunker  │──▶│ embedder   │──▶│ vector store │
│ (raw)    │   │ (500 char│   │ (nomic-    │   │ (numpy +     │
│          │   │  chunks) │   │ embed-text)│   │ cosine sim)  │
└─────────┘   └──────────┘   └────────────┘   └──────────────┘
                                                      │
                 ┌──────────┐   ┌──────────┐          ▼
                 │  You ask │──▶│ embed    │   ┌──────────────┐
                 │ a question│  │ question │──▶│ top-k chunks │
                 └──────────┘   └──────────┘   └──────────────┘
                                                      │
                                                      ▼
                 ┌──────────┐   ┌───────────────────────────┐
                 │  Answer  │◀──│ llama3.2 (prompt = context│
                 │          │   │ + question)               │
                 └──────────┘   └───────────────────────────┘

## The Code

| File | What it does |
|---|---|
| `src/load_book.py` | Extracts raw text from the PDF using pypdf |
| `src/chunker.py` | Splits text into 500-char chunks with 50-char overlap so no sentence is cut in half |
| `src/embedder.py` | Converts chunks into 768-dimension vectors using `nomic-embed-text` via Ollama |
| `src/vector_store.py` | Stores vectors, computes cosine similarity, returns the k most similar chunks |
| `src/rag.py` | The pipeline: retrieve chunks → build prompt → generate answer with llama3.2 |
| `src/main.py` | Interactive chat loop |

## What I Learned

- Embeddings turn meaning into geometry: similar text → similar vectors
- Cosine similarity measures the angle between vectors = semantic closeness
- **RAG answers are only as good as retrieved chunks.** Example: asking
  "Why did Victor create the monster?" retrieves revenge-themed passages,
  so the LLM answers about revenge — even though the real motive was
  scientific ambition. Increasing k didn't help because the *relevant*
  chunks weren't in the top results. This is known as query-context mismatch.
- and so on...