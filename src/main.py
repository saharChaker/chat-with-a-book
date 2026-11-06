"""Chat with the book: ask questions in a loop until you type 'quit'."""
from load_book import load_book
from chunker import chunk_text
from vector_store import VectorStore
from rag import answer_question


def main() -> None:
    print("Loading index...")
    chunks = chunk_text(load_book("book/frankenstein.pdf"))
    store = VectorStore()
    store.load(chunks)
    print("Ready! Ask anything about Frankenstein. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not question:
            continue
        print("Book:", answer_question(store, question), "\n")


if __name__ == "__main__":
    main()