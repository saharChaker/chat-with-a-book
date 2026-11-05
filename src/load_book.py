"""Load a book file and return its full text as a string."""
from pypdf import PdfReader


def load_book(path: str) -> str:
    """Read a PDF file and return all its text as one string."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


if __name__ == "__main__":
    # This block runs only when you execute the file directly (python src/load_book.py)
    # It's how we TEST the file without a real app around it.
    text = load_book("book/frankenstein.pdf")
    print(f"Total characters: {len(text)}")
    print("---- First 500 characters ----")
    print(text[:500])