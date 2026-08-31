from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    PDF_PATH = "sample.pdf"

    docs = load_pdf(PDF_PATH)
    print(f"Loaded {len(docs)} pages from {PDF_PATH}")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:200])
