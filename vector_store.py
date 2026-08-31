import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_chunker import load_pdf, chunk_documents


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, embeddings


def save_vector_store(vector_store, path="vector_store"):
    vector_store.save_local(path)
    print(f"Vector store saved to {path}/")


def load_vector_store(path="vector_store", model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from {path}/")
    return vector_store, embeddings


def search_similar(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    PDF_PATH = "sample.pdf"
    STORE_PATH = "vector_store"

    docs = load_pdf(PDF_PATH)
    print(f"Loaded {len(docs)} pages")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vector_store, _ = create_embeddings(chunks)
    print("Embeddings created")

    save_vector_store(vector_store, STORE_PATH)

    query = "What is this document about?"
    results = search_similar(vector_store, query, k=3)
    print(f"\nQuery: {query}")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(res.page_content[:200])
