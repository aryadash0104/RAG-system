# RAG System

PDF loading, chunking, embedding, and vector store for Retrieval-Augmented Generation.

## Dependencies

```bash
pip install langchain langchain-community langchain-huggingface pypdf faiss-cpu sentence-transformers
```

## Modules

### pdf_chunker.py
Loads PDF and splits into chunks.

```bash
python pdf_chunker.py
```

### vector_store.py
Creates embeddings and stores in FAISS for similarity search.

```bash
python vector_store.py
```

## Configuration

| Parameter        | Default | Description                        |
|------------------|---------|------------------------------------|
| `chunk_size`     | 1000    | Maximum characters per chunk       |
| `chunk_overlap`  | 200     | Overlap between consecutive chunks |
| `model_name`     | all-MiniLM-L6-v2 | HuggingFace embedding model |

## Output

- `vector_store/` - Saved FAISS index and embeddings
