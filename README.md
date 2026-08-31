# PDF Chunker with LangChain

A simple script to load PDF files and split them into manageable chunks using LangChain.

## Dependencies

```bash
pip install langchain langchain-community pypdf
```

## Usage

1. Place your PDF file in the project directory
2. Update `PDF_PATH` in `pdf_chunker.py` if needed
3. Run the script:

```bash
python pdf_chunker.py
```

## Configuration

| Parameter        | Default | Description                        |
|------------------|---------|------------------------------------|
| `chunk_size`     | 1000    | Maximum characters per chunk       |
| `chunk_overlap`  | 200     | Overlap between consecutive chunks |

## Output

The script prints:
- Total pages loaded from the PDF
- Total number of chunks created
- Preview of the first 3 chunks
