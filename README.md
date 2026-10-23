# Codebase Intelligence RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for answering natural-language questions about the Rich Python codebase using source-code retrieval and grounded LLM generation.

## Architecture

```text
Rich Python Repository
        |
        v
Python File Ingestion
        |
        v
Code Chunking + Line Metadata
        |
        v
+-------------------+-------------------+
| Dense Embeddings  | BM25 Keyword Search|
+-------------------+-------------------+
        |                   |
        +--------+----------+
                 |
                 v
            RRF Fusion
                 |
                 v
          Top 20 Candidates
                 |
                 v
       Lightweight Reranking
                 |
                 v
             Top 5 Chunks
                 |
                 v
          LLM Generation
                 |
                 v
     Grounded Answer + Citations
                 |
                 v
        Citation Validation
                 |
                 v
        Streamlit Interface
```

## Key Features

- Dense semantic retrieval using OpenAI embeddings
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Lightweight lexical reranking
- Grounded LLM responses
- Source-code citations with file and line ranges
- Citation validation
- Streamlit interface
- Docker support

## Dataset

The system indexes the `rich/rich` Python source directory from the [Rich repository](https://github.com/Textualize/rich).

- **Python files:** 100
- **Code chunks:** 842

## Evaluation

A fixed benchmark of 10 codebase questions was used to compare retrieval approaches.

| Retrieval Method | Recall@5 |
|---|---:|
| Dense retrieval | 80% |
| Dense + BM25 + RRF | 90% |
| Hybrid + reranking | **100%** |

These results apply to the project's 10-question benchmark and do not guarantee the same performance on other codebases or questions.

## Example

**Question:** How does Rich render text objects?

The system retrieves relevant source-code chunks, reranks the candidates, and generates an answer grounded in the retrieved code.

Example citation format:

```text
[text.py:689-705]
[text.py:707-717]
```

## Tech Stack

- Python
- LangChain
- OpenAI embeddings and LLM
- ChromaDB
- BM25
- Streamlit
- Docker

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/Rineesha700/codebase-intelligence-rag.git
cd codebase-intelligence-rag
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

Copy `.env.example` to `.env` and add your own API key to the `.env` file.

The `.env` file is private and should never be committed to GitHub.

### 5. Prepare the Rich codebase

The application expects the Rich source code at:

```text
data/repository/rich/rich/
```

If the source code is not already present, clone the upstream repository:

```bash
git clone --depth 1 https://github.com/Textualize/rich.git data/repository/rich
```

### 6. Run the application

```bash
streamlit run app.py
```

## Project Structure

```text
codebase-intelligence-rag/
├── app.py
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── hybrid_retrieval.py
│   ├── reranker.py
│   ├── generator.py
│   └── citation.py
├── evaluation/
│   ├── benchmark.json
│   ├── evaluate_baseline.py
│   ├── evaluate_hybrid.py
│   └── evaluate_reranker.py
└── data/
    └── repository/
        └── rich/
```

## Evaluation Scripts

Run the retrieval experiments from the project root:

```bash
python evaluation/evaluate_baseline.py
python evaluation/evaluate_hybrid.py
python evaluation/evaluate_reranker.py
```

## Future Improvements

- Cross-encoder reranking
- A larger evaluation benchmark
- Support for additional programming languages
- More advanced code-structure retrieval

## License and Attribution

This project uses the [Rich Python library](https://github.com/Textualize/rich). Rich is distributed under the MIT License. See the upstream repository for its license and attribution details.