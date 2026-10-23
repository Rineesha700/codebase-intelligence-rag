\# Codebase Intelligence RAG Assistant



A Retrieval-Augmented Generation (RAG) assistant for answering natural-language questions about the Rich Python codebase using source-code retrieval and grounded LLM generation.



\## Architecture



```text

Rich Python Repository

&#x20;       ↓

Python File Ingestion

&#x20;       ↓

Code Chunking + Line Metadata

&#x20;       ↓

Dense Embeddings ────── BM25

&#x20;       ↓                  ↓

&#x20;       └──── RRF Fusion ──┘

&#x20;                ↓

&#x20;         Top 20 Candidates

&#x20;                ↓

&#x20;      Lightweight Reranking

&#x20;                ↓

&#x20;            Top 5 Chunks

&#x20;                ↓

&#x20;         LLM Generation

&#x20;                ↓

&#x20;    Grounded Answer + Citations

&#x20;                ↓

&#x20;         Streamlit Interface

\## Key Features



\- Dense semantic retrieval using OpenAI embeddings

\- BM25 lexical retrieval

\- Reciprocal Rank Fusion (RRF)

\- Lightweight lexical reranking

\- Grounded LLM responses

\- Source-code citations with file and line ranges

\- Citation validation

\- Streamlit interface

\- Docker support



\## Dataset



The system indexes the `rich/rich` Python source directory from the Rich repository.



\- Python files: 100

\- Code chunks: 842



\## Evaluation



A fixed benchmark of 10 codebase questions was used to compare retrieval approaches.



| Retrieval Method | Recall@5 |

|---|---:|

| Dense retrieval | 80% |

| Dense + BM25 + RRF | 90% |

| Hybrid + reranking | \*\*100%\*\* |



\## Example



\*\*Question\*\*



> How does Rich render text objects?



The system retrieves relevant source-code chunks from `text.py`, reranks the candidates, and generates a grounded explanation with source citations such as:



```text

\[text.py:689-705]

\[text.py:707-717]

\## Tech Stack



\- Python

\- LangChain

\- OpenAI Embeddings

\- OpenAI LLM

\- ChromaDB

\- BM25

\- Streamlit

\- Docker



\## Setup



\### 1. Clone the repository



```bash

git clone <your-github-repository-url>

cd codebase-intelligence-rag
### 2. Create a virtual environment



```bash

python -m venv .venv

\### 3. Install dependencies



```bash

pip install -r requirements.txt

\### 4. Configure the OpenAI API key



Copy `.env.example` to `.env` and add your API key:



```text

OPENAI\_API\_KEY=your\_openai\_api\_key\_here
### 5. Run the application



```bash

streamlit run app.py

## Project Structure



```text

codebase-intelligence-rag/

├── app.py

├── Dockerfile

├── requirements.txt

├── .env.example

├── .gitignore

├── src/

│   ├── \_\_init\_\_.py

│   ├── ingestion.py

│   ├── chunking.py

│   ├── vectorstore.py

│   ├── hybrid\_retrieval.py

│   ├── reranker.py

│   ├── generator.py

│   └── citation.py

├── evaluation/

│   ├── benchmark.json

│   ├── evaluate\_baseline.py

│   ├── evaluate\_hybrid.py

│   └── evaluate\_reranker.py

└── data/

&#x20;   └── repository/

&#x20;       └── rich/

## Evaluation Scripts



The retrieval experiments can be reproduced using:



```bash

python evaluation/evaluate\_baseline.py

python evaluation/evaluate\_hybrid.py

python evaluation/evaluate\_reranker.py

## Future Improvements



\- Cross-encoder reranking

\- Larger evaluation benchmark

\- Support for additional programming languages

\- More advanced code-structure retrieval

