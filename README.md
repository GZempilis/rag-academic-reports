# RAG System for Academic Reports Analysis

A Retrieval-Augmented Generation (RAG) pipeline for question answering over academic PDF documents. Combines **local embeddings** with a **remote LLM** to produce context-grounded answers with explicit source attribution, packaged in Docker for reproducibility.

---

## Features

- **Local embeddings** via `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — no embedding API costs, works offline
- **Persistent vector store** using ChromaDB with cosine similarity
- **Context-grounded answers** through the DeepSeek-R1-Distill-Qwen-7B model via HuggingFace Inference API
- **Hallucination guard** — the LLM is instructed to say "I don't know" when the context lacks the answer
- **Custom evaluation framework** using cosine similarity against a hand-crafted ground-truth dataset
- **Fully Dockerized** — reproducible across platforms
- **Interactive chat loop** with `exit` and `exit_eval` commands

---

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  PDF files  │ →  │  Chunking    │ →  │  Embeddings  │ →  │  ChromaDB   │
│  (raw_data) │    │  (800/100)   │    │  (local BGE) │    │  (cosine)   │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│   Answer    │ ←  │  DeepSeek    │ ←  │   Prompt     │ ←  │  Top-K      │
│             │    │  (HF API)    │    │  (ctx + Q)   │    │  retrieval  │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

**Pipeline stages:**
1. **Ingestion** — PDFs are loaded, split into chunks, embedded locally, and stored in ChromaDB
2. **Retrieval** — the user query is embedded with the same model and compared against stored vectors
3. **Generation** — the top-K chunks are injected into a constrained prompt and sent to the LLM
4. **Evaluation** — generated answers are compared against ground truth via cosine similarity

---

## Project Structure

```
.
├── src/
│   ├── config.py         # Central configuration and paths
│   ├── rag.py            # Ingestion + retrieval
│   ├── llm.py            # LLM wrapper (HuggingFace Inference API)
│   ├── main.py           # Interactive chat loop
│   └── evaluation.py     # Custom evaluation pipeline
├── raw_data/             # Source PDF documents
├── test/
│   └── Ground_truth.json # Evaluation dataset (15 Q&A pairs)
├── chroma_db/            # Vector store (generated, gitignored)
├── .hf_cache/            # HuggingFace model cache (generated, gitignored)
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Docker Desktop
- A HuggingFace account with an [access token](https://huggingface.co/settings/tokens)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/GZempilis/rag-academic-reports.git
   cd rag-academic-reports
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your HF_TOKEN
   ```

3. **Create the model cache directory**
   ```bash
   mkdir -p .hf_cache
   ```

4. **Build the Docker image**
   ```bash
   docker build -t my-rag-app -f Dockerfile .
   ```

---

## Usage

### Interactive chat

```bash
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/raw_data:/app/raw_data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/.hf_cache:/root/.cache/huggingface \
  my-rag-app python -m src.main
```

Commands inside the chat:
- Type a question to get an answer grounded in the PDFs
- Type `exit` to quit
- Type `exit_eval` to run the evaluation pipeline

### Run evaluation only

```bash
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/test:/app/test \
  -v $(pwd)/raw_data:/app/raw_data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/.hf_cache:/root/.cache/huggingface \
  my-rag-app python -m src.evaluation
```

Results are written to `test/results.csv` with per-question similarity scores.

### Test retrieval only

```bash
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/raw_data:/app/raw_data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/.hf_cache:/root/.cache/huggingface \
  my-rag-app python -m src.rag
```

---

## Configuration

All key parameters live in `src/config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | LLM for generation |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection name |

To change models, edit `.env` (no code changes needed).

---

## Evaluation

The project uses a **custom evaluation framework** instead of RAGAS to avoid external API costs and dependencies.

**Method:**
- A ground-truth dataset (`test/Ground_truth.json`) with 15 questions across 3 documents
- Answers generated by the RAG are compared to ground truth using **cosine similarity** on the same embedding model
- Scores are aggregated (mean, median, worst, best) and written to CSV

**Results (baseline run):**

| Metric | Value |
|:---|:---|
| Questions evaluated | 15 |
| Mean similarity | ~0.76 |
| Peak similarity | 0.87 |
| Worst case | 0.01 (retrieval failure on one factual question) |

> Note: The evaluation hit the HuggingFace Inference API free-tier limit (HTTP 402) after 9 questions. The scores above reflect the partial run.

---

## Tech Stack

| Component | Technology |
|:---|:---|
| PDF loading | `pypdf` + `langchain-community` |
| Text splitting | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Vector store | `chromadb` (PersistentClient, cosine) |
| LLM | `huggingface_hub.InferenceClient` |
| Data handling | `pandas`, `numpy` |
| Containerization | Docker (Python 3.11-slim, CPU-only torch) |

---

## Design Decisions

- **Local embeddings over API** — Zero cost, no rate limits, works offline, full control.
- **Custom evaluation over RAGAS** — RAGAS requires LLM-as-judge (more API calls) and defaults to OpenAI. Cosine similarity is fast, free, and interpretable.
- **Three-file separation** (`rag.py` / `llm.py` / `main.py`) — Each file has one responsibility, making the pipeline testable and easy to swap components.
- **Volume mounts over image baking** — The vector store, PDF data, and model cache live on the host, avoiding expensive rebuilds during development.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**George Zempilis**
- GitHub: [@GZempilis](https://github.com/GZempilis)
- Email: g.zempilis@gmail.com
