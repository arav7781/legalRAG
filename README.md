# LegalRAG

Question answering over Indian court judgments. Give it links to judgment PDFs and a list of questions; it
downloads and indexes the documents once, retrieves the relevant passages with hybrid search, and answers each
question from them.

## How it works

```
PDF/DOCX URLs ─▶ download + extract (pdfplumber, docx2txt) ─▶ clean + chunk (1,000 chars, 100 overlap)
            ─▶ dense vectors (all-MiniLM-L6-v2) + sparse vectors (TF-IDF over a legal vocabulary)
            ─▶ Qdrant collection with named dense and sparse vectors
question ─▶ legal query expansion ─▶ dense search + sparse search ─▶ reciprocal rank fusion ─▶ LLM answer
```

- **Hybrid retrieval.** Each chunk is stored with a dense and a sparse vector. Queries run both searches and merge
  them with reciprocal rank fusion (dense weight 0.7, sparse 0.3), which keeps exact section numbers and case names
  from getting lost in semantic search.
- **Idempotent ingestion.** Documents are keyed by a content hash, so a URL that was already indexed is skipped.
- **Bounded work.** Up to 20 questions per request, 50 MB per document, 45 s download timeout with retries,
  batched upserts on a thread pool.
- **Auth.** The API is protected with an HTTP bearer token.

## API

`POST /legal/analyze` (bearer token)

```json
{
  "documents": "https://api.sci.gov.in/supremecourt/2019/3513/3513_2019_11_1501_59090_Judgement_04-Feb-2025.pdf",
  "questions": ["What offence was involved?", "What remedy did the respondent receive under Section 138 NI Act?"]
}
```

Response: `{ "legal_analysis": ["…", "…"] }`

## Repository layout

| Path | What |
| --- | --- |
| `api_main.py` | Main FastAPI service: ingestion, hybrid search, `/legal/analyze` (`main.py` is an earlier copy) |
| `backend/` | Agentic RAG variant in LangGraph: retrieve, grade the documents, rewrite the question if they are not relevant, then generate. Endpoints `/upload-pdf`, `/chat`, `/collections`, `/health` |
| `app.py` | Flask research service: case search through SerpAPI, page extraction with Firecrawl, analysis with Gemini, SQLite cache. Endpoints `/search`, `/analyze`, `/document/<id>`, `/news`, `/courts` |
| `frontend/` | Next.js research UI (`/legal_search`) |
| `test.py` | Load and correctness tester for the API |

## Run locally

```bash
docker run -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
ollama pull gemma3:1b
pip install fastapi uvicorn qdrant-client sentence-transformers pdfplumber docx2txt scikit-learn langchain langchain-community
export QDRANT_URL=http://localhost:6333
python api_main.py            # serves on $PORT (default 8000)
```

A deployed front end for this project lives in
[LegalSearch-CASE-FILES](https://github.com/arav7781/LegalSearch-CASE-FILES).

## Tech

Python · FastAPI · Qdrant · sentence-transformers · scikit-learn · LangChain · LangGraph · Ollama · Flask · Next.js
