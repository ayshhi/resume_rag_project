# Resume Interview Assistant

A retrieval-augmented Streamlit app: upload a resume PDF, ask questions about it, and get
grounded answers plus a candidate summary, strengths/gaps, tailored interview questions, and
resume improvement suggestions — all generated from the resume's actual content, not generic
templates.

## How it works

1. **Extract**: `pdfplumber` (with a `pypdf` fallback and a word-spacing repair pass) pulls text
   out of the uploaded PDF.
2. **Chunk**: text is split into overlapping ~500-word chunks.
3. **Index**: chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a
   local FAISS index.
4. **Retrieve**: the top-matching chunks for a question are pulled from the index.
5. **Generate**: if an OpenRouter API key is configured, retrieved chunks are sent to an LLM
   (via [OpenRouter](https://openrouter.ai)) to produce a grounded answer, summary, strengths/gaps,
   interview questions, and suggestions. Without a key, the app falls back to local
   keyword/regex heuristics so it still works offline.

## Project layout

```
app.py                     Streamlit UI / entry point
resume_assistant/
  rag_pipeline.py           PDF extraction, chunking, FAISS index, OpenRouter calls, fallbacks
test_rag_pipeline.py        Unit tests for the offline/pure-logic pieces
requirements.txt
.env.example                 Template for required environment variables
OPENROUTER_SETUP.md          How to get and configure an OpenRouter API key
data/                        Generated FAISS index + chunks (gitignored, rebuilt per session)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

cp .env.example .env          # then edit .env and add your OPENROUTER_API_KEY
streamlit run app.py
```

See [OPENROUTER_SETUP.md](./OPENROUTER_SETUP.md) for how to get a key.

## Running tests

```bash
pytest test_rag_pipeline.py -v
```

Tests only cover offline logic (chunking, the glued-word repair heuristic, and the
no-API-key fallback paths) — they don't call OpenRouter or load the embedding model, so they
run fast and without a key or network access.

## Notes / known limitations

- Each question triggers up to 5 OpenRouter calls (answer + summary + strengths/gaps +
  interview questions + suggestions), which adds latency and cost. These could be collapsed
  into a single structured call if that becomes a priority.
- The FAISS index is rebuilt from scratch on each new PDF upload and lives only in
  `st.session_state` + `./data` for the current session.
- PDF extraction quality depends on how the source PDF encodes text; scanned/image-only PDFs
  won't extract any text and will show an error in the UI.