# Aurora Labs knowledge assistant

A retrieval augmented generation service that answers questions about a set of
documents and shows its work. Every answer carries the source document, the
exact location of the supporting text and a verbatim excerpt, and the service
says so plainly when the answer is not in the documents at all.

The system is deliberately strict about one thing: it would rather return
`The requested information is unavailable in the provided documents.` than
invent a plausible answer. That property is enforced by code before the language
model is ever called, not by asking the model nicely.

## Contents

1. [What this is](#what-this-is)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [API usage](#api-usage)
5. [Architecture](#architecture)
6. [Architectural decisions](#architectural-decisions)
7. [Evaluation](#evaluation)
8. [Testing](#testing)
9. [Limitations](#limitations)
10. [Deliverables map](#deliverables-map)

---

## What this is

A FastAPI service that ingests PDF, TXT and Markdown documents into a Chroma
vector store and answers questions about them using Gemini, with evidence
attached to every answer.

Against the assignment brief:

| Requirement | Status |
|---|---|
| Ingest PDF, TXT and Markdown into chunks | Met, 4 sample documents covering all three formats |
| Semantic search with embeddings and a vector store | Met, Gemini embeddings at 768 dimensions, Chroma with cosine distance |
| Answers grounded in retrieved content | Met, strict prompt, structured output and citation validation |
| Evidence with document names and excerpts | Met, verbatim excerpts with page, chunk and character offsets |
| State that information is unavailable rather than guessing | Met, deterministic relevance gate plus post-generation validation |
| Ingestion and question answering endpoints | Met, five endpoints |
| Evaluation questions across answerable, unanswerable and multi evidence categories | Met and exceeded, 30 questions at 10 per category |
| Python with a clean modular architecture | Met, five flat modules |
| Environment variables for keys and configuration | Met, `.env` with 13 variables |
| Tests for retrieval, API behaviour and failure cases | Met, 110 tests, three suites, fully offline |
| Docker support and setup documentation | Met, Dockerfile and compose service |
| An existing LLM with no training | Met, Gemini free tier |

---

## Setup

You need a [Gemini API key](https://aistudio.google.com/apikey). Everything else
is installed for you. There are no model weights to download, because the model
is a remote service.

### Docker, recommended

```bash
cp .env.example .env
# edit .env and paste your key into GEMINI_API_KEY
docker compose up --build
```

Then open <http://localhost:8000/docs> and use the interface, or drive it with
curl as shown below.

Ingest the sample documents. The recommended order is through the HTTP API while
the service is running:

```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@data/documents/employee_handbook.pdf"
```

Re-uploading a file with the same name replaces the previous version rather than
adding a duplicate, so it is safe to run again.

#### Ingestion order matters, read this once

The service holds one Chroma handle for the lifetime of the process. If a second
container writes to the index underneath it, every later query fails with an
`Internal error: Error creating hnsw segment reader` until the service
restarts. Three safe orders exist:

1. **Ingest through the HTTP API while the service runs.** The write happens in
   the process that holds the handle. This is the recommended path.
2. **Run the ingest script first, start the service afterwards:**

   ```bash
   docker compose run --rm app python scripts/ingest_samples.py
   docker compose up -d
   ```

3. **Run the script while the service is up, then restart it immediately:**

   ```bash
   docker compose run --rm app python scripts/ingest_samples.py
   docker compose restart app
   ```

Do not combine options 2 or 3 without the restart. That failure mode is real and
has bitten this project during verification.

### Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS and Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env      # Windows: copy .env.example .env
# add your key to .env
uvicorn app.main:app --reload
```

The service starts on <http://localhost:8000>. Interactive documentation is at
`/docs`, and the machine readable schema is at `/openapi.json`.

### Commands

| Command | What it does |
|---|---|
| `uvicorn app.main:app --reload` | Run the service locally |
| `python -m pytest -q` | Offline test suite, no key and no network needed |
| `python -m pytest -q --cov=app` | The same with coverage |
| `python -m pytest -m live` | Optional live smoke test, needs a key |
| `python scripts/ingest_samples.py` | Ingest the four sample documents |
| `python scripts/run_eval.py` | Run all 30 evaluation questions |
| `python scripts/run_eval.py --sweep` | Threshold sweep, no generation calls |
| `python scripts/make_sample_pdfs.py` | Regenerate the sample PDF |

---

## Configuration

Settings load once through `pydantic-settings` from the environment and from
`.env`. Every variable appears in `.env.example`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes at run time | none | Key from Google AI Studio. Gates chat and embeddings. |
| `GEMINI_CHAT_MODEL` | No | `gemini-3.1-flash-lite` | Answer generation model. |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embedding model. |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma` | Index location, mounted as a volume in Docker. |
| `CHUNK_SIZE` | No | `800` | Maximum chunk length in characters. |
| `CHUNK_OVERLAP` | No | `150` | Overlap between consecutive chunks. |
| `RETRIEVAL_TOP_K` | No | `8` | Chunks retrieved per question. |
| `RELEVANCE_MAX_DISTANCE` | No | `0.33` | Relevance gate threshold and evidence admission. |
| `LLM_TEMPERATURE` | No | `0.0` | Zero keeps evaluation runs repeatable. |
| `API_MAX_RETRIES` | No | `3` | Retries on rate limit and server errors, backing off 1, 2 then 4 seconds. |
| `HOST` | No | `0.0.0.0` | Uvicorn binding. |
| `PORT` | No | `8000` | Uvicorn binding. |
| `LANGSMITH_TRACING` | No | `false` | No call data leaves the machine. |

**Fixed constants.** These are not environment variables, because each knob is
one more thing to misconfigure and nothing here needs tuning:

| Constant | Value |
|---|---|
| Embedding dimensions | 768 |
| Collection name | `knowledge_docs` |
| Maximum upload size | 20 MB |
| Maximum output tokens | 1024 |
| Per request timeout | 30 seconds |
| Unavailable message | `The requested information is unavailable in the provided documents.` |

**A missing key does not stop the service.** `/healthz` reports `degraded`, and
the endpoints that need Gemini return 503 with an actionable message. Listing,
deleting, the health check and the interactive documentation all keep working, so
the service can be inspected before any secret exists.

**Changing the embedding model or the dimension requires re-indexing.** The
collection is created with a fixed dimensionality and a fixed distance space.
Delete `CHROMA_PERSIST_DIR` and ingest again, or queries will fail against a
mismatched index.

---

## API usage

Base path `/api`. Errors are JSON objects with a `detail` field. Interactive
documentation is at `/docs`.

| Method and path | Purpose | Success | Errors |
|---|---|---|---|
| `POST /api/documents` | Ingest a multipart file in the `file` field | 201 | 415 unsupported type, 413 too large, 422 empty or unreadable, 503 model service down |
| `GET /api/documents` | List ingested documents | 200 | none |
| `DELETE /api/documents/{id}` | Remove a document and its chunks | 204 | 404 unknown id |
| `POST /api/ask` | Ask a question | 200 | 422 validation, 503 model unavailable, 502 malformed output after retry |
| `GET /healthz` | Readiness | 200 | 503 if the store is unreachable |

### Ingest

```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@data/documents/benefits_guide.md"
```

```json
{
  "document_id": "benefits_guide",
  "filename": "benefits_guide.md",
  "media_type": "text/markdown",
  "chunk_count": 16
}
```

The `document_id` is a slug of the filename without its extension. It is stable
across re-ingestion on purpose, because evidence locations and delete requests
reference it, and a random identifier would break that continuity.

Ingestion embeds before it deletes. If embedding fails part way through, for
example because the daily free tier quota is spent, the previously indexed
version of the document is left intact rather than destroyed on the way to a
failure.

### List

```bash
curl http://localhost:8000/api/documents
```

```json
{
  "documents": [
    {"document_id": "benefits_guide", "filename": "benefits_guide.md", "chunk_count": 16},
    {"document_id": "employee_handbook", "filename": "employee_handbook.pdf", "chunk_count": 20},
    {"document_id": "it_security_policy", "filename": "it_security_policy.txt", "chunk_count": 17},
    {"document_id": "remote_work_policy", "filename": "remote_work_policy.md", "chunk_count": 15}
  ]
}
```

Those counts are what the four committed sample documents produce at the default
chunk size, 68 chunks in total.

### Delete

```bash
curl -X DELETE http://localhost:8000/api/documents/benefits_guide
```

Returns 204 with no body. Deleting an identifier that is not indexed returns 404.

### Ask

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much paid time off does a full time employee accrue?"}'
```

The body takes a `question` of 1 to 2000 characters and an optional `top_k`
between 1 and 20.

```json
{
  "answer": "Every regular full time employee accrues paid time off at a rate of ten hours for each full month worked, which is one hundred and twenty hours, or fifteen days, across a full year. Accrual begins on the first day of employment [1].",
  "answerable": true,
  "sources": [
    {
      "document_name": "employee_handbook.pdf",
      "document_id": "employee_handbook",
      "excerpt": "Pay adjustments take effect from the first of the month following the review and are announced to the\nemployee in a one to one, never by email. An employee who disagrees with a rating may ask for it to be\nreconsidered within ten working days of receiving it, and People Operations reviews the case.\n4. Paid time off\nEvery regular full time employee accrues paid time off at a rate of ten hours for each full month worked, which\nis one hundred and twenty hours, or fifteen days, across a full year. Pa",
      "location": {
        "page_number": 2,
        "chunk_index": 6,
        "char_start": 600,
        "char_end": 1356
      },
      "score": 0.7
    }
  ],
  "retrieved_count": 3,
  "evidence_count": 1,
  "distinct_document_count": 1,
  "model": "gemini-3.1-flash-lite",
  "latency_ms": 3936.0
}
```

**About this example.** The evidence fields are real and reproducible. The
excerpt is a verbatim slice of chunk 6 of the committed `employee_handbook.pdf`,
at page 2, characters 600 to 1356, matching a recorded live run. You can
reproduce it exactly by loading that file with the committed splitter and
printing the chunk with `chunk_index` 6. The `answer` text shows the shape of a
grounded response and the citation format; the wording is illustrative rather
than a captured transcript.

Four details in that response matter more than they look:

- **The excerpt is never written by the model.** It is sliced from the retrieved
  chunk text outside the generation path, then capped at 500 characters. Search
  it in the PDF and you will find it byte for byte. This is the single most
  important property of the whole design.
- **`char_end` describes the whole chunk, not the excerpt.** The excerpt is
  truncated at 500 characters while the location still addresses the full chunk,
  so `char_end` can be greater than the excerpt length. Both numbers are honest
  about different things.
- **For PDFs, offsets are relative to the page text, not the whole file.** Each
  page is split independently, so the location always reports the page number
  alongside the offsets. For TXT and Markdown there is no `page_number` field.
- **`retrieved_count` can be positive while `sources` is empty.** That is the
  correct shape for a refused question, not a bug. Retrieval ran and returned
  nearby chunks, the gate rejected all of them, and the model was never called.

### When the answer is not available

```json
{
  "answer": "The requested information is unavailable in the provided documents.",
  "answerable": false,
  "sources": [],
  "retrieved_count": 2,
  "evidence_count": 0,
  "distinct_document_count": 0,
  "model": "gemini-3.1-flash-lite",
  "latency_ms": 4.1
}
```

The `answer`, `answerable` and `sources` values above are the contract, not an
example of it: the message is fixed, the flag is always false and the source
list is always empty. `retrieved_count` and `latency_ms` vary per question, and
the values shown are illustrative. A `retrieved_count` of zero means the gate
stopped the question outright; a higher count means chunks were returned, the
gate rejected them, and the model still declined to invent an answer.

That sentence is a constant in `app/config.py`, not something the model
produced. It is the same string whether the model abstained, the gate rejected
everything, the key was missing, or the quota was spent.

---

## Architecture

```
  HTTP request
      |
      v
  +------------------------------------------------------------+
  | app/main.py    routes for documents, ask and health.         |
  |                Validates input and maps errors to codes.     |
  +------------------------------------------------------------+
      |
      v
  +------------------------------------------------------------+
  | app/rag.py     Service layer. Runs ingest and ask flows,     |
  |                applies the grounding rules, builds evidence. |
  +------------------------------------------------------------+
      |
      v
  +------------------------------------------------------------+
  | app/chains.py  LangChain factories and the answer runnable.  |
  +------------------------------------------------------------+
      |
      v
  Gemini API for chat and embeddings, Chroma on disk
```

The application is five modules. There is no nested route package, no `core`
package and no separate ingestion, retrieval or generation packages, because
LangChain already supplies those layers.

```
knowledge-assistant/
|-- app/
|   |-- main.py            FastAPI app, routes, exception handlers
|   |-- config.py          environment settings and fixed constants
|   |-- schemas.py         request and response models
|   |-- chains.py          LangChain factories and the answer runnable
|   `-- rag.py             ingest, ask, list, delete, grounding and evidence
|-- data/
|   |-- documents/         4 medium length sample documents
|   |-- questions.json     30 evaluation questions
|   `-- chroma/            persisted index, ignored by git
|-- scripts/
|   |-- ingest_samples.py  loads the sample documents into the index
|   |-- run_eval.py        runs the 30 questions and writes the results file
|   `-- make_sample_pdfs.py  regenerates the sample PDF
|-- tests/
|   |-- conftest.py        fakes, temporary store, client fixtures
|   |-- test_retrieval.py  retrieval behaviour
|   |-- test_api.py        API behaviour
|   `-- test_failures.py   failure cases
|-- docs/
|   |-- test_results.md    captured pytest output
|   `-- evaluation_results.md  per category evaluation metrics
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

### Ingest flow

A multipart upload arrives. The extension and size are validated. The bytes go
to a temporary file, because LangChain loaders read from paths. The loader for
the extension returns one `Document` per page for PDF and one for TXT and
Markdown. The splitter turns them into chunks and records exact character
offsets. The new chunks are embedded, the superseded chunks for the same
document are deleted, and the new chunks are added. The response returns the
document identifier, filename, media type and chunk count.

### Ask flow

The question is embedded with the retrieval query task type and the store
returns the closest chunks with their raw cosine distances. Scores are
`1 - distance`. The relevance gate keeps only chunks at or below
`RELEVANCE_MAX_DISTANCE`. When nothing passes, the response is the canonical
unavailable message and the model is never called.

Otherwise numbered context blocks are built from the surviving chunks, the
answer runnable is invoked, citations are validated against the blocks that were
actually sent, and the response carries verbatim evidence. Evidence is taken
from the chunks on disk, outside the model path.

### Grounding, in three steps

1. **Relevance gate.** If no chunk is at or below the threshold, return the
   unavailable message without calling the model. This is deterministic and
   free.
2. **Strict prompt at zero temperature with structured output.** The model is
   instructed to abstain and constrained to an `Answer` schema with `answer`,
   `answerable` and `citations` fields.
3. **Citation validation after generation.** Any citation number not present in
   the context that was sent is dropped. If the model claimed an answer but no
   valid citation survives, the response is demoted to unavailable. Model
   abstention wording is normalised to the canonical message.

The final `answerable` value is the result of all three.

### LangChain components

| Need | Component | Notes |
|---|---|---|
| PDF parsing | `PyPDFLoader` | One `Document` per page, zero based page value in metadata, converted to one based. No OCR. |
| TXT and Markdown parsing | `TextLoader` | Read with `autodetect_encoding=True`. Headings stay in the text and act as retrieval signal. |
| Chunking | `RecursiveCharacterTextSplitter` | Size 800, overlap 150, `add_start_index=True`, separators on paragraph, line, sentence, word then character. |
| Embeddings | `GoogleGenerativeAIEmbeddings` | Task types `retrieval_document` and `retrieval_query`, `output_dimensionality=768`, 30 second timeout. |
| Vector store | `Chroma` | Persist directory, collection `knowledge_docs`, `hnsw:space` set to cosine. |
| Retrieval | `Chroma.similarity_search_with_score` | Returns the raw cosine distance, which the gate needs. |
| Prompt | `ChatPromptTemplate` | System and human parts. |
| Model | `ChatGoogleGenerativeAI` | Temperature 0, 1024 output tokens, retries from `API_MAX_RETRIES`. |
| Structured output | `with_structured_output` | Returns the validated `Answer` model, replacing manual JSON parsing. |
| Composition | `RunnablePassthrough`, `RunnableLambda` | Composes prompt, model and parser into one runnable. |

LangChain legacy chain classes are not used. In LangChain 1.x they live in a
separate package and are being retired in favour of runnable composition.

Each component is built by a small factory in `chains.py` that accepts an
alternative object: `build_loader`, `build_splitter`, `build_embeddings`,
`build_store`, `build_answer_chain`. That is what lets the test suite swap in
deterministic fakes without touching a route.

---

## Architectural decisions

**LangChain components instead of hand written loaders, splitters, embeddings
and store wrappers.** The built-in components replace roughly thirteen modules
that would each need their own tests, and they are the standard way to do this.
Rejected: the hand written layer set, which was larger and needed more tests.
LlamaIndex solves the same problem and only one of the two is needed.
LangGraph arrives as a transitive dependency but is only justified by multi hop
or agent reasoning, which this project does not have.

**Gemini through `langchain-google-genai` for both chat and embeddings, with one
key.** No model weights in the image keeps Docker smaller, and one key removes a
second failure mode. Rejected: local Ollama, which needs a multi gigabyte
download and weaker small models; the OpenCode Zen free tier, which answers with
403 when called from outside OpenCode; and OpenAI, which publishes no free model.

**Chroma through `langchain-chroma`, embedded.** No extra container, cosine
distance built in, and metadata filtering that makes delete by document a one
line call. Rejected: Qdrant or Postgres with pgvector, which add a service to
run and to operate, and FAISS, which needs manual persistence and metadata
handling.

**LangChain Expression Language with structured output for generation, while
retrieval and gating stay explicit calls.** The prompt, model and output parser
compose into a single runnable, and `with_structured_output` returns a validated
pydantic object, so a malformed answer is impossible by construction. Retrieval
stays an explicit call because the gate needs the raw distance and must be able
to skip the model entirely. Rejected: a ready made question answering chain
class, which hides both the distance and the gate; a graph based orchestration
layer that this linear pipeline does not need; and manual JSON parsing with
repair logic, which adds failure modes and code.

**One small factory per LangChain component instead of provider interface
classes.** The factories accept alternative objects, which is what makes the
offline suite deterministic and lets a reviewer swap an implementation without
touching a route. Interface classes were judged unnecessary because the
LangChain base classes already define the contract.

**Three grounding steps and one distance threshold.** The gate abstains
deterministically before any model call, the strict prompt and structured output
constrain the model, and citation validation guarantees that every citation
number exists in the context that was sent. Rejected: two tier thresholds and a
minimum source count rule, both removed to save time, with multi document
correctness asserted directly in the evaluation instead.

**A flat five module application.** LangChain already supplies the ingestion,
retrieval and generation layers, so nested packages would only add navigation
cost. Rejected: a layered package tree where import depth exceeded the logic in
each file.

---

## Evaluation

Thirty questions, ten in each of three categories, run through the real
ingestion, retrieval and generation path.

```bash
python scripts/ingest_samples.py     # once, to populate the index
python scripts/run_eval.py           # writes docs/evaluation_results.md
```

The runner creates a fresh temporary store, ingests the corpus, asks all thirty
questions and scores the answers deterministically. No model judges any answer,
which keeps the run repeatable. Rate limits are retried and calls are paced at
thirteen seconds, because the application's own backoff of one, two and four
seconds cannot clear a free tier retry delay of forty seconds.

**A full run needs about 30 generation calls.** Check the daily quota in AI
Studio first.

### How the question set is designed

- **Answerable, 10.** Fully answered by one document.
- **Unanswerable, 10.** The fact is absent from the whole corpus, but the
  subject matter is close enough that retrieval will return chunks. This is what
  forces the gate and the abstention path to do real work rather than passing
  trivially on an empty result set.
- **Multi evidence, 10.** Each needs at least two documents, using deliberate
  cross references. Parental leave length lives in `benefits_guide.md` while the
  notification path lives in `employee_handbook.pdf`. The reimbursable home
  office equipment list lives in `remote_work_policy.md` while the expense
  limits and approval chain live in `employee_handbook.pdf`. The device
  enrolment deadline lives in `it_security_policy.txt` while new hire equipment
  readiness lives in `employee_handbook.pdf`.

The corpus is the internal knowledge base of a fictional company, Aurora Labs, so
no correct answer can arrive from general world knowledge. That removes a whole
class of false positive from the scoring.

### Results

Recorded on 2026-10-13 with `gemini-3.1-flash-lite` and `gemini-embedding-001`
at 768 dimensions, chunk size 800 and overlap 150, top k 8, threshold 0.33,
temperature 0, over 68 indexed chunks. Full detail is in
[`docs/evaluation_results.md`](docs/evaluation_results.md).

| Metric | Target | Result |
|---|---|---|
| Retrieval hit rate, answerable | at least 80 percent | 10 of 10, 100 percent |
| Retrieval hit rate, multi evidence | at least 80 percent | 10 of 10, 100 percent |
| Retrieval hit rate, overall | at least 80 percent | 20 of 20, 100 percent |
| Answerable accuracy, all gold keywords present | at least 70 percent | 9 of 10, 90 percent |
| Unanswerable accuracy, canonical abstention | 10 of 10 | 10 of 10, 100 percent |
| Hallucination rate on unanswerable questions | 0 | 0 |
| Multi evidence rate | at least 70 percent | 7 of 10, 70 percent |
| Citation validity, verbatim excerpts | 100 percent | 34 of 34, 100 percent |

The multi evidence rate is reported on its own line and is not folded into the
overall retrieval number, so a collapse in multi document answers cannot hide
behind a strong answerable score.

Retrieval returned at least one gated chunk for 6 of the 10 unanswerable
questions, so those abstentions were produced by the model and the citation
rules rather than by an empty result set. That is the harder and more meaningful
half of the behaviour.

### The calibrated threshold

`RELEVANCE_MAX_DISTANCE` is set to **0.33**, chosen from a retrieval only sweep
that spent no model calls. Each candidate was applied to cached cosine distances
for all thirty questions.

| Threshold | Unanswerable stopped by gate | Multi evidence keeping both gold docs |
|---|---|---|
| 0.30 | 9/10 | 7/10 |
| 0.32 | 6/10 | 9/10 |
| **0.33** | **4/10** | **10/10** |
| 0.35 | 3/10 | 10/10 |
| 0.38 | 0/10 | 10/10 |

Below 0.32 the gate rejects one multi evidence question outright, because its
best matching chunk sits at a distance of 0.3204. Above 0.33 nothing improves,
and by 0.38 the gate rejects nothing at all, so every unanswerable question
reaches the model and the deterministic layer stops protecting anything. 0.33 is
the point where the gate does the most work it can without costing a multi
evidence question.

You can reproduce the sweep without spending quota:

```bash
python scripts/run_eval.py --sweep
```

---

## Testing

```bash
python -m pytest -q                    # offline default
python -m pytest -q --cov=app          # with coverage
python -m pytest -m live               # optional, needs a key
```

The default suite needs no network access and no API key. `conftest.py` provides
fake embeddings built on the LangChain `Embeddings` interface, using
deterministic hashed vectors, and a stub answer runnable built with
`RunnableLambda`. Both are injected through the same factories used in
production, and every test uses a temporary store directory.

**110 passed, 1 skipped, 95 percent statement coverage.** The skip is the live
smoke test, which is skipped automatically without a key. Counts break down as
42 in `test_retrieval.py`, 33 in `test_api.py` and 35 in `test_failures.py`.

| Suite | Covers |
|---|---|
| `test_retrieval.py` | Rank one retrieval, score mapping from distance, the gate in both directions, delete and replace, chunk size, overlap and offset invariants, verbatim evidence and locations |
| `test_api.py` | The full endpoint and status matrix, response schemas, ask validation, health states, `/docs` and the OpenAPI schema |
| `test_failures.py` | Rejected uploads, model outage, retries exhausted and recovered, malformed output, missing key, a failed re-ingest leaving the previous version intact, grounding guarantees, the prompt contract |

Captured output is in [`docs/test_results.md`](docs/test_results.md).

**What this suite does not prove.** The fake embedder hashes words into vectors,
so it rewards word overlap rather than meaning. It therefore verifies mechanics:
ranking, thresholds, offsets, deletion, evidence and status codes. Whether the
assistant answers a *paraphrased* question correctly is measured by the live
evaluation run and nowhere else. Where a test needs to reach a specific chunk, it
builds the query from that chunk's own words for exactly this reason.

---

## Limitations

| Limitation | Mitigation |
|---|---|
| Gemini free tier rate limits, and quotas vary by project | Retries with exponential backoff, batched embeddings through the LangChain wrapper, and a gate that avoids the model call for unanswerable questions |
| Ingestion is expensive on the free tier. Three full ingestions of the four sample documents on one day exhausted the daily embedding quota of 1000 requests, and a fourth attempt failed with a retry delay of ten hours | The backoff cannot clear a delay that long. Ingest once and expect 503 with an actionable message when the quota is spent. Ingestion is not idempotent in cost, so re-ingesting to "refresh" is not free |
| The generation model is a flash lite variant, chosen because the free tier quota for the stronger flash model is smaller than a single evaluation run needs | Both identifiers are environment variables. This is the most likely reason the multi evidence rate sits exactly on target rather than above it |
| The multi evidence rate is 7 of 10, exactly on the 70 percent target with no margin | Both failures are citation discipline, not retrieval. One question cited a single document for a two document question; another abstained with eight relevant chunks in context. A stronger model is expected to close both. The category is reported on its own line so the margin is visible |
| The default chat model may be retired for new accounts, as `gemini-2.5-flash` already is | Every Google model identifier moves. `GEMINI_CHAT_MODEL` is the single knob |
| The embedding model identifier may be superseded | Confirm the current free tier identifier in AI Studio before the first run. A model or dimension change requires deleting the persist directory and re-ingesting |
| A single threshold that is badly calibrated produces either false unavailable answers or too much context | Calibrated against the 30 question set with a retrieval only sweep that cost no model calls, and recorded at 0.33 with its reasoning |
| `PyPDFLoader` has no optical character recognition, and layout parsing is limited for columns and tables | Stated plainly. The sample PDF keeps a simple single column layout |
| Character offsets in PDF chunks are relative to the page, not the file | Evidence reports the page number together with the offsets inside that page |
| The model may paraphrase evidence in the answer text | Excerpts are taken from the retrieved chunks outside the model path, and the answer text is a separate field from the sources list |
| The embedded store is single process and has no authentication | Out of scope for an assignment deliverable |
| The network is required at run time | 503 with an actionable message, retries, and a test suite that runs fully offline |
| `langchain-community` reports that it is being sunset in favour of standalone integration packages | `PyPDFLoader` and `TextLoader` remain the supported path today. If they move, the change is confined to `build_loader` in `app/chains.py`, the single place that imports them |
| Hashed fake embeddings cannot measure semantic similarity | Offline tests verify mechanics only. Semantic quality comes from the live evaluation run |
| LangChain packages move quickly | Eight packages are pinned with `==`, including `langchain-core` because the test fakes are built on its base classes. An upgrade is a deliberate pin change followed by a full test run |

### Troubleshooting

**Every query returns 500 with `Error creating hnsw segment reader: Nothing
found on disk`.** A second process wrote to the index while the service was
running. See the ingestion order notes above, then restart the service.

**503 with a quota message.** The daily free tier quota is spent. This is
expected and is not a defect. The service stays inspectable; listing, deleting
and `/healthz` all keep working, and unanswerable questions still return the
canonical message without any model call.

**Results look wrong after changing the embedding model.** Delete
`CHROMA_PERSIST_DIR` and ingest again. The existing collection keeps its original
dimension and distance space.

**`PermissionError` on the pytest temporary directory on Windows.** A stale
`pytest-of-<user>` folder in the system temp directory can become unreadable.
Point pytest elsewhere with `--basetemp`:

```bash
python -m pytest -q --basetemp ./.pytest-tmp
```

---

## Deliverables map

| Deliverable | File |
|---|---|
| D1 Git repository | This repository, 10 commits |
| D2 README with setup and decisions | [`README.md`](README.md), this file |
| D3a Sample documents | [`data/documents/`](data/documents), 4 files, 68 chunks at default settings |
| D3b Evaluation questions | [`data/questions.json`](data/questions.json), 30 questions |
| D4a Automated test results | [`docs/test_results.md`](docs/test_results.md) |
| D4b Evaluation results | [`docs/evaluation_results.md`](docs/evaluation_results.md) |
| D4c Limitations | [Limitations](#limitations), above |
| D5 Docker | [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml), [`.dockerignore`](.dockerignore) |
| D6 Working notes excluded | [`.gitignore`](.gitignore) |
| Application | [`app/`](app), 5 modules |
| Sample PDF generator | [`scripts/make_sample_pdfs.py`](scripts/make_sample_pdfs.py) |
| Evaluation runner | [`scripts/run_eval.py`](scripts/run_eval.py) |
| Ingest script | [`scripts/ingest_samples.py`](scripts/ingest_samples.py) |
| Tests | [`tests/`](tests), 3 suites |
| Dependencies | [`requirements.txt`](requirements.txt), [`requirements-dev.txt`](requirements-dev.txt) |
| Configuration template | [`.env.example`](.env.example) |