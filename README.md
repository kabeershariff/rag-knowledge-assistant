# RAG API knowledge assistant

A FastAPI service that ingests PDF, TXT and Markdown documents into a Chroma
vector store and answers questions about them with Gemini. Every answer carries
the source document, the exact location of the supporting text, and a verbatim
excerpt. When the answer is not in the documents, it says so instead of guessing,
and it says so before the language model is ever called rather than by asking
the model nicely.

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
10. [Deliverables](#deliverables)

---

## What this is

A FastAPI service that ingests PDF, TXT and Markdown documents into a Chroma
vector store and answers questions about them using Gemini, with evidence
attached to every answer. It would rather return `The requested information is
unavailable in the provided documents.` than invent a plausible answer, and that
property is enforced by code before the language model is ever called.

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
| Python with a clean modular architecture | Met, seven flat modules, no package tree |
| Environment variables for keys and configuration | Met, `.env.example` lists every variable the app or the container entrypoint reads |
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

The image runs as an unprivileged user, not root. The index lives inside the
image so the container is useful with no volume, and Compose mounts a named
volume over `data/chroma`; either way the service writes as that user.

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
| `python scripts/run_eval.py` | Fresh run of all 30 evaluation questions, writes `docs/evaluation_results.md` |
| `python scripts/run_eval.py --resume` | Continue an interrupted run from its saved answers |
| `python scripts/run_eval.py --sweep` | Threshold sweep, no generation calls |
| `python scripts/make_sample_pdfs.py` | Regenerate the sample PDF |

---

## Configuration

Settings load once through `pydantic-settings` from the environment and from
`.env`. Every variable the application reads appears in `.env.example`. Unknown
variables are ignored rather than rejected, so an extra line in `.env` cannot
stop the service from starting.

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
| `HOST` | No | `0.0.0.0` | Uvicorn bind address, read by the container entrypoint. |
| `PORT` | No | `8000` | Uvicorn port, read by the container entrypoint. |

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

**`LANGSMITH_TRACING` is not a setting.** It is not a `Settings` field, so this
application never reads it and `.env` cannot change it. It is pinned to `false`
in the `Dockerfile` only so that any transitive LangSmith auto-instrumentation
stays off inside the image. It says nothing about where queries go: asking a
question sends the question text to Google's embedding API. Nothing is retained
on the local machine, and nothing is sent anywhere else.

**A missing key does not stop the service.** `/healthz` reports `degraded`, and
the endpoints that need Gemini return 503 with an actionable message. Listing,
deleting, the health check and `/docs` all keep working, so the service can be
inspected before any secret exists. **Changing the embedding model or the
dimension requires re-indexing**: delete `CHROMA_PERSIST_DIR` and ingest again,
or queries will fail against a mismatched index.

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
reference it, and a random identifier would break that continuity. Ingestion
embeds before it deletes: if embedding fails part way through, for example
because the daily free tier quota is spent, the previously indexed version of the
document is left intact rather than destroyed on the way to a failure.

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

**About this example.** The evidence fields are real and reproducible: the
excerpt is a verbatim slice of chunk 6 of the committed `employee_handbook.pdf`,
at page 2, characters 600 to 1356, matching a recorded live run. The `answer`
text shows the shape of a grounded response and the citation format; its wording
is illustrative rather than a captured transcript.

Four details in that response matter more than they look:

- **The excerpt is never written by the model.** It is sliced from the retrieved
  chunk text outside the generation path, then capped at 500 characters. Search
  it in the PDF and you will find it byte for byte. This is the single most
  important property of the whole design.
- **`char_end` describes the whole chunk, not the excerpt.** The excerpt is
  truncated at 500 characters while the location still addresses the full chunk,
  so `char_end` can exceed the excerpt length. Both numbers are honest about
  different things.
- **For PDFs, offsets are relative to the page text, not the whole file.** Each
  page is split independently, so the location always reports the page number
  alongside the offsets. For TXT and Markdown there is no `page_number` field.
- **A refused question can return evidence.** See the next section.

### When the answer is not available

There are exactly two shapes, and they are not the same thing.

**Case 1 — the gate stopped the question.** No chunk was at or below the
threshold, so the model was never called. Every count is zero:

```json
{
  "answer": "The requested information is unavailable in the provided documents.",
  "answerable": false,
  "sources": [],
  "retrieved_count": 0,
  "evidence_count": 0,
  "distinct_document_count": 0,
  "model": "gemini-3.1-flash-lite",
  "latency_ms": 2.2
}
```

Only `latency_ms` varies. This is question `u03` in the recorded evaluation run
([`docs/evaluation_results.md`](docs/evaluation_results.md)).

**Case 2 — the model declined, or its citations did not validate.** The gate let
chunks through, the model ran, and the answer was refused. `answer` and
`answerable` are identical to case 1, but the counts are not:

| Field | Case 2 |
|---|---|
| `answer` | The canonical message. A constant in `app/config.py`, not model output. |
| `answerable` | `false`. |
| `sources` | The retrieved-but-rejected chunks, if the model cited any that validated. May be populated. |
| `retrieved_count` | Chunks that passed the gate. Greater than zero. |
| `evidence_count` | Citations the model made that were in range. Often `0`, because a declining model cites nothing. |
| `distinct_document_count` | Distinct documents in `sources`. |

Recorded examples from the same run: `u02` returned `answerable: false` with two
gated chunks and one cited document, `u06` and `u09` returned one gated chunk and
one cited document each, and `m10` abstained with eight gated chunks in context.
The table is the contract; those counts are the recorded values.

The contract, stated once: **`answer` is the canonical message and `answerable`
is `false` whenever the service refuses. `sources` is populated only when chunks
were retrieved, and its emptiness is not by itself a statement about the gate.**
A `retrieved_count` of zero always means the gate stopped the question outright,
because that is the only code path that returns it. A positive
`retrieved_count` with `answerable: false` means the model and the citation
rules carried the refusal. The message is the same string whether the model
abstained, the gate rejected everything, the key was missing, or the quota was
spent.

---

## Architecture

```
  HTTP request
      |
      v
  +------------------------------------------------------------+
  | app/main.py       routes, validation, error to status code   |
  +------------------------------------------------------------+
      |
      +-----------------------------+-----------------------+
      v                             v                       |
  +--------------------------+  +-------------------------+
  | app/documents.py         |  | app/rag.py              |
  | registry, upload check,  |  | retrieve, gate, ask,    |
  | ingest, list, delete     |  | citations, evidence     |
  +--------------------------+  +-------------------------+
      |                             |
      +-----------+-----------------+
                  v
      +-----------------------------+
      | app/retries.py              |  retry policy, error classification
      | app/chains.py               |  LangChain factories, Chroma adapter
      +-----------------------------+
                  |
                  v
      Gemini API for chat and embeddings, Chroma on disk
```

Seven flat modules, no package tree. `documents.py` and `rag.py` never import each
other; both go down through `retries.py` and `chains.py`.

### Grounding, in three steps

1. **Relevance gate.** The question is embedded, the store returns chunks with
   raw cosine distances, scores are `1 - distance`, and only chunks at or below
   `RELEVANCE_MAX_DISTANCE` survive. If none do, return the unavailable message
   without calling the model. Deterministic and free.
2. **Strict prompt at zero temperature with structured output.** Surviving
   chunks become numbered context blocks. The model is instructed to abstain and
   constrained to an `Answer` schema with `answer`, `answerable` and `citations`
   fields.
3. **Citation validation after generation.** Any citation number not present in
   the context that was sent is dropped. If the model claimed an answer but no
   valid citation survives, the response is demoted to unavailable. Model
   abstention wording is normalised to the canonical message.

The final `answerable` value is the result of all three. None of them checks
whether the answer's sentences are actually supported by the chunks it cites —
see [Limitations](#limitations).

### LangChain components

PDF parsing is `PyPDFLoader`, TXT and Markdown are `TextLoader` with
`autodetect_encoding=True`, chunking is `RecursiveCharacterTextSplitter` at the
size and overlap in [Configuration](#configuration), embeddings are
`GoogleGenerativeAIEmbeddings`, the store is `Chroma` with `hnsw:space` set to
cosine, retrieval is `similarity_search_with_score` because the gate needs the
raw distance, and generation is `ChatPromptTemplate` composed with
`ChatGoogleGenerativeAI` and `with_structured_output`, which replaces manual JSON
parsing with a validated object.

LangChain legacy chain classes are not used. In LangChain 1.x they live in a
separate package and are being retired in favour of runnable composition. Each
component is built by a small factory that accepts an alternative object —
`build_loader`, `build_splitter` and `load_and_split` in `documents.py`, where
they are used, and `build_embeddings`, `build_store` and `build_answer_chain` in
`chains.py`. That is what lets the test suite swap in deterministic fakes without
touching a route.

`langchain-chroma` exposes no public API that returns stored documents and
metadata together, which listing, deleting and citation checking all need. The
private `_collection` attribute is therefore reached in exactly one place,
`chains.collection_of`, and nothing else in the application may touch it. See
[Limitations](#limitations) for what that coupling costs on upgrade.

---

## Architectural decisions

**LangChain components instead of hand written loaders, splitters, embeddings
and store wrappers.** The brief asks for "a vector store of your choice" and
prohibits only training a model from scratch, so an off the shelf integration
library is inside the terms rather than a loophole. The built-in components
replace roughly thirteen modules that would each need their own tests, and they
are the standard way to do this. Rejected: the hand written layer set, which was
larger and needed more tests. LlamaIndex solves the same problem and only one of
the two is needed. LangGraph arrives as a transitive dependency but is only
justified by multi hop or agent reasoning, which this project does not have.

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

**A flat seven module application.** LangChain already supplies the ingestion,
retrieval and generation layers, so nested packages would only add navigation
cost. Rejected: a layered package tree where import depth exceeded the logic in
each file.

**One file per responsibility, and the seam chosen by what changes together.**
`documents.py` and `rag.py` split because a 353 line file was doing twelve jobs,
not because a diagram called for two boxes. Rejected: a fourth module for a
responsibility that fits, and rejected again the layered `api/`, `core/`,
`services/` tree that the import ceremony would have outgrown.

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

### A run is fresh by default

Every answer is saved to a cache as it arrives, so a quota wall does not throw
away work that has already been paid for. That cache is why a run can quietly
stop being a run, so the default is the opposite of the obvious one:

| Flag | Behaviour |
|---|---|
| *(none)* | **Fresh.** Deletes the saved answers and asks all thirty questions again. |
| `--resume` | Reuses the saved answers and asks only the missing ones. |
| `--cache PATH` | Where the answers are saved. Defaults to a file in the system temp directory. |

A report states which it was. `Fresh run` is `yes` only when every question was
actually asked; otherwise a banner at the top of the report says how many answers
were replayed. Replaying a cache reproduces the numbers exactly, which is
precisely what makes it dangerous to publish without the stamp.

### What the two retrieval numbers mean

| Metric | Question it answers |
|---|---|
| Retrieval recall at k | Did a gold document survive the relevance gate, measured **before** generation? |
| Gold document citation rate | Did the model go on to **cite** a gold document? |

Citation recall cannot fall below retrieval recall for the same question, so a
gap between them is citation discipline failing rather than retrieval failing,
and reporting only the first number hides which of the two broke.

### How the question set is designed

Ten each of **answerable** (fully answered by one document), **unanswerable**
(the fact is absent from the whole corpus, but the subject matter is close
enough that retrieval will return chunks, which is what forces the gate and the
abstention path to do real work), and **multi evidence** (deliberate cross
references, so parental leave lives in `benefits_guide.md` while the notification
path lives in `employee_handbook.pdf`, and the home office equipment list lives
in `remote_work_policy.md` while the expense limits live in the handbook).

All four sample documents describe the same invented organisation, so no correct
answer can arrive from general world knowledge. That removes a whole class of
false positive from the scoring, and it is why several questions name the company
directly: an invented name cannot be answered from memory, so a correct answer
proves the answer came from the index rather than from the model.

### Results

The numbers live in
[`docs/evaluation_results.md`](docs/evaluation_results.md) and nowhere else,
because that file is written by the runner and this one is written by hand. A
hand-copied table is a table that drifts; regenerating is one command.

```bash
python scripts/run_eval.py --calibration docs/calibration_sweep.txt
```

**The threshold finding, stated plainly.** The relevance gate's score
distributions for answerable and unanswerable questions overlap. At 0.33 the
gate stops 4 of the 10 unanswerable questions before any model call, and the
remaining six are handled by the prompt and the deterministic citation rules.
Abstention is carried by layers two and three, not by the gate alone. That is the
weakest number in the system and it is reported here rather than left in a
results file.

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
production, and every test uses a temporary store directory. Captured output is
in [`docs/test_results.md`](docs/test_results.md).

| Suite | Covers |
|---|---|
| `test_retrieval.py` | Rank one retrieval, score mapping from distance, the gate in both directions, delete and replace, chunk size, overlap and offset invariants, verbatim evidence and locations |
| `test_api.py` | The full endpoint and status matrix, response schemas, ask validation, health states, `/docs` and the OpenAPI schema |
| `test_failures.py` | Rejected uploads, model outage, retries exhausted and recovered, malformed output, missing key, a failed re-ingest leaving the previous version intact, grounding guarantees, the prompt contract |

### What the offline suite does not prove

The fake embedder is SHA-256 word hashing: two sentences with the same words get
the same vector, and two sentences that mean the same thing get **nothing like
it**. Read the pass count as a statement about mechanics, not about retrieval
quality.

- **It does prove:** ranking order, threshold behaviour in both directions, chunk
  offsets, re-ingest replacing rather than duplicating, verbatim evidence, status
  codes, retry counts and ordering.
- **It cannot prove:** that a paraphrased question reaches the right chunk. Every
  test that needs to reach a specific chunk builds the query from that chunk's own
  words, precisely because a paraphrase does not survive the hash.
- **Semantic quality is measured in exactly one place**, the live evaluation run
  against real Gemini embeddings, and nowhere else. A suite of 110 green tests
  that cannot tell a paraphrase from a non-paraphrase is not 110 tests of
  retrieval quality, and should not be counted as one.

---

## Limitations

The weaknesses below are stated first, because the mitigation column is worth
nothing without them. Read top to bottom as a decreasing order of how much they
matter.

### The ones that actually matter

**Citation validation is not answer faithfulness.** The three grounding layers
check that a citation number existed in the context sent, that the excerpt is a
verbatim slice of the indexed chunk, and that an answer without a surviving
citation is demoted. None of them check that the sentences in the answer are
supported by the chunks it cites. A response with `answerable: true`,
`citations: [1]` and fluent, well-formed, entirely unsupported prose passes all
three layers and gets served with a real-looking excerpt attached. Nothing here
is an entailment check and no amount of citation bookkeeping becomes one.

**Gold keyword scoring is strict token containment, and it has already cost a
point.** A gold keyword counts as present only when every one of its tokens
appears in the answer after normalisation. Question `a05` was scored **wrong**
solely because the answer said "People Operations" as two words where the
keyword expected them joined, while stating the fifteen week rule correctly. That
is a scorer artefact, not a wrong answer, and it is the single biggest
methodological weakness in the evaluation.

**The relevance gate does not do as much as it looks like it does.** Score
distributions for answerable and unanswerable questions overlap. At the
calibrated 0.33 the gate stops 4 of the 10 unanswerable questions before any
model call; the other 6 reach the model and are stopped by the prompt and the
citation rules. Abstention is carried by layers two and three.

**The published evaluation numbers are one run of one model on one day.** They
are stamped with `Fresh run` and a replay count precisely because a cached replay
produces identical numbers and identical confidence. There is no confidence
interval, because one run cannot produce one.

**`langchain-chroma` has no public API for reading stored documents and metadata
together.** Listing, deleting and citation checking all need it, so the
application reaches the private `_collection` attribute behind
`chains.collection_of`. A Chroma upgrade that moves it breaks three features at
once, silently, with no test able to catch it, because the tests use the same
adapter.

**The free tier is the binding constraint, not the code.** Ingestion is not
idempotent in cost: three full ingestions of four documents on one day exhausted
a daily embedding quota of 1000 requests, and a fourth failed with a retry delay
of ten hours that no backoff can clear.

### Everything else

| Limitation | Mitigation |
|---|---|
| The multi evidence rate is 7 of 10, exactly on the 70 percent target with no margin | Both failures are citation discipline, not retrieval: one question cited a single document for a two document question, another abstained with eight relevant chunks in context. The category is reported on its own line so the margin stays visible |
| The generation model is a flash lite variant, chosen because the free tier quota for the stronger flash model is smaller than a single evaluation run needs | Both identifiers are environment variables. This is the most likely reason multi evidence sits exactly on target rather than above it |
| Hashed fake embeddings cannot measure semantic similarity | The offline suite verifies mechanics only. Semantic quality comes from the live evaluation run and nowhere else |
| Google model identifiers move; `gemini-2.5-flash` is already retired | `GEMINI_CHAT_MODEL` and `GEMINI_EMBEDDING_MODEL` are the only knobs |
| Changing the embedding model or the dimension needs a re-index | Delete `CHROMA_PERSIST_DIR` and ingest again; the collection keeps its original dimension and distance space |
| A single badly calibrated threshold gives either false unavailable answers or too much context | Calibrated against the 30 question set with a retrieval only sweep that spent no generation calls. The table and the rule that picked 0.33 are in [`docs/calibration_sweep.txt`](docs/calibration_sweep.txt) |
| `PyPDFLoader` has no OCR, and layout parsing is limited for columns and tables | Stated plainly. The sample PDF keeps a simple single column layout |
| PDF offsets are relative to the page, and `char_end` addresses the whole chunk while the excerpt is capped at 500 characters | Evidence reports the page number alongside the offsets, and the caption says which number describes what |
| The model may paraphrase evidence in the answer text | Excerpts are sliced from the retrieved chunks outside the model path, and the answer text is a separate field from the sources list |
| The embedded store is single process, unauthenticated, and needs the network | Out of scope for an assignment deliverable. 503 with an actionable message, retries, and a test suite that runs fully offline |
| `langchain-community` reports that it is being sunset in favour of standalone integration packages | `PyPDFLoader` and `TextLoader` remain supported today. If they move, the change is confined to `build_loader` in `app/documents.py`, the one place that imports them |
| LangChain packages move quickly | All thirteen runtime dependencies are pinned with `==`, including `langchain-core` because the test fakes are built on its base classes. An upgrade is a deliberate pin change followed by a full test run |

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
Point pytest elsewhere: `python -m pytest -q --basetemp ./.pytest-tmp`.

---

## Deliverables

This document is itself deliverable D2, so it is not listed here.

| Deliverable | Where it is |
|---|---|
| D1 Git repository | [github.com/kabeershariff/rag-knowledge-assistant](https://github.com/kabeershariff/rag-knowledge-assistant) |
| D3a Sample documents | [`data/documents/`](data/documents), 4 files, 68 chunks at default settings |
| D3b Evaluation questions | [`data/questions.json`](data/questions.json), 30 questions |
| D4a Automated test results | [`docs/test_results.md`](docs/test_results.md) |
| D4b Evaluation results | [`docs/evaluation_results.md`](docs/evaluation_results.md) |
| D4c Limitations | [Limitations](#limitations), above |
| D5 Docker | [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml), [`.dockerignore`](.dockerignore) |
| D6 Working notes excluded | [`.gitignore`](.gitignore) |
| Threshold calibration | [`docs/calibration_sweep.txt`](docs/calibration_sweep.txt) |