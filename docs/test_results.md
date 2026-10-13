# Automated test results

The default suite runs fully offline. It needs no network access and no API key.
Captured on Python 3.12.10 on Windows.

## Commands

```text
python -m pytest -q                    # offline default
python -m pytest -q --cov=app          # with coverage
python -m pytest -m live               # optional, needs GEMINI_API_KEY
```

## Result

110 passed, 1 skipped. Counts add up to 42, 33 and 35, plus the one skip. The
skip is the live smoke test, which carries the `live` marker and is skipped
without a key.

| Suite | Collected | Covers |
|---|---|---|
| `test_retrieval.py` | 42 | Rank one retrieval, score mapping from distance, the gate in both directions, delete and replace, chunk size, overlap and offset invariants, verbatim evidence and locations |
| `test_api.py` | 33 | The full endpoint and status matrix, response schemas, ask validation, health states, `/docs` and the OpenAPI schema |
| `test_failures.py` | 36, of which 35 pass and 1 skips | Rejected uploads, model outage, retries exhausted and recovered, malformed output, missing key, a failed re-ingest leaving the previous version intact, grounding guarantees, the prompt contract |

## Coverage

96 percent of statements across `app`. The uncovered lines are the Gemini model
construction and its default arguments, the non-injected branch of a retry, and a
few defensive raises that the suite reaches through other paths.

## What this suite does not prove

The fake embedder hashes words into vectors, so it rewards word overlap rather
than meaning. It cannot tell a paraphrase from a non-paraphrase. The suite
therefore verifies mechanics: ranking, thresholds, offsets, deletion, evidence
and status codes. Whether the assistant answers a paraphrased question correctly
is measured by the live evaluation run and nowhere else.

## Captured output

```text
python -m pytest -q --cov=app --cov-report=term-missing

....................................................................s... [ 64%]
.......................................                                  [100%]
============================== warnings summary ===============================
app\documents.py:18
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.12.10-final-0 _______________

Name               Stmts   Miss  Cover   Missing
------------------------------------------------
app\chains.py         37      2    95%   30, 65
app\config.py         28      0   100%
app\documents.py     116      5    96%   70, 91, 147, 184, 213
app\main.py           72      1    99%   42
app\rag.py            94      5    95%   55, 70-71, 82, 179
app\retries.py        56      5    91%   44, 46, 66, 93, 97
app\schemas.py        41      0   100%
------------------------------------------------
TOTAL                444     18    96%
110 passed, 1 skipped, 1 warning in 37.87s (0:00:37.87)
```

## A note on the deprecation warning

`langchain-community` reports that it is being sunset in favour of standalone
integration packages. `PyPDFLoader` and `TextLoader` are still the supported path
today. If they move, the change is confined to `build_loader` in
`app/documents.py`, which is the single place that imports them. That import
moved out of `chains.py` during the module split, because ingestion is the only
caller of a loader.