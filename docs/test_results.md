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

110 passed, 1 skipped. Counts add up to 42, 33 and 35. The one skip is the
live smoke test, which carries the `live` marker and is skipped without a key.

| Suite | Tests | Covers |
|---|---|---|
| `test_retrieval.py` | 42 | Rank one retrieval, score mapping from distance, the gate in both directions, delete and replace, chunk size, overlap and offset invariants, verbatim evidence and locations |
| `test_api.py` | 33 | The full endpoint and status matrix, response schemas, ask validation, health states, `/docs` and the OpenAPI schema |
| `test_failures.py` | 35 passed, 1 skipped | Rejected uploads, model outage, retries exhausted and recovered, malformed output, missing key, a failed re-ingest leaving the previous version intact, grounding guarantees, the prompt contract |

## Coverage

95 percent of statements across `app`, shown at the end of the captured output.
The uncovered lines are the Gemini model construction and its default
arguments, which cannot be exercised without a key.

## What this suite does not prove

The fake embedder hashes words into vectors, so it rewards word overlap rather
than meaning. The suite therefore verifies mechanics: ranking, thresholds,
offsets, deletion, evidence and status codes. Whether the assistant answers a
paraphrased question correctly is measured by the live evaluation run, not here.

## Captured output

```text
python -m pytest -q --cov=app --cov-report=term-missing

....................................................................s... [ 64%]
.......................................                                  [100%]
============================== warnings summary ===============================
app\chains.py:13
  C:\Dev\Re\app\chains.py:13: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.12.10-final-0 _______________

Name             Stmts   Miss  Cover   Missing
----------------------------------------------
app\chains.py       83     6    93%   52, 108, 130, 143, 145, 199
app\config.py       28     0   100%
app\main.py         72     1    99%   42
app\rag.py         206    13    94%   66, 81-82, 93, 121, 130, 139, 176, 252, 255, 380, 384, 393
app\schemas.py      41     0   100%
----------------------------------------------
TOTAL              430    20    95%
110 passed, 1 skipped, 1 warning in 39.95s (0:00:39.95)
```

## A note on the deprecation warning

`langchain-community` reports that it is being sunset in favour of standalone
integration packages. `PyPDFLoader` and `TextLoader` are still the supported
path today. If they move, the change is confined to `build_loader` in
`app/chains.py`, which is the single place that imports them.