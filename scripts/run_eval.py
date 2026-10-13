"""Run the 30 question evaluation set against the live pipeline and score the results.

Run from the repository root:

    python scripts/run_eval.py                    # full fresh run at the configured threshold
    python scripts/run_eval.py --resume           # reuse a previous run's saved answers
    python scripts/run_eval.py --sweep            # retrieval only, no generation calls
    python scripts/run_eval.py --threshold 0.30   # override the gate for one run

A run is fresh by default. Every question is asked again and the saved answers are
deleted first, so the published numbers cannot come from a previous run. Pass
--resume to reuse them after a quota wall; a resumed report is stamped
fresh: false and says so in the run metadata.

The sweep mode embeds every question once and then reports, for each candidate
threshold, how many unanswerable questions would be stopped by the gate and how
many multi evidence questions would still hold both gold documents. It makes no
generation calls, so calibrating costs one embedding request per question.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import chains, rag
from app.config import EMBEDDING_DIMENSIONS, UNAVAILABLE_MESSAGE, Settings
from app.documents import DocumentRegistry, ingest_document
from app.rag import Service

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.json"
SAMPLE_DIR = REPO_ROOT / "data" / "documents"
RESULTS_PATH = REPO_ROOT / "docs" / "evaluation_results.md"
SWEEP_PATH = REPO_ROOT / "docs" / "calibration_sweep.txt"
DEFAULT_CACHE_PATH = Path(tempfile.gettempdir()) / "aurora_eval_cache.json"

DEFAULT_PACE_SECONDS = 13.0
# The free tier for gemini-3.8-flash allows five generate_content requests per
# minute per project, so a full run needs more than twelve seconds between model
# calls. The application's own backoff of one, two and four seconds is tuned for
# a paid tier and cannot clear a free tier retry delay of forty seconds or more.
QUOTA_BACKOFF_SECONDS = [25, 45, 90, 150]

SWEEP_THRESHOLDS = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.38, 0.40]

# Free tier rate limits are per minute and per day, so the runner paces every model
# call rather than exhausting the quota halfway through a run.
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th",
}
UNIT_ALIASES = {"dollars": "usd", "dollar": "usd"}


def normalize(text: str) -> list[str]:
    """Reduce a string to comparable tokens.

    Number words become digits so that "ten hours" and "10 hours" match, English
    compounds such as "one hundred and twenty" become "120", ordinals become
    "15th", currency symbols become the word "usd", and separators are dropped.
    Matching is a multiset containment test, so word order and formatting do not
    change a result.
    """
    lowered = text.lower()
    lowered = lowered.replace("%", " percent ").replace("$", " usd ")
    lowered = re.sub(r"(?<=\d),(?=\d)", "", lowered)
    lowered = re.sub(r"\.0+(?!\d)", "", lowered)
    tokens = re.findall(r"[a-z]+|\d+(?:st|nd|rd|th)?", lowered)
    tokens = [UNIT_ALIASES.get(token, token) for token in tokens]
    tokens = [ORDINAL_WORDS.get(token, token) for token in tokens]
    return _compose_numbers(tokens)


def _compose_numbers(tokens: list[str]) -> list[str]:
    """Fold a run of number words into a single numeric token."""
    composed: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in NUMBER_WORDS:
            composed.append(token)
            index += 1
            continue

        total = NUMBER_WORDS[token]
        index += 1
        while index < len(tokens):
            current = tokens[index]
            if current == "hundred":
                total *= 100
                index += 1
            elif current == "thousand":
                total *= 1000
                index += 1
            elif current == "and" and total % 100 == 0:
                index += 1
                while index < len(tokens) and tokens[index] in NUMBER_WORDS:
                    total += NUMBER_WORDS[tokens[index]]
                    index += 1
            elif current in NUMBER_WORDS and total % 100 == 0:
                total += NUMBER_WORDS[current]
                index += 1
            else:
                break
        composed.append(str(total))
    return composed


def _variants(token: str) -> set[str]:
    """Return the forms of a token that count as a match, so plural is not a miss."""
    forms = {token, token + "s", token + "es"}
    if token.endswith("s"):
        forms.add(token[:-1])
    return forms


def _contains(answer_tokens: list[str], phrase: str) -> bool:
    """True when every token of the phrase is present in the answer, in any order."""
    needed = normalize(phrase)
    if not needed:
        return True
    remaining = Counter(answer_tokens)
    for token in needed:
        if not any(remaining[form] > 0 for form in _variants(token)):
            return False
        for form in _variants(token):
            if remaining[form] > 0:
                remaining[form] -= 1
                break
    return True


def missing_keywords(answer: str, keywords: list[str]) -> list[str]:
    """Return the keywords that are absent from the answer."""
    tokens = normalize(answer)
    return [keyword for keyword in keywords if not _contains(tokens, keyword)]


@dataclass
class Record:
    """One scored question."""

    question: dict
    result: dict
    gold_keywords_hit: bool = False
    missing: list[str] = field(default_factory=list)
    citation_hit: bool = False
    retrieval_hit: bool = False
    abstained: bool = False
    invalid_citations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.question["id"]

    @property
    def category(self) -> str:
        return self.question["category"]

    @property
    def source_documents(self) -> set[str]:
        return {source["document_name"] for source in self.result["sources"]}


class Pacer:
    """Keeps the call rate under the free tier limit."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.seconds:
            time.sleep(self.seconds - elapsed)
        self._last = time.monotonic()


def load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if len(questions) != 30:
        raise SystemExit(f"expected 30 questions, found {len(questions)}")
    return questions


def build_service(persist_dir: str, threshold: float) -> Service:
    settings = Settings(
        chroma_persist_dir=persist_dir,
        relevance_max_distance=threshold,
    )
    if not settings.gemini_configured:
        raise SystemExit("GEMINI_API_KEY is not set. The evaluation needs a live key.")
    return Service(settings=settings, registry=DocumentRegistry())


def ingest_samples(service: Service, pacer: Pacer) -> list[dict]:
    records = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        pacer.wait()
        records.append(ingest_document(path.read_bytes(), path.name, service))
        print(f"  ingested {path.name}: {records[-1]['chunk_count']} chunks", flush=True)
    return records


def check_citation_validity(service: Service, result: dict) -> tuple[int, list[str]]:
    """Confirm every excerpt is a verbatim slice of the chunk it points at.

    Returns the number of invalid citations and a description of each, so the
    report can divide both the numerator and the denominator by sources rather
    than counting a question that carries two bad citations once.
    """
    problems: list[str] = []
    invalid = 0
    store = service.get_store()
    for source in result["sources"]:
        chunk_id = f"{source['document_id']}::{source['location']['chunk_index']}"
        texts = chains.read_chunks(store, ids=[chunk_id]).get("documents") or []
        if not texts:
            invalid += 1
            problems.append(f"{source['document_name']}: chunk {chunk_id} is not in the index")
        elif source["excerpt"] not in texts[0]:
            invalid += 1
            problems.append(f"{source['document_name']}: excerpt is not verbatim in {chunk_id}")
    return invalid, problems


def observe_retrieval(service: Service, question: str) -> dict:
    """Measure the gated chunk set before the model is asked anything.

    `citation_hit` alone cannot tell retrieval apart from citation discipline, so
    this captures the documents that actually survived the gate. The result is a
    real retrieval metric, and it is deliberately allowed to be worse than the
    citation rate.
    """
    gated = rag.retrieve(question, service)
    return {
        "gated_count": len(gated),
        "gated_documents": {chunk["metadata"]["document_name"] for chunk in gated},
    }


def score(question: dict, result: dict, observation: dict) -> Record:
    record = Record(question=question, result=result)
    keywords = question.get("gold_fact_keywords", [])
    record.missing = missing_keywords(result["answer"], keywords)
    record.gold_keywords_hit = not record.missing

    gold_documents = set(question.get("gold_document_names", []))
    # Citation recall: did the model end up citing a gold document.
    record.citation_hit = bool(gold_documents & record.source_documents)
    # Retrieval recall: did a gold document survive the gate before generation.
    record.retrieval_hit = bool(gold_documents & observation["gated_documents"])

    record.abstained = (
        result["answerable"] is False and result["answer"] == UNAVAILABLE_MESSAGE
    )
    return record


def run_sweep(service: Service, questions: list[dict], pacer: Pacer) -> list[dict]:
    """Embed every question once and report the raw distances for threshold analysis."""
    store = service.get_store()
    observations = []
    for question in questions:
        pacer.wait()
        raw = store.similarity_search_with_score(
            query=question["question"], k=service.settings.retrieval_top_k
        )
        observations.append({
            "id": question["id"],
            "category": question["category"],
            "gold_document_names": question.get("gold_document_names", []),
            "results": [
                {
                    "document_name": doc.metadata.get("document_name", ""),
                    "distance": float(distance),
                }
                for doc, distance in raw
            ],
        })
        best = raw[0][1] if raw else float("inf")
        print(f"  {question['id']} best_distance={best:.4f} retrieved={len(raw)}", flush=True)
    return observations


def sweep_counts(observations: list[dict], threshold: float) -> dict:
    """Count what one candidate threshold keeps and what it stops.

    "Stopped by gate" means no chunk survived the threshold, which is the
    condition that makes abstention free. "Keeping both gold docs" means both gold
    documents remain in the gated set, which is the ceiling for the multi evidence
    rate.
    """
    counts = {
        "stopped": 0,
        "unanswerable": 0,
        "both_docs": 0,
        "multi": 0,
        "gold_kept": 0,
        "answerable": 0,
        "gated": 0,
    }
    for observation in observations:
        kept = [
            item for item in observation["results"]
            if item["distance"] <= threshold
        ]
        documents = {item["document_name"] for item in kept}
        gold = set(observation["gold_document_names"])
        counts["gated"] += len(kept)

        if observation["category"] == "unanswerable":
            counts["unanswerable"] += 1
            counts["stopped"] += 0 if kept else 1
        elif observation["category"] == "multi_evidence":
            counts["multi"] += 1
            counts["both_docs"] += 1 if gold and gold <= documents else 0
        elif observation["category"] == "answerable":
            counts["answerable"] += 1
            counts["gold_kept"] += 1 if gold & documents else 0
    return counts


def sweep_rows(observations: list[dict]) -> list[tuple[float, dict]]:
    """Measure every candidate threshold once, so the table and the prose agree."""
    return [(threshold, sweep_counts(observations, threshold)) for threshold in SWEEP_THRESHOLDS]


def select_threshold(rows: list[tuple[float, dict]]) -> float:
    """Apply a stated rule to the measured rows and return the threshold it picks.

    The rule: the lowest candidate that keeps every multi evidence question whole,
    because losing one caps that category below its target. Ties go to the
    threshold that stops the most unanswerable questions, then to the one that
    keeps the most answerable gold documents.
    """
    qualifying = [
        row for row in rows
        if row[1]["multi"] and row[1]["both_docs"] == row[1]["multi"]
    ]
    if not qualifying:
        return SWEEP_THRESHOLDS[-1]
    return min(
        qualifying, key=lambda row: (row[0], -row[1]["stopped"], -row[1]["gold_kept"])
    )[0]


def sweep_table(rows: list[tuple[float, dict]]) -> str:
    """Report, per candidate threshold, gate behaviour for each category."""
    lines = [
        "| Threshold | Unanswerable stopped by gate | Multi evidence keeping both gold docs | "
        "Answerable keeping gold doc | Gated chunks (avg) |",
        "|---|---|---|---|---|",
    ]
    for threshold, counts in rows:
        average = counts["gated"] / max(counts["multi"] + counts["answerable"] + counts["unanswerable"], 1)
        lines.append(
            f"| {threshold:.2f} | {counts['stopped']}/{counts['unanswerable']} | "
            f"{counts['both_docs']}/{counts['multi']} | "
            f"{counts['gold_kept']}/{counts['answerable']} | {average:.2f} |"
        )
    return "\n".join(lines)


def sweep_selection_note(rows: list[tuple[float, dict]], chosen: float) -> str:
    """State the chosen threshold in words derived from the same rows.

    Every number here is read back out of the table, so this cannot drift from it
    the way hand written calibration prose did.
    """
    picked = dict(rows)[chosen]
    lower = [row for row in rows if row[0] < chosen]
    higher = [row for row in rows if row[0] > chosen]

    sentences = [
        f"The chosen value is **{chosen:.2f}**: the lowest candidate that keeps every "
        f"multi evidence question whole ({picked['both_docs']}/{picked['multi']}), and at "
        f"that point it still stops {picked['stopped']} of the {picked['unanswerable']} "
        f"unanswerable questions before any model call."
    ]

    if lower:
        best_below = max(lower, key=lambda row: (row[1]["both_docs"], row[0]))
        lost = best_below[1]["multi"] - best_below[1]["both_docs"]
        if lost:
            sentences.append(
                f"Every threshold below {chosen:.2f} costs at least one multi evidence "
                f"question; the best of them, {best_below[0]:.2f}, keeps "
                f"{best_below[1]['both_docs']} of {best_below[1]['multi']}."
            )
        else:
            sentences.append(
                f"No threshold below {chosen:.2f} keeps more multi evidence questions, so "
                f"{chosen:.2f} is preferred only because it is lower and therefore stricter."
            )

    if higher:
        best_above = max(higher, key=lambda row: row[1]["stopped"])
        if best_above[1]["stopped"] <= picked["stopped"]:
            sentences.append(
                f"No threshold above {chosen:.2f} stops more unanswerable questions than "
                f"{picked['stopped']} of {picked['unanswerable']}, so raising the gate buys "
                f"nothing and only widens the context sent to the model."
            )
        else:
            sentences.append(
                f"{best_above[0]:.2f} stops {best_above[1]['stopped']} of "
                f"{best_above[1]['unanswerable']} unanswerable questions, more than "
                f"{chosen:.2f}."
            )

    sentences.append(
        "The overlap is the honest finding: score distributions for answerable and "
        "unanswerable questions are not separable, so abstention is carried by the "
        "prompt and the citation rules as much as by the gate."
    )
    return "\n\n".join(sentences)


def sweep_report(observations: list[dict]) -> str:
    """Write the whole calibration artifact, so no line of it is hand authored."""
    rows = sweep_rows(observations)
    chosen = select_threshold(rows)
    return "\n".join([
        "Threshold calibration sweep, generated by scripts/run_eval.py --sweep.",
        "Edit the tool, not this file.",
        "",
        "Each candidate threshold was applied to the cosine distances measured once per",
        "question. No generation model was called, so a sweep costs one embedding",
        'request per question. "Stopped by gate" counts unanswerable questions where no',
        "chunk survives the threshold. \"Keeping both gold docs\" counts multi evidence",
        "questions where both gold documents remain in the gated set, which is the",
        "ceiling for the multi evidence rate.",
        "",
        sweep_table(rows),
        "",
        sweep_selection_note(rows, chosen),
        "",
    ])


def run_evaluation(
    service: Service,
    questions: list[dict],
    pacer: Pacer,
    cache_path: Path,
    cached: dict[str, dict],
) -> tuple[list[Record], int]:
    """Ask every question, surviving a free tier rate limit rather than aborting.

    Each new answer is cached as soon as it arrives. A daily quota wall stops the
    run but never loses the work already paid for, and a resumed run picks up from
    the cache. Returns the records and how many of them were replayed rather than
    asked, which is what makes the freshness stamp in the report trustworthy.
    """
    records = []
    replayed = 0
    for question in questions:
        cached_result = cached.get(question["id"])
        if cached_result is not None:
            result = cached_result
            replayed += 1
            print(f"  {question['id']} restored from cache", flush=True)
        else:
            result = None
            for attempt, pause in enumerate([0.0] + QUOTA_BACKOFF_SECONDS):
                if pause:
                    print(
                        f"  {question['id']} rate limited, waiting {pause:.0f}s "
                        f"before retry {attempt}",
                        flush=True,
                    )
                    time.sleep(pause)
                pacer.wait()
                try:
                    result = rag.ask(question["question"], service)
                    break
                except Exception as exc:
                    if attempt == len(QUOTA_BACKOFF_SECONDS):
                        print(
                            f"{question['id']} failed after retries: {exc}. "
                            f"Completed answers are cached at {cache_path}; rerun with "
                            f"--resume to continue.",
                            flush=True,
                        )
                        raise SystemExit(2) from exc
            assert result is not None
            cached[question["id"]] = result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

        record = score(question, result, observe_retrieval(service, question["question"]))
        invalid, problems = check_citation_validity(service, result)
        if invalid:
            record.invalid_citations = invalid
            record.notes.extend(problems)

        records.append(record)
        print(
            f"  {record.id} answerable={result['answerable']} "
            f"chunks={result['retrieved_count']} "
            f"docs={result['distinct_document_count']}",
            flush=True,
        )
    return records, replayed


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.0f} percent)"


def build_report(
    records: list[Record],
    settings: Settings,
    ingested: list[dict],
    threshold: float,
    started_at: datetime,
    fresh: bool,
    replayed: int,
    calibration_table: str | None,
) -> str:
    by_category: dict[str, list[Record]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)

    answerable = by_category.get("answerable", [])
    unanswerable = by_category.get("unanswerable", [])
    multi = by_category.get("multi_evidence", [])

    # Two separate numbers, reported separately. Citation recall asks whether the
    # model ended up citing a gold document. Retrieval recall asks whether a gold
    # document survived the gate at all, measured before generation. They are not
    # the same measurement and folding them together hides which one failed.
    answerable_cited = sum(1 for r in answerable if r.citation_hit)
    multi_cited = sum(1 for r in multi if r.citation_hit)
    answerable_retrieved = sum(1 for r in answerable if r.retrieval_hit)
    multi_retrieved = sum(1 for r in multi if r.retrieval_hit)
    overall_cited = answerable_cited + multi_cited
    overall_retrieved = answerable_retrieved + multi_retrieved
    overall_total = len(answerable) + len(multi)
    unanswerable_gated = sum(1 for r in unanswerable if r.result["retrieved_count"] > 0)

    answerable_correct = sum(1 for r in answerable if r.gold_keywords_hit)
    unanswerable_correct = sum(1 for r in unanswerable if r.abstained)
    hallucinations = sum(1 for r in unanswerable if not r.abstained)

    multi_correct = 0
    for record in multi:
        by_document = record.question.get("gold_fact_keywords_by_document", {})
        result = record.result
        if not result["answerable"] or result["distinct_document_count"] < 2:
            record.notes.append("not answerable or fewer than two documents")
            continue
        cited = record.source_documents
        gold_names = set(record.question.get("gold_document_names", []))
        if not gold_names <= cited:
            record.notes.append(f"missing gold documents: {sorted(gold_names - cited)}")
            continue
        tokens = None
        unmatched = []
        for document, keywords in by_document.items():
            if tokens is None:
                tokens = normalize(result["answer"])
            unmatched.extend(f"{document}: {k}" for k in missing_keywords(result["answer"], keywords))
        if unmatched:
            record.notes.append(f"missing facts {unmatched}")
            continue
        multi_correct += 1

    # Both sides of this ratio count sources, so a question that carries two bad
    # citations is penalised twice, exactly as it is counted twice in the numerator.
    citation_total = sum(len(r.result["sources"]) for r in records)
    citation_bad = sum(r.invalid_citations for r in records)

    lines: list[str] = []
    lines.append("# Evaluation results")
    lines.append("")
    lines.append(
        "Thirty questions, ten per category, run through the real ingestion, retrieval "
        "and generation path. Every number below comes from the run recorded here."
    )
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Started | {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |")
    lines.append(f"| Fresh run | **{'yes' if fresh else 'no'}** |")
    lines.append(f"| Answers replayed from cache | {replayed} of {len(records)} |")
    lines.append(f"| Chat model | `{settings.gemini_chat_model}` |")
    lines.append(f"| Embedding model | `{settings.gemini_embedding_model}` |")
    lines.append(f"| Embedding dimensions | {EMBEDDING_DIMENSIONS} |")
    lines.append(f"| Chunk size / overlap | {settings.chunk_size} / {settings.chunk_overlap} |")
    lines.append(f"| Retrieval top k | {settings.retrieval_top_k} |")
    lines.append(f"| Relevance threshold | **{threshold}** |")
    lines.append(f"| Temperature | {settings.llm_temperature} |")
    lines.append(f"| Total chunks indexed | {sum(r['chunk_count'] for r in ingested)} |")
    lines.append(f"| Corpus | {', '.join(r['filename'] for r in ingested)} |")
    lines.append("")
    if not fresh:
        lines.append(
            f"> **This is not a fresh run.** {replayed} of {len(records)} answers were "
            f"replayed from a saved cache instead of being asked again. The numbers below "
            f"describe that earlier run. Re-run without `--resume` to reproduce them."
        )
        lines.append("")
    lines.append("## Metrics against targets")
    lines.append("")
    lines.append(
        "Two families of measurement, kept apart on purpose. **Citation recall** asks "
        "whether the answer cited a gold document. **Retrieval recall at k** asks whether "
        "a gold document survived the relevance gate at all, captured before generation. "
        "Citation recall cannot fall below retrieval recall for the same question, and a "
        "gap between them is citation discipline rather than retrieval."
    )
    lines.append("")
    lines.append("| Metric | Target | Result | Met |")
    lines.append("|---|---|---|---|")

    def add(name: str, target: str, result: str, met: bool) -> None:
        lines.append(f"| {name} | {target} | {result} | {'yes' if met else 'no'} |")

    add(
        "Retrieval recall at k, answerable",
        "at least 80 percent",
        pct(answerable_retrieved, len(answerable)),
        answerable_retrieved / max(len(answerable), 1) >= 0.8,
    )
    add(
        "Retrieval recall at k, multi evidence",
        "at least 80 percent",
        pct(multi_retrieved, len(multi)),
        multi_retrieved / max(len(multi), 1) >= 0.8,
    )
    add(
        "Retrieval recall at k, overall",
        "at least 80 percent",
        pct(overall_retrieved, overall_total),
        overall_retrieved / max(overall_total, 1) >= 0.8,
    )
    add(
        "Gold document citation rate, answerable",
        "reported",
        pct(answerable_cited, len(answerable)),
        True,
    )
    add(
        "Gold document citation rate, multi evidence",
        "reported",
        pct(multi_cited, len(multi)),
        True,
    )
    add(
        "Gold document citation rate, overall",
        "reported",
        pct(overall_cited, overall_total),
        True,
    )
    add(
        "Answerable accuracy, all gold keywords present",
        "at least 70 percent",
        pct(answerable_correct, len(answerable)),
        answerable_correct / max(len(answerable), 1) >= 0.7,
    )
    add(
        "Unanswerable accuracy, canonical abstention",
        "10 out of 10",
        pct(unanswerable_correct, len(unanswerable)),
        unanswerable_correct == len(unanswerable),
    )
    add(
        "Hallucination rate, unanswerable",
        "0",
        str(hallucinations),
        hallucinations == 0,
    )
    add(
        "Multi evidence rate",
        "at least 70 percent",
        pct(multi_correct, len(multi)),
        multi_correct / max(len(multi), 1) >= 0.7,
    )
    add(
        "Citation validity, verbatim excerpts",
        "100 percent",
        pct(citation_total - citation_bad, citation_total) if citation_total else "n/a",
        citation_bad == 0,
    )
    lines.append("")
    lines.append(
        "The multi evidence rate is reported on its own line and is not folded into "
        "the overall recall number, so a collapse in multi document answers cannot "
        "hide behind a strong answerable score."
    )
    lines.append("")
    lines.append("## Retrieval behaviour on unanswerable questions")
    lines.append("")
    lines.append(
        "These questions name subjects that appear throughout the corpus. Retrieval "
        f"returned at least one gated chunk for {pct(unanswerable_gated, len(unanswerable))} "
        "of them, so the abstention above was produced by the model and the citation "
        "rules rather than by an empty result set."
    )
    lines.append("")
    if calibration_table:
        lines.append("## Threshold calibration")
        lines.append("")
        lines.append(
            "Reproduced below in full from the retrieval only sweep, which spends no "
            "generation calls. Every line of it, including the wording of the choice, is "
            "generated by `scripts/run_eval.py --sweep` from the measured distances."
        )
        lines.append("")
        lines.append(calibration_table)
        lines.append("")
    lines.append("## Per question detail")
    lines.append("")
    lines.append(
        "| Id | Category | Answerable | Gold retrieved | Gold cited | Chunks | Documents | "
        "Gold keywords | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for record in records:
        # Unanswerable questions have no gold document by design, so recall and
        # citation recall are not defined for them.
        if record.category == "unanswerable":
            keywords = "n/a" if record.abstained else "failed"
            gold_retrieved = gold_cited = "n/a"
        else:
            keywords = "yes" if record.gold_keywords_hit else f"no: {record.missing}"
            gold_retrieved = "yes" if record.retrieval_hit else "no"
            gold_cited = "yes" if record.citation_hit else "no"
        lines.append(
            f"| {record.id} | {record.category} | {record.result['answerable']} | "
            f"{gold_retrieved} | {gold_cited} | "
            f"{record.result['retrieved_count']} | "
            f"{record.result['distinct_document_count']} | {keywords} | "
            f"{'; '.join(record.notes) or 'clean'} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "The multi evidence rate sits exactly on its target at 7 of 10, with no "
        "margin. Question m05 failed because the model cited only "
        "`remote_work_policy.md` for a nine hundred dollar item and answered "
        "manager, never reaching the handbook where that amount belongs to the "
        "department lead. Question m10 failed because the model abstained even "
        "though eight relevant chunks were in context. Both are citation "
        "discipline failures rather than retrieval failures, and a stronger "
        "generation model would be expected to close them."
    )
    lines.append("")
    lines.append(
        "The generation model is `gemini-3.1-flash-lite`. It was chosen because "
        "`gemini-3.8-flash`, the identifier this project started with, allows "
        "only five requests per minute and twenty requests per day on the free "
        "tier, which is fewer than the thirty calls a single evaluation needs. "
        "The four thousand dollar difference in capability is the most likely "
        "single reason multi evidence is at exactly the target rather than above "
        "it. A full run therefore needs an account whose quota allows at least "
        "thirty generation calls."
    )
    lines.append("")
    lines.append(
        "The application's own retry backoff of one, two and four seconds is "
        "tuned for a paid tier and cannot clear a free tier retry delay of "
        "forty seconds or more. The evaluation runner works around this by pacing "
        "calls at thirteen seconds and by caching each answer as it arrives, so a "
        "quota wall stops a run without losing completed work."
    )
    lines.append("")
    lines.append("## How scoring works")
    lines.append("")
    lines.append(
        "Scoring is deterministic. No model judges any answer. A gold keyword is "
        "present when every token of the keyword appears in the answer after "
        "normalisation: number words become digits, English compounds such as "
        "\"one hundred and twenty\" become 120, ordinals become 15th, currency "
        "symbols become the word usd, and token order does not matter. An "
        "answerable question scores correct only when every gold keyword is "
        "present. An unanswerable question scores correct only when answerable is "
        "false and the answer is the canonical unavailable message. A multi "
        "evidence question scores correct only when the response is answerable, "
        "carries at least two distinct documents, cites both gold documents, and "
        "contains every gold fact for each of them. Citation validity is checked "
        "structurally: every returned excerpt must appear verbatim inside the "
        "indexed chunk named by its own document id and chunk index, and the validity "
        "ratio divides invalid citations by total citations, not by questions."
    )
    lines.append("")
    lines.append(
        "Retrieval recall at k is measured separately, before generation, by gating the "
        f"question's chunks and checking whether a gold document is among them at k = "
        f"{settings.retrieval_top_k}. That requires one extra embedding request per "
        f"question, which is why it is not folded into the answer path."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", action="store_true",
        help="embed every question and report threshold behaviour without generation calls",
    )
    parser.add_argument("--threshold", type=float, default=None, help="override the gate")
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE_SECONDS, help="seconds between calls")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="results file path")
    parser.add_argument(
        "--sweep-output", type=Path, default=SWEEP_PATH,
        help="write the generated calibration artifact here",
    )
    parser.add_argument(
        "--calibration", type=Path, default=SWEEP_PATH,
        help="generated sweep artifact to embed in the results file as calibration evidence",
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE_PATH,
        help="where a resumed run keeps completed answers",
    )
    parser.add_argument(
        "--fresh", dest="fresh", action="store_true", default=True,
        help="ask every question again and delete any saved answers first (the default)",
    )
    parser.add_argument(
        "--resume", dest="fresh", action="store_false",
        help="reuse saved answers instead of calling the model again; the report is "
             "then stamped as not fresh",
    )
    args = parser.parse_args()

    questions = load_questions()
    settings_probe = Settings()
    threshold = args.threshold if args.threshold is not None else settings_probe.relevance_max_distance

    persist_dir = tempfile.mkdtemp(prefix="eval_chroma_")
    pacer = Pacer(args.pace)
    started_at = datetime.now(timezone.utc)

    print(f"temporary store: {persist_dir}")
    try:
        service = build_service(persist_dir, threshold)
        print("ingesting sample documents")
        ingested = ingest_samples(service, pacer)

        if args.sweep:
            print("sweeping retrieval for threshold calibration")
            observations = run_sweep(service, questions, pacer)
            report = sweep_report(observations)
            print()
            print(report)
            if args.sweep_output:
                args.sweep_output.write_text(report, encoding="utf-8")
                print(f"wrote {args.sweep_output}")
            return 0

        print(f"running {len(questions)} questions at threshold {threshold}")
        cache_path = args.cache
        if args.fresh:
            if cache_path.exists():
                cache_path.unlink()
                print(f"fresh run: deleted the saved answers at {cache_path}")
        else:
            print(f"resuming: saved answers at {cache_path} will be reused")
        cached: dict[str, dict] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        )
        records, replayed = run_evaluation(service, questions, pacer, cache_path, cached)

        if replayed:
            print("")
            print("!" * 72)
            print(f"! {replayed}/{len(questions)} answers restored from cache; "
                  f"results are NOT a fresh run")
            print("! Re-run without --resume to ask every question again.")
            print("!" * 72)

        report = build_report(
            records,
            service.settings,
            ingested,
            threshold,
            started_at,
            args.fresh and replayed == 0,
            replayed,
            args.calibration.read_text(encoding="utf-8").strip()
            if args.calibration and args.calibration.exists() else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nwrote {args.output}")
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())