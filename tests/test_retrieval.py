"""Retrieval behavior, the relevance gate and the chunking invariants.

The fake embedder hashes words into vectors, so it rewards overlap rather than
meaning. A paraphrased question does not reliably reach the chunk it describes.
Queries here are therefore built from chunk text through the `verbatim_query`
fixture, which makes ranking, thresholds, offsets and evidence deterministic.
Whether the assistant paraphrases well is answered by the live evaluation run.
"""

import pytest

from app import chains, rag
from app.config import UNAVAILABLE_MESSAGE
from app.rag import Service

from tests.conftest import OFF_CORPUS_QUESTIONS, SAMPLE_DIR


def ingest_text(text: str, filename: str, service: Service) -> int:
    """Ingest raw text as if it had been uploaded, returning the chunk count."""
    record = rag.ingest_document(text.encode("utf-8"), filename, service)
    return record["chunk_count"]


# Ranking


def test_a_known_question_returns_the_gold_chunk_at_rank_one(corpus, verbatim_query):
    """The chunk a question was built from must come back first."""
    question = verbatim_query("benefits_guide", 7)

    gated = rag.retrieve(question, corpus)

    assert gated, "a query taken from an indexed chunk must clear the gate"
    assert gated[0]["metadata"]["chunk_id"] == "benefits_guide::7"


def test_rank_one_holds_across_the_whole_corpus(corpus, verbatim_query, chunk_reader):
    """Sampled chunks from every document must rank themselves first."""
    checked = 0
    for document_id in (
        "employee_handbook",
        "it_security_policy",
        "benefits_guide",
        "remote_work_policy",
    ):
        rows = chunk_reader(document_id)
        for _, _, metadata in rows[::5]:
            question = verbatim_query(document_id, metadata["chunk_index"])
            gated = rag.retrieve(question, corpus)
            assert gated, f"{document_id} chunk {metadata['chunk_index']} returned nothing"
            assert gated[0]["metadata"]["chunk_id"] == f"{document_id}::{metadata['chunk_index']}"
            checked += 1
    assert checked >= 12, "the ranking sample should cover a meaningful number of chunks"


def test_results_are_sorted_by_ascending_distance(corpus, verbatim_query):
    question = verbatim_query("it_security_policy", 3)

    gated = rag.retrieve(question, corpus)

    distances = [result["distance"] for result in gated]
    assert distances == sorted(distances)


def test_score_is_one_minus_distance(corpus, verbatim_query):
    question = verbatim_query("remote_work_policy", 2)

    gated = rag.retrieve(question, corpus)

    assert gated
    for result in gated:
        assert result["score"] == pytest.approx(1.0 - result["distance"])


def test_top_k_limits_the_number_of_returned_chunks(corpus, verbatim_query):
    question = verbatim_query("benefits_guide", 7)

    gated = rag.retrieve(question, corpus, top_k=2)

    assert len(gated) <= 2


# The relevance gate


def test_only_chunks_at_or_below_the_threshold_survive_the_gate(corpus, verbatim_query):
    question = verbatim_query("benefits_guide", 7)

    gated = rag.retrieve(question, corpus)

    assert gated
    limit = corpus.settings.relevance_max_distance
    assert all(result["distance"] <= limit for result in gated)


@pytest.mark.parametrize("question", OFF_CORPUS_QUESTIONS)
def test_off_corpus_questions_are_rejected_by_the_gate(corpus, stub, question):
    """A subject the corpus never mentions must produce nothing."""
    assert rag.retrieve(question, corpus) == []


@pytest.mark.parametrize("question", OFF_CORPUS_QUESTIONS)
def test_rejected_questions_never_call_the_model(corpus, stub, question):
    rag.ask(question, corpus)

    assert stub.call_count == 0


def test_gate_short_circuit_returns_the_canonical_message(corpus, stub):
    result = rag.ask(OFF_CORPUS_QUESTIONS[0], corpus)

    assert result["answerable"] is False
    assert result["answer"] == UNAVAILABLE_MESSAGE
    assert result["sources"] == []
    assert result["retrieved_count"] == 0
    assert result["evidence_count"] == 0
    assert result["distinct_document_count"] == 0
    assert stub.call_count == 0


def test_a_looser_threshold_admits_what_the_gate_rejected(corpus):
    """The distance is what rejects an off-corpus question, not a hidden filter."""
    question = OFF_CORPUS_QUESTIONS[0]
    assert rag.retrieve(question, corpus) == []

    corpus.settings.relevance_max_distance = 1.5

    assert rag.retrieve(question, corpus), "with no threshold the store always returns candidates"


def test_a_tighter_threshold_admits_nothing_even_for_a_perfect_query(corpus, verbatim_query):
    """The same query is gated out when the threshold drops below its distance."""
    question = verbatim_query("benefits_guide", 7)
    assert rag.retrieve(question, corpus), "the query passes at the default threshold"

    corpus.settings.relevance_max_distance = -1.0

    assert rag.retrieve(question, corpus) == []


def test_question_against_an_empty_corpus_returns_unavailable(service, stub):
    result = rag.ask("What is the parental leave entitlement?", service)

    assert result["answerable"] is False
    assert result["answer"] == UNAVAILABLE_MESSAGE
    assert result["retrieved_count"] == 0
    assert stub.call_count == 0


def test_retrieval_requires_a_model_service(settings):
    """Without a key and without an injected embedder, retrieval is refused."""
    settings.gemini_api_key = None
    bare = Service(settings=settings)

    with pytest.raises(rag.IngestError) as caught:
        rag.retrieve("anything", bare)

    assert caught.value.status_code == 503
    assert "GEMINI_API_KEY" in caught.value.detail


# Document lifecycle


def test_delete_removes_every_chunk_of_a_document(corpus, store_of):
    document_id = "benefits_guide"
    assert chains.read_chunks(store_of(), where={"document_id": document_id})["ids"]

    assert rag.delete_document(document_id, corpus) is True

    assert not chains.read_chunks(store_of(), where={"document_id": document_id})["ids"]
    survivor = "employee_handbook"
    assert chains.read_chunks(store_of(), where={"document_id": survivor})["ids"], (
        "deleting one document must leave the others alone"
    )


def test_deleting_an_unknown_document_reports_false(corpus):
    assert rag.delete_document("no_such_document", corpus) is False


def test_reingesting_the_same_name_replaces_chunks_instead_of_duplicating_them(
    corpus, store_of
):
    from tests.conftest import sample_bytes

    first = rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", corpus)
    second = rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", corpus)

    assert first["chunk_count"] == second["chunk_count"]
    assert first["document_id"] == second["document_id"] == "benefits_guide"
    stored = chains.read_chunks(store_of(), where={"document_id": "benefits_guide"})
    assert len(stored["ids"]) == second["chunk_count"], "re-ingest must not accumulate chunks"


def test_listing_reports_one_entry_per_document_with_its_chunk_count(corpus):
    listed = {item["document_id"]: item for item in rag.list_documents(corpus)}

    assert set(listed) == {
        "employee_handbook",
        "it_security_policy",
        "benefits_guide",
        "remote_work_policy",
    }
    for document_id, item in listed.items():
        assert item["chunk_count"] >= 12, f"{document_id} produced {item['chunk_count']} chunks"
        assert item["filename"].endswith((".pdf", ".txt", ".md"))


def test_the_whole_corpus_produces_at_least_sixty_chunks(corpus):
    total = sum(item["chunk_count"] for item in rag.list_documents(corpus))

    assert total >= 60


# Chunking invariants


def test_chunks_respect_the_configured_size(corpus, store_of):
    records = chains.read_chunks(
        store_of(), where={"document_id": "employee_handbook"}, include=["documents"]
    )

    for text in records["documents"]:
        assert len(text) <= corpus.settings.chunk_size


def test_chunks_carry_exact_start_offsets(service, write_temp, sample_text):
    """`add_start_index` must point at the real slice of the source text."""
    chunks = chains.load_and_split(
        write_temp(sample_text, "handbook.txt"),
        ".txt",
        service.settings,
        {
            "document_id": "handbook",
            "document_name": "handbook.txt",
            "media_type": "text/plain",
        },
    )

    assert chunks
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        assert sample_text[start : start + len(chunk.page_content)] == chunk.page_content


def test_consecutive_chunks_overlap_or_touch(service, write_temp, sample_text):
    """With a configured overlap, the splitter never leaves a positive gap."""
    chunks = chains.load_and_split(
        write_temp(sample_text, "handbook.txt"),
        ".txt",
        service.settings,
        {
            "document_id": "handbook",
            "document_name": "handbook.txt",
            "media_type": "text/plain",
        },
    )

    assert len(chunks) > 1, "the fixture document must produce more than one chunk"
    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous.metadata["start_index"] + len(previous.page_content)
        gap = current.metadata["start_index"] - previous_end
        assert gap <= 0, f"unexpected gap of {gap} characters between chunks"


def test_overlap_is_present_between_consecutive_chunks(service, write_temp, sample_text):
    """A non-zero overlap means some tail text is repeated, not merely adjacent."""
    chunks = chains.load_and_split(
        write_temp(sample_text, "handbook.txt"),
        ".txt",
        service.settings,
        {
            "document_id": "handbook",
            "document_name": "handbook.txt",
            "media_type": "text/plain",
        },
    )

    overlapped = 0
    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous.metadata["start_index"] + len(previous.page_content)
        shared = previous_end - current.metadata["start_index"]
        if shared > 0:
            overlapped += 1
    assert overlapped, "expected at least one overlapping boundary"


def test_chunk_identifiers_are_explicit_and_unique(corpus, store_of):
    records = chains.read_chunks(
        store_of(), where={"document_id": "it_security_policy"}, include=["metadatas"]
    )

    ids = records["ids"]
    assert len(ids) == len(set(ids)), "chunk identifiers must be unique"
    for identifier in ids:
        assert identifier.startswith("it_security_policy::")
    indexes = sorted(metadata["chunk_index"] for metadata in records["metadatas"])
    assert indexes == list(range(len(indexes))), "chunk indexes must run 0..n-1"


def test_whitespace_only_chunks_are_dropped(settings, write_temp):
    chunks = chains.load_and_split(
        write_temp("   \n\n   \t  \n", "blank.txt"),
        ".txt",
        settings,
        {
            "document_id": "blank",
            "document_name": "blank.txt",
            "media_type": "text/plain",
        },
    )

    assert chunks == []


def test_pdf_chunks_carry_a_one_based_page_number(corpus, chunk_reader):
    rows = chunk_reader("employee_handbook")

    assert rows, "the sample PDF should be indexed"
    pages = {metadata["page_number"] for _, _, metadata in rows}
    assert min(pages) >= 1, "page numbers are one based for a reader"


def test_text_documents_carry_no_page_number(corpus, chunk_reader):
    for document_id in ("benefits_guide", "remote_work_policy", "it_security_policy"):
        for _, _, metadata in chunk_reader(document_id):
            assert "page_number" not in metadata, f"{document_id} is not a PDF"


# Evidence


def test_evidence_excerpts_are_verbatim_slices_of_indexed_chunks(corpus, verbatim_query, store_of):
    question = verbatim_query("remote_work_policy", 4)

    result = rag.ask(question, corpus)

    assert result["sources"], "a passing query must return evidence"
    stored = chains.read_chunks(store_of(), include=["documents"])
    for source in result["sources"]:
        assert any(
            source["excerpt"] in text for text in stored["documents"]
        ), "an excerpt must appear verbatim inside an indexed chunk"


def test_evidence_offsets_address_the_source_document(corpus, verbatim_query, store_of):
    """`char_start` and `char_end` must index into the document, not into the chunk.

    A text document is loaded as one string, so the offsets it produces address
    the whole file. The span they describe is exactly the stored chunk.
    """
    question = verbatim_query("remote_work_policy", 4)

    result = rag.ask(question, corpus)

    assert result["sources"]
    for source in result["sources"]:
        if source["document_name"].endswith(".pdf"):
            continue
        location = source["location"]
        stored = chains.read_chunks(
            store_of(),
            ids=[f"{source['document_id']}::{location['chunk_index']}"],
            include=["documents"],
        )
        chunk_text = stored["documents"][0]
        span_length = location["char_end"] - location["char_start"]
        assert span_length == len(chunk_text), "the span must be the chunk length"
        document_text = (SAMPLE_DIR / source["document_name"]).read_text(encoding="utf-8")
        span = document_text[location["char_start"] : location["char_end"]]
        assert span == chunk_text, "the offsets must address the real slice of the file"
        assert source["excerpt"] == chunk_text[: len(source["excerpt"])]


def test_evidence_offsets_for_a_pdf_address_the_page_text(corpus, verbatim_query, store_of):
    """PDF offsets are page relative, which is why evidence also carries a page."""
    question = verbatim_query("employee_handbook", 3)

    result = rag.ask(question, corpus)

    assert result["sources"]
    for source in result["sources"]:
        location = source["location"]
        assert "page_number" in location and location["page_number"] >= 1
        stored = chains.read_chunks(
            store_of(),
            ids=[f"{source['document_id']}::{location['chunk_index']}"],
            include=["documents", "metadatas"],
        )
        chunk_text = stored["documents"][0]
        assert location["char_end"] - location["char_start"] == len(chunk_text)
        assert source["excerpt"] == chunk_text[: len(source["excerpt"])]


def test_evidence_only_cites_chunks_that_passed_the_gate(corpus, verbatim_query):
    """A citable chunk that failed the threshold can never appear as evidence."""
    question = verbatim_query("remote_work_policy", 4)
    gated = {result["metadata"]["chunk_id"] for result in rag.retrieve(question, corpus)}

    result = rag.ask(question, corpus)

    for source in result["sources"]:
        chunk_id = f"{source['document_id']}::{source['location']['chunk_index']}"
        assert chunk_id in gated


def test_evidence_caps_the_excerpt_length(corpus, verbatim_query):
    question = verbatim_query("remote_work_policy", 4)

    result = rag.ask(question, corpus)

    for source in result["sources"]:
        assert len(source["excerpt"]) <= 500


def test_counts_describe_the_response(corpus, verbatim_query):
    question = verbatim_query("it_security_policy", 5)

    result = rag.ask(question, corpus)

    assert result["answerable"] is True
    assert result["evidence_count"] == len(result["sources"])
    assert result["retrieved_count"] >= result["evidence_count"]
    assert result["distinct_document_count"] == len(
        {source["document_id"] for source in result["sources"]}
    )


# The fake embedder itself


def test_identical_text_embeds_identically_offline(embeddings):
    """Determinism is what makes the offline suite stable."""
    first = embeddings.embed_query("accrues monthly at 10 hours")
    second = embeddings.embed_query("accrues monthly at 10 hours")

    assert first == second
    assert len(first) == embeddings.dimensions


def test_shared_words_raise_cosine_similarity(embeddings):
    """The fake embedder is overlap sensitive, not merely random."""
    text = "unused hours carry over into the next calendar year"
    shared = embeddings.embed_query(text)

    shared_distance = 1 - sum(
        a * b for a, b in zip(shared, embeddings.embed_query(text))
    )
    unrelated_distance = 1 - sum(
        a * b
        for a, b in zip(
            shared,
            embeddings.embed_query("rotor diameter coastal wind farm turbines"),
        )
    )

    assert shared_distance == pytest.approx(0.0, abs=1e-9)
    assert unrelated_distance > 0.5