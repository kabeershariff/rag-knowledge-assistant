"""Load the sample documents through the ingestion pipeline and report chunk counts.

Run from the repository root:

    python scripts/ingest_samples.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.documents import DocumentRegistry, ingest_document
from app.rag import Service

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def main() -> int:
    settings = get_settings()
    service = Service(settings=settings, registry=DocumentRegistry())

    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        content = path.read_bytes()
        record = ingest_document(content, path.name, service)
        print(
            f"{record['filename']}: {record['chunk_count']} chunks "
            f"({record['media_type']})"
        )

    print(f"total documents: {service.registry.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
