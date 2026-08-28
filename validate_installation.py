"""Non-destructive deployment and data-integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from create_embeddings import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, DEFAULT_INPUT, iter_bids
from gem_hybrid_retrieval import HybridRetriever


PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_DIR / "downloads"
EXPECTED_DIMENSION = 384
REQUIRED_PACKAGES = {
    "streamlit": "1.60.0",
    "chromadb": "1.5.9",
    "sentence-transformers": "5.6.1",
    "PyMuPDF": "1.28.0",
    "requests": "2.34.2",
}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def pass_(self, message: str) -> None:
        print(f"[PASS] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(full: bool = False, allow_empty: bool = False) -> int:
    report = Report()
    print(f"Validating: {PROJECT_DIR}")

    if sys.version_info[:2] == (3, 11):
        report.pass_(f"Python {sys.version.split()[0]}")
    else:
        report.warn(f"Python {sys.version.split()[0]} is active; deployment is tested with Python 3.11.")

    for package, expected in REQUIRED_PACKAGES.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report.fail(f"Missing package: {package}=={expected}")
            continue
        if installed == expected:
            report.pass_(f"{package}=={installed}")
        else:
            report.warn(f"{package}=={installed}; deployment pins {expected}")

    manifest = DOWNLOADS_DIR / "downloaded_bid_manifest.json"
    bids_dir = DOWNLOADS_DIR / "bids"
    pending = DOWNLOADS_DIR / "pending_chunk_sync.json"
    if pending.exists():
        report.fail("An unacknowledged embedding sync exists; run maintenance to recover it.")

    if not manifest.is_file():
        (report.warn if allow_empty else report.fail)(f"Downloader manifest missing: {manifest}")
    else:
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            tracked = len(manifest_data.get("downloaded_bid_ids", []))
            report.pass_(f"Downloader manifest is valid and tracks {tracked:,} bid IDs")
        except (OSError, ValueError, TypeError) as error:
            report.fail(f"Downloader manifest is invalid: {error}")

    pdf_count = sum(1 for _ in bids_dir.rglob("*.pdf")) if bids_dir.is_dir() else 0
    if pdf_count:
        report.pass_(f"Found {pdf_count:,} local PDF files")
    else:
        (report.warn if allow_empty else report.fail)(f"No PDFs found under {bids_dir}")

    scanned_bids = 0
    scanned_chunks = 0
    duplicate_ids = 0
    mismatched_duplicate_ids = 0
    seen_text_hash_by_id: dict[str, str] = {}
    if not DEFAULT_INPUT.is_file():
        (report.warn if allow_empty else report.fail)(f"Chunk file missing: {DEFAULT_INPUT}")
    else:
        try:
            for bid in iter_bids(DEFAULT_INPUT):
                scanned_bids += 1
                for chunk in bid.get("chunks", []):
                    scanned_chunks += 1
                    chunk_id = str(chunk.get("chunk_id", ""))
                    text_hash = hashlib.sha256(str(chunk.get("text", "")).encode("utf-8")).hexdigest()
                    if chunk_id in seen_text_hash_by_id:
                        duplicate_ids += 1
                        if seen_text_hash_by_id[chunk_id] != text_hash:
                            mismatched_duplicate_ids += 1
                    else:
                        seen_text_hash_by_id[chunk_id] = text_hash
                if not full and scanned_bids >= 1_000:
                    break
            scope = "complete corpus" if full else "first 1,000 bids"
            report.pass_(f"Parsed {scanned_bids:,} bids and {scanned_chunks:,} chunks ({scope})")
            if mismatched_duplicate_ids:
                report.warn(
                    f"Found {mismatched_duplicate_ids:,} duplicate chunk IDs with different text. "
                    "The imported index can run, but rechunk before any complete index rebuild."
                )
            elif duplicate_ids:
                report.warn(f"Found {duplicate_ids:,} identical duplicate chunk IDs")
        except Exception as error:
            report.fail(f"Chunk file could not be streamed: {error}")

    collection_count = None
    if not (DEFAULT_CHROMA_PATH / "chroma.sqlite3").is_file():
        (report.warn if allow_empty else report.fail)(f"Chroma database missing: {DEFAULT_CHROMA_PATH}")
    else:
        try:
            import chromadb

            collection = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_PATH)).get_collection(DEFAULT_COLLECTION)
            collection_count = collection.count()
            if collection_count or allow_empty:
                report.pass_(f"Chroma collection contains {collection_count:,} vectors")
            else:
                report.fail("Chroma collection is empty")
            if collection_count:
                embeddings = collection.peek(limit=1).get("embeddings")
                dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else None
                if dimension == EXPECTED_DIMENSION:
                    report.pass_(f"Embedding dimension is {dimension} (BGE-small compatible)")
                else:
                    report.fail(f"Embedding dimension is {dimension}; expected {EXPECTED_DIMENSION}")
        except Exception as error:
            report.fail(f"Could not open Chroma collection {DEFAULT_COLLECTION!r}: {error}")

    if full and collection_count is not None and seen_text_hash_by_id:
        unique_chunk_count = len(seen_text_hash_by_id)
        if unique_chunk_count == collection_count:
            report.pass_("Unique JSON chunk count matches Chroma vector count")
        else:
            report.warn(
                f"JSON has {unique_chunk_count:,} unique chunk IDs but Chroma has {collection_count:,} vectors."
            )

    if collection_count is not None:
        try:
            retriever = HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION)
            lexical_state = retriever.lexical.state()
            if (
                lexical_state.get("collection") == DEFAULT_COLLECTION
                and lexical_state.get("chunk_count") == collection_count
            ):
                report.pass_("Lexical index matches the Chroma collection")
            else:
                report.warn("Lexical index is missing or stale; maintenance can rebuild it")
        except Exception as error:
            report.warn(f"Could not inspect lexical index: {error}")

    print(f"\nValidation complete: {report.failures} failure(s), {report.warnings} warning(s).")
    return 1 if report.failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Lets Bid deployment without changing data.")
    parser.add_argument("--full", action="store_true", help="Scan the complete chunk file instead of a sample.")
    parser.add_argument("--allow-empty", action="store_true", help="Use before importing or building runtime data.")
    args = parser.parse_args()
    raise SystemExit(validate(full=args.full, allow_empty=args.allow_empty))


if __name__ == "__main__":
    main()
