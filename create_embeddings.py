"""Build the Chroma vector collection from ``bid_chunks.json``.

The chunk export is intentionally read as a stream: the current file is
large enough that ``json.load`` would need several gigabytes of memory.  Each
batch is embedded with BAAI/bge-small-en-v1.5 and then upserted into Chroma,
so rerunning this script safely resumes or refreshes existing chunk IDs.

Examples
--------
Create/update the default collection::

    python create_embeddings.py

Try a small batch first::

    python create_embeddings.py --limit 100

Rebuild the collection from scratch (deletes the named collection)::

    python create_embeddings.py --reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "bid_chunks.json"
DEFAULT_CHROMA_PATH = PROJECT_DIR / "chroma_db"
DEFAULT_BOILERPLATE_HASHES = PROJECT_DIR / "downloads" / "boilerplate_hashes.json"
DEFAULT_COLLECTION = "gem_bid_chunks"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Pin one reviewed Hugging Face snapshot so corpus and query embeddings cannot
# drift after a model-repository update.
DEFAULT_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def iter_bids(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from the top-level ``bids`` array without loading it all.

    This parser is deliberately limited to the stable output format produced
    by ``gem_extract_and_chunks.py``: ``{"bids": [{...}, ...]}``.
    """
    decoder = json.JSONDecoder()
    marker = re.compile(r'"bids"\s*:\s*\[')
    buffer = ""
    array_started = False

    with path.open("r", encoding="utf-8") as source:
        while True:
            data = source.read(1024 * 1024)
            if data:
                buffer += data
            end_of_file = not data

            if not array_started:
                match = marker.search(buffer)
                if match is None:
                    if end_of_file:
                        raise ValueError(f"No top-level 'bids' array found in {path}")
                    # Keep enough text to recognise a marker split between reads.
                    buffer = buffer[-32:]
                    continue
                buffer = buffer[match.end():]
                array_started = True

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if not buffer:
                    break
                try:
                    bid, consumed = decoder.raw_decode(buffer)
                except json.JSONDecodeError as error:
                    if end_of_file:
                        raise ValueError(f"Malformed JSON in {path}: {error}") from error
                    break
                if not isinstance(bid, dict):
                    raise ValueError("Each item in the 'bids' array must be an object.")
                yield bid
                buffer = buffer[consumed:]

            if end_of_file:
                raise ValueError(f"The 'bids' array in {path} was not closed.")


def normalised_text_hash(text: str) -> str:
    normalised = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def load_boilerplate_hashes(path: Path = DEFAULT_BOILERPLATE_HASHES) -> set[str]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("text_hashes", []) if isinstance(data, dict) else data
        return {str(value) for value in values}
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Could not read boilerplate registry {path}: {error}") from error


def chroma_metadata(
    metadata: Mapping[str, Any],
    chunk: Mapping[str, Any],
    *,
    is_boilerplate: bool = False,
) -> dict[str, str | int | float | bool]:
    """Return metadata accepted by Chroma (scalar values only, no ``None``)."""
    result: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        else:
            result[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    # These fields describe the individual chunk rather than its parent bid.
    result["chunk_type"] = str(chunk.get("chunk_type", "unknown"))
    if chunk.get("clause_no") is not None:
        result["clause_no"] = str(chunk["clause_no"])
    if is_boilerplate:
        result["is_boilerplate"] = True
    return result


def embed_and_upsert(
    collection: Any,
    model: Any,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
) -> int:
    """Create normalized BGE vectors for one batch and persist them."""
    if not ids:
        return 0
    vectors = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors.tolist(),
    )
    return len(ids)


def batched(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def validate_chunk_ids_before_reset(path: Path) -> None:
    """Protect an existing collection from being reset for invalid input."""
    seen: dict[str, str] = {}
    mismatched = []
    for bid in iter_bids(path):
        for chunk in bid.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                continue
            digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            previous = seen.get(chunk_id)
            if previous is not None and previous != digest:
                mismatched.append(chunk_id)
                if len(mismatched) >= 10:
                    break
            seen.setdefault(chunk_id, digest)
        if len(mismatched) >= 10:
            break
    if mismatched:
        sample = ", ".join(repr(value) for value in mismatched[:3])
        raise RuntimeError(
            "Refusing to reset Chroma because duplicate chunk IDs contain different text. "
            f"Rechunk with the current schema first. Examples: {sample}"
        )


def build_embeddings(args: argparse.Namespace) -> int:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Missing dependency. Activate the project environment and install "
            "'chromadb sentence-transformers'."
        ) from error

    if not args.input.is_file():
        raise SystemExit(f"Chunk file not found: {args.input}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    selected_chunk_ids: set[str] | None = None
    deleted_chunk_ids: list[str] = []
    if args.sync_file:
        if not args.sync_file.is_file():
            raise SystemExit(f"Chunk sync file not found: {args.sync_file}")
        try:
            sync = json.loads(args.sync_file.read_text(encoding="utf-8"))
            selected_chunk_ids = {str(chunk_id) for chunk_id in sync.get("upsert_chunk_ids", [])}
            deleted_chunk_ids = [str(chunk_id) for chunk_id in sync.get("deleted_chunk_ids", [])]
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"Could not read chunk sync file: {error}") from error

    if args.reset:
        print("Preflighting chunk IDs before resetting Chroma...", flush=True)
        validate_chunk_ids_before_reset(args.input)

    model = None
    model_dimension = None
    if selected_chunk_ids is None or selected_chunk_ids:
        model_revision = getattr(args, "model_revision", DEFAULT_MODEL_REVISION)
        print(f"Loading embedding model: {args.model} @ {model_revision}", flush=True)
        model = SentenceTransformer(args.model, revision=model_revision)
        model_dimension = model.get_sentence_embedding_dimension()

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except ValueError:
            # Chroma raises ValueError when the requested collection is absent.
            pass
        else:
            print(f"Deleted existing collection: {args.collection}", flush=True)

    # Embeddings are supplied explicitly below.  This mirrors HybridRetriever,
    # which supplies its own query vectors and does not use Chroma's default EF.
    collection = client.get_or_create_collection(name=args.collection)

    if selected_chunk_ids is not None and not selected_chunk_ids:
        for ids_to_delete in batched(deleted_chunk_ids, 1_000):
            collection.delete(ids=ids_to_delete)
        if deleted_chunk_ids:
            print(f"Removed {len(deleted_chunk_ids):,} stale chunk(s) from Chroma", flush=True)
        print(
            f"No new or changed chunks to embed. Collection '{args.collection}' contains "
            f"{collection.count():,} chunks.",
            flush=True,
        )
        return 0

    if not args.reset and collection.count():
        stored_embeddings = collection.peek(limit=1).get("embeddings")
        if stored_embeddings is not None and len(stored_embeddings):
            stored_dimension = len(stored_embeddings[0])
            if model_dimension and stored_dimension != model_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch: collection={stored_dimension}, "
                    f"model={model_dimension}. Use the same model as the initial build."
                )
    if model is None:
        raise RuntimeError("Embedding model was not initialised for a non-empty upsert.")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    processed = 0
    skipped = 0
    seen_text_hash_by_id: dict[str, str] = {}
    boilerplate_hashes = load_boilerplate_hashes()

    for bid in iter_bids(args.input):
        for chunk in bid.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not chunk_id or not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            if selected_chunk_ids is not None and chunk_id not in selected_chunk_ids:
                continue
            clean_text = text.strip()
            clean_text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            previous_text_hash = seen_text_hash_by_id.get(chunk_id)
            if previous_text_hash is not None:
                if previous_text_hash == clean_text_hash:
                    skipped += 1
                    continue
                raise RuntimeError(
                    f"Chunk ID {chunk_id!r} is used by different text. Re-run chunking "
                    "with the current chunk schema before embedding."
                )
            seen_text_hash_by_id[chunk_id] = clean_text_hash

            raw_metadata = chunk.get("metadata")
            if not isinstance(raw_metadata, Mapping):
                raw_metadata = bid.get("metadata") if isinstance(bid.get("metadata"), Mapping) else {}
            ids.append(chunk_id)
            documents.append(clean_text)
            metadatas.append(chroma_metadata(
                raw_metadata,
                chunk,
                is_boilerplate=normalised_text_hash(clean_text) in boilerplate_hashes,
            ))

            target_reached = args.limit and processed + len(ids) >= args.limit
            if len(ids) >= args.batch_size or target_reached:
                if target_reached:
                    remaining = args.limit - processed
                    ids, documents, metadatas = ids[:remaining], documents[:remaining], metadatas[:remaining]
                processed += embed_and_upsert(collection, model, ids, documents, metadatas)
                print(f"Embedded and stored {processed:,} chunks", flush=True)
                ids.clear()
                documents.clear()
                metadatas.clear()
                if target_reached:
                    break
        if args.limit and processed >= args.limit:
            break

    if ids and (not args.limit or processed < args.limit):
        if args.limit:
            remaining = args.limit - processed
            ids, documents, metadatas = ids[:remaining], documents[:remaining], metadatas[:remaining]
        processed += embed_and_upsert(collection, model, ids, documents, metadatas)
        print(f"Embedded and stored {processed:,} chunks", flush=True)

    # Deletions happen only after every requested upsert succeeds. If model
    # loading or embedding fails, the previous searchable vectors remain.
    for ids_to_delete in batched(deleted_chunk_ids, 1_000):
        collection.delete(ids=ids_to_delete)
    if deleted_chunk_ids:
        print(f"Removed {len(deleted_chunk_ids):,} stale chunk(s) from Chroma", flush=True)

    print(
        f"Complete. Processed {processed:,} chunks; skipped {skipped:,}. "
        f"Collection '{args.collection}' now contains {collection.count():,} chunks.",
        flush=True,
    )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Chroma embeddings from GeM bid chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to bid_chunks.json")
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH, help="Chroma persistence directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks embedded per batch")
    parser.add_argument("--limit", type=int, default=0, help="Maximum chunks to embed (0 means all)")
    parser.add_argument(
        "--sync-file", type=Path,
        help="Only embed chunk IDs listed by gem_extract_and_chunks.py; also remove listed stale IDs.",
    )
    parser.add_argument("--reset", action="store_true", help="Delete the named collection before embedding")
    return parser.parse_args()


if __name__ == "__main__":
    build_embeddings(parse_args())
