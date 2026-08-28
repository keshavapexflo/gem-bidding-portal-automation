"""Build a durable boilerplate registry and synchronize Chroma metadata."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from create_embeddings import (
    DEFAULT_BOILERPLATE_HASHES,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_INPUT,
    iter_bids,
    normalised_text_hash,
)
from gem_hybrid_retrieval import HybridRetriever


def build_registry(input_path: Path, output_path: Path, minimum_distinct_bids: int) -> set[str]:
    if minimum_distinct_bids < 2:
        raise ValueError("minimum_distinct_bids must be at least 2")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix="boilerplate_", suffix=".sqlite3", dir=output_path.parent)
    os.close(handle)
    analysis_db = Path(temporary_name)
    try:
        with closing(sqlite3.connect(analysis_db)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE appearances (text_hash TEXT NOT NULL, bid_id TEXT NOT NULL, "
                "PRIMARY KEY(text_hash, bid_id)) WITHOUT ROWID"
            )
            pending: list[tuple[str, str]] = []
            bid_count = 0
            chunk_count = 0
            for bid in iter_bids(input_path):
                bid_count += 1
                bid_id = str(bid.get("bid_id", ""))
                for chunk in bid.get("chunks", []):
                    text = chunk.get("text")
                    if isinstance(text, str) and text.strip():
                        pending.append((normalised_text_hash(text), bid_id))
                        chunk_count += 1
                    if len(pending) >= 10_000:
                        db.executemany("INSERT OR IGNORE INTO appearances VALUES (?, ?)", pending)
                        db.commit()
                        pending.clear()
                if bid_count % 5_000 == 0:
                    print(f"Analysed {bid_count:,} bids / {chunk_count:,} chunks", flush=True)
            if pending:
                db.executemany("INSERT OR IGNORE INTO appearances VALUES (?, ?)", pending)
                db.commit()

            rows = db.execute(
                "SELECT text_hash FROM appearances GROUP BY text_hash HAVING COUNT(*) >= ?",
                (minimum_distinct_bids,),
            ).fetchall()
            hashes = {str(row[0]) for row in rows}

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "minimum_distinct_bids": minimum_distinct_bids,
            "text_hashes": sorted(hashes),
        }
        temporary_output = output_path.with_suffix(".tmp")
        temporary_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary_output, output_path)
        print(f"Registry contains {len(hashes):,} boilerplate text hashes: {output_path}", flush=True)
        return hashes
    finally:
        analysis_db.unlink(missing_ok=True)
        analysis_db.with_name(f"{analysis_db.name}-wal").unlink(missing_ok=True)
        analysis_db.with_name(f"{analysis_db.name}-shm").unlink(missing_ok=True)


def update_chroma(hashes: set[str], batch_size: int = 1_000) -> int:
    import chromadb

    collection = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_PATH)).get_collection(DEFAULT_COLLECTION)
    total = collection.count()
    updated = 0
    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        changed_ids = []
        changed_metadatas = []
        for chunk_id, document, metadata in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            revised = dict(metadata or {})
            should_flag = normalised_text_hash(str(document or "")) in hashes
            is_flagged = revised.get("is_boilerplate") is True
            if should_flag == is_flagged:
                continue
            if should_flag:
                revised["is_boilerplate"] = True
            else:
                revised.pop("is_boilerplate", None)
            changed_ids.append(chunk_id)
            changed_metadatas.append(revised)
        if changed_ids:
            collection.update(ids=changed_ids, metadatas=changed_metadatas)
            updated += len(changed_ids)
        print(f"Checked {min(offset + batch_size, total):,}/{total:,} vectors", flush=True)
    print(f"Updated boilerplate metadata on {updated:,} vectors", flush=True)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh exact-text boilerplate classification.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_BOILERPLATE_HASHES)
    parser.add_argument("--minimum-distinct-bids", type=int, default=8)
    parser.add_argument("--skip-chroma-update", action="store_true")
    args = parser.parse_args()

    hashes = build_registry(args.input, args.output, args.minimum_distinct_bids)
    if not args.skip_chroma_update:
        update_chroma(hashes)
        HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION).build_lexical_index(rebuild=True)


if __name__ == "__main__":
    main()
