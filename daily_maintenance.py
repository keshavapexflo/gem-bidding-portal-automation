"""Restart-safe scheduled ingestion, embedding, and expiry maintenance."""

from __future__ import annotations

import argparse
import json
import os
import socket
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from create_embeddings import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, DEFAULT_MODEL, build_embeddings
from downloading_v2 import main as download_new_bids
from gem_expiry_cleanup import archive_expired_bids
from gem_extract_and_chunks import process_directory
from gem_hybrid_retrieval import HybridRetriever


PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_DIR / "downloads"
CHUNK_FILE = PROJECT_DIR / "bid_chunks.json"
NEXT_CHUNK_FILE = PROJECT_DIR / "bid_chunks.next.json"
PENDING_SYNC_FILE = DOWNLOADS_DIR / "pending_chunk_sync.json"
STAGED_SYNC_FILE = DOWNLOADS_DIR / "chunk_sync.next.json"
STATE_FILE = DOWNLOADS_DIR / "maintenance_state.json"
LOCK_FILE = DOWNLOADS_DIR / ".maintenance.lock"
LOCK_STALE_AFTER = timedelta(hours=18)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def save_state(state: dict) -> None:
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def expiry_is_due(state: dict) -> bool:
    value = state.get("last_expiry_check_at")
    if not value:
        return True
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return utcnow() - parsed.astimezone(timezone.utc) >= timedelta(days=7)
    except (TypeError, ValueError):
        return True


def acquire_lock() -> None:
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": utcnow().isoformat(),
    }
    for attempt in range(2):
        try:
            handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            try:
                existing = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
                started = datetime.fromisoformat(str(existing.get("started_at", "")).replace("Z", "+00:00"))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                is_stale = utcnow() - started.astimezone(timezone.utc) >= LOCK_STALE_AFTER
            except (OSError, ValueError, TypeError):
                try:
                    modified = datetime.fromtimestamp(LOCK_FILE.stat().st_mtime, tz=timezone.utc)
                    is_stale = utcnow() - modified >= LOCK_STALE_AFTER
                except OSError:
                    is_stale = False
            if not is_stale or attempt:
                raise RuntimeError(
                    f"Another maintenance run is active, or its lock is less than "
                    f"{LOCK_STALE_AFTER.total_seconds() / 3600:.0f} hours old: {LOCK_FILE}"
                ) from error
            LOCK_FILE.unlink(missing_ok=True)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as lock:
            json.dump(payload, lock, indent=2)
        return
    raise RuntimeError(f"Could not acquire maintenance lock: {LOCK_FILE}")


def sync_change_count(sync_file: Path) -> int:
    sync = json.loads(sync_file.read_text(encoding="utf-8"))
    return len(sync.get("upsert_chunk_ids", [])) + len(sync.get("deleted_chunk_ids", []))


def apply_pending_sync() -> int:
    """Apply and acknowledge a journaled Chroma change set."""
    if not PENDING_SYNC_FILE.is_file():
        return 0
    if NEXT_CHUNK_FILE.is_file():
        os.replace(NEXT_CHUNK_FILE, CHUNK_FILE)
    if not CHUNK_FILE.is_file():
        raise RuntimeError("A pending embedding sync exists but bid_chunks.json is missing.")

    changed_chunks = sync_change_count(PENDING_SYNC_FILE)
    build_embeddings(Namespace(
        input=CHUNK_FILE,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection=DEFAULT_COLLECTION,
        model=DEFAULT_MODEL,
        batch_size=64,
        limit=0,
        reset=False,
        sync_file=PENDING_SYNC_FILE,
    ))

    retriever = HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION)
    lexical_state = retriever.lexical.state()
    lexical_is_current = (
        lexical_state.get("collection") == DEFAULT_COLLECTION
        and lexical_state.get("chunk_count") == retriever.collection.count()
    )
    if changed_chunks or not lexical_is_current:
        print("Refreshing the lexical index...", flush=True)
        retriever.build_lexical_index(rebuild=True)

    # Removing this file acknowledges the complete vector + lexical update.
    PENDING_SYNC_FILE.unlink()
    return changed_chunks


def stage_chunk_update() -> None:
    NEXT_CHUNK_FILE.unlink(missing_ok=True)
    STAGED_SYNC_FILE.unlink(missing_ok=True)
    process_directory(
        str(DOWNLOADS_DIR / "bids"),
        str(DOWNLOADS_DIR / "downloaded_bid_manifest.json"),
        str(NEXT_CHUNK_FILE),
        str(STAGED_SYNC_FILE),
        state_path=str(CHUNK_FILE) if CHUNK_FILE.is_file() else None,
    )
    # Journal intended vector changes before promoting the new chunk state.
    os.replace(STAGED_SYNC_FILE, PENDING_SYNC_FILE)
    os.replace(NEXT_CHUNK_FILE, CHUNK_FILE)


def run_maintenance(
    force_expiry: bool = False,
    skip_expiry: bool = False,
    apply_expiry: bool = False,
) -> None:
    acquire_lock()
    try:
        if PENDING_SYNC_FILE.is_file():
            print("[Recovery] Retrying the previous incomplete embedding sync...", flush=True)
            apply_pending_sync()

        print("[1/4] Downloading newly listed GeM bid PDFs...", flush=True)
        download_new_bids()

        print("[2/4] Chunking new or changed PDFs...", flush=True)
        stage_chunk_update()

        print("[3/4] Applying the journaled embedding changes...", flush=True)
        changed_chunks = apply_pending_sync()

        state = load_state()
        if not skip_expiry and (force_expiry or expiry_is_due(state)):
            mode = "apply" if apply_expiry else "dry run"
            print(f"[4/4] Running weekly GeM expiry check ({mode})...", flush=True)
            report = archive_expired_bids(apply=apply_expiry)
            state["last_expiry_check_at"] = utcnow().isoformat()
            state["last_expiry_report"] = report
            if apply_expiry and report["expired_chunk_count"]:
                print("Refreshing lexical index after expiry cleanup...", flush=True)
                HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION).build_lexical_index(rebuild=True)
        else:
            print("[4/4] Weekly expiry check is not due or was skipped.", flush=True)

        state["last_daily_run_at"] = utcnow().isoformat()
        state["last_changed_chunk_count"] = changed_chunks
        save_state(state)
        print("Maintenance complete.", flush=True)
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily GeM portal maintenance.")
    parser.add_argument("--force-expiry", action="store_true", help="Run the weekly expiry scan now.")
    parser.add_argument("--skip-expiry", action="store_true", help="Skip expiry scanning for this run.")
    parser.add_argument(
        "--apply-expiry",
        action="store_true",
        help="Actually archive expired bids. Without this, expiry is a safe dry run.",
    )
    args = parser.parse_args()
    run_maintenance(
        force_expiry=args.force_expiry,
        skip_expiry=args.skip_expiry,
        apply_expiry=args.apply_expiry,
    )


if __name__ == "__main__":
    main()
