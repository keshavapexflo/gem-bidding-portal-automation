"""One command-line entry point for initialisation and ongoing operations."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from create_embeddings import (
    DEFAULT_BOILERPLATE_HASHES,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    build_embeddings,
)
from daily_maintenance import DOWNLOADS_DIR, run_maintenance
from downloading_v2 import main as download_bids
from gem_expiry_cleanup import archive_expired_bids
from gem_extract_and_chunks import process_directory
from gem_hybrid_retrieval import HybridRetriever
from refresh_boilerplate import build_registry, update_chroma


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = DOWNLOADS_DIR / "downloaded_bid_manifest.json"
BIDS_DIR = DOWNLOADS_DIR / "bids"
SYNC_FILE = DOWNLOADS_DIR / "initial_chunk_sync.json"


def ensure_runtime_directories() -> None:
    for path in (
        BIDS_DIR,
        DOWNLOADS_DIR / "expired",
        DOWNLOADS_DIR / "logs",
        PROJECT_DIR / "static",
        DEFAULT_CHROMA_PATH,
    ):
        path.mkdir(parents=True, exist_ok=True)


def initialise(args: argparse.Namespace) -> None:
    if args.reset_index and args.limit:
        raise SystemExit("Do not combine --reset-index with --limit; that would create a partial production index.")
    ensure_runtime_directories()
    if not args.skip_download:
        print("[1/5] Downloading the initial GeM corpus...", flush=True)
        download_bids()
    elif not BIDS_DIR.exists():
        raise SystemExit(f"--skip-download was used but the bids directory is missing: {BIDS_DIR}")

    print("[2/5] Extracting metadata and chunks...", flush=True)
    process_directory(
        str(BIDS_DIR),
        str(MANIFEST_FILE),
        str(DEFAULT_INPUT),
        str(SYNC_FILE),
        force_reprocess=args.force_rechunk,
    )

    print("[3/5] Building the vector collection...", flush=True)
    build_embeddings(Namespace(
        input=DEFAULT_INPUT,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection=DEFAULT_COLLECTION,
        model=DEFAULT_MODEL,
        batch_size=args.batch_size,
        limit=args.limit,
        reset=args.reset_index,
        sync_file=None if args.reset_index else SYNC_FILE,
    ))

    if not args.skip_boilerplate:
        print("[4/5] Classifying repeated boilerplate text...", flush=True)
        hashes = build_registry(DEFAULT_INPUT, DEFAULT_BOILERPLATE_HASHES, 8)
        update_chroma(hashes)
    else:
        print("[4/5] Boilerplate classification skipped by request.", flush=True)

    print("[5/5] Building the lexical index...", flush=True)
    HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION).build_lexical_index(rebuild=True)
    print("Initialisation complete. Run .\\start_portal.ps1 to open the portal.", flush=True)


def rebuild_index(args: argparse.Namespace) -> None:
    if not DEFAULT_INPUT.is_file():
        raise SystemExit(f"Chunk file not found: {DEFAULT_INPUT}")
    build_embeddings(Namespace(
        input=DEFAULT_INPUT,
        chroma_path=DEFAULT_CHROMA_PATH,
        collection=DEFAULT_COLLECTION,
        model=DEFAULT_MODEL,
        batch_size=args.batch_size,
        limit=0,
        reset=True,
        sync_file=None,
    ))
    HybridRetriever(DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION).build_lexical_index(rebuild=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lets Bid unified deployment pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("initialise", help="Run the Phase 1 initial build.")
    init_parser.add_argument("--skip-download", action="store_true", help="Use PDFs already in downloads/bids.")
    init_parser.add_argument("--reset-index", action="store_true", help="Recreate the Chroma collection.")
    init_parser.add_argument("--batch-size", type=int, default=64)
    init_parser.add_argument("--limit", type=int, default=0, help="Test limit; zero means all chunks.")
    init_parser.add_argument("--skip-boilerplate", action="store_true")
    init_parser.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Reprocess every PDF using the current chunk-ID schema.",
    )
    init_parser.set_defaults(handler=initialise)

    maintenance_parser = subparsers.add_parser("maintain", help="Run incremental Phase 2 maintenance.")
    maintenance_parser.add_argument("--force-expiry", action="store_true")
    maintenance_parser.add_argument("--skip-expiry", action="store_true")
    maintenance_parser.add_argument("--apply-expiry", action="store_true")
    maintenance_parser.set_defaults(handler=lambda values: run_maintenance(
        force_expiry=values.force_expiry,
        skip_expiry=values.skip_expiry,
        apply_expiry=values.apply_expiry,
    ))

    rebuild_parser = subparsers.add_parser("rebuild-index", help="Rebuild Chroma from bid_chunks.json.")
    rebuild_parser.add_argument("--batch-size", type=int, default=64)
    rebuild_parser.set_defaults(handler=rebuild_index)

    expiry_parser = subparsers.add_parser("expiry-report", help="Scan expiry without changing local data.")
    expiry_parser.set_defaults(handler=lambda _values: print(archive_expired_bids(apply=False)))

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
