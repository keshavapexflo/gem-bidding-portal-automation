# Lets Bid - unified deployment

This directory combines the Phase 1 initial ingestion/build workflow with the
Phase 2 local portal and incremental maintenance workflow.

## Recommended deployment with an existing corpus

Open PowerShell in this directory and run:

```powershell
.\setup_new_laptop.ps1 -DataSource "E:\LetsBidData"
.\validate.ps1 -Full
.\start_portal.ps1
```

The data source must contain:

```text
bid_chunks.json
downloads\
chroma_db\
```

After testing searches and PDF downloads, run one maintenance cycle without
destructive expiry cleanup:

```powershell
.\run_maintenance.ps1 -SkipExpiry
.\enable_automation.ps1 -Time 11:00
```

Scheduled expiry checks are dry-run only by default. Enable automatic archival
only after reviewing the reports and taking a backup:

```powershell
.\backup.ps1
.\enable_automation.ps1 -Time 11:00 -ApplyExpiry
```

## Build Phase 1 from scratch

```powershell
.\setup_new_laptop.ps1
.\initialize_phase1.ps1 -ResetIndex
```

The initial embedding job can be lengthy on a CPU. A CUDA-capable environment
is recommended for a corpus with hundreds of thousands of chunks.

If a standalone initial embedding run is interrupted, preserve `chroma_db` and
resume by embedding only chunk IDs that are not already stored:

```powershell
.\.venv\Scripts\python.exe .\create_embeddings.py --batch-size 64 --resume
```

Do not combine `--resume` with `--reset` or `--sync-file`.

## Main commands

- `initialize_phase1.ps1` - initial download, chunk, embed, boilerplate and lexical build.
- `start_portal.ps1` - start the Streamlit search application.
- `run_maintenance.ps1` - manually run incremental maintenance.
- `enable_automation.ps1` - install the daily Windows scheduled task.
- `disable_automation.ps1` - remove only the scheduled task.
- `validate.ps1` - perform non-destructive integrity checks.
- `backup.ps1` - copy runtime state into a timestamped backup.

See `DEPLOYMENT_GUIDE.md` for the complete laptop handoff procedure.
For a GPU-assisted initial build, see `COLAB_INITIAL_BUILD.md`.
