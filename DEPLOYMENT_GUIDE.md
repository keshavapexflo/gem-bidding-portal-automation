# Deployment guide

## 1. Prepare the source package

Keep application code and runtime data conceptually separate. The code in this
directory can be copied or cloned normally. The large runtime artifacts must be
transferred separately because they are excluded by `.gitignore`:

```text
bid_chunks.json
downloads\
chroma_db\
```

Before copying data, run `backup.ps1` and ensure no maintenance job is active.

## 2. Target-laptop prerequisites

- Windows 10 or Windows 11, 64 bit.
- 64-bit Python 3.11 with the Python Launcher enabled.
- Enough free space for the PDFs, Chroma database, a backup, and temporary
  maintenance files. Plan for at least twice the current runtime-data size.
- Internet access to GeM and Hugging Face during setup or model download.

Install the application in a stable local path such as `C:\LetsBid`. Avoid a
OneDrive-synchronised directory and do not run it directly from a removable disk.

## 3. Install and import existing Phase 1 output

Copy this code directory to the laptop. Put the three runtime artifacts on an
external disk or another local staging folder, then run PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd C:\LetsBid
.\setup_new_laptop.ps1 -DataSource "E:\LetsBidData"
.\validate.ps1 -Full
```

Validation warnings about legacy duplicate chunk IDs describe the old notebook
corpus and do not prevent using its existing Chroma database. Do not perform a
full index rebuild until the PDFs have been re-chunked with the current schema.
When you intentionally schedule that one-time clean rebuild, take a backup and
run:

```powershell
.\initialize_phase1.ps1 -SkipDownload -ForceRechunk -ResetIndex
```

If the imported Chroma database came from `embedding_40k.ipynb`, make its
boilerplate classification durable for future incremental upserts:

```powershell
.\.venv\Scripts\python.exe .\refresh_boilerplate.py
```

This is a one-time, potentially lengthy corpus scan.

The deployed code pins both the Python packages and the BGE model snapshot.
These values are recorded in `deployment_manifest.json`; keep that file with
every laptop installation and backup.

## 4. Manual acceptance test

Start the portal:

```powershell
.\start_portal.ps1
```

Confirm all of the following before enabling automation:

1. The sidebar reports the expected collection and vector count.
2. Exact GeM bid-number search works.
3. Natural-language product search returns sensible results.
4. Boilerplate exclusion improves product searches.
5. A PDF can be downloaded and opened.
6. CSV export opens correctly in Excel.

Then stop the portal and test non-expiry maintenance:

```powershell
$env:MAX_NEW_DOWNLOADS_PER_RUN = '10'
.\run_maintenance.ps1 -SkipExpiry
Remove-Item Env:MAX_NEW_DOWNLOADS_PER_RUN
.\validate.ps1
```

The temporary limit is only for the acceptance test. Normal scheduled runs are
uncapped and incremental.

## 5. Enable daily Phase 2 automation

Install the task with dry-run expiry reporting:

```powershell
.\enable_automation.ps1 -Time 11:00
```

Logs are written to `downloads\logs\daily_maintenance.log`. The scheduled
wrapper always changes to the project directory, so Chroma and chunk paths do
not depend on Task Scheduler's working directory.

After at least one expiry report has been reviewed:

```powershell
.\backup.ps1
.\enable_automation.ps1 -Time 11:00 -ApplyExpiry
```

Cleanup refuses to run if GeM reports zero or implausibly few active bids, or if
more than 25% of the local corpus would be removed without an explicit manual
override.

## 6. Recovery and administration

- Run `validate.ps1` after data transfer, upgrades, and maintenance failures.
- A pending embedding journal is retried automatically on the next run.
- Maintenance locks older than 18 hours are treated as stale and recovered.
- Run `backup.ps1` before upgrades or enabling expiry archival.

## Manual weekly or fortnightly expiry cleanup

First generate a read-only report:

```powershell
.\weekly_expiry_cleanup.ps1
```

After reviewing the active/expired counts, stop the portal and explicitly apply
the cleanup:

```powershell
.\weekly_expiry_cleanup.ps1 -Apply
```

Type `APPLY` when prompted. Expired PDFs are moved into `downloads\expired`,
their chunks and vectors are removed from search, and the lexical index is
rebuilt. The operation refuses zero or implausibly few active GeM results and
blocks removal of more than 25% of the local corpus without separate manual
review.
- Run `disable_automation.ps1` before moving or uninstalling the directory.
- Disabling automation never deletes PDFs, chunks, vectors, or backups.
- Run `.\.venv\Scripts\python.exe .\refresh_boilerplate.py` monthly, or after a
  large corpus import, so newly recurring templates enter the durable registry.

To inspect expiry without changing data:

```powershell
.\.venv\Scripts\python.exe .\portal_pipeline.py expiry-report
```

To manually approve an unusually large, verified cleanup:

```powershell
.\.venv\Scripts\python.exe .\gem_expiry_cleanup.py --apply --allow-large-cleanup
```
