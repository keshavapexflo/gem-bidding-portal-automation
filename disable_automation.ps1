<# Remove the Lets Bid scheduled task without deleting application data. #>
param()
$TaskName = 'LetsBidDailyMaintenance'
schtasks.exe /Query /TN $TaskName *> $null
if ($LASTEXITCODE -eq 0) {
    schtasks.exe /Delete /TN $TaskName /F | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Could not delete scheduled task $TaskName" }
    Write-Host "Removed scheduled task $TaskName. Application data was not changed."
} else {
    Write-Host "Scheduled task $TaskName is not installed."
}
