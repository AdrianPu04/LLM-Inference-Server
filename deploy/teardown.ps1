# Delete the cluster (stops all compute billing). -DeleteImages also removes the Artifact Registry repo.
param([switch]$DeleteImages)

. "$PSScriptRoot\config.ps1"

Write-Host "==> Deleting cluster '$Cluster' in $Zone" -ForegroundColor Cyan
gcloud container clusters delete $Cluster --zone $Zone --project $Project --quiet
if ($LASTEXITCODE -ne 0) { throw "Cluster delete failed - check the GCP console so nothing is left running." }

if ($DeleteImages) {
    Write-Host "==> Deleting Artifact Registry repo '$Repo'" -ForegroundColor Cyan
    gcloud artifacts repositories delete $Repo --location $Region --project $Project --quiet
}

Write-Host "Teardown complete." -ForegroundColor Green
