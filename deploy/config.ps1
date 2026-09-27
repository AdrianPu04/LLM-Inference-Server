# Shared settings for setup.ps1 / bench.ps1 / teardown.ps1. Override with env vars.
$Project = if ($env:GCP_PROJECT) { $env:GCP_PROJECT } else { (gcloud config get-value project 2>$null) }
$Region  = if ($env:GCP_REGION)  { $env:GCP_REGION }  else { "us-central1" }
$Zone    = if ($env:GCP_ZONE)    { $env:GCP_ZONE }    else { "us-central1-a" }
$Cluster = if ($env:GKE_CLUSTER) { $env:GKE_CLUSTER } else { "llm-bench" }
$Repo    = "llm"

if (-not $Project) { throw "No GCP project. Run 'gcloud config set project <id>' or set GCP_PROJECT." }

$Image = "$Region-docker.pkg.dev/$Project/$Repo/llm-inference-server:latest"
