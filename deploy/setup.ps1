# One-time setup: Artifact Registry repo, image push, GKE cluster with an L4 pool.
# Usage: .\deploy\setup.ps1 [-OnDemand]   (default GPU node is Spot, ~1/3 the price)
param([switch]$OnDemand)

. "$PSScriptRoot\config.ps1"

function Invoke-Step([string]$Desc, [scriptblock]$Cmd) {
    Write-Host "==> $Desc" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Desc" }
}

Invoke-Step "Enable GKE + Artifact Registry APIs" {
    gcloud services enable container.googleapis.com artifactregistry.googleapis.com --project $Project
}

gcloud artifacts repositories describe $Repo --location $Region --project $Project --format "value(name)" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Invoke-Step "Create Artifact Registry repo '$Repo'" {
        gcloud artifacts repositories create $Repo --repository-format docker --location $Region --project $Project
    }
}

Invoke-Step "Configure Docker auth" { gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet }
Invoke-Step "Build image" { docker build -t $Image (Join-Path $PSScriptRoot "..") }
Invoke-Step "Push image" { docker push $Image }

gcloud container clusters describe $Cluster --zone $Zone --project $Project --format "value(name)" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    # Small CPU node for system pods and the load-test pod; GPU nodes live in their own pool.
    Invoke-Step "Create cluster '$Cluster'" {
        gcloud container clusters create $Cluster --zone $Zone --project $Project `
            --num-nodes 1 --machine-type e2-standard-2 --release-channel regular
    }

    # max-nodes 1 caps spend at one L4 and means servers are benchmarked one at a time.
    $gpuArgs = @(
        "container", "node-pools", "create", "gpu-pool",
        "--cluster", $Cluster, "--zone", $Zone, "--project", $Project,
        "--machine-type", "g2-standard-8",
        "--accelerator", "type=nvidia-l4,count=1,gpu-driver-version=latest",
        "--num-nodes", "0", "--enable-autoscaling", "--min-nodes", "0", "--max-nodes", "1"
    )
    if (-not $OnDemand) { $gpuArgs += "--spot" }
    Invoke-Step "Create L4 node pool (scales 0-1)" { gcloud @gpuArgs }
}

Invoke-Step "Fetch kubectl credentials" {
    gcloud container clusters get-credentials $Cluster --zone $Zone --project $Project
}

Write-Host "`nReady. Next: .\deploy\bench.ps1 -Target server   (then -Target vllm)" -ForegroundColor Green
Write-Host "When finished: .\deploy\teardown.ps1" -ForegroundColor Yellow
