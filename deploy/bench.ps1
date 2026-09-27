# Deploy one server onto the L4 node and benchmark it from a pod inside the cluster
# (no port-forward / internet hop in the latency numbers). Writes results/gke_<label>[_mixed].json.
#
#   .\deploy\bench.ps1 -Target server                                   # continuous (default)
#   .\deploy\bench.ps1 -Target server -Server static_batch_server
#   .\deploy\bench.ps1 -Target vllm
#   .\deploy\bench.ps1 -Target vllm -VllmArgs "--max-num-seqs 8" -Label vllm_seqs8
param(
    [ValidateSet("server", "vllm")][string]$Target = "server",
    [string]$Server = "continuous_batch_server",
    [string]$ModelName = "Qwen/Qwen2.5-1.5B-Instruct",
    [string]$VllmArgs = "",
    [string]$Label = "",
    [int]$Requests = 100,
    [int[]]$Concurrency = @(1, 2, 4, 8, 16, 32)
)

. "$PSScriptRoot\config.ps1"

function Assert-Ok([string]$Desc) { if ($LASTEXITCODE -ne 0) { throw "Failed: $Desc" } }

if (-not $Label) {
    $Label = if ($Target -eq "vllm") { "vllm" } else { $Server -replace "_server$", "" }
}

Push-Location (Join-Path $PSScriptRoot "..")
try {
    # Only one L4 in the pool: remove the other deployment so its GPU frees up.
    if ($Target -eq "server") {
        kubectl delete deployment vllm --ignore-not-found
        $manifest = (Get-Content deploy/server.yaml -Raw) `
            -replace "__IMAGE__", $Image -replace "__SERVER__", $Server -replace "__MODEL__", $ModelName
        $deploy = "llm-server"; $url = "http://llm-server:8000/generate"; $api = "native"
    } else {
        kubectl delete deployment llm-server --ignore-not-found
        $manifest = (Get-Content deploy/vllm.yaml -Raw) `
            -replace "__MODEL__", $ModelName -replace "__EXTRA_ARGS__", $VllmArgs
        $deploy = "vllm"; $url = "http://vllm:8000/v1/completions"; $api = "openai"
    }

    Write-Host "==> Deploying $deploy ($Label)" -ForegroundColor Cyan
    $manifest | kubectl apply -f -
    Assert-Ok "kubectl apply"

    # Covers GPU node scale-up from zero + driver install + model download/load.
    Write-Host "==> Waiting for $deploy to become ready (first run can take ~10-15 min)" -ForegroundColor Cyan
    kubectl rollout status "deployment/$deploy" --timeout=30m
    Assert-Ok "rollout $deploy"

    kubectl get pod loadtest 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> Starting load-test pod" -ForegroundColor Cyan
        kubectl run loadtest --image $Image --restart Never --command -- sleep infinity
        Assert-Ok "start loadtest pod"
    }
    kubectl wait --for=condition=Ready pod/loadtest --timeout=10m
    Assert-Ok "loadtest pod ready"

    $runs = @(
        @{ Suffix = "";       Tokens = @(128) },
        @{ Suffix = "_mixed"; Tokens = @(32, 128, 256) }
    )
    foreach ($run in $runs) {
        $tokens = $run.Tokens
        $out = "results/gke_$Label$($run.Suffix).json"
        Write-Host "==> Load test -> $out" -ForegroundColor Cyan
        kubectl exec loadtest -- python scripts/load_test.py --api $api --url $url --model $ModelName `
            --concurrency $Concurrency --requests-per-level $Requests --max-new-tokens $tokens --out /tmp/out.json
        Assert-Ok "load test"
        kubectl cp loadtest:/tmp/out.json $out
        Assert-Ok "copy results"
    }
} finally {
    Pop-Location
}

Write-Host "`nDone. The GPU node stays up while a deployment exists; run .\deploy\teardown.ps1 when finished." -ForegroundColor Yellow
