# Local development only — runs the full KFP pipeline setup on a Kind cluster.
# Run setup_kind.ps1 first if the cluster does not exist yet.
# Requires $env:GITHUB_TOKEN to be set with write:packages scope.


param(
    [string]$CLUSTER_NAME = "kubeflow",
    [string]$REGISTRY = "ghcr.io/habh2"
)

$ErrorActionPreference = "Stop"

$IMAGES = @(
    "recommender-system-ingest",
    "recommender-system-embed",
    "recommender-system-train-topic",
    "recommender-system-compute-dist",
    "recommender-system-train-pref"
)

Write-Host "Logging in to GitHub Container Registry..."
docker login ghcr.io -u habh2 -p $env:GITHUB_TOKEN

Write-Host "Pushing images to GHCR (skipping unchanged)..."
foreach ($img in $IMAGES) {
    $localId  = docker inspect --format "{{.Id}}" "${img}:latest"
    if ($LASTEXITCODE -ne 0) { $localId = $null }
    $taggedId = docker inspect --format "{{.Id}}" "${REGISTRY}/${img}:latest"
    if ($LASTEXITCODE -ne 0) { $taggedId = $null }
    if ($localId -and $localId -eq $taggedId) {
        Write-Host "  ${img}: up to date, skipping"
        continue
    }
    Write-Host "  ${img}: pushing..."
    docker tag "${img}:latest" "${REGISTRY}/${img}:latest"
    docker push "${REGISTRY}/${img}:latest"
}

Write-Host "Applying shared volumes and config..."
kubectl apply -f "k8s\volumes.yaml"
kubectl apply -f "k8s\qdrant.yaml"
kubectl apply -f "k8s\config.yaml"

Write-Host "Compiling KFP pipeline..."
python "k8s\pipeline.py"

Write-Host "Submitting experiment and pipeline to KFP..."
$pf = Start-Job { kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 }
Start-Sleep -Seconds 3
python "k8s\submit_pipeline.py"
Stop-Job $pf
Remove-Job $pf

Write-Host "Access the UI with:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
