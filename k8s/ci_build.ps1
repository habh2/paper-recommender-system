# CI / author script — builds all Docker images and pushes them to GHCR.
# Requires $env:GITHUB_TOKEN with write:packages scope.
# Run deploy.ps1 afterwards to apply the updated images to the cluster.


param(
    [string]$REGISTRY = "ghcr.io/habh2",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_TOKEN) {
    Write-Error "GITHUB_TOKEN is not set. Run: `$env:GITHUB_TOKEN = 'ghp_...'"
    exit 1
}

$IMAGES = @(
    "recommender-system-ingest",
    "recommender-system-embed",
    "recommender-system-train-topic",
    "recommender-system-compute-dist",
    "recommender-system-train-pref",
    "recommender-system-app"
)

if (-not $SkipBuild) {
    Write-Host "Building images..."
    docker build -t recommender-base -f Dockerfile.base .
    docker compose --profile pipeline build
}

Write-Host "Logging in to GitHub Container Registry..."
$GITHUB_USER = $REGISTRY.Split("/")[1]
docker login ghcr.io -u $GITHUB_USER -p $env:GITHUB_TOKEN

Write-Host "Pushing images to GHCR (skipping unchanged)..."
foreach ($img in $IMAGES) {
    $ErrorActionPreference = "Continue"
    $localId  = docker inspect --format "{{.Id}}" "${img}:latest"
    if ($LASTEXITCODE -ne 0) { $localId = $null }
    $taggedId = docker inspect --format "{{.Id}}" "${REGISTRY}/${img}:latest"
    if ($LASTEXITCODE -ne 0) { $taggedId = $null }
    $ErrorActionPreference = "Stop"
    if ($localId -and $localId -eq $taggedId) {
        Write-Host "  ${img}: up to date, skipping"
        continue
    }
    Write-Host "  ${img}: pushing..."
    docker tag "${img}:latest" "${REGISTRY}/${img}:latest"
    docker push "${REGISTRY}/${img}:latest"
}

Write-Host "Done. Run deploy.ps1 to apply to the cluster."
