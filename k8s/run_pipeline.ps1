# Local development only — runs the full KFP pipeline setup on a Kind cluster.
# Run setup_kind.ps1 first if the cluster does not exist yet.


param(
    [string]$CLUSTER_NAME = "kubeflow",
    [string]$REGISTRY = "127.0.0.1:5001"
)

$ErrorActionPreference = "Stop"

$IMAGES = @(
    "recommender-system-ingest",
    "recommender-system-embed",
    "recommender-system-train-topic",
    "recommender-system-compute-dist",
    "recommender-system-train-pref"
)

Write-Host "Pushing images to local registry..."
foreach ($img in $IMAGES) {
    docker tag "${img}:latest" "${REGISTRY}/${img}:latest"
    docker push "${REGISTRY}/${img}:latest"
}

Write-Host "Applying shared volumes and config..."
kubectl apply -f "k8s\volumes.yaml"
kubectl apply -f "k8s\qdrant.yaml"
kubectl apply -f "k8s\config.yaml"

Write-Host "Compiling KFP pipeline..."
python "k8s\pipeline.py"

Write-Host "Done. Upload k8s\pipeline.yaml via the KFP UI to run the pipeline."
Write-Host "Access the UI with:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
