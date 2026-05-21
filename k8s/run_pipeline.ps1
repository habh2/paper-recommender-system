# Local development only — runs the full KFP pipeline setup on a Kind cluster.
# Run setup_kind.ps1 first if the cluster does not exist yet.


param(
    [string]$CLUSTER_NAME = "kubeflow"
)
$ErrorActionPreference = "Stop"


Write-Host "Applying shared volumes..."
kubectl apply -f "$PSScriptRoot\volumes.yaml"
kubectl apply -f "$PSScriptRoot\qdrant.yaml"

Write-Host "Loading images into Kind..."
kind load docker-image recommender-system-ingest:latest --name $CLUSTER_NAME
kind load docker-image recommender-system-embed:latest --name $CLUSTER_NAME
kind load docker-image recommender-system-train-topic:latest --name $CLUSTER_NAME
kind load docker-image recommender-system-compute-dist:latest --name $CLUSTER_NAME
kind load docker-image recommender-system-train-pref:latest --name $CLUSTER_NAME

Write-Host "Compiling KFP pipeline..."
python "$PSScriptRoot\pipeline.py"

Write-Host "Done. Upload k8s\pipeline.yaml via the KFP UI to run the pipeline."
Write-Host "Access the UI with:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
