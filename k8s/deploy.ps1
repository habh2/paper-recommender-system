# Applies all Kubernetes manifests, compiles the KFP pipeline, and submits it.
# Works for both reviewers (using pre-built public images) and CI (after ci_build.ps1).
# Requires: Kind cluster running (setup_cluster.ps1), kubectl, Python 3.10+.


param(
    [string]$CLUSTER_NAME = "kubeflow"
)

$ErrorActionPreference = "Stop"

Write-Host "Applying shared volumes and config..."
kubectl apply -f "k8s\volumes.yaml"
kubectl apply -f "k8s\qdrant.yaml"
kubectl apply -f "k8s\config.yaml"
kubectl apply -f "k8s\app.yaml"
kubectl apply -f "k8s\rbac.yaml"

Write-Host "Compiling KFP pipeline..."
python "k8s\pipeline.py"

Write-Host "Submitting experiment and pipeline to KFP..."
$pf = Start-Job { kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 }
Start-Sleep -Seconds 8
python "k8s\submit_pipeline.py"
Stop-Job $pf
Remove-Job $pf

Write-Host "Done."
Write-Host "  KFP UI:  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
Write-Host "  App:     kubectl port-forward -n kubeflow svc/recommender-app 8000:8000"
