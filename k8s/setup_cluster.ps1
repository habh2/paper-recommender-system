# Local development only — do not run against a production cluster.
# Kind creates a throwaway Kubernetes cluster inside Docker for testing KFP pipelines locally.


param(
    [string]$KFP_VERSION = "2.15.0",
    [string]$CLUSTER_NAME = "kubeflow",
    [switch]$SkipCluster
)


$ErrorActionPreference = "Stop"


if (-not $SkipCluster) {
    Write-Host "Deleting existing cluster (if any)..."
    kind delete cluster --name $CLUSTER_NAME

    Write-Host "Creating Kind cluster..."
    kind create cluster --name $CLUSTER_NAME
}

Write-Host "Installing KFP cluster-scoped resources..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"

Write-Host "Waiting for CRDs..."
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

Write-Host "Installing KFP..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$KFP_VERSION"

Write-Host "Waiting for pods to be ready (this takes a few minutes)..."
kubectl wait --for=condition=ready pod --all -n kubeflow --timeout=1200s
kubectl get pods -n kubeflow  

Write-Host "Done. Access the KFP UI with:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
Write-Host "Then open http://localhost:8080"
