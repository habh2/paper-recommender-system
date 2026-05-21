# Local development only — do not run against a production cluster.
# Kind creates a throwaway Kubernetes cluster inside Docker for testing KFP pipelines locally.


param(
    [string]$KFP_VERSION = "2.15.0",
    [string]$CLUSTER_NAME = "kubeflow",
    [string]$REGISTRY_NAME = "kind-registry",
    [int]$REGISTRY_PORT = 5001,
    [switch]$SkipCluster
)


$ErrorActionPreference = "Stop"


# Start local registry if not already running
if (-not (docker ps --filter "name=$REGISTRY_NAME" --format "{{.Names}}" | Select-String $REGISTRY_NAME)) {
    Write-Host "Starting local registry on port $REGISTRY_PORT..."
    docker run -d --restart=always -p "${REGISTRY_PORT}:5000" --name $REGISTRY_NAME registry:2
} else {
    Write-Host "Local registry already running."
}

if (-not $SkipCluster) {
    Write-Host "Deleting existing cluster (if any)..."
    kind delete cluster --name $CLUSTER_NAME

    Write-Host "Creating Kind cluster..."
    kind create cluster --name $CLUSTER_NAME
}

Write-Host "Connecting registry to Kind network..."
$networks = docker inspect $REGISTRY_NAME --format "{{json .NetworkSettings.Networks}}"
if ($networks -notmatch '"kind"') {
    docker network connect kind $REGISTRY_NAME
} else {
    Write-Host "Registry already connected to Kind network."
}

Write-Host "Configuring containerd to trust local registry..."
$hostsToml = "[host.`"http://${REGISTRY_NAME}:5000`"]`n  capabilities = [`"pull`", `"resolve`"]`n  skip_verify = true"
docker exec "${CLUSTER_NAME}-control-plane" mkdir -p "/etc/containerd/certs.d/localhost:${REGISTRY_PORT}"
docker exec "${CLUSTER_NAME}-control-plane" bash -c "echo '$hostsToml' > /etc/containerd/certs.d/localhost:${REGISTRY_PORT}/hosts.toml"
docker exec "${CLUSTER_NAME}-control-plane" systemctl restart containerd
Start-Sleep -Seconds 5

Write-Host "Installing KFP cluster-scoped resources..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"

Write-Host "Waiting for CRDs..."
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

Write-Host "Installing KFP..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$KFP_VERSION"

<# Write-Host "Waiting for pods to be ready (this takes a few minutes)..."
kubectl wait --for=condition=ready pod --all -n kubeflow --timeout=300s #>
#kubectl get pods -n kubeflow  

Write-Host "Done. Access the KFP UI with:"
Write-Host "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
Write-Host "Then open http://localhost:8080"
