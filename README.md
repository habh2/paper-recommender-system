# Paper Recommender System

A recommender system that learns user taste from pairwise abstract comparisons. The user is shown two paper abstracts and picks the more interesting one. The system learns which topics the user favours and reranks candidates accordingly.

Built as a portfolio project demonstrating production ML engineering: data pipelines, embeddings, preference learning, containerisation, and Kubeflow orchestration on Kubernetes.

## Architecture

```
Semantic Scholar API
        │
        ▼
   Ingest (SQLite)
        │
        ├──► SPECTER2 embeddings ──► Qdrant (ANN retrieval)
        │
        └──► BERTopic (80 topics) ──► Topic distributions per paper
                                              │
                                    User pairwise choices
                                              │
                                              ▼
                                    Logistic regression
                                    (chosen − rejected topic vectors)
                                              │
                                              ▼
                                    Preference score per paper
                                              │
                              ANN retrieval (top 200) → rerank → top K
```

Two-stage retrieval: Qdrant finds the 200 most semantically similar papers fast (ANN), then the preference model reranks them by predicted user interest.

## Stack

| Layer | Technology |
|---|---|
| Data storage | SQLite (papers), Qdrant (vectors) |
| Embeddings | SPECTER2 (768-dim, domain-specific for scientific papers) |
| Topic model | BERTopic (HDBSCAN + c-TF-IDF, 80 topics) |
| Preference model | Logistic regression on topic difference vectors |
| API | FastAPI |
| Containerisation | Docker, Docker Compose |
| Orchestration | Kubeflow Pipelines v2 on Kind (local Kubernetes) |
| Registry | GitHub Container Registry (ghcr.io) |

## Pipelines

Two KFP pipelines run on a local Kind cluster:

**`data-pipeline`** — run once to build the corpus:
```
ingest → embed → train-topic → compute-dist
```

**`preference-pipeline`** — run after collecting user choices:
```
train-pref → evaluate → restart-app
```

The `evaluate` step computes pairwise accuracy on a held-out 20% of choices and fails the pipeline if the new model scores below 50% (worse than random), keeping the old model in place. If it passes, `restart-app` triggers a zero-downtime rolling restart via the Kubernetes API — scaling to 2 pods, waiting for the new one to become ready, then scaling back to 1.

## Local Development (Docker Compose)

```bash
# Build images
docker build -t recommender-base -f Dockerfile.base .
docker compose build

# Run the full data pipeline
docker compose --profile pipeline run --rm ingest
docker compose --profile pipeline run --rm embed
docker compose --profile pipeline run --rm train-topic
docker compose --profile pipeline run --rm compute-dist

# Start the app
docker compose up app
```

Open `http://localhost:8000`, make pairwise choices, then train the preference model:

```bash
docker compose --profile pipeline run --rm train-pref
docker compose restart app
```

## Kubernetes (Kind)

Prerequisites: Docker Desktop (8GB RAM), Kind, kubectl, Python 3.10+.

All pipeline images are pre-built and publicly available on GHCR — no GitHub token required to run.

```powershell
# One-time cluster setup
.\k8s\setup_cluster.ps1

# Deploy (applies manifests, compiles pipeline, submits to KFP)
.\k8s\deploy.ps1

# Access the KFP UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Access the app
kubectl port-forward -n kubeflow svc/recommender-app 8000:8000
```

Trigger `data-pipeline` from the KFP UI, then interact with the app at `http://localhost:8000`. Once you have enough choices, trigger `preference-pipeline` — it retrains the model and restarts the app automatically.

### For development (building and pushing images)

Requires a GitHub token with `write:packages` scope.

```powershell
$env:GITHUB_TOKEN = "ghp_..."
.\k8s\ci_build.ps1   # build + push to GHCR
.\k8s\deploy.ps1     # apply to cluster
```

## Project Structure

```
├── ingestion/          # Semantic Scholar ingest → SQLite
├── embeddings/         # SPECTER2 embed → Qdrant
├── preference/
│   ├── train_topic_model.py      # BERTopic training
│   ├── extract_embeddings.py     # Pulls embeddings from Qdrant for topic model input
│   ├── compute_distributions.py  # Topic distributions per paper
│   ├── train_preference_model.py # Logistic regression on choices
│   ├── evaluate.py               # Pairwise accuracy (offline evaluation gate)
│   ├── rerank.py                 # ANN retrieval + preference rerank
│   └── app.py                    # FastAPI serving layer
├── k8s/                # Kubeflow pipeline definitions and manifests
├── Dockerfile.base     # Shared base image
└── docker-compose.yml  # Local development orchestration
```

## Design Decisions & Known Limitations

**Single-user by design.** SQLite handles concurrent reads fine but not concurrent writes. There is no authentication and one shared preference model. This is intentional for a portfolio demo — a multi-user system would need a proper RDBMS and per-user models.

**Topic count (80) is empirical.** The number of BERTopic topics was chosen by inspecting a sample of the corpus, not by formal coherence scoring (e.g. NPMI). BERTopic is relatively robust to this choice, but a production system would tune it with an automated metric.

**No model rollback.** The `evaluate` step prevents deploying a model worse than random (< 50% pairwise accuracy), but there is no versioned rollback to a specific previous model. The production solution is a model registry (e.g. MLflow) that versions artefacts and supports explicit promotion and rollback.

**No monitoring.** The app exposes a `/health` endpoint but exports no metrics. Prometheus + Grafana would be the natural next step for tracking recommendation quality and latency over time.

**Preference model retrains on all choices.** Each `preference-pipeline` run fits from scratch on the full choice history. With many choices this becomes slow. A production system would use online learning or mini-batch updates.
