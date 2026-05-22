from kfp import dsl, compiler, kubernetes

REGISTRY = "ghcr.io/habh2"
INGEST_IMAGE = f"{REGISTRY}/recommender-system-ingest:latest"
EMBED_IMAGE = f"{REGISTRY}/recommender-system-embed:latest"
TRAIN_TOPIC_IMAGE = f"{REGISTRY}/recommender-system-train-topic:latest"
COMPUTE_DIST_IMAGE = f"{REGISTRY}/recommender-system-compute-dist:latest"
TRAIN_PREF_IMAGE = f"{REGISTRY}/recommender-system-train-pref:latest"
EVALUATE_IMAGE = TRAIN_PREF_IMAGE

CONFIG_MAP = "pipeline-config"


@dsl.container_component
def ingest():
    return dsl.ContainerSpec(image=INGEST_IMAGE, command=["python", "ingestion/ingest.py"])


@dsl.container_component
def embed():
    return dsl.ContainerSpec(image=EMBED_IMAGE, command=["python", "embeddings/embed.py"])


@dsl.container_component
def train_topic():
    return dsl.ContainerSpec(image=TRAIN_TOPIC_IMAGE, command=["python", "preference/train_topic_model.py"])


@dsl.container_component
def compute_dist():
    return dsl.ContainerSpec(image=COMPUTE_DIST_IMAGE, command=["python", "preference/compute_distributions.py"])


@dsl.container_component
def train_pref():
    return dsl.ContainerSpec(image=TRAIN_PREF_IMAGE, command=["python", "preference/train_preference_model.py"])


@dsl.container_component
def evaluate():
    return dsl.ContainerSpec(image=EVALUATE_IMAGE, command=["python", "preference/evaluate.py"])


@dsl.container_component
def restart_app():
    return dsl.ContainerSpec(
        image="bitnami/kubectl:latest",
        command=["sh", "-c"],
        args=[
            "kubectl scale deployment/recommender-app --replicas=2 -n kubeflow && "
            "kubectl rollout status deployment/recommender-app -n kubeflow && "
            "kubectl scale deployment/recommender-app --replicas=1 -n kubeflow"
        ],
    )


def _mount_qdrant_env(task):
    kubernetes.use_config_map_as_env(
        task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={"QDRANT_URL": "QDRANT_URL"},
    )


@dsl.pipeline(name="data-pipeline")
def data_pipeline():
    ingest_task = ingest()
    kubernetes.mount_pvc(ingest_task, pvc_name="papers-db-pvc", mount_path="/app/data")

    embed_task = embed().after(ingest_task)
    kubernetes.mount_pvc(embed_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(embed_task, pvc_name="hf-cache-pvc", mount_path="/root/.cache/huggingface")
    _mount_qdrant_env(embed_task)

    train_topic_task = train_topic().after(embed_task)
    kubernetes.mount_pvc(train_topic_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(train_topic_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    _mount_qdrant_env(train_topic_task)

    compute_dist_task = compute_dist().after(train_topic_task)
    kubernetes.mount_pvc(compute_dist_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(compute_dist_task, pvc_name="models-pvc", mount_path="/app/preference/models")


@dsl.pipeline(name="preference-pipeline")
def preference_pipeline():
    train_pref_task = train_pref()
    kubernetes.mount_pvc(train_pref_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(train_pref_task, pvc_name="models-pvc", mount_path="/app/preference/models")

    evaluate_task = evaluate().after(train_pref_task)
    kubernetes.mount_pvc(evaluate_task, pvc_name="papers-db-pvc", mount_path="/app/data")

    restart_app().after(evaluate_task)


if __name__ == "__main__":
    compiler.Compiler().compile(data_pipeline, "k8s/data_pipeline.yaml")
    compiler.Compiler().compile(preference_pipeline, "k8s/preference_pipeline.yaml")
