from kfp import dsl, compiler, kubernetes

REGISTRY = "localhost:5001"
INGEST_IMAGE = f"{REGISTRY}/recommender-system-ingest:latest"
EMBED_IMAGE = f"{REGISTRY}/recommender-system-embed:latest"
TRAIN_TOPIC_IMAGE = f"{REGISTRY}/recommender-system-train-topic:latest"
COMPUTE_DIST_IMAGE = f"{REGISTRY}/recommender-system-compute-dist:latest"
TRAIN_PREF_IMAGE = f"{REGISTRY}/recommender-system-train-pref:latest"

CONFIG_MAP = "pipeline-config"


@dsl.container_component
def ingest():
    return dsl.ContainerSpec(image=INGEST_IMAGE)


@dsl.container_component
def embed():
    return dsl.ContainerSpec(image=EMBED_IMAGE)


@dsl.container_component
def train_topic():
    return dsl.ContainerSpec(image=TRAIN_TOPIC_IMAGE)


@dsl.container_component
def compute_dist():
    return dsl.ContainerSpec(image=COMPUTE_DIST_IMAGE)


@dsl.container_component
def train_pref():
    return dsl.ContainerSpec(image=TRAIN_PREF_IMAGE)


def _mount_qdrant_env(task):
    kubernetes.use_config_map_as_env(
        task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={"QDRANT_URL": "QDRANT_URL"},
    )


@dsl.pipeline(name="recommender-pipeline")
def recommender_pipeline():
    ingest_task = ingest()
    kubernetes.mount_pvc(ingest_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.set_image_pull_policy(ingest_task, "IfNotPresent")

    embed_task = embed().after(ingest_task)
    kubernetes.mount_pvc(embed_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(embed_task, pvc_name="hf-cache-pvc", mount_path="/root/.cache/huggingface")
    kubernetes.set_image_pull_policy(embed_task, "IfNotPresent")
    _mount_qdrant_env(embed_task)

    train_topic_task = train_topic().after(embed_task)
    kubernetes.mount_pvc(train_topic_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(train_topic_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    kubernetes.set_image_pull_policy(train_topic_task, "IfNotPresent")
    _mount_qdrant_env(train_topic_task)

    compute_dist_task = compute_dist().after(train_topic_task)
    kubernetes.mount_pvc(compute_dist_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(compute_dist_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    kubernetes.set_image_pull_policy(compute_dist_task, "IfNotPresent")

    train_pref_task = train_pref().after(compute_dist_task)
    kubernetes.mount_pvc(train_pref_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    kubernetes.mount_pvc(train_pref_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    kubernetes.set_image_pull_policy(train_pref_task, "IfNotPresent")


if __name__ == "__main__":
    compiler.Compiler().compile(recommender_pipeline, "k8s/pipeline.yaml")
