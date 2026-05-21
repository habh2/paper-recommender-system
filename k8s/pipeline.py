from kfp import dsl, compiler
from kfp_kubernetes import mount_pvc, set_image_pull_policy

INGEST_IMAGE = "recommender-system-ingest:latest"
EMBED_IMAGE = "recommender-system-embed:latest"
TRAIN_TOPIC_IMAGE = "recommender-system-train-topic:latest"
COMPUTE_DIST_IMAGE = "recommender-system-compute-dist:latest"
TRAIN_PREF_IMAGE = "recommender-system-train-pref:latest"

QDRANT_URL = "http://qdrant:6333"


@dsl.container_component
def ingest():
    return dsl.ContainerSpec(image=INGEST_IMAGE)


@dsl.container_component
def embed():
    return dsl.ContainerSpec(
        image=EMBED_IMAGE,
        env=[dsl.EnvVar(name="QDRANT_URL", value=QDRANT_URL)],
    )


@dsl.container_component
def train_topic():
    return dsl.ContainerSpec(
        image=TRAIN_TOPIC_IMAGE,
        env=[dsl.EnvVar(name="QDRANT_URL", value=QDRANT_URL)],
    )


@dsl.container_component
def compute_dist():
    return dsl.ContainerSpec(image=COMPUTE_DIST_IMAGE)


@dsl.container_component
def train_pref():
    return dsl.ContainerSpec(image=TRAIN_PREF_IMAGE)


@dsl.pipeline(name="recommender-pipeline")
def recommender_pipeline():
    ingest_task = ingest()
    mount_pvc(ingest_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    set_image_pull_policy(ingest_task, "IfNotPresent")

    embed_task = embed().after(ingest_task)
    mount_pvc(embed_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    mount_pvc(embed_task, pvc_name="hf-cache-pvc", mount_path="/root/.cache/huggingface")
    set_image_pull_policy(embed_task, "IfNotPresent")

    train_topic_task = train_topic().after(embed_task)
    mount_pvc(train_topic_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    mount_pvc(train_topic_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    set_image_pull_policy(train_topic_task, "IfNotPresent")

    compute_dist_task = compute_dist().after(train_topic_task)
    mount_pvc(compute_dist_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    mount_pvc(compute_dist_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    set_image_pull_policy(compute_dist_task, "IfNotPresent")

    train_pref_task = train_pref().after(compute_dist_task)
    mount_pvc(train_pref_task, pvc_name="papers-db-pvc", mount_path="/app/data")
    mount_pvc(train_pref_task, pvc_name="models-pvc", mount_path="/app/preference/models")
    set_image_pull_policy(train_pref_task, "IfNotPresent")


if __name__ == "__main__":
    compiler.Compiler().compile(recommender_pipeline, "k8s/pipeline.yaml")
