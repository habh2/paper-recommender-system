import hashlib
import kfp
from pathlib import Path

EXPERIMENT = "recommender"
PIPELINE_NAME = "recommender-pipeline"
PIPELINE_FILE = "k8s/pipeline.yaml"
HASH_FILE = "k8s/.pipeline_hash"
KFP_HOST = "http://localhost:8888"

client = kfp.Client(host=KFP_HOST)

client.create_experiment(EXPERIMENT)

current_hash = hashlib.md5(Path(PIPELINE_FILE).read_bytes()).hexdigest()
last_hash = Path(HASH_FILE).read_text().strip() if Path(HASH_FILE).exists() else ""

pipeline_id = client.get_pipeline_id(PIPELINE_NAME)

if pipeline_id is None:
    client.upload_pipeline(pipeline_package_path=PIPELINE_FILE, pipeline_name=PIPELINE_NAME)
    pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
    Path(HASH_FILE).write_text(current_hash)
    print(f"Pipeline '{PIPELINE_NAME}' created (v1).")
elif current_hash == last_hash:
    print(f"Pipeline '{PIPELINE_NAME}' unchanged, skipping upload.")
else:
    versions = client.list_pipeline_versions(pipeline_id=pipeline_id, page_size=100)
    existing = [v.name for v in (versions.pipeline_versions or [])]
    max_v = max((int(n[1:]) for n in existing if n.startswith("v") and n[1:].isdigit()), default=1)
    next_version = f"v{max_v + 1}"
    client.upload_pipeline_version(
        pipeline_package_path=PIPELINE_FILE,
        pipeline_id=pipeline_id,
        pipeline_version_name=next_version,
    )
    Path(HASH_FILE).write_text(current_hash)
    print(f"Pipeline version '{next_version}' uploaded.")

print(f"Trigger a run at: {KFP_HOST}")
