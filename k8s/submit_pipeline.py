import hashlib
import kfp
from pathlib import Path

EXPERIMENT = "recommender"
KFP_HOST = "http://localhost:8888"

PIPELINES = [
    {"name": "data-pipeline",       "file": "k8s/data_pipeline.yaml",       "hash_file": "k8s/.data_pipeline_hash"},
    {"name": "preference-pipeline", "file": "k8s/preference_pipeline.yaml", "hash_file": "k8s/.preference_pipeline_hash"},
]

client = kfp.Client(host=KFP_HOST)

client.create_experiment(EXPERIMENT)

for p in PIPELINES:
    current_hash = hashlib.md5(Path(p["file"]).read_bytes()).hexdigest()
    last_hash = Path(p["hash_file"]).read_text().strip() if Path(p["hash_file"]).exists() else ""

    pipeline_id = client.get_pipeline_id(p["name"])

    if pipeline_id is None:
        client.upload_pipeline(pipeline_package_path=p["file"], pipeline_name=p["name"])
        pipeline_id = client.get_pipeline_id(p["name"])
        Path(p["hash_file"]).write_text(current_hash)
        print(f"Pipeline '{p['name']}' created (v1).")
    elif current_hash == last_hash:
        print(f"Pipeline '{p['name']}' unchanged, skipping upload.")
    else:
        versions = client.list_pipeline_versions(pipeline_id=pipeline_id, page_size=100)
        existing = [v.name for v in (versions.pipeline_versions or [])]
        max_v = max((int(n[1:]) for n in existing if n.startswith("v") and n[1:].isdigit()), default=1)
        next_version = f"v{max_v + 1}"
        client.upload_pipeline_version(
            pipeline_package_path=p["file"],
            pipeline_id=pipeline_id,
            pipeline_version_name=next_version,
        )
        Path(p["hash_file"]).write_text(current_hash)
        print(f"Pipeline '{p['name']}' version '{next_version}' uploaded.")

print(f"Trigger runs at: {KFP_HOST}")
