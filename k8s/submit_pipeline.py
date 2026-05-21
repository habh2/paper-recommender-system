import kfp

EXPERIMENT = "recommender"
PIPELINE_NAME = "recommender-pipeline"
PIPELINE_FILE = "k8s/pipeline.yaml"
KFP_HOST = "http://localhost:8888"

client = kfp.Client(host=KFP_HOST)

client.create_experiment(EXPERIMENT)

pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
if pipeline_id:
    client.upload_pipeline_version(
        pipeline_package_path=PIPELINE_FILE,
        pipeline_id=pipeline_id,
        pipeline_version_name="latest",
    )
else:
    client.upload_pipeline(
        pipeline_package_path=PIPELINE_FILE,
        pipeline_name=PIPELINE_NAME,
    )

print(f"Experiment '{EXPERIMENT}' and pipeline '{PIPELINE_NAME}' ready.")
print(f"Trigger a run at: {KFP_HOST}")
