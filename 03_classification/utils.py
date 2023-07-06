from mlflow import MlflowClient


def yield_artifacts(
    run_id: str,
    path: str = None,
):
    """
    run_idに対応するrunのartifactをyieldする.
    """
    client = MlflowClient()
    for artifact in client.list_artifacts(run_id=run_id, path=path):
        if artifact.is_dir:
            yield from yield_artifacts(run_id, artifact.path)
        else:
            yield artifact.path
            
            
def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts
    }