from pprint import pprint

from mlflow.tracking import MlflowClient

client = MlflowClient()
for mv in client.search_model_versions("name='group1_synonym_scoring'"):
    if mv.current_stage == "Production":
        run_id = mv.run_id
        artifact = client.list_artifacts(run_id=run_id, path="group1_model")[0]
        dst = client.download_artifacts(run_id=run_id, path=artifact.path)
        print(dst)
