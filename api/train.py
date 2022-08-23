import argparse
import time
from typing import List, Tuple

import pandas as pd
from mlflow.tracking import MlflowClient

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from contextual_synonym_recommender.feature_extractor import (
    FeatureExtractor,
    DEFAULT_SBERT_MODEL,
    DEFAULT_W2V_MODEL,
    Features,
    DEFAULT_FEATURES,
)
from contextual_synonym_recommender.model import ModelTrainer, SynonymScoringModel
from contextual_synonym_recommender.confusion_matrix import ConfusionMatrix

import mlflow

import numpy as np

MLFLOW_MODEL_NAME = "group1_synonym_scoring"
MLFLOW_ARTIFACT_PATH = "group1_model"

# set the random seed in order to get reproducible results
np.random.seed(10)
import random
random.seed(10)

mlflow_client = MlflowClient()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSVs", nargs="+")
    parser.add_argument("-t", "--test", required=True, help="Test CSV")
    parser.add_argument(
        "--fe-w2v-model", default=DEFAULT_W2V_MODEL, help="Name of the w2v model to use for feature extraction"
    )
    parser.add_argument(
        "--fe-sbert-model",
        default=DEFAULT_SBERT_MODEL,
        help="Name of the Sbert model to use for feature extraction",
    )
    parser.add_argument(
        "--feature-list", default=None, help="Path to feature list file"
    )
    parser.add_argument("--output", "-o", required=True, help="Store model at")
    args = parser.parse_args()

    # define the parameters for the MLPClassifier grid search
    parameters = {
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "hidden_layer_sizes": [
            (10, 10, 5),
            (20, 20, 10),
            (10, 30, 10),
            (20, 25, 5),
            (20, 5),
            (100,),
            (100, 20),
            (100, 50, 20),
            (50, 20, 20),
            (50, 50, 50),
            (30, 100, 5),
            (40, 20, 5),
            (30, 30, 30, 20, 20, 10, 10),
        ],
        "alpha": [
            0.1,
            0.01,
            0.001,
            0.0001,
            0.0005,
            0.00005,
            1e-05,
            1e-06,
            1e-07,
            1e-08,
            1e-09,
        ],
        "solver": ["adam", "sgd", "lbfgs"],
        "max_iter": [1000],
    }
    clf = GridSearchCV(
        estimator=MLPClassifier(), param_grid=parameters, n_jobs=1, verbose=2, cv=5
    )
    pipeline = Pipeline([("min_max", MinMaxScaler()), ("classifier", clf)])

    fe = FeatureExtractor(
        features=read_feature_list(args.feature_list)
        if args.feature_list is not None
        else DEFAULT_FEATURES,
        w2v_name=args.fe_w2v_model,
        sbert_name=args.fe_sbert_model,
    )
    model = SynonymScoringModel(fe, clf)
    trainer = ModelTrainer()

    dataset, labels = read_ds(args.input)
    test_ds, test_y = read_ds([args.test])

    with mlflow.start_run() as run:
        trainer.train(model, dataset, labels)
        with open(args.output, "wb") as f:
            f.write(model.serialize())

        metrics = test_clf(model, test_ds, test_y)
        print(metrics)
        for param, val in clf.best_params_.items():
            mlflow.log_param(param, str(val))
        mlflow.log_param("pipeline", ", ".join([step for step, _ in pipeline.steps]))

        for param, val in fe.get_params().items():
            if param == "features":
                continue
            mlflow.log_param("fe-%s" % param, str(val))
        for feature in fe.features:
            mlflow.log_param("feature-%s" % feature.name, True)
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        mlflow.log_artifact(args.output, artifact_path=MLFLOW_ARTIFACT_PATH)
        model_info = mlflow.sklearn.log_model(
            clf,
            MLFLOW_MODEL_NAME,
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        version = mlflow_client.search_model_versions(
            "run_id = '%s'" % run.info.run_id
        )[0]
        print("Model version: ", version)

        prod_metrics = get_mlflow_production_metrics()
        print("Current production metrics:", prod_metrics)

        if (
            metrics["precision"] >= prod_metrics["precision"]
            and metrics["f1_score"] >= 0.9 * prod_metrics["f1_score"]
        ):
            print("New metrics are better than production metrics. Update model...")
            mlflow_client.transition_model_version_stage(
                name=version.name,
                version=version.version,
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            print(
                "Production metrics are better than for the newly trained model. Skip stage transition..."
            )


def get_mlflow_production_metrics():
    for mv in mlflow_client.search_model_versions("name='%s'" % MLFLOW_MODEL_NAME):
        if mv.current_stage == "Production":
            run_id = mv.run_id
            precision = mlflow_client.get_metric_history(
                run_id=run_id, key="precision"
            )[0].value
            recall = mlflow_client.get_metric_history(run_id=run_id, key="recall")[
                0
            ].value
            f1 = mlflow_client.get_metric_history(run_id=run_id, key="f1_score")[
                0
            ].value
            return {"precision": precision, "recall": recall, "f1_score": f1}


def read_feature_list(path: str) -> List[Features]:
    features = []
    with open(path, "r") as f:
        for line in f.readlines():
            features += [Features(line.lower().strip())]
    return features


def custom_loss(y_true, y_pred):
    metrics = ConfusionMatrix.from_predictions(
        y_pred, y_true
    ).calc_unweighted_measurements()
    return metrics["precision"] * metrics["f1_score"]


def read_ds(paths: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.concat([pd.read_csv(path) for path in paths])
    df.reset_index(inplace=True, drop=True)
    df = df[["masked_sentence", "original", "replacement", "score"]]
    labels = df.apply(lambda row: row["score"], axis=1)
    return df, labels


def test_clf(model, test_ds, test_y):
    start_time = time.time()
    test_X = model.feature_extractor.extract_features(test_ds)
    preds = model.clf.predict(test_X)
    duration = time.time() - start_time
    metrics = ConfusionMatrix.from_predictions(
        preds, test_y
    ).calc_unweighted_measurements()
    metrics["latency_per_sentence"] = duration / len(test_X)
    return metrics


if __name__ == "__main__":
    main()
