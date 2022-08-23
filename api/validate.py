"""
This script can be used for validating an existing model, i.e. run it on a test set.
It outputs the same set of metrics as in the training stage.

There are two usecases for this script:

1. Re-evaluate an existing model if the test set has changed
2. Test the Baseline Model on a test set (use --baseline for this)

The metrics are output in stdout and in a metrics.csv file.
"""

import argparse
import pprint

import pandas as pd

from contextual_synonym_recommender.model import SynonymScoringModel, SynonymScore

from contextual_synonym_recommender.confusion_matrix import ConfusionMatrix
from contextual_synonym_recommender.baseline import BaselineModel
from train import read_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model file(s)", nargs="+")
    parser.add_argument("-t", "--test", help="Test CSV file", required=True)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument(
        "--threshold", help="Baseline threshold", default=0.95, type=float
    )
    parser.add_argument("--iter-thresholds", action="store_true", default=False)
    parser.add_argument(
        "--w2v-model", help="Baseline w2v-model", default="glove-wiki-gigaword-50"
    )
    parser.add_argument(
        "--sentence-model", help="Baseline Sentence", default="all-MiniLM-L6-v2"
    )
    parser.add_argument("-o", "--output", default="validation_metrics.csv")
    args = parser.parse_args()

    if args.baseline:
        if args.iter_thresholds:
            model1 = BaselineModel(args.w2v_model, args.sentence_model, 0)
            models = [
                BaselineModel(
                    args.w2v_model,
                    args.sentence_model,
                    threshold=t,
                    w2v_model=model1.w2v,
                    sentence_model=model1.sentence_model,
                )
                for t in [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
            ]
            models = [model1] + models
        else:
            models = [
                BaselineModel(args.w2v_model, args.sentence_model, args.threshold)
            ]

    else:
        models = []
        for model_path in args.model:
            with open(model_path, "rb") as f:
                models.append(SynonymScoringModel.load(f.read()))

    metrics_aggregated = []
    df, labels = read_ds([args.test])
    for model in models:
        metrics = validate_model(model, df, labels)
        metrics["name"] = model.get_name()
        pprint.pprint(metrics)
        metrics_aggregated.append(metrics)
    metrics_df = pd.DataFrame(metrics_aggregated)
    metrics_df.to_csv(args.output)


def validate_model(model: SynonymScoringModel, data: pd.DataFrame, ground_truth):
    preds = []
    for _, row in data.iterrows():
        sentence = row["masked_sentence"]
        original = row["original"]
        replacement = row["replacement"]
        prediction, confidence = model.predict(sentence, original, [replacement])[0]
        pred_value = 1 if prediction == SynonymScore.SYNONYM else 0
        preds.append(pred_value)
    metrics = ConfusionMatrix.from_predictions(
        preds, ground_truth
    ).calc_unweighted_measurements()
    return metrics


if __name__ == "__main__":
    main()
