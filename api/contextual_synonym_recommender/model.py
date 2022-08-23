import enum
from typing import List, Tuple

import pandas as pd
import pickle

from contextual_synonym_recommender.feature_extractor import FeatureExtractor


class SynonymScore(enum.Enum):
    NO_SYNONYM = 0
    SYNONYM = 1


class SynonymScoringModel:
    """
    Scores raw text input (i.e. a masked sentence, the original token and the replacement token) as either synonyms or no synonyms.

    Both the training and inference pipelines work in a similar manner: First, the features from the raw input are extracted.
    Then, a classifier (which is the actual Machine Learning component) assigns the label "SYNONYM" or "NO_SYNONYM" to each feature vector.

    This class can be de/serialized in order to be pushed and fetched from mlflow.
    """
    def __init__(self, feature_extractor: FeatureExtractor, clf):
        self.clf = clf
        self.feature_extractor = feature_extractor

    def predict(
        self, sentence: str, original: str, replacements: List[str]
    ) -> List[Tuple[SynonymScore, float]]:
        df = pd.DataFrame(
            [
                {
                    "masked_sentence": sentence,
                    "original": original,
                    "replacement": replacement_token,
                }
                for replacement_token in replacements
            ]
        )
        features = self.feature_extractor.extract_features(df)
        predictions = self.clf.predict_proba(features)
        scores = []
        for i in range(len(df)):
            if predictions[i][0] < predictions[i][1]:
                scores.append((SynonymScore.SYNONYM, predictions[i][1]))
            else:
                scores.append((SynonymScore.NO_SYNONYM, predictions[i][0]))
        return scores

    def serialize(self) -> bytes:
        return pickle.dumps(
            {"clf": self.clf, "feature_extractor": self.feature_extractor.get_params()}
        )

    def get_name(self):
        return "ScoringModel(fe=%s, clf=%s)" % (
            self.feature_extractor.get_name(),
            self.clf.__class__.__name__,
        )

    @staticmethod
    def load(serialized: bytes) -> "SynonymScoringModel":
        unserialized = pickle.loads(serialized)
        return SynonymScoringModel(
            clf=unserialized["clf"],
            feature_extractor=FeatureExtractor(**unserialized["feature_extractor"]),
        )


class ModelTrainer:
    """
    Trains a model.
    """
    def __init__(self):
        pass

    def train(
        self, model: SynonymScoringModel, training_data: pd.DataFrame, labels: pd.Series
    ):
        features = model.feature_extractor.extract_features(training_data)
        model.clf.fit(features, labels)
