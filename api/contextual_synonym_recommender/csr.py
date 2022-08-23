import logging

from contextual_synonym_recommender.baseline import BaselineModel
from contextual_synonym_recommender.model import SynonymScoringModel, SynonymScore

from contextual_synonym_recommender.token_gen import TokenGenerator


class ContextualSynonymRecommender:
    """
    Recommends synonyms for a word in a sentence. This is the "main" entrypoint for the synonym recommendation logic.
    It contains a TokenGenerator, which generates suitable words which can replace the original token and a
    SynonymScoringModel that scores all those suggestions regarding of whether they are synonyms to the original token.
    """
    def __init__(self, unmasker_model_name="bert-large-cased", model_path=None):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing CSR")

        if model_path is not None:
            self.logger.debug(f"Loading model: {model_path}")
            with open(model_path, "rb") as f:
                self.model = SynonymScoringModel.load(f.read())
        else:
            self.logger.debug(f"Loading BaselineModel")
            self.model = BaselineModel()
        self.logger.debug(f"Loaded scoring model")

        self.token_gen = TokenGenerator(unmasker_model_name)
        self.logger.debug(f"Loaded token generator")

    def recommend(self, masked_sentence, original_token):
        self.logger.debug(
            f"recommend() – sentence: {masked_sentence}; original_token: {original_token}"
        )
        synonyms = []
        replacements = self.token_gen.generate_tokens(masked_sentence, original_token)
        print(replacements)
        syn_scores = self.model.predict(masked_sentence, original_token, replacements)
        for replacement, syn_score in zip(replacements, syn_scores):
            score, confidence = syn_score
            if score == SynonymScore.SYNONYM:
                synonyms.append((replacement, confidence))

        synonyms.sort(reverse=True, key=lambda item: item[1])
        without_confidence = [synonym for synonym, confidence in synonyms]
        return without_confidence
