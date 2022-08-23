from typing import List

from gensim import downloader as gensim_downloader
from sentence_transformers import SentenceTransformer, util

from contextual_synonym_recommender.model import SynonymScore


class BaselineModel:
    """
    This implements a simple heuristic for contextually scoring synonyms. There is no Machine Learning going on here,
     i.e. no data is needed for "training" or building the model.

    The heuristic simply calculates the SBERT embeddings for both the original and replacement sentence and calculates
    their cosine similarity. If the similarity exceeds a (configurable) threshold, the replacement is classified as
    a synonym, otherwise not.

    It is designed to replace a SynonymScoringModel in order to provide a baseline for comparison.
    """
    def __init__(
        self,
        w2v_name="glove-wiki-gigaword-50",
        sentence_model_name="all-MiniLM-L6-v2",
        threshold=0.55,
        w2v_model=None,
        sentence_model=None,
    ):
        if w2v_model is None:
            self.w2v = gensim_downloader.load(w2v_name)
        else:
            self.w2v = w2v_model
        if sentence_model is None:
            self.sentence_model = SentenceTransformer(sentence_model_name)
        else:
            self.sentence_model = sentence_model
        self.w2v_name = w2v_name
        self.sentence_model_name = sentence_model_name
        self.threshold = threshold

    def predict(self, sentence, original, replacements: List[str], mask_token="[MASK]"):
        original_sentence = sentence.replace(mask_token, original)
        original_emb = self.sentence_model.encode(original_sentence)
        predictions = []
        for replacement in replacements:
            replacement_sentence = sentence.replace(mask_token, replacement)
            replacement_emb = self.sentence_model.encode(replacement_sentence)
            similarity = util.cos_sim(original_emb, replacement_emb)
            if similarity > self.threshold:
                predictions += [(SynonymScore.SYNONYM, similarity)]
            else:
                predictions += [(SynonymScore.NO_SYNONYM, 1 - similarity)]
        return predictions

    def get_name(self):
        return "BaselineModel(threshold=%s, w2v_model=%s, sentence_model=%s)" % (
            self.threshold,
            self.w2v_name,
            self.sentence_model_name,
        )
