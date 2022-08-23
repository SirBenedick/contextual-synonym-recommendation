import typing

import nltk
import transformers
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer


class TokenGenerator:
    """
    Predicts tokens that might be suitable replacements for a word in a sentence.

    Those suggestions come from two sources:
    1. an unmasking model from huggingface (e.g. BERT); it is context-sensitive, but cannot predict synonyms
    2. NLTK's synonym wordnet; it can predict synonyms for a word, but does not consider its context

    Duplicates and words with the same stem (according to NLTK's porter stemmer) are filtered out. The maximum number
    of suggestions can be configured with 'max_size'.
    """
    def __init__(self, unmasker_model_name, max_size=200):
        self.unmasker = transformers.pipeline("fill-mask", model=unmasker_model_name)
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        self.max_size = max_size
        self.stemmer = PorterStemmer()

    def generate_tokens(self, masked_sentence: str, original_token: str):
        predictions = self.unmasker(masked_sentence, top_k=150)
        replacements = [
            pred["token_str"]
            for pred in predictions
            if pred["token_str"].lower() != original_token.lower()
        ]
        unmasker_repl_set = set(replacements)
        wordnet_syns = self.get_wordnet_syns(original_token, depth=3)
        without_duplicates = [s for s in wordnet_syns if s not in unmasker_repl_set]
        print(replacements, without_duplicates)
        filtered_stems = self.filter_stems(
            replacements + without_duplicates, original_token, max_length=self.max_size
        )
        return filtered_stems

    def get_wordnet_syns(self, word, depth=3, current_depth=0):
        if current_depth >= depth:
            return set()
        synsets = wordnet.synsets(word.lower())
        syns = set()
        for synset in synsets:
            syns = syns.union(set([lemma.name().lower() for lemma in synset.lemmas()]))
        for syn in syns:
            syns = syns.union(
                self.get_wordnet_syns(syn, depth=depth, current_depth=current_depth + 1)
            )
        return set([s for s in syns if s.lower() != word.lower()])

    def filter_stems(
        self, tokens: typing.List[str], original_token: str, max_length: int
    ) -> typing.List[str]:
        original_stem = [self.stemmer.stem(original_token)]
        stems = set(original_stem)
        filtered_tokens = []
        for t in tokens:
            token_stem = self.stemmer.stem(t.lower())
            if token_stem in stems:
                continue
            filtered_tokens.append(t)
            stems.add(token_stem)
            if len(filtered_tokens) >= max_length:
                break
        return filtered_tokens
