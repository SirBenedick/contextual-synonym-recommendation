import enum
import itertools
import pickle
import random
from typing import List

import pandas
import pandas as pd
import numpy as np
from gensim import downloader as gensim_downloader
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk


class Features(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    W2V_HAS_ORIGINAL = enum.auto()
    W2V_HAS_REPLACEMENT = enum.auto()
    W2V_SIM_ORIGINAL_REPLACEMENT = enum.auto()
    W2V_SIM_WINDOW_8_ORIGINAL = enum.auto()
    W2V_SIM_WINDOW_4_ORIGINAL = enum.auto()
    W2V_SIM_WINDOW_2_ORIGINAL = enum.auto()
    W2V_SIM_MEAN_ORIGINAL = enum.auto()
    W2V_SIM_MEAN_REPLACEMENT = enum.auto()
    W2V_SIM_WINDOW_8_REPLACEMENT = enum.auto()
    W2V_SIM_WINDOW_4_REPLACEMENT = enum.auto()
    W2V_SIM_WINDOW_2_REPLACEMENT = enum.auto()
    W2V_SIM_MEAN_DIFF = enum.auto()
    W2V_DIFF = enum.auto()
    W2V_ORIGINAL = enum.auto()
    W2V_REPLACEMENT = enum.auto()
    W2V_SIM_NN = enum.auto()
    W2V_SIM_VB = enum.auto()
    W2V_SIM_ADJ = enum.auto()
    SBERT_SIM_ONLY_TOKENS = enum.auto()
    SBERT_SIM_WINDOW_2 = enum.auto()
    SBERT_SIM_WINDOW_4 = enum.auto()
    SBERT_SIM_WINDOW_8 = enum.auto()
    SBERT_SIM_WHOLE_SENTENCE = enum.auto()
    WORDNET_IS_IN_SYNS = enum.auto()
    WORDNET_IS_IN_ANTONYMS = enum.auto()
    WORDNET_NUMBER_SYNSETS = enum.auto()
    NLTK_POS_CAT = enum.auto()


DEFAULT_W2V_MODEL = "glove-wiki-gigaword-50"
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_FEATURES = [feature for feature in Features]


class FeatureExtractor:
    """
    Extracts numeric features from raw text input (i.e. a masked sentence, the original token and the replacement token).

    The features that should be extracted upon calling the method 'extract_features()' can be specified with the
    `features` parameter. By default, all features are extracted. Furthermore, the word2vec and sbert model name must
    be specified, which will be used for the feature extraction process.
    """
    def __init__(
        self,
        features: List[Features] = DEFAULT_FEATURES,
        w2v_name=DEFAULT_W2V_MODEL,
        sbert_name=DEFAULT_SBERT_MODEL,
        mask_token="[MASK]",
    ):
        self.w2v_name = w2v_name
        self.sbert_model_name = sbert_name
        self.features = features
        self.w2v = gensim_downloader.load(w2v_name)
        self.sbert_model = SentenceTransformer(sbert_name)
        self.mask_token = mask_token
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("punkt")

    def get_w2v_vectors(self, token):
        try:
            return self.w2v[token]
        except KeyError as e:
            return np.zeros(self.w2v.vector_size)

    def get_mask_index(self, tokens, mask_token=None):
        if mask_token is None:
            mask_token = self.mask_token
        mask_tokens = [i for i, tok in enumerate(tokens) if mask_token in tok]
        if len(mask_tokens) != 1:
            raise ValueError("There are %s mask tokens" % len(mask_tokens))
        return mask_tokens[0]

    def get_sbert_window(self, tokens, original, replacement, window_length):
        mask_index = self.get_mask_index(tokens)
        window = tokens[
            max(0, mask_index - window_length) : min(
                mask_index + window_length, len(tokens) - 1
            )
        ]
        window_original = " ".join(window).replace(self.mask_token, original)
        window_replacement = " ".join(window).replace(self.mask_token, replacement)
        return [window_original, window_replacement]

    def get_w2v_features(self, sentence, original, replacement):
        features = []
        tokens = self.tokenize(sentence)
        mask_index = self.get_mask_index(tokens)
        w2v_vectors = [
            self.get_w2v_vectors(token)
            for i, token in enumerate(tokens)
            if i != mask_index
        ]

        mean = np.mean(w2v_vectors, axis=0)
        w2v_original = self.get_w2v_vectors(original)
        w2v_replacement = self.get_w2v_vectors(replacement)
        original_replacement_diff = w2v_original - w2v_replacement

        if Features.W2V_HAS_ORIGINAL in self.features:
            features += [1 if original in self.w2v else 0]
        if Features.W2V_HAS_REPLACEMENT in self.features:
            features += [1 if replacement in self.w2v else 0]

        if Features.W2V_SIM_ORIGINAL_REPLACEMENT in self.features:
            features += [cos_sim(w2v_original, w2v_replacement)]
        if Features.W2V_SIM_MEAN_ORIGINAL in self.features:
            features += [cos_sim(w2v_original, mean)]
        if Features.W2V_SIM_MEAN_REPLACEMENT in self.features:
            features += [cos_sim(w2v_replacement, mean)]
        if Features.W2V_SIM_MEAN_DIFF in self.features:
            features += [cos_sim(original_replacement_diff, mean)]

        if Features.W2V_SIM_WINDOW_2_ORIGINAL in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=2,
                    compare_with_vec=w2v_original,
                )
            ]
        if Features.W2V_SIM_WINDOW_2_REPLACEMENT in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=2,
                    compare_with_vec=w2v_replacement,
                )
            ]
        if Features.W2V_SIM_WINDOW_4_ORIGINAL in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=4,
                    compare_with_vec=w2v_original,
                )
            ]
        if Features.W2V_SIM_WINDOW_4_REPLACEMENT in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=4,
                    compare_with_vec=w2v_replacement,
                )
            ]
        if Features.W2V_SIM_WINDOW_8_ORIGINAL in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=8,
                    compare_with_vec=w2v_original,
                )
            ]
        if Features.W2V_SIM_WINDOW_8_REPLACEMENT in self.features:
            features += [
                self.get_w2v_window_sim_feature(
                    w2v_vectors,
                    mask_index,
                    window_length=8,
                    compare_with_vec=w2v_replacement,
                )
            ]

        if Features.W2V_DIFF in self.features:
            features += original_replacement_diff.tolist()
        if Features.W2V_ORIGINAL in self.features:
            features += w2v_original.tolist()
        if Features.W2V_REPLACEMENT in self.features:
            features += w2v_replacement.tolist()

        original_sentence_tokens = [
            t if i != self.mask_token else original for i, t in enumerate(tokens)
        ]
        pos = nltk.pos_tag(original_sentence_tokens)
        if Features.W2V_SIM_NN in self.features:
            features += [
                self.get_w2v_sim_for_pos(
                    pos, pos_contains="NN", compare_w2v=w2v_replacement
                )
            ]
        if Features.W2V_SIM_ADJ in self.features:
            features += [
                self.get_w2v_sim_for_pos(
                    pos, pos_contains="JJ", compare_w2v=w2v_replacement
                )
            ]
        if Features.W2V_SIM_VB in self.features:
            features += [
                self.get_w2v_sim_for_pos(
                    pos, pos_contains="VB", compare_w2v=w2v_replacement
                )
            ]
        if Features.NLTK_POS_CAT in self.features:
            _, tag = pos[mask_index]
            if "NN" in tag:
                one_hot = [1, 0, 0]
            elif "VB" in tag:
                one_hot = [0, 1, 0]
            elif "JJ" in tag:
                one_hot = [0, 0, 1]
            else:
                one_hot = [0, 0, 0]
            features += one_hot

        return np.array(features)

    def get_w2v_sim_for_pos(self, pos_tags, pos_contains: str, compare_w2v):
        filtered_tags = [t for t, p in pos_tags if pos_contains in p]
        w2v_vecs = [self.get_w2v_vectors(t) for t in filtered_tags]
        if len(w2v_vecs) == 0:
            return 0
        w2v_mean = np.mean(w2v_vecs, axis=0)
        return cos_sim(w2v_mean, compare_w2v)

    def get_w2v_window_sim_feature(
        self, w2v_vectors, mask_index, window_length, compare_with_vec
    ):
        window = w2v_vectors[
            max(0, mask_index - window_length) : min(
                mask_index + window_length, len(w2v_vectors) - 1
            )
        ]
        mean = np.mean(window, axis=0)
        return cos_sim(mean, compare_with_vec)

    def get_wordnet_features(self, sentence, original, replacement):
        features = []
        synsets = wordnet.synsets(original)
        if Features.WORDNET_NUMBER_SYNSETS in self.features:
            features += [len(synsets)]
        if Features.WORDNET_IS_IN_SYNS in self.features:
            features += [1 if self.is_wordnet_syn(synsets, replacement) else 0]
        if Features.WORDNET_IS_IN_ANTONYMS in self.features:
            features += [1 if self.is_wordnet_antonym(synsets, replacement) else 0]
        return np.array(features)

    def is_wordnet_syn(self, synsets, token):
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name().lower() == token.lower():
                    return True
        return False

    def is_wordnet_antonym(self, synsets, token):
        for synset in synsets:
            for lemma in synset.lemmas():
                antonyms = lemma.antonyms()
                antonym_synsets = [a.synset() for a in antonyms]
                is_antonym_syn = self.is_wordnet_syn(antonym_synsets, token)
                if is_antonym_syn:
                    print("antonym", token, synsets)
                    return True
        return False

    def tokenize(self, sentence) -> List[str]:
        replacement_mask_token = "hfsiudfh89eh98fhew98fh9s8dhf9"
        while replacement_mask_token in sentence:
            replacement_mask_token += str(random.randint(0, 10000))
        tokens = word_tokenize(
            sentence.replace(self.mask_token, replacement_mask_token)
        )
        return [t if t != replacement_mask_token else self.mask_token for t in tokens]

    def get_sbert_features_ds(self, df: pandas.DataFrame) -> pd.DataFrame:
        sentences = {
            Features.SBERT_SIM_WHOLE_SENTENCE: [],
            Features.SBERT_SIM_WINDOW_2: [],
            Features.SBERT_SIM_WINDOW_4: [],
            Features.SBERT_SIM_WINDOW_8: [],
            Features.SBERT_SIM_ONLY_TOKENS: [],
        }
        for _, row in df.iterrows():
            original = row["original"]
            replacement = row["replacement"]
            sentence = row["masked_sentence"]
            tokens = self.tokenize(sentence)
            if Features.SBERT_SIM_WHOLE_SENTENCE in self.features:
                sentences[Features.SBERT_SIM_WHOLE_SENTENCE].append(
                    sentence.replace(self.mask_token, original)
                )
                sentences[Features.SBERT_SIM_WHOLE_SENTENCE].append(
                    sentence.replace(self.mask_token, replacement)
                )
            if Features.SBERT_SIM_WINDOW_2 in self.features:
                sentences[Features.SBERT_SIM_WINDOW_2] += self.get_sbert_window(
                    tokens, original, replacement, 2
                )
            if Features.SBERT_SIM_WINDOW_4 in self.features:
                sentences[Features.SBERT_SIM_WINDOW_4] += self.get_sbert_window(
                    tokens, original, replacement, 4
                )
            if Features.SBERT_SIM_WINDOW_8 in self.features:
                sentences[Features.SBERT_SIM_WINDOW_8] += self.get_sbert_window(
                    tokens, original, replacement, 8
                )
            if Features.SBERT_SIM_ONLY_TOKENS in self.features:
                sentences[Features.SBERT_SIM_ONLY_TOKENS] += [original, replacement]
        sentences = list(itertools.chain(*sentences.values()))
        embeddings = self.sbert_model.encode(sentences)
        n = len(df)
        features = pandas.DataFrame()

        offset = 0

        def get_next_sims():
            sims = []
            for i in range(n):
                emb1 = embeddings[offset + 2 * i]
                emb2 = embeddings[offset + 2 * i + 1]
                sims.append(util.cos_sim(emb1, emb2).detach().numpy()[0][0])
            return sims

        if Features.SBERT_SIM_WHOLE_SENTENCE in self.features:
            features["SBERT_SIM_WHOLE_SENTENCE"] = get_next_sims()
            offset += 2 * n
        if Features.SBERT_SIM_WINDOW_2 in self.features:
            features["SBERT_SIM_WINDOW_2"] = get_next_sims()
            offset += 2 * n
        if Features.SBERT_SIM_WINDOW_4 in self.features:
            features["SBERT_SIM_WINDOW_4"] = get_next_sims()
            offset += 2 * n
        if Features.SBERT_SIM_WINDOW_8 in self.features:
            features["SBERT_SIM_WINDOW_8"] = get_next_sims()
            offset += 2 * n
        if Features.SBERT_SIM_ONLY_TOKENS in self.features:
            features["SBERT_SIM_ONLY_TOKENS"] = get_next_sims()
            offset += 2 * n
        return features

    def extract_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        expand_to_df(
            features,
            "w2v",
            dataset.apply(
                lambda row: self.get_w2v_features(
                    row["masked_sentence"], row["original"], row["replacement"]
                ),
                axis=1,
                result_type="expand",
            ),
        )
        features = pandas.concat(
            [features, self.get_sbert_features_ds(dataset)], axis=1
        )
        expand_to_df(
            features,
            "wordnet",
            dataset.apply(
                lambda row: self.get_wordnet_features(
                    row["masked_sentence"], row["original"], row["replacement"]
                ),
                axis=1,
                result_type="expand",
            ),
        )
        return features

    def get_params(self):
        return {
            "w2v_name": self.w2v_name,
            "sbert_name": self.sbert_model_name,
            "features": self.features,
        }

    def get_name(self):
        return "feature extractor"


def expand_to_df(df: pd.DataFrame, col_prefix: str, columns):
    cols = [f"{col_prefix}_{i}" for i in range(columns.shape[1])]
    df[cols] = columns


def cos_sim(a, b):
    val = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if np.isnan(val):
        return -1
    return val
