import transformers
import typing
import spacy
import argparse
import random
import pandas
from gensim import downloader
import nltk

from nltk.corpus import wordnet

SUPPORTED_MASK_MODELS = {
    "bert-base-cased": "[MASK]",
    "bert-large-cased": "[MASK]",
    "roberta-large": "<mask>",
    "roberta-base": "<mask>",
    "distilbert-base-cased": "[MASK]",
}

ds_mask_token = "[MASK]"

nltk.download("wordnet")
nltk.download("omw-1.4")


class DatesetGenerator:
    def __init__(
        self,
        model_name: str = "bert-large-cased",
        spacy_model: str = "en_core_web_lg",
        gensim_model="word2vec-google-news-300",
        top_k_unmasked_tokens=100,
    ):
        if model_name not in SUPPORTED_MASK_MODELS:
            raise ValueError(
                f"{model_name} is no valid or supported model. Choose of {SUPPORTED_MASK_MODELS.keys()}"
            )
        self.mask_token = SUPPORTED_MASK_MODELS[model_name]
        print("Load spacy model...")
        self.pos_tagger = spacy.load(spacy_model)
        self.mask_token = SUPPORTED_MASK_MODELS[model_name]
        self.top_k_unmasked_tokens = top_k_unmasked_tokens
        print("Load w2v gensim model...")
        self.w2v = downloader.load(gensim_model)
        print("Load unmasking model...")
        self.unmasker = transformers.pipeline("fill-mask", model=model_name)
        self.replace_pos_tags = {"ADJ", "ADV", "NOUN", "VERB"}
        self.ignore_tokens = {"", ";", " ", "\t", ".", ",", "(", ")", "[", "]"}
        self.dataset = []

    def input_sentence(self, sentence: str):
        if self.mask_token in sentence:
            print(f"{self.mask_token} is present in the input sentence. Skip.")
            return
        tokens = self.pos_tagger(sentence)
        indexes = [
            i for i, tok in enumerate(tokens) if tok.pos_ in self.replace_pos_tags
        ]

        while True:
            index = random.choice(indexes)
            original_token = tokens[index].text.lower()
            masked_sentence = self._make_new_sentence(tokens, index, self.mask_token)
            suggestions = self._get_best_predictions(masked_sentence, original_token)
            suggestions += get_nltk_synonyms(original_token)[:5]
            # suggestions += self._get_w2v_similar_tokens(original_token)
            suggestions = list(set(suggestions))
            print(
                f"Next sentence: {masked_sentence} \n"
                f" with {self.mask_token}={original_token} \n"
                f" suggested synonyms: {suggestions}"
            )
            choice = self._choice_input(
                "[o]k; [d]ifferent token; [s]kip sentence?", choices=["d", "s", "o"]
            )
            if choice == "s":
                return
            if choice == "o":
                break

        print(masked_sentence)
        print("Original token: ", original_token)
        for tok in suggestions:
            print(tok)
            score = self._get_score()
            self.dataset.append(
                {
                    "masked_sentence": masked_sentence.replace(
                        self.mask_token, ds_mask_token
                    ),
                    "original": original_token,
                    "replacement": tok,
                    "score": score,
                }
            )

    def _get_best_predictions(
        self, masked_sentence, original_token, best_k: int = 10
    ) -> typing.List[str]:
        predictions = self.unmasker(masked_sentence, top_k=self.top_k_unmasked_tokens)

        for pred in predictions:
            if pred["token_str"].lower() == original_token:
                original_token_score = pred["score"]
                break
        similarities = []
        for pred in predictions:
            tok = pred["token_str"].strip()
            if tok.lower() == original_token.lower():
                continue
            try:
                similarity = self.w2v.similarity(original_token, tok)
            except KeyError:
                print(f"Token {tok} is not present the w2v model")
                similarity = -1
            similarities.append((tok, similarity, pred["score"]))

        similarities = sorted(
            similarities, key=lambda x: x[1] * x[1] * x[2], reverse=True
        )
        return [token for token, _, _ in similarities[:best_k]]

    def _get_score(self) -> int:
        valid_range = {0, 1, 2, 3}
        score = -1
        while score not in valid_range:
            score = input("Score? %s" % valid_range)
            try:
                score = int(score)
            except:
                pass
        return score

    def _choice_input(self, question: str, choices: typing.Iterable[str]) -> str:
        choices = [c.lower() for c in choices]
        read_input = None
        while read_input not in choices:
            read_input = input(f"{question}").lower()
        return read_input

    def _make_new_sentence(self, tokens, index: int, new_token: str):
        new_tokens = [
            tok.text if i != index else new_token for i, tok in enumerate(tokens)
        ]
        return " ".join(new_tokens)

    def write(self, path):
        df = pandas.DataFrame(self.dataset)
        with open(path, "a"):
            df.to_csv(path, index=False)

    def _get_w2v_similar_tokens(self, original_token):
        try:
            predictions = self.w2v.most_similar(original_token, topn=5)
            return [token for token, _ in predictions[:5]]
        except KeyError:
            return []


def get_nltk_synonyms(token) -> typing.List[str]:
    syns = []
    for syn in wordnet.synsets(token):
        for lemm in syn.lemmas():
            syns.append(lemm.name())
    return [s for s in set(syns) if s.lower() != token.lower()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file. Will be read line-wise.",
        type=str,
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output csv file.", type=str
    )
    parser.add_argument("--start", default=1, help="Start at line x", type=int)
    parser.add_argument(
        "--mask-model",
        default="bert-base-cased",
        help=f"Mask model to use. Choose one of: {[m for m in SUPPORTED_MASK_MODELS.keys()]}",
        type=str,
    )
    parser.add_argument(
        "--w2v-model",
        default="word2vec-google-news-300",
        help="w2v gensim model used for calculating the similarity between predicted tokens.",
        type=str,
    )
    parser.add_argument(
        "--top-k",
        default=100,
        help="score the first k tokens from the unmasking model",
        type=int,
    )
    args = parser.parse_args()
    gen = DatesetGenerator(
        model_name=args.mask_model,
        gensim_model=args.w2v_model,
        top_k_unmasked_tokens=args.top_k,
    )
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i < args.start:
                continue
            sentence = line.strip()
            if sentence == "":
                continue
            try:
                gen.input_sentence(sentence)
                gen.write(args.output)
            except Exception as e:
                print("Error occured:", e)


if __name__ == "__main__":
    main()
