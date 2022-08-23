from typing import Union, Optional
from pydantic import BaseModel, validator


class Config(BaseModel):
    model_path: Optional[str]


class SentenceRequestObject(BaseModel):
    masked_sentence: str
    original_token: str

    @validator("masked_sentence")
    def masked_sentence_must_include_mask_token(cls, v):
        if v.count("[MASK]") != 1:
            raise ValueError("must contain exactly one [MASK] token")

        return v

    @validator("original_token")
    def original_token_must_be_single_word(cls, v):
        if " " in v:
            raise ValueError("must not contain a space ")
        if "!" in v:
            raise ValueError("must not contain a ! ")
        if "?" in v:
            raise ValueError("must not contain a ? ")
        if ":" in v:
            raise ValueError("must not contain a : ")
        return v


class FeedbackObject(BaseModel):
    masked_sentence: str
    original_token: str
    replacement: str
    score: int
    rank_of_recommendation: Union[int, None] = None

    @validator("masked_sentence")
    def masked_sentence_must_include_mask_token(cls, v):
        if v.count("[MASK]") != 1:
            raise ValueError("must contain exactly one [MASK] token")
        return v

    @validator("original_token")
    def original_token_must_be_single_word(cls, v):
        if " " in v:
            raise ValueError("must not contain a space ")
        if "!" in v:
            raise ValueError("must not contain a ! ")
        if "?" in v:
            raise ValueError("must not contain a ? ")
        if ":" in v:
            raise ValueError("must not contain a : ")
        return v

    @validator("replacement")
    def replacement_must_be_single_word(cls, v):
        if " " in v:
            raise ValueError("must not contain a space ")
        if "!" in v:
            raise ValueError("must not contain a ! ")
        if "?" in v:
            raise ValueError("must not contain a ? ")
        if ":" in v:
            raise ValueError("must not contain a : ")
        return v

    @validator("score")
    def score_must_be_0_or_1(cls, v):
        if v != 0 and v != 1:
            raise ValueError("score must be 0 or 1 ")
        return v

    @validator("rank_of_recommendation")
    def score_must_be_between_0_or_1000(cls, v):
        if v < 0 or v > 1000:
            raise ValueError("rank_of_recommendation must be between 0 and 1000 ")
        return v
