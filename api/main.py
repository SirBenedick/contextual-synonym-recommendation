import os
import logging
import logging.config
import timeit
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import csv

from contextual_synonym_recommender.csr import ContextualSynonymRecommender
from db import (
    addFeedback,
    getAllFeedback,
    logRequest,
    getAllRequestsMonitoring,
    get_overview_db,
)
from classes import Config, SentenceRequestObject, FeedbackObject

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_config() -> Config:
    try:
        model_path = os.environ["CSR_MODEL_PATH"]
    except KeyError:
        logging.debug(
            "CSR_MODEL_PATH environment variable is not set, use baseline model"
        )
        model_path = None
    return Config(model_path=model_path)


config = load_config()
csr = ContextualSynonymRecommender(model_path=config.model_path)
logger.debug("Loaded Contextual Synonym Recommender")

app = FastAPI(debug=True, docs_url="/api/docs")
logger.debug("Started FastAPI")

# Handle Cross-Origin-Conflicts, should not be the way to handle it in production, but enough for this project
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:3000",
    "http://localhost:8000",
    "URL/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
Subpath "/api" is being added by the kubernets ingress network when forwarding the requests to the csr-api-service.
This leads to this api receving "/api/predict" calls and not just "/predict".
There sems to be no easy way to configure ingress to stop forwarding with the subpath.
Azure seems to have it figured out (https://stackoverflow.com/questions/69370957/kubernetes-ingress-pass-only-sub-path-to-backend-and-not-full-path).

For now every route in the FastAPI appliaction has to start with "/api/".

This leads to the docs not being available when in production.
"""


@app.get("/api/")
def read_root():
    return "Nothing to see here"


@app.get("/api/feedback/all")
def get_feedback_as_csv():
    data = getAllFeedback()
    return generate_csv_file(
        file_name="feedback.csv",
        headers=[
            "masked_sentence",
            "original",
            "replacement",
            "score",
            "rank_of_recommendation",
            "timestamp",
        ],
        data=data,
    )


@app.get("/api/feedback/yesterday")
def get_feedback_yesterday_as_csv():
    data = getAllFeedback(only_yesterday=True)
    return generate_csv_file(
        file_name="feedback.csv",
        headers=[
            "masked_sentence",
            "original",
            "replacement",
            "score",
            "rank_of_recommendation",
            "timestamp",
        ],
        data=data,
    )


@app.post("/api/feedback")
def add_feedback(feedback: FeedbackObject):
    try:
        addFeedback(
            masked_sentence=feedback.masked_sentence,
            original=feedback.original_token,
            replacement=feedback.replacement,
            score=feedback.score,
            rank_of_recommendation=feedback.rank_of_recommendation,
        )
    except:
        logger.debug("Failed to add feedback")
        return False

    return True


@app.get("/api/monitoring/overview")
def get_overview():

    overview = get_overview_db()

    return overview


@app.get("/api/monitoring/requests/all")
def get_requests_monitoring_as_csv():
    data = getAllRequestsMonitoring()
    return generate_csv_file(
        file_name="monitoring_requests.csv",
        headers=[
            "masked_sentence",
            "original_token",
            "recommended_synonyms",
            "time_in_s",
            "timestamp",
        ],
        data=data,
    )


@app.get("/api/monitoring/requests/yesterday")
def get_requests_monitoring_yesterday_as_csv():
    data = getAllRequestsMonitoring(only_yesterday=True)
    return generate_csv_file(
        file_name="monitoring_requests.csv",
        headers=[
            "masked_sentence",
            "original_token",
            "recommended_synonyms",
            "time_in_s",
            "timestamp",
        ],
        data=data,
    )


@app.post("/api/predict")
def predict_sentence(sentenceObject: SentenceRequestObject):
    # Stop the time needed for the recommendaion
    start = timeit.default_timer()
    synonyms = csr.recommend(
        masked_sentence=sentenceObject.masked_sentence,
        original_token=sentenceObject.original_token,
    )
    stop = timeit.default_timer()
    time_difference_in_seconds = stop - start
    time_difference_in_seconds = round(time_difference_in_seconds, 6)

    # Log duration of predict request for a minimum '__monitoring__' effect
    try:
        logRequest(
            masked_sentence=sentenceObject.masked_sentence,
            original_token=sentenceObject.original_token,
            recommended_synonyms=synonyms,
            time_in_s=time_difference_in_seconds,
        )
    except:
        logger.debug("Failed to add logRequest")

    return synonyms

@app.post("/api/predict_performance_test")
def predict_sentence_performance_test(sentenceObject: SentenceRequestObject):
    # Interface is used for loadtests
    # returns the time needed for a recommendation
    # does not log the requests 
    start = timeit.default_timer()
    synonyms = csr.recommend(
        masked_sentence=sentenceObject.masked_sentence,
        original_token=sentenceObject.original_token,
    )
    stop = timeit.default_timer()
    time_difference_in_seconds = stop - start
    time_difference_in_seconds = round(time_difference_in_seconds, 6)


    return {"synonyms": synonyms, "time_needed_for_prediction": time_difference_in_seconds}


# Helper Functions
def generate_csv_file(file_name, headers, data):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    return FileResponse(path=file_name, media_type="text/csv", filename=file_name)
