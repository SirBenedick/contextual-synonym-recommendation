FROM python:3.8-slim-buster

WORKDIR /code

COPY ./requirements-cuda.txt /code/requirements-cuda.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements-cuda.txt

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install mlflow dvc dvc[s3]

RUN apt-get update && apt-get install -y curl git perl

COPY . /code