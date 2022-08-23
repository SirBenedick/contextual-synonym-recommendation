FROM python:3.8-slim-buster

WORKDIR /code

COPY ./requirements-cuda.txt /code/requirements-cuda.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements-cuda.txt

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

ADD production.model /production.model

ENV CSR_MODEL_PATH="/production.model"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--root-path=/api"]