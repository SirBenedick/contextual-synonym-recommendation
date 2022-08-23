#!/bin/bash

set -eux

# prepare git commits/push from CI, see https://stackoverflow.com/a/51466047
mkdir -pvm 0700 ~/.ssh
printf "%s" "$SSH_DEPLOY_KEY" | base64 -d > ~/.ssh/id_rsa
env
echo ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
ssh-keyscan -H 'git.URL.de' >> ~/.ssh/known_hosts
git clone "git@git.URL.git" repo
cd repo/api
git checkout main
git pull
# prepare dvc
dvc remote add -d minio s3://group1 -f --local
dvc remote modify minio endpointurl "$MINIO_ENDPOINT" --local
dvc remote modify minio secret_access_key $MINIO_SECRET_KEY --local
dvc remote modify minio access_key_id $MINIO_ACCESS_KEY --local
dvc pull -f

TODAY=$(date '+%Y-%m-%d')
export MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD"
export MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME"
export MLFLOW_EXPERIMENT_ID="$MLFLOW_EXPERIMENT_ID"
python train.py -i ../datasets/training.csv ../datasets/feedback.csv -t ../datasets/test.csv -o $TODAY-k8s.model --feature-list features.txt

# tag the commit the model was trained with.
git tag "model-$TODAY"
git pull
git push --tags

