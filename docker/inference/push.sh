#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

cd "$SCRIPT_DIR"

# Get the name of the containing folder
FOLDER_NAME=$(basename "$PWD")

docker login

docker push "matthieujln/bird-sound-classif:${FOLDER_NAME}"