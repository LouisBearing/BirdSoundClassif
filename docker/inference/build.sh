#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

cd "$SCRIPT_DIR"

# Get the name of the containing folder
FOLDER_NAME=$(basename "$PWD")

# Ensure the destination directory exists
mkdir -p ./inference

# Copy the contents of the source inference directory to the already created destination directory
cp -r ../../app/inference/. ./inference/

# Copy model weights to the build context
ls ../../models/
mkdir -p ./models
cp -r ../../models/detr_noneg_100q_bs20_r50dc5 ./models/detr_noneg_100q_bs20_r50dc5

# Build the Docker image with the folder name as the tag
docker build -t "matthieujln/bird-sound-classif:${FOLDER_NAME}" -f "Dockerfile.${FOLDER_NAME}" .

# Cleanup: Remove copied directories
rm -rf ./models
rm -rf ./inference