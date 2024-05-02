#!/bin/bash
SCRIPT_DIR=$(dirname "$0")


cd "$SCRIPT_DIR"

# Get the name of the containing folder
FOLDER_NAME=$(basename "$PWD")

# Copy src files to the build context
cp -r ../../src ./src
cp ../../setup.py ./setup.py

# Build the Docker image with the folder name as the tag
docker build -t "matthieujln/bird-sound-classif:${FOLDER_NAME}" -f "Dockerfile.${FOLDER_NAME}" .

# Cleanup: Remove copied files
rm -rf ./src
rm ./setup.py