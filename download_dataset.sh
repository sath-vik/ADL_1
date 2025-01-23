#!/bin/bash

# Create the Downloads directory if it doesn't exist
mkdir -p data

# Download the dataset
curl -L -o data/fetal-head-abnormalities-classification.zip \
  https://www.kaggle.com/api/v1/datasets/download/mohammedakheelsb/fetal-head-abnormalities-classification

