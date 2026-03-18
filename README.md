# 5_flower_classification_with_1_hidden_layer_with_128_nodes-hyperparameters
# Flower Classification using CNN + W&B

## Overview
This project performs image classification on 5 flower classes using TensorFlow and Weights & Biases (W&B).

## Dataset
- 5 classes: Lilly, Tulip, Sunflower, Lotus, Orchid
- Loaded using kagglehub

## Features
- Custom data pipeline using tf.data
- CSV-based dataset loading
- Hyperparameter tuning using W&B sweeps
- Experiment tracking (accuracy, loss, epoch vs validation accuracy)

## Hyperparameters Tuned
- batch_size: [8, 16]
- learning_rate: [0.001, 0.0001]
- hidden_nodes: [64, 128]
- image_size: [16, 224]
- epochs: [5, 10]

## How to Run
1. Open 5-flowers-classification.ipynb in Google Colab
2. Run all cells

## Requirements
- tensorflow
- wandb
- matplotlib
- pandas
- scikit-learn
- kagglehub

## Dataset Processing

The dataset is downloaded using kagglehub.  
Image paths and labels are extracted and stored in CSV files:

- train_set.csv → training data  
- eval_set.csv → validation data  

These CSV files are used to create an efficient data pipeline using TensorFlow.

## Files
- 5_flower_classification.ipynb → main code
- train_set.csv → training data
- eval_set.csv → validation data

## Results
- Validation Accuracy: 66%
- Best configuration found using W&B sweeps

## Key Highlights
- Implemented hyperparameter tuning using W&B sweeps  
- Built scalable data pipeline using tf.data API  
- Experimented with multiple configurations to optimize performance

## W&B Dashboard
https://wandb.ai/jaicky-iit-ism-dhanbad/5-flowers-classification?nw=nwuserjaikeeanand007&panelDisplayName=batch%2Faccuracy&panelSectionName=batch

## 👨‍💻 Author
Avinash Kumar Bharti
