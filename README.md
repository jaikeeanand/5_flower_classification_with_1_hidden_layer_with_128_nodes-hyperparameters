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


## Results
- Validation Accuracy: 66%
- Best configuration found using W&B sweeps

## Key Highlights
- Implemented hyperparameter tuning using W&B sweeps  
- Built scalable data pipeline using tf.data API  
- Experimented with multiple configurations to optimize performance

## W&B Dashboard
https://wandb.ai/jaicky-iit-ism-dhanbad/5-flowers-classification?nw=nwuserjaikeeanand007&panelDisplayName=batch%2Faccuracy&panelSectionName=batch

## Data Pipeline

The dataset is converted into CSV format containing image paths and labels.
These CSV files are used to create an efficient data pipeline using TensorFlow.




```python
train_dataset = (
    tf.data.TextLineDataset("train_set.csv")
    .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(config.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

eval_dataset = (
    tf.data.TextLineDataset("eval_set.csv")
    .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(config.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
```
A simple neural network model with one hidden layer is used for classification.
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    keras.layers.Dense(config.hidden_nodes, activation="relu"),
    keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
])
```
The model is compiled using Adam optimizer and trained using sparse categorical crossentropy loss.
```pyhton
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
```

Weights & Biases is used to track experiments and log metrics.
```python
model.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=config.epochs,
    callbacks=[WandbMetricsLogger(log_freq=5)]
)
```

## 👨‍💻 Author
Avinash Kumar Bharti
