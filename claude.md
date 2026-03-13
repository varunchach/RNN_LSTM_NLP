# Project Overview

This repository is an educational NLP project designed to help students
understand how modern text processing pipelines and sequence models work
in practice.

The project will build a complete end-to-end sentiment analysis system
using sequence models.

Topics covered:

Sequence Models\
Sentiment Analysis\
RNN, LSTM, GRU\
Word Embeddings (Word2Vec / GloVe style embeddings)\
IMDb Sentiment Classification\
Text Pipeline Engineering\
Model Training and Evaluation\
Model Monitoring\
FastAPI Deployment for inference

The main goal is to help students understand the entire journey from raw
text to deployed NLP system.

# Target Audience

Students may come from both technical and non-technical backgrounds.

Therefore:

Code must be simple\
Code must be heavily commented\
Concepts should be explained clearly\
Avoid unnecessary complexity\
Prefer clarity over engineering sophistication

# Technology Stack

Development Environment

VS Code\
Python Virtual Environment (venv)

Deep Learning Framework

PyTorch preferred\
TensorFlow acceptable if necessary

Libraries

torch / tensorflow\
torchtext or equivalent text preprocessing utilities\
pandas\
numpy\
matplotlib\
seaborn\
scikit-learn\
fastapi\
uvicorn\
tensorboard

# Dataset

One dataset must be used across the entire project.

Preferred dataset:

IMDb Movie Review Sentiment Dataset

Possible sources:

TensorFlow datasets\
HuggingFace datasets\
Kaggle\
Torchtext

The same dataset must be used consistently across:

EDA\
training\
evaluation\
API inference

# Exploratory Data Analysis

The project must include exploratory data analysis to help students
understand the dataset.

Display the following:

Sample raw reviews from the dataset\
Random positive and negative examples\
Dataset size and structure\
Class distribution\
Review length distribution\
Word frequency insights

Example outputs:

df.head()\
sample text rows\
text length statistics\
class balance

Visualizations should include:

bar charts\
histograms\
review length distributions\
word frequency plots

These visualizations should help learners understand the nature of text
data before modeling begins.

# Project Structure

Use a modular and structured code architecture.

Example structure:

project_root/

config/\
configuration settings

data/\
dataset download utilities

ingestion/\
dataset loading logic

processing/\
tokenization\
vocabulary building\
padding\
batching

embeddings/\
embedding layer definitions

models/\
RNN implementation\
LSTM implementation\
GRU implementation

training/\
training loop\
validation loop\
early stopping

evaluation/\
metrics\
plots\
model comparison

monitoring/\
TensorBoard logging utilities

inference/\
prediction pipeline

api/\
FastAPI application

notebooks/\
teaching notebook for students

main.py\
training entry point

requirements.txt

# Text Pipeline Engineering

The text processing pipeline must clearly demonstrate NLP engineering
concepts.

These include:

tokenization\
vocabulary creation\
sequence padding\
batching\
token buckets\
efficient dataloaders

Students should clearly see the transformation pipeline:

raw text\
→ tokens\
→ token ids\
→ padded sequences\
→ model input

# Models To Implement

Implement and compare the following sequence models:

RNN\
LSTM\
GRU

Use embedding layers similar to Word2Vec style embeddings.

If feasible, optionally demonstrate pretrained embeddings.

# Training Pipeline

The dataset must be split into:

train\
validation\
test

Shuffle the dataset properly.

Training should include:

limited epochs\
dropout regularization\
validation monitoring\
early stopping if overfitting occurs

# Model Monitoring

Use TensorBoard for monitoring model training and experiments.

TensorBoard should log the following metrics:

training loss\
validation loss\
training accuracy\
validation accuracy

Where useful, also log:

confusion matrix images\
sample predictions

TensorBoard logs should be stored in a runs/ directory.

Students should be able to launch TensorBoard locally using:

tensorboard --logdir=runs

Visualization helps students observe:

how loss decreases during training\
how models converge\
differences between RNN, LSTM, and GRU

# Evaluation

Evaluation must be visual and easy for students to interpret.

Display:

training loss curves\
validation loss curves\
accuracy curves\
confusion matrix\
model comparison tables

Highlight observations such as:

which model performed best\
differences in training stability\
impact of model complexity

# FastAPI Deployment

Create a FastAPI application exposing an inference endpoint.

The endpoint should:

accept a text sentence\
run preprocessing pipeline\
run model inference\
return sentiment prediction

Example endpoint:

POST /predict

Input:

text sentence

Output:

predicted sentiment

# Code Style Requirements

Code must be:

simple\
clean\
highly readable\
well commented

Prefer small functions.

Avoid complex abstractions.

Each module should have clear responsibilities.

# Jupyter Notebook (Teaching Notebook)

Create a notebook that walks students through the entire NLP pipeline
step by step.

The notebook must include:

clear markdown explanations\
intermediate outputs\
visualizations\
stepwise learning

# Notebook Rules

Each code cell should contain no more than 20 lines of code.

The notebook should frequently display outputs such as:

df.head()\
sample reviews\
tokenization examples\
vocabulary mappings\
embedding vectors\
padded sequences\
training logs\
accuracy graphs\
loss curves\
evaluation tables

# Educational Flow

The notebook should guide students through the following journey:

1 Data loading\
2 Dataset exploration (EDA)\
3 Sample text inspection\
4 Text preprocessing\
5 Tokenization\
6 Vocabulary creation\
7 Padding and batching\
8 Embedding layer explanation\
9 RNN architecture\
10 LSTM architecture\
11 GRU architecture\
12 Model training\
13 Model comparison\
14 Evaluation visualization\
15 Inference pipeline\
16 API deployment

# Final Goal

By the end of the project students should understand:

how text becomes model input\
how embeddings work\
how sequence models process text\
how models are trained and evaluated\
how models are monitored using TensorBoard\
how models are deployed through APIs
