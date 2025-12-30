# ML-Based RAG Configuration Optimizer 

## Overview

This project explores a core but under-addressed problem in Retrieval-Augmented Generation (RAG) systems:

**How do we choose the right RAG configuration for a given question?**

Instead of relying on static heuristics (fixed retrievers, fixed chunk sizes, fixed pipelines), this project aims to **learn** which RAG configuration works best based on question characteristics.

The system treats RAG as a configurable system and frames configuration selection as a machine learning problem.

This repository is currently under active development.

---

## Motivation

While building RAG systems, a recurring challenge emerged:

- BM25 works better for some questions, FAISS for others
- Extractive QA models fail on multi-chunk or list-based answers
- LLMs are powerful but expensive and sensitive to context quality
- Small parameter changes (chunk size, k, reranker) can drastically affect answer quality

---

## Core Idea

For each question:
1. Run multiple RAG configurations
2. Measure answer quality
3. Log configuration parameters and question features
4. Train a model to predict the best configuration for future questions

The ML model does not generate answers.  
It selects the most suitable RAG setup.

---

## RAG Configurations

Each configuration is defined by:

- Retriever type: BM25, FAISS, Hybrid
- Chunk size
- Top-k retrieval depth
- Reranker on or off (cross-encoder)
- Answering model: Extractive QA or LLM

Invalid or low-signal combinations are explicitly filtered to reduce noise.

---

## Dataset Generation Pipeline

For each question and configuration pair, the following are logged:

- Question text
- Question type (factoid, list, explanatory, summary)
- Question features:
  - Token length
  - Named entity count
  - Contains numbers
  - Question embedding
- RAG configuration parameters:
  - Retrieved chunk IDs
  - Model prediction
  - Ground truth answer
  - Evaluation metrics (Exact Match, F1, semantic similarity)

This results in a structured dataset suitable for supervised learning.

---

## Evaluation Metrics

Answer quality is evaluated using:

- Exact Match for strict factoid questions
- Token-level F1 using SQuAD-style normalization
- Semantic similarity for explanatory answers
- Manual scoring for ambiguous cases

Metrics are chosen based on question type to avoid misleading supervision.

---

## Machine Learning Objective

The ML task is defined as:

Given a question and its features, predict the best-performing RAG configuration.

Initial models will include:
- Logistic Regression
- Random Forest
- Gradient Boosted Trees

The ML model is evaluated against static heuristics and always-hybrid baselines.

---

## Current Status

- Question feature extraction implemented
- Proposal model for predicting top-7 config is completed 
- Training ranker for choosing best config is under progress 


Model training and evaluation are upcoming stages.

---

## Planned Work

- Finalize dataset generation at scale
- Train baseline ML models
- Analyze feature importance
- Compare ML routing versus heuristic routing
- Add detailed experiment reports

---
