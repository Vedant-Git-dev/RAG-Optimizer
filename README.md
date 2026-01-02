## RAG Configuration Optimizer

This project builds an automatic RAG configuration optimizer that selects the best retrieval and generation setup for a given question.
Instead of using a fixed RAG pipeline, the system learns from past runs to dynamically choose the most suitable configuration.

The core idea is a two-stage decision system:

- A Proposer filters promising RAG configurations.

- A Ranker selects the best configuration among those candidates.

---

## Problem Motivation

RAG performance is highly sensitive to configuration choices such as:

- retriever type (BM25, FAISS, etc.)

- chunk size and overlap

- top-k retrieval depth

- reranking usage

- answer generation model

A single static configuration performs well on average but fails on specific question types.
This project treats RAG configuration selection as a learning-to-rank problem.

---

## Dataset Structure

Each row represents a (question, configuration) pair with an observed reward.

### Key columns

question id – question identifier (grouping key)

Question features:

- token length

- entity count

- question type

- presence of digits (isdigit)
  
Configuration features:

- retriever

- chunk size

- chunk overlap

- reranker flag (True or False)

- answer model

- reward – performance score for that configuration

Each question is evaluated with multiple configurations.

---

## Models

1. Proposer (Candidate Generator)

- Model: LightGBMClassifier

- Objective: binary classification

- Training target: whether a configuration belongs to the top-N configs for a question

- Purpose: high recall, not precision

The proposer ensures the optimal configuration is almost always included in the candidate set.

2. Ranker (Final Selector)

- Model: LightGBMRanker (LambdaRank)

- Objective: learning-to-rank with NDCG

- Training target: discretized relevance derived from reward

- Grouping: per-question (question id)

The ranker learns relative ordering between configurations for the same question.

---

## Training Strategy

- Strict separation of questions between splits

- Row ordering enforced before ranker training

- Categorical features encoded using fixed mappings

---

## Evaluation

1. Ranker Quality

- NDCG@k to measure ordering quality

- Hit@k to measure near-optimal selection

2. End-to-End System

- Regret = (oracle reward − chosen reward)

- Median regret ≈ 0 for most questions

- Mean regret affected by rare tail cases (feature-limited)

The system performs strongly on typical cases while remaining explainable.

---

## Exported Artifacts

The final system exports:

models/

├── proposer.txt          # LightGBM proposer

├── ranker.txt            # LightGBM ranker

└── feature_cols.joblib   # feature order/schema

This makes the optimizer portable across notebooks and environments.

---

## Inference Flow

- Extract question features

- Enumerate all RAG configurations

- Score configurations with proposer

- Select top-N candidates

- Rank candidates with ranker

- Choose best configuration

- Execute RAG with chosen config

---

## Key Design Decisions

- Two-stage architecture to balance coverage and precision

- Learning-to-rank instead of regression or classification

- Group-aware training to avoid data leakage

- Explicit feature versioning for reproducibility

---

## Limitations

- Tail failures exist when reward differences are driven by signals not present in features

- Performance depends on the diversity of training configurations

- Designed for structured RAG config spaces, not arbitrary pipelines

---

## Future Improvements

Add retrieval-quality signals (similarity statistics)

Online learning from new RAG executions

Per-domain specialization

---

## Summary

This project demonstrates how RAG configuration selection can be framed as a learning problem rather than a manual tuning task.
The resulting system is modular, explainable, and extensible to more complex retrieval and generation pipelines.

---

<div align = "center">

**Made with ❤️ by [Vedant](https://github.com/Vedant-Git-dev)**

If you find this project helpful, please consider giving it a ⭐️!

</div>

---