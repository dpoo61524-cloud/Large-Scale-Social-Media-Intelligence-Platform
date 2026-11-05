# System Architecture: Large Scale Social Media Intelligence Platform

## 1. Overview
The **Large Scale Social Media Intelligence Platform** is designed to ingest, process, analyze, and visualize high-volume conversational data from social media platforms (Reddit and YouTube).

This architecture follows the 7-step incremental roadmap defined in the internship specification:

```
[Reddit & YouTube APIs]
          │ (Raw JSON)
          ▼
   [Step 1: Ingestion & Raw Storage (data/raw)]
          │
          ▼
   [Step 2: Text Preprocessing & Cleaning (Spark NLP)]
          │
   ┌──────┴───────────────────────────────┬───────────────────────────────┐
   ▼                                      ▼                               ▼
[Step 3: Multilingual Sentiment]   [Step 4: Topic Modeling]   [Step 5: Interaction Graph]
  (Transformer Models)                (BERTopic)               (Neo4j & NetworkX)
   │                                      │                               │
   └──────────────────────┬───────────────┴───────────────────────────────┘
                          │
                          ▼
            [Step 6: Misinformation Classifier]
              (Trained on LIAR / FakeNewsNet)
                          │
                          ▼
            [Step 7: Search & Visual Dashboards]
              (Elasticsearch & Kibana)
```

---

## 2. Planned Components & Responsibilities

### Component 1: Data Ingestion (Step 1)
- **Responsibility**: Ingest public posts, comments, and video metadata from Reddit and YouTube.
- **Protocols & Rules**: Respect rate limits, pagination, and platform terms of service.
- **Storage**: Store unmodified raw payloads in `data/raw/` in standard JSON format for auditability and replayability.

### Component 2: Text Preprocessing & Cleaning (Step 2)
- **Responsibility**: Clean raw conversational social media text for downstream natural language processing.
- **Key Tasks**: Remove noise, normalize whitespaces and URLs, handle emojis, and perform language detection.
- **Technology**: Python and Spark NLP for scalable text processing.

### Component 3: Multilingual Sentiment Analysis (Step 3)
- **Responsibility**: Detect sentiment polarity (positive, negative, neutral) across multilingual posts and comments.
- **Key Tasks**: Track sentiment shifts over time across topics and queries.
- **Technology & Datasets**: Transformer models; reference datasets include Sentiment140, TweetEval, and Multilingual Amazon Reviews Corpus.

### Component 4: Dynamic Topic Discovery (Step 4)
- **Responsibility**: Automatically extract emerging themes and track how topics rise and fall over time.
- **Technology**: BERTopic with transformer-based embeddings and c-TF-IDF.

### Component 5: User Interaction Graph & Network Analysis (Step 5)
- **Responsibility**: Model social relationships (replies, mentions, subreddit cross-posts) as a directed graph.
- **Key Tasks**: Identify key influencers via PageRank and uncover communities using community detection algorithms.
- **Technology & Datasets**: Neo4j graph database, NetworkX graph library; reference datasets include Stanford SNAP Reddit Hyperlink Network.

### Component 6: Misinformation / Fake News Classification (Step 6)
- **Responsibility**: Flag potentially misleading or false claims using a trained classification model.
- **Technology & Datasets**: Supervised text classifier trained on standard benchmark datasets (LIAR, FakeNewsNet, ISOT). Model artifacts saved to `models/`.

### Component 7: Search, Indexing & Dashboards (Step 7)
- **Responsibility**: Centralized search, aggregated metric indexing, and interactive visual reporting.
- **Technology**: Elasticsearch (indexing cleaned texts, sentiments, and topics) and Kibana (dashboards for real-time trends, search, and alerts).

---

## 3. Directory Layout and Security Model

- **Configuration (`config/`)**: Environment-driven configuration via `settings.py`. Sensitive credentials (API keys, passwords) are loaded via `.env` and masked from all outputs and logs.
- **Raw Data (`data/raw/`)**: Stores ingested JSON data; ignored by Git.
- **Processed Data (`data/processed/`)**: Stores cleaned datasets; ignored by Git.
- **Models (`models/`)**: Stores trained model weights and pipeline binaries; ignored by Git.
- **Source (`src/`)**: Modular Python source packages implementing each pipeline step incrementally.
- **Tests (`tests/`)**: Unit and integration test suites validating each component independently.
