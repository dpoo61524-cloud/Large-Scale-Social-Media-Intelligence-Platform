# Large-Scale Social Media Intelligence Platform
*(Repository: [Large-Scale-Social-Media-Intelligence-Platform](https://github.com/dpoo61524-cloud/Large-Scale-Social-Media-Intelligence-Platform))*

An end-to-end data engineering and intelligence pipeline designed to ingest, clean, analyze, enrich, index, and visualize multi-source social media data from **Reddit** and **YouTube**.

---

## 1. Project Overview & Objectives

The platform provides a modular, contract-driven architecture capable of processing unstructured social media content through natural language processing, dynamic topic modeling, interaction network analysis, heuristic misinformation risk assessment, multi-source intelligence consolidation, and search indexing.

### Core Capabilities
* **Multi-Source Ingestion**: Native OAuth2 and API v3 data collection for Reddit and YouTube with automatic fallback to offline development fixtures.
* **Text Preprocessing**: Robust text normalization, demojization, entity unescaping, and language identification.
* **Multilingual Sentiment Tracking**: Transformer-based sentiment polarity inference (`cardiffnlp/twitter-roberta-base-sentiment-latest`) with non-destructive preservation of unsupported records.
* **Dynamic Topic Discovery**: BERTopic theme extraction with deterministic outlier classification (`topic_id: -1`, `topic_status: "outlier"`).
* **Interaction Graph Modeling**: Structural network analysis (PageRank, betweenness centrality, degree metrics) and Neo4j Cypher persistence.
* **Heuristic Misinformation Risk Assessment**: Stylometric feature extraction and LIAR benchmark machine learning classification producing heuristic risk indices.
* **Multi-Source Intelligence Consolidation**: Strict Schema-v2 record unification merging Reddit and YouTube signals into canonical intelligence records.
* **Search & Analytics Indexing**: Elasticsearch bulk indexing adapter with strict type mappings, field sanitization, and indexing manifest tracking.
* **Dashboard Validation & Deployment**: Structural validation and HTTP import bundle for 41 Kibana saved objects (dashboards, visualizations, search views).
* **Unified Pipeline Orchestration**: 12-stage linear pipeline executor (`--run-all`) with isolated run directories (`data/runs/<run_id>/`), OS-level advisory concurrency locking, single-platform graceful degradation, and exception-safe atomic canonical promotion.

---

## 2. Canonical 12-Stage Pipeline Architecture

The platform executes processing in a strict, linear dependency order across 12 canonical stages:

```text
[Raw Ingestion]
  ├── 1. reddit_ingestion   ──────> data/raw/reddit_raw_posts.json
  └── 2. youtube_ingestion  ──────> data/raw/youtube_videos_raw.json
          │
[Text Preprocessing]
  ├── 3. reddit_preprocessing   ──> data/processed/reddit_posts_processed.json
  └── 4. youtube_preprocessing  ──> data/processed/youtube_videos_processed.json
          │
[Analytical Enrichment]
  ├── 5. sentiment          ──────> data/processed/sentiment_analysis_results.json
  ├── 6. topics             ──────> data/processed/topic_modeling_results.json
  ├── 7. graph              ──────> data/processed/graph_analysis_results.json
  │       │
  │       └── 8. neo4j_persistence ──> Graph Database (Neo4j Cypher Ingestion)
  │
  └── 9. misinformation     ──────> data/processed/misinformation_analysis_results.json
          │
[Consolidation & Search Integration]
  ├── 10. consolidation     ──────> data/processed/consolidated_intelligence.json
  ├── 11. elasticsearch_indexing ─> Search Engine (Elasticsearch Bulk Indexing)
  └── 12. kibana_validation ──────> Offline Bundle Validation (41 Saved Objects)
```

### Stage Summary & Data Contracts

| Stage # | Stage Identifier | Functional Scope | Primary Output Artifact | Prerequisite Inputs |
|:---:|---|---|---|---|
| **1** | `reddit_ingestion` | Fetch Reddit posts (live API or mock fixture) | `reddit_raw_posts.json` | Network API or sample fixture |
| **2** | `youtube_ingestion` | Fetch YouTube metadata (live API or mock fixture) | `youtube_videos_raw.json` | Network API or sample fixture |
| **3** | `reddit_preprocessing` | Clean text, demojize, detect language | `reddit_posts_processed.json` | `reddit_raw_posts.json` |
| **4** | `youtube_preprocessing` | Clean text, demojize, detect language | `youtube_videos_processed.json` | `youtube_videos_raw.json` |
| **5** | `sentiment` | RoBERTa sentiment polarity scoring | `sentiment_analysis_results.json` | Preprocessed Reddit and/or YouTube |
| **6** | `topics` | BERTopic thematic cluster extraction | `topic_modeling_results.json` | Preprocessed Reddit and/or YouTube |
| **7** | `graph` | Entity interaction graph & PageRank | `graph_analysis_results.json` | Preprocessed Reddit and/or YouTube |
| **8** | `neo4j_persistence` | Cypher transactional graph insertion | Neo4j Graph Database | `graph_analysis_results.json` |
| **9** | `misinformation` | Stylometric & ML risk assessment | `misinformation_analysis_results.json` | Preprocessed Reddit and/or YouTube |
| **10** | `consolidation` | Schema-v2 multi-source record join | `consolidated_intelligence.json` | Preprocessed data + Stages 5, 6, 7, 9 |
| **11** | `elasticsearch_indexing` | Bulk document transformation & indexing | Elasticsearch Cluster | `consolidated_intelligence.json` |
| **12** | `kibana_validation` | Offline schema & reference bundle validation | 41 Saved Objects Verified | `social_intelligence_dashboards.ndjson` |

---

## 3. Installation & Runtime Prerequisites

### 3.1 Python Runtime Requirements

To ensure clear operational expectations, the repository distinguishes the environments actually tested from broader compatibility claims:

* **Core Execution & Test Environment (Actually Tested)**:
  * **Python 3.14.7** (Windows 10 x64) is the active project environment where all non-JVM core pipeline modules (Stages 1, 2, 5–13) and CLI orchestration run. The Stage 13 focused regression suites passed 904 tests on Python 3.14.7. This result does not establish that the entire repository test suite has been executed.
* **Spark NLP Preprocessing Environment (Actually Tested)**:
  * **Python 3.12.10** (in an isolated virtual environment, e.g., `venv-spark`) is used exclusively for PySpark and Spark NLP preprocessing (Stages 3 & 4). Spark preprocessing is run in this separate Python 3.12.10 environment.
* **Declared Dependency Compatibility Range**:
  * The packages in `requirements.txt` declare standard Python package compatibility (e.g., Python >= 3.10), but **only Python 3.14.7 (core) and Python 3.12.10 (Spark) have been physically tested and validated** in this workspace. Broader version matrix compatibility has not been verified via CI.

### 3.2 Java Runtime Requirements (For Spark NLP Preprocessing)
* Required exclusively when executing PySpark NLP preprocessing: **Microsoft OpenJDK 17** configured on `JAVA_HOME` and `PATH`.
* On Windows, Hadoop native binaries (`winutils.exe` and `hadoop.dll`) under `%HADOOP_HOME%\bin` are required for POSIX permission simulation.
* Core non-JVM stages (Stages 1, 2, 5–13) do not require Java or Spark.

### 3.3 Dependency Installation

#### Base Installation (Core NLP, Analytics, and Offline Testing)
Installs all dependencies required for core pipeline execution, machine learning inference, graph modeling, and complete offline test suites:
```bash
pip install -r requirements.txt
```
*(Includes `torch`, `transformers`, `bertopic`, `networkx`, `neo4j`, `scikit-learn`, `joblib`, `python-dotenv`, `emoji`, `pytest`)*

#### Optional Live Cluster Integration (Elasticsearch Only)
If connecting to an active, running Elasticsearch cluster for live bulk indexing (`--index-elasticsearch`), install the extended dependency manifest:
```bash
pip install -r requirements-live.txt
```
*(Extends base requirements with `elasticsearch>=8.0.0,<9.0.0`. Note: This optional file covers Elasticsearch only. Neo4j is already covered in base requirements, and Kibana utilizes standard library HTTP transport with zero additional packages).*

---

## 4. Environment Configuration & Security

The platform utilizes a secure, zero-hardcoded-credentials configuration pattern via `config/settings.py` and `.env`.

### 4.1 Setup `.env`
Copy the provided template and supply real credentials as needed:
```bash
# Linux/macOS
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

### 4.2 Configuration Parameters

| Environment Variable | Default Value | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Environment mode (`development`, `testing`, `production`) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `REDDIT_CLIENT_ID` | Empty | Reddit OAuth2 script app client ID |
| `REDDIT_CLIENT_SECRET` | Empty | Reddit OAuth2 client secret |
| `REDDIT_USER_AGENT` | `social-media-intelligence:v1.0` | Reddit compliance user agent string |
| `YOUTUBE_API_KEY` | Empty | Google Cloud YouTube Data API v3 key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j authentication user |
| `NEO4J_PASSWORD` | Empty | Neo4j authentication password |
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | Elasticsearch REST endpoint |
| `ELASTICSEARCH_USERNAME` | `elastic` | Elasticsearch basic authentication user |
| `ELASTICSEARCH_PASSWORD` | Empty | Elasticsearch basic authentication password |
| `KIBANA_HOST` | `http://localhost:5601` | Kibana endpoint for saved objects import |

### 4.3 Security & Secret Masking Guarantee
* **No Plaintext Logging**: Credentials (`reddit_client_secret`, `youtube_api_key`, `neo4j_password`, `elasticsearch_password`) are declared with `field(repr=False)` in `AppConfig`. String representations and application logs never output plaintext secrets.
* **Safe Summary Helper**: `AppConfig.get_safe_summary()` returns boolean readiness flags (`reddit_configured`, `elasticsearch_configured`) rather than raw tokens.
* **Git Protection**: `.gitignore` strictly blocks `.env`, `.env.*`, `*.lock`, `data/runs/`, and local dataset files from version control.

---

## 5. Mock vs. Live Ingestion Behavior

The ingestion modules automatically distinguish between mock development environments and authenticated live API calls:

1. **Reddit Ingestion (`src/ingestion/reddit_ingestion.py`)**:
   * If `REDDIT_CLIENT_ID` or `REDDIT_CLIENT_SECRET` is unset or contains `your_...` placeholders, the pipeline falls back to loading synthetic mock data from `data/sample_reddit_posts.json`.
   * When valid credentials are supplied, it requests OAuth2 bearer tokens via standard library HTTPS calls (`https://www.reddit.com/api/v1/access_token`), fetching live posts from configured subreddits.

2. **YouTube Ingestion (`src/ingestion/youtube_ingestion.py`)**:
   * If `YOUTUBE_API_KEY` is missing or placeholder, mock video records are generated for local offline testing.
   * When a key is supplied, it issues `search.list` and `videos.list` requests against `https://www.googleapis.com/youtube/v3/`, tracking daily quotas and rate limits defensibly.

---

## 6. Command-Line Interface (CLI) Usage

The application entry point is `app.py`. All operations are invoked via command-line arguments.

### 6.1 Unified 12-Stage Pipeline Execution (`--run-all`)

To execute the complete 12-stage intelligence pipeline:
```bash
python app.py --run-all
```

#### Skip Flags (Selective Platform Degradation)
Skip flags allow skipping upstream ingestion/preprocessing or downstream external targets during a unified run:
```bash
# Skip YouTube ingestion and preprocessing (runs in Reddit-only single-platform mode)
python app.py --run-all --skip-youtube

# Skip Reddit ingestion and preprocessing (runs in YouTube-only single-platform mode)
python app.py --run-all --skip-reddit

# Skip Neo4j database persistence
python app.py --run-all --skip-neo4j

# Skip Kibana dashboard validation
python app.py --run-all --skip-kibana

# Combine multiple skip flags
python app.py --run-all --skip-youtube --skip-neo4j
```

#### Custom Execution Manifest Path
By default, the manifest is saved to `data/runs/<run_id>/manifest.json` and mirrored to `data/processed/execution_manifest.json`. You can provide a custom destination:
```bash
python app.py --run-all --manifest-path data/runs/my_custom_manifest.json
```

### 6.2 Standalone Execution Commands (Legacy Actions)

Individual components can be invoked directly for focused testing or manual batch processing:
```bash
# Ingest mock Reddit dataset into data/raw/
python app.py --ingest-reddit
python app.py --ingest                 # Alias for --ingest-reddit

# Ingest live Reddit posts via OAuth2 API
python app.py --ingest-reddit-live

# Ingest live YouTube videos via Data API v3
python app.py --ingest-youtube

# Preprocess ingested Reddit posts
python app.py --preprocess-reddit

# Preprocess ingested YouTube videos
python app.py --preprocess-youtube

# Bulk index consolidated intelligence into Elasticsearch
python app.py --index-elasticsearch

# Validate Kibana dashboard export bundle offline
python app.py --validate-kibana
```

### 6.3 Argument Parser Rules & Mutual Exclusion
The CLI parser (`parse_cli_arguments`) enforces strict validation contracts:
1. **Rule 1 (Unified vs. Legacy Exclusivity)**: `--run-all` **cannot** be combined with any legacy primary action (`--ingest-reddit`, `--preprocess-reddit`, `--index-elasticsearch`, `--validate-kibana`).
2. **Rule 2 (Legacy Mutual Exclusivity)**: Legacy primary actions cannot be combined with each other in a single command.
3. **Rule 3 (Scope-Restricted Flags)**: `--skip-reddit`, `--skip-youtube`, `--skip-neo4j`, `--skip-kibana`, and `--manifest-path` **require** `--run-all`. Supplying them with legacy actions or standalone results in a parser error (exit code 2).

---

## 7. Execution Isolation, Locking & Canonical Promotion

### 7.1 Run-Isolated Directories (`data/runs/<run_id>/`)
Every orchestrated execution generates a cryptographically unique run identifier (`run_YYYYMMDD_HHMMSS_<8hex>`) and an isolated directory:
```text
data/runs/run_20261112_143000_1a2b3c4d/
├── manifest.json
├── reddit_raw_posts.json
├── youtube_videos_raw.json
├── reddit_posts_processed.json
├── youtube_videos_processed.json
├── sentiment_analysis_results.json
├── topic_modeling_results.json
├── graph_analysis_results.json
├── misinformation_analysis_results.json
└── consolidated_intelligence.json
```
Downstream stages resolve inputs **strictly from the active run directory**. If an upstream stage fails or is skipped, downstream stages will never fall back to stale files in `data/processed/`.

### 7.2 OS-Backed Advisory Concurrency Locking
To prevent concurrent pipeline executions from corrupting shared local storage, `OrchestrationLock` acquires a kernel-level advisory lock on `data/.orchestration.lock`:
* **Windows**: Uses `msvcrt.locking(LK_NBLCK)` at byte offset `1_000_000`.
* **POSIX**: Uses `fcntl.flock(LOCK_EX | LOCK_NB)`.
* **Dead-Process Recovery**: If a prior process died abruptly, the OS kernel automatically releases file locks. The lock manager safely reclaims stale locks without Time-of-Check to Time-of-Use (TOCTOU) file-unlinking races.

### 7.3 Exception-Safe Canonical Promotion
Upon successful pipeline completion (or graceful single-platform partial completion), run artifacts are promoted to canonical storage (`data/processed/`):
* **Staging & Pre-Validation**: Files are pre-validated before promotion begins.
* **Automated Rollback Ledger**: If an unexpected Python exception or filesystem error occurs mid-promotion, the rollback ledger restores previously existing canonical files and cleans up staging files.
* **Crash-Safety Boundary**: The promotion replaces files sequentially on disk. While fully exception-safe within Python execution, it is not POSIX crash-safe against sudden OS power failures or `SIGKILL`.

---

## 8. Kibana Dashboard Import & Offline Validation

The platform includes an export bundle containing 41 Kibana saved objects:
* **Bundle Location**: `config/kibana/social_intelligence_dashboards.ndjson`
* **Object Inventory**: 1 Dashboard, 12 Visualizations (Lens & TSVB), 1 Search view, 2 Index Patterns, and 25 Tag/Formatting objects.

### 8.1 Offline Validation (Stage 12)
Validates bundle schema, NDJSON structure, index-pattern references, and Elasticsearch field type conformance without contacting a live Kibana server:
```bash
python app.py --validate-kibana
```

### 8.2 Live Dashboard Import Tool
An optional HTTP utility (`src/search/kibana_dashboard_importer.py`) deploys the bundle directly into Kibana 8.x via the Saved Objects Import API:
```bash
python -m src.search.kibana_dashboard_importer --host http://localhost:5601
```
*(Requires active Kibana service and proper network connectivity)*

---

## 9. Analytical Integrity & Responsible Disclosures

The platform's analytical modules implement specific methodological boundaries that must be accurately understood:

### 9.1 Misinformation Risk Assessment (Stage 9)
* **Heuristic Risk Scoring**: The classifier combines stylometric features (punctuation density, lexical diversity, capitalization patterns) with a model trained on the LIAR benchmark.
* **Non-Factual Verification**: The system produces **heuristic-assisted misinformation risk assessments**, NOT ground truth. It does not perform factual claim verification, web scraping for fact-checks, or real-world truth determination.
* **Non-Calibrated Scores**: Output credibility scores are normalized indices in $[0.0, 1.0]$. They are **not calibrated probabilities**.
* **Development Fixtures**: In the absence of an external pretrained model, an offline 24-sample fixture is used for structural verification; it is not representative of real-world production accuracy.

### 9.2 Interaction Graph Analytics (Stage 7 & 8)
* **Structural Metrics**: Influence rankings (PageRank, betweenness centrality, degree centrality) measure structural graph properties within the observed interactions dataset.
* **Non-Causal Representation**: Graph metrics reflect network centrality within collected sample posts and comments; they do not establish proven real-world persuasion, authority, or causal influence.

### 9.3 Multilingual Sentiment Polarity (Stage 5)
* **Model**: CardiffNLP RoBERTa (`twitter-roberta-base-sentiment-latest`).
* **Non-Destructive Retention**: Posts containing empty text, emoji-only strings, or unsupported languages are classified as `label: "UNDETERMINED"`, `score: 0.0`, `status: "empty_text"`. The pipeline preserves records in output datasets rather than silently deleting non-English or unparseable data.

### 9.4 Topic Modeling (Stage 6)
* **Outlier Transparency**: BERTopic unassigned records are deterministically tagged with `topic_id: -1` and `topic_status: "outlier"`.
* **Type Safety**: Float, boolean, or corrupted topic IDs map deterministically to `topic_status: "error"`, maintaining strict Schema-v2 record integrity without dropping rows.

---

## 10. Verification & Test Execution

### 10.1 Running Focused Regression Suites
Core pipeline contracts, manifest calculations, stage cascades, and advisory locking mechanics are verified offline via `pytest`. The Stage 13 focused regression suites passed 904 tests on Python 3.14.7 across three targeted test files (`tests/test_orchestration_runner.py`, `tests/test_app.py`, `tests/test_topic_pipeline.py`). This result does not establish that the entire repository test suite has been executed:

```bash
# Run the three Stage 13 targeted regression test suites (904 tests)
python -m pytest tests/test_orchestration_runner.py tests/test_app.py tests/test_topic_pipeline.py -q

# Run consolidation and search pipeline integration tests
python -m pytest tests/test_data_consolidator.py tests/test_indexing_pipeline.py -q

# Run misinformation detector and schema tests
python -m pytest tests/test_misinformation_pipeline.py tests/test_misinformation_validator.py -q
```

### 10.2 Test Verification Scope
* **Offline Contract Verification**: All unit and integration test suites operate strictly offline using mock data, local temporary files, and injected client stubs.
* **Live Service Notice**: Running the test suites validates local software contracts; it does **not** test network connectivity or authentication against live external Elasticsearch, Neo4j, or Kibana services.

---

## 11. Known Operational Boundaries & Deployment Prerequisites

The platform is designed as an advanced data engineering reference implementation. Before considering production deployment, the following operational boundaries must be addressed:

1. **Live External Services Unverified**: End-to-end integration against live, external clusters (Elasticsearch 8.x, Neo4j 5.x, Kibana 8.x) has not been validated in a live staging/production environment.
2. **Missing Live Dependencies**: Live bulk indexing requires installing the optional `requirements-live.txt` package (`elasticsearch>=8.0.0`).
3. **Local vs. Distributed Locking**: `OrchestrationLock` provides kernel-level advisory locking for a single operating system instance. Distributed multi-server worker deployments require a distributed lock manager (e.g., Redis, ZooKeeper, or etcd).
4. **Shared Database State**: Isolated run directories (`data/runs/<run_id>/`) isolate filesystem artifacts, but Stage 8 and Stage 11 write to shared external database endpoints. Multi-worker deployments require database-level namespace isolation.
5. **API Quotas & Rate Limits**: Live Reddit and YouTube ingestion are subject to third-party rate limits, developer credential approvals, daily search quota caps, and network availability.
6. **Non-Crash-Safe Promotion**: Multi-file canonical promotion is exception-safe with an automated rollback ledger within Python execution, but cannot guarantee crash-safety against sudden OS process termination (`SIGKILL`, power failure).
7. **Production Status**: The platform is **not production-ready** without operational infrastructure deployment, monitoring, and live integration testing.

---

## 12. Repository Hygiene & Licensing

* **Clean Repository State**: Runtime output directories (`data/runs/`) and lock files (`*.lock`, `data/.orchestration.lock`) are excluded from version control via `.gitignore`.
* **Zero Hardcoded Secrets**: Scanned and verified to contain no exposed credentials, API keys, or private tokens.
* **License**: Private Development / Academic Reference.
