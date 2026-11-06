# Large Scale Social Media Intelligence Platform
*(Repository: [Large-Scale-Social-Media-Intelligence-Platform](https://github.com/dpoo61524-cloud/Large-Scale-Social-Media-Intelligence-Platform))*

An end-to-end data science and engineering platform designed to ingest, clean, analyze, and visualize high-volume social media data from **Reddit** and **YouTube**.

---

## 1. Project Introduction

Social media platforms generate millions of user interactions every hour. Extracting actionable intelligence from this unstructured data requires a reliable pipeline capable of handling data ingestion, natural language processing, network analysis, and real-time visualization.

This project is built incrementally following professional software engineering and data science practices. **Block 1** establishes the project foundation: environment setup, modular directory architecture, secure configuration handling, test automation, and documentation.

---

## 2. Project Objectives

The project strictly follows the 7-step roadmap specified for the platform:

1. **Data Ingestion**: Collect live posts and comments via the Reddit and YouTube APIs (respecting rate limits and terms of service) and store raw JSON data.
2. **Text Cleaning & Preprocessing**: Clean noisy text, normalize emojis, strip URLs/unwanted characters, and detect languages.
3. **Multilingual Sentiment Analysis**: Analyze sentiment polarity with transformer models and track how sentiment evolves over time.
4. **Topic Modeling**: Discover hidden themes and track the rise and fall of topics using BERTopic.
5. **Graph & Network Analysis**: Model user interactions in Neo4j, rank key influencers using PageRank, and detect user communities using NetworkX.
6. **Misinformation Classification**: Train and evaluate a machine learning classifier on labeled fake-news datasets (e.g., LIAR).
7. **Search & Visualization**: Index processed data in Elasticsearch and construct interactive Kibana dashboards for search, trend monitoring, and automated alerts.

---

## 3. Technology Overview

The planned stack is aligned directly with the project roadmap:

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Data Ingestion** | Reddit API, YouTube Data API v3 | Collect public posts, comments, and metadata |
| **NLP & Preprocessing** | Python, Spark NLP | Scalable text cleaning, tokenization, language detection |
| **Topic Modeling** | BERTopic | Dynamic topic extraction and temporal trend analysis |
| **Graph Analytics** | Neo4j, NetworkX | Interaction graph modeling, PageRank, community detection |
| **Misinformation ML** | Scikit-learn / Transformers, LIAR Dataset | Fake news and claim verification classification |
| **Search & Dashboards**| Elasticsearch, Kibana | Full-text indexing, aggregation, and visual monitoring |
| **Foundation (Block 1)**| `python-dotenv`, `pytest` | Environment management, secure config, and testing |

---

## 4. Project Structure

```text
Large-Scale Social Media Intelligence Platform/
│
├── app.py                  # Main entry point and status inspection
├── requirements.txt        # Python dependency manifest
├── .env.example            # Template for environment variables and API credentials
├── .gitignore              # Excludes virtual environments, datasets, and secrets
├── README.md               # Beginner-friendly project documentation
│
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Secure environment variable reader with secret masking
│
├── src/                    # Source code package
│   ├── __init__.py
│   └── ingestion/          # Data ingestion pipelines (Step 1)
│       ├── __init__.py
│       ├── reddit_ingestion.py  # Reddit mock & OAuth2 live ingestion
│       └── youtube_ingestion.py # YouTube Data API v3 live ingestion
│
├── data/                   # Data storage
│   ├── sample_reddit_posts.json # Synthetic Reddit posts fixture
│   ├── raw/                # Unprocessed JSON payloads (git-ignored)
│   └── processed/          # Cleaned, tokenized, and enriched datasets (git-ignored)
│
├── models/                 # Model weights and pipeline binaries (git-ignored)
│
├── tests/                  # Unit and validation tests
│   ├── __init__.py
│   ├── test_app.py         # Application entry point tests
│   ├── test_reddit_ingestion.py  # Reddit schema & API tests
│   ├── test_youtube_ingestion.py # YouTube API & quota tests
│   └── test_settings.py    # Configuration and security masking tests
│
└── docs/                   # Architecture and technical documentation
    └── architecture.md     # System architecture and component roadmap
```

---

## 5. Installation Instructions

Follow these step-by-step instructions to set up the project on your local machine:

### Step 1: Clone or Navigate to the Project
Open your terminal (PowerShell on Windows, or bash on macOS/Linux):
```bash
cd "Large-Scale Social Media Intelligence Platform"
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment ensures dependencies remain isolated:
```bash
# Windows
python -m venv venv

# macOS / Linux
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment
```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### Step 4: Install Dependencies
Install the required foundational packages:
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (CMD)
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
> **Security Reminder**: Never put real credentials inside `.env.example`, and never commit `.env` to Git.

---

## 6. How to Run the Application

Execute the application entry point using Python:

```bash
python app.py
```

### Running Validation Tests
Run the automated test suite with pytest:

```bash
pytest
```

---

## 7. Data Ingestion (Company Step 1: Reddit & YouTube APIs)

The data ingestion package (`src/ingestion/`) implements raw data collection for public Reddit discussions and YouTube video metadata without applying text cleaning, emoji modification, or language detection (those operations belong to Step 2).

### 7.1 Data Sources & Storage

| Source | Collection Mechanism | Output File | Git Status |
| :--- | :--- | :--- | :--- |
| **Reddit (Mock)** | Synthetic JSON fixture (`data/sample_reddit_posts.json`) | `data/raw/reddit_posts_raw.json` | Excluded by `.gitignore` |
| **Reddit (Live API)** | OAuth2 Client-Credentials grant via `oauth.reddit.com` | `data/raw/reddit_posts_raw.json` | Excluded by `.gitignore` |
| **YouTube (Live API)** | YouTube Data API v3 (`search` and `videos` endpoints) | `data/raw/youtube_videos_raw.json` | Excluded by `.gitignore` |

> **Non-destructive Storage**: Both raw storage functions (`save_raw_posts` and `save_raw_youtube_videos`) merge new records by their primary identifiers (`post_id` and `video_id`) to prevent accidental data loss across multiple ingestion sessions.

---

### 7.2 API Credentials & Approval Requirements

#### A. Reddit API Access & Approval
* **Important Developer Policy Notice**: Navigating to `reddit.com/prefs/apps` is a legacy interface and is **not** a guaranteed self-serve method for new automated API access. Under Reddit's updated Data API Developer Terms, external API access requires developer registration and formal use-case approval through the **Reddit Developer Platform** ([https://support.reddithelp.com/](https://support.reddithelp.com/) / [https://www.reddit.com/dev/api](https://www.reddit.com/dev/api)).
* **Mock Testing vs. Live API Verification**:
  * **Mock Mode (Recommended for Learning & Development)**: `python app.py --ingest-reddit` requires **no API keys, no approvals, and no network requests**. It runs 100% offline using `data/sample_reddit_posts.json` to allow full platform development without external dependencies.
  * **Live API Mode (Optional / Advanced)**: Requires verified OAuth credentials from an approved Reddit Developer application:
    1. Submit and receive approval for your application under Reddit's Developer Terms.
    2. Obtain your approved **Client ID** and **Client Secret**.
    3. Configure a compliant User-Agent in `.env` adhering strictly to Reddit's specification:
       `<platform>:<app_id>:<version> (by /u/<your_approved_username>)`
    4. Set in `.env`:
       ```env
       REDDIT_CLIENT_ID=your_approved_client_id
       REDDIT_CLIENT_SECRET=your_approved_client_secret
       REDDIT_USER_AGENT=social-media-intelligence:v1.0 (by /u/your_username)
       ```
    5. If approval or credentials are not yet active, live collection remains safely disabled to protect against unauthorized access violations.

#### B. YouTube Data API v3 Setup
1. Create a project on the [Google Cloud Console](https://console.cloud.google.com/) and enable the **YouTube Data API v3**.
2. Create an API Key in Credentials and optionally restrict it to YouTube Data API v3.
3. Configure in `.env`:
   ```env
   YOUTUBE_API_KEY=your_actual_youtube_api_key
   ```
4. Note that search requests cost **1 quota unit** each against a separate default limit of 100 search calls per day. Video details consume **1 quota unit** each from the separate 10,000 units/day other-methods allocation. Pagination stops automatically if either limit or an API quota error is reached.

---

### 7.3 YouTube Data API v3 Quota Accounting & Cost Model

YouTube Data API v3 enforces quota usage based on **quota units** with distinct limits for search and other methods ([Google Developers Documentation](https://developers.google.com/youtube/data/determine_quota_cost)):

| API Operation | Endpoint | Quota Cost | Quota Bucket / Allocation | Ingestion Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Search List** | `/youtube/v3/search` | **1 unit** per call | **Separate limit**: 100 calls/day | Discovers video IDs matching the query (each page = 1 call) |
| **Videos List** | `/youtube/v3/videos` | **1 unit** per call | **Other methods**: 10,000 units/day | Fetches complete snippet and statistics (batches up to 50 IDs) |

#### Accurate Quota Rules & Independent Tracking
1. **`search.list` Cost**: Costs **1 quota unit** per request.
2. **Independent Search Limit**: `search.list` operates under a separate default allocation of **100 requests per day**.
3. **Per-Page Consumption**: Every additional search page requested via `pageToken` consumes another search request from that separate 100-request daily limit.
4. **`videos.list` Cost**: Costs **1 quota unit** per call from the **other daily quota allocation** (default: **10,000 units per day**).
5. **Video Batching**: Up to 50 video IDs are batched per `videos.list` request, consuming exactly 1 unit per 50-item batch.
6. **Compound Session Accounting**:
   - 1 search page + 1 video batch: $1 \text{ search request (1 unit)} + 1 \text{ video batch (1 unit)} = \mathbf{2 \text{ total units}}$.
   - 3 search pages + 3 video batches: $3 \text{ search requests} + 3 \text{ video batches} = \mathbf{6 \text{ total units}}$.
   - 100 search pages: Hits the 100-search-request daily limit and pagination halts safely.
7. **Local Estimation vs. Google Cloud Console**:
   - The platform tracks estimated local usage against standard defaults (100 search requests/day, 10,000 other-methods units/day).
   - The YouTube API does not return remaining project quota in HTTP response headers; actual project-wide quota consumption is managed externally in the Google Cloud Console. The pipeline never falsely claims to know external remaining quota.
8. **Automated Limit Protection**: Pagination stops immediately when either the search request limit (100 calls) or other-methods quota limit (10,000 units) is reached, or when an HTTP 403 `quotaExceeded` error is returned.

---

### 7.4 Reddit Data Retention & Deletion Compliance

Under Reddit's Developer Terms and Data API policies ([https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)), developers who store Reddit content must respect user privacy, retention limits, and deletion requests:

1. **No Assumption of 90-Day Retention**:
   - Developers must **not** assume a 90-day retention period is compliant.
   - Reddit Data API terms recommend that stored data be **routinely deleted within 48 hours** unless active deletion reconciliation is performed or explicit authorized agreements exist.
2. **48-Hour Routine Retention Policy**:
   - `apply_reddit_retention_policy(max_age_hours=48)`: Purges records whose `collected_at` timestamp exceeds the retention window (default: 48 hours).
   - Purged records are permanently removed from storage to ensure deleted/expired data cannot be retained or reused.
3. **Collection Timestamp Tracking (`collected_at`)**:
   - Every collected or saved Reddit post is stamped with an ISO 8601 UTC timestamp (`collected_at`).
   - The original raw text, title, and metadata of the post remain completely uncleaned and unmodified during ingestion (text normalization belongs to Step 2).
4. **Author-Identifying Information Anonymization**:
   - `anonymize_reddit_author(author_to_anonymize)`: Replaces the `author` field with `"[deleted]"` across all stored records matching the requested user, honoring user account deletions and privacy requests while preserving text for aggregate sentiment analysis.
5. **Targeted Post Purging**:
   - `purge_reddit_posts(post_ids_to_purge)`: Permanently deletes specific post records from the raw JSON file when requested.
6. **Live Deletion Reconciliation**:
   - `reconcile_reddit_deletions(access_token, user_agent)`: Reconciles local post IDs against Reddit's live API (`/by_id/{fullnames}`). Records identified as deleted or removed are automatically purged.
   - **Crucial Compliance Notice**: Stored Reddit records **cannot** automatically know when posts or accounts are deleted without actively querying Reddit's authorized API or receiving deletion notices.
7. **Strict Prevention of Reuse**:
   - Posts confirmed deleted or expired are deleted from disk storage immediately and cannot be retained, merged back, or reused.

---

### 7.5 YouTube vs. Reddit Retention Requirements Distinction

YouTube and Reddit impose fundamentally distinct legal and operational data retention rules:

| Dimension | YouTube Data API v3 | Reddit Data API |
| :--- | :--- | :--- |
| **Governing Policy** | YouTube Developer Terms of Service (Section III.E) | Reddit Developer Terms & Data API Wiki |
| **Core Retention Rule** | **30-Day Cache Limit**: API data must not be stored/cached for longer than 30 calendar days without refreshing from the API. | **Routine 48-Hour Deletion & User Deletion Compliance**: Routine deletion within 48 hours recommended; deleted posts must be purged. |
| **Expired Data Reuse** | Records older than 30 days are purged during retention runs and filtered out during merges so they are never reused. | Records older than 48 hours or identified as deleted are permanently removed and cannot be reused. |
| **Automated Detection** | Time-based expiration (`collected_at > 30 days`). | Time-based 48-hour routine deletion, plus active API checks (`/by_id/`) for user deletions. |
| **Implementation** | `apply_youtube_retention_policy(max_cache_days=30)` | `apply_reddit_retention_policy(max_age_hours=48)`, `reconcile_reddit_deletions()`, `purge_reddit_posts()`, `anonymize_reddit_author()` |

---

### 7.6 Rate Limits & Quota Handling Summary

* **Reddit API**:
  * Enforces rate limits per OAuth client. Live requests inspect response headers (`X-Ratelimit-Remaining`, `X-Ratelimit-Reset`, `X-Ratelimit-Used`).
  * If a `429 Too Many Requests` error occurs or remaining quota reaches zero, ingestion halts immediately to protect developer standing.
* **YouTube Data API v3**:
  * Tracks 1-unit search costs against the 100 search requests/day limit.
  * Tracks 1-unit video detail batch costs against the 10,000 units/day other-methods allocation.
  * Stops pagination automatically when limits are reached or on HTTP `403 quotaExceeded`.
* **Safe Fallback**: If credentials are missing or left as template placeholders (`your_*`), live ingestion remains safely disabled and reports what configuration is required.

---

### 7.7 How to Run Ingestion

Execute any ingestion pipeline through the main entry point:

```bash
# 1. Run offline mock Reddit ingestion (10 synthetic posts)
python app.py --ingest-reddit

# 2. Run live Reddit API collection (requires credentials in .env)
python app.py --ingest-reddit-live

# 3. Run live YouTube video collection (requires YOUTUBE_API_KEY in .env)
python app.py --ingest-youtube
```

Or execute modules directly:
```bash
python -m src.ingestion.reddit_ingestion
python -m src.ingestion.youtube_ingestion
```

---

## 8. Text Preprocessing & Cleaning (Company Step 2: Spark NLP)

The text preprocessing package (`src/preprocessing/`) implements scalable, modular text cleaning, emoji conversion, and Spark NLP language identification for Reddit posts and YouTube videos.

### 8.1 Preprocessing Architecture & Components

```text
src/preprocessing/
├── __init__.py
├── spark_session.py         # Reusable Spark NLP session & prerequisite validator
├── text_cleaner.py          # HTML, URL, whitespace & noise cleaner (preserves case, hashtags, numbers)
├── emoji_handler.py         # Emoji & emoticon converter (😍🔥 -> smiling_face_with_heart_eyes fire)
├── language_detector.py     # Spark NLP LanguageDetectorDL integration & short-text handling
└── preprocessing_pipeline.py# Batched dataset pipeline reading data/raw/ and saving to data/processed/
```

| Component | Module | Key Operations |
| :--- | :--- | :--- |
| **Spark NLP Session** | `spark_session.py` | Manages singleton local SparkSession; handles Java 17 and Python <= 3.12 compatibility validation; provides clean shutdown. |
| **Text Cleaner** | `text_cleaner.py` | Strips HTML tags, unescapes entities, removes URLs, scrubs invisible noise/control chars, normalizes elongated character runs (`sooooo` -> `soo`). Preserves numbers, technical terms (`GPT-4`), hashtags (`#NLP`), mentions (`@dev`), and case. |
| **Emoji Handler** | `emoji_handler.py` | Demojizes Unicode emojis and ASCII emoticons (`:)`, `<3`) into semantic tokens (`smiling_face_with_heart_eyes`, `red_heart`) with whitespace normalization. |
| **Language Detector** | `language_detector.py` | Integrates Spark NLP `LanguageDetectorDL`; handles empty text, short-text uncertainty (<15 chars), and unsupported languages without false assumptions. |
| **Batched Pipeline** | `preprocessing_pipeline.py` | Processes records in configurable batches (default 100); preserves raw JSON files; appends enrichment fields (`title_cleaned`, `text_cleaned`/`description_cleaned`, `detected_language`, `language_code`, `processed_at`). |

---

### 8.2 Environment & Runtime Prerequisites (Python & Java)

Spark NLP and PySpark impose strict runtime constraints:
* **Python Runtime**: Validated for **Python 3.8 through 3.12**. Python 3.14 does not currently have binary wheels for PySpark / Py4J / gRPC.
* **Java Runtime**: Requires **Java 17** (or Java 8/11) on `PATH` or `JAVA_HOME` (e.g. `Microsoft OpenJDK 17`).
* **Windows Native Binaries**: On Windows, Hadoop requires `winutils.exe` and `hadoop.dll` under `%HADOOP_HOME%\bin` to perform POSIX permissions on local temp files.
* **Conservative Memory Tuning for Resource-Constrained Environments**:
  On machines with <= 4 GB total RAM (~500-900 MB free), standard Spark defaults (`memory="4G"` or `16G`) cause JVM object heap reservation failures. The pipeline configures Spark defensively:
  - Driver memory default: `1024M` (configurable via `SPARK_DRIVER_MEMORY` environment variable).
  - Cores: `spark.driver.cores=1` (`master("local[1]")`) to avoid multi-threaded NetBIOS heartbeat contention.
  - Driver host and bind address: `127.0.0.1` (`localhost`).
  - Max result size: `512M`.
* **Isolated Environment Setup**:
  To run Spark NLP with distributed acceleration, set up an isolated Python 3.12 environment:
  ```powershell
  # 1. Install OpenJDK 17 & Python 3.12 (Windows)
  winget install Microsoft.OpenJDK.17
  winget install Python.Python.3.12

  # 2. Create and configure isolated Python 3.12 environment
  py -3.12 -m venv venv-spark

  # 3. Install pinned Spark NLP dependencies
  .\venv-spark\Scripts\pip install -r requirements-sparknlp.txt
  ```
* **Graceful Degradation**: If Java or Python <= 3.12 is missing in the primary execution environment (such as Python 3.14), the pipeline catches prerequisite failures cleanly, reports them transparently in summaries, and skips JVM startup without false claims.

---

### 8.3 Processed Storage & Non-Destructive Flow

Raw JSON files (`data/raw/`) are treated as **read-only** and are never modified or overwritten by preprocessing. Output datasets are saved to:
* `data/processed/reddit_posts_processed.json`
* `data/processed/youtube_videos_processed.json`

Both files are excluded from Git via `.gitignore`.

---

### 8.4 How to Run Preprocessing & Integration Tests

Execute preprocessing and verification in the isolated Spark environment:

```powershell
# Preprocess ingested Reddit posts using Spark NLP LanguageDetectorDL
.\venv-spark\Scripts\python.exe app.py --preprocess-reddit

# Preprocess ingested YouTube videos using Spark NLP LanguageDetectorDL
.\venv-spark\Scripts\python.exe app.py --preprocess-youtube

# Run the real Spark NLP integration test suite (executes actual pretrained LanguageDetectorDL model)
.\venv-spark\Scripts\python.exe -m pytest tests/test_sparknlp_integration.py -v

# Run the full project unit test suite (93 unit tests pass across both environments)
.\venv\Scripts\python.exe -m pytest -v
```

---

## 9. Security Best Practices Followed

- **Zero Hardcoded Secrets**: All future API keys, tokens, and database passwords are read from the environment.
- **Secret Masking**: `config/settings.py` includes a safe summary utility that never exposes raw credentials in application logs or standard output.
- **Git Protection**: `.gitignore` is pre-configured to block virtual environments, `.env` files, large dataset files (`data/raw/`, `data/processed/`), and trained model artifacts (`models/`).
