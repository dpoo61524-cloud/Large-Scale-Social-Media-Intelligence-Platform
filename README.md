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
├── src/                    # Source code package (for future pipeline blocks)
│   └── __init__.py
│
├── data/                   # Data storage (git-ignored, preserved with .gitkeep)
│   ├── raw/                # Unprocessed JSON payloads from Reddit/YouTube
│   └── processed/          # Cleaned, tokenized, and enriched datasets
│
├── models/                 # Model weights and pipeline binaries (git-ignored)
│
├── tests/                  # Unit and validation tests
│   ├── __init__.py
│   ├── test_app.py         # Application entry point tests
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

## 7. Security Best Practices Followed

- **Zero Hardcoded Secrets**: All future API keys, tokens, and database passwords are read from the environment.
- **Secret Masking**: `config/settings.py` includes a safe summary utility that never exposes raw credentials in application logs or standard output.
- **Git Protection**: `.gitignore` is pre-configured to block virtual environments, `.env` files, large dataset files (`data/raw/`, `data/processed/`), and trained model artifacts (`models/`).
