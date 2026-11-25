# JobLens 🔍

> **Personalised Job & Placement Search Engine** — custom inverted indexing, BM25F field-weighted ranking, and candidate profile alignment built from scratch.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)

## Overview

JobLens is an Information Retrieval (IR) and Big Data Analytics engine for job seekers and university placement preparation. It implements custom inverted indexing and field-weighted BM25F ranking entirely from scratch — no Elasticsearch, no external search library.

## Features

- **Custom Inverted Index** — built from raw job postings; multi-field (title, description, company, location) lookup with per-field IDF weighting
- **BM25F Ranking** — field-weighted variant of BM25; tunable k1 and per-field boost parameters
- **Candidate Profile Alignment** — matches user skills/experience against job postings; outputs a ranked candidate-fit score
- **Keyword Baseline** — TF-based engine for comparison against BM25F
- **NLP Preprocessing** — NLTK stopword removal, stemming, tokenization
- **Streamlit UI** — search bar, ranked results, relevance metrics, profile alignment view
- **Evaluation Suite** — pytest measuring Precision@K, Recall@K across held-out query sets

## Tech Stack

| Component | Technology |
|---|---|
| Search Engine | Custom BM25F, custom inverted index (from scratch) |
| NLP | NLTK (tokenization, stopwords, stemming) |
| ML / Scoring | scikit-learn, numpy, pandas |
| Frontend | Streamlit |
| Data | PyArrow (Parquet job dataset) |
| Visualisation | Matplotlib |
| Tests | pytest |

## Project Structure

```
JobLens/
├── src/
│   ├── indexer.py          # Inverted index builder
│   ├── bm25f.py            # BM25F ranking implementation
│   ├── query_processor.py  # Query parsing & NLP pipeline
│   ├── profile_aligner.py  # Candidate-job profile matching
│   └── evaluation.py       # Precision@K, Recall@K metrics
├── app/                    # Streamlit web app
├── data/                   # Job posting dataset (Parquet)
├── scripts/                # Index build scripts
├── tests/                  # pytest evaluation suite
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/mangalrohan05/JobLens
cd JobLens
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
python scripts/build_index.py
streamlit run app/main.py
```

## Known Limitations

- Keyword baseline lacks IDF weighting; keyword-heavy postings rank disproportionately high
- Streamlit Cloud containers are stateless — SQLite candidate profiles reset on restart