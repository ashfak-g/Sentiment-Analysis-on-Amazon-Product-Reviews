# 🛒 Amazon Product Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade Natural Language Processing (NLP) and Machine Learning system** designed to analyze and classify Amazon product review feedback into **Positive** or **Negative** sentiments. Built for enterprise scalability, modularity, real-time web serving, and REST API integration.

---

## 👨‍💻 Developer Profile

- **Developer**: **Ashfakur Rahman**
- **Degree**: B.Sc. in Computer Science & Engineering (CSE)
- **Institution**: Green University of Bangladesh
- **Domain Focus**: Machine Learning, Natural Language Processing (NLP), MLOps, & Full-Stack AI Application Development

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Repository Structure](#-system-architecture--repository-structure)
- [NLP Preprocessing & Feature Engineering](#-nlp-preprocessing--feature-engineering)
- [Model Benchmarks & Technical Justification](#-model-benchmarks--technical-justification)
- [Visitor Quick Start & Live Testing Guide](#-visitor-quick-start--live-testing-guide)
- [FastAPI REST Service Usage](#-fastapi-rest-service-usage)
- [Automated Testing & Containerization](#-automated-testing--containerization)
- [Future Roadmap](#-future-roadmap)
- [License & Contact](#-license--contact)

---

## 💡 Project Overview

In e-commerce platforms like Amazon, millions of customer reviews are submitted daily. Manually evaluating customer sentiment across vast volumes of unstructured text is inefficient and unscalable.

This project delivers an automated, high-accuracy AI classification system that:
1. Ingests and cleans unstructured customer review text.
2. Applies custom NLP transformations preserving critical sentiment indicators (such as negations).
3. Evaluates 5 machine learning models and performs hyperparameter tuning.
4. Serves predictions via an **interactive Streamlit Web Dashboard** and a **FastAPI REST Service** with real-time probability confidence scores.

---

## 🌟 Key Features

- **Advanced NLP Preprocessing**: Text normalization, HTML/URL stripping, punctuation removal, **negation-aware stop word filtering** (`not`, `no`, `never` retained), and WordNet lemmatization.
- **TF-IDF Feature Engineering**: Sublinear TF-IDF vectorization capturing unigrams and bigrams `(1, 2)`.
- **Multi-Algorithm Model Benchmarking**: Comparative training across Logistic Regression, Multinomial Naïve Bayes, Support Vector Machines (LinearSVC), Random Forest, and XGBoost Classifier.
- **Automated Hyperparameter Tuning**: 5-fold Grid Search Cross-Validation for parameter optimization.
- **Interactive Web App (Streamlit)**: Single review inference, batch CSV classification with downloadable predictions, and real-time model confidence breakdown.
- **Production REST API (FastAPI)**: Swagger UI OpenAPI documentation (`/docs`), `/predict`, `/batch-predict`, and `/health` status endpoints.
- **Enterprise Engineering Best Practices**: Unit tested (`pytest`), containerized (`Dockerfile`), and automated CI/CD (`GitHub Actions`).

---

## 🏗️ System Architecture & Repository Structure

```
Sentiment Analysis on Amazon Product Reviews/
│
├── .github/workflows/ci.yml         # GitHub Actions CI/CD Pipeline
├── data/                            # Dataset storage
│   ├── raw/amazon.csv               # Raw Amazon 20,000 reviews dataset
│   └── processed/                   # Train/Test split CSVs
├── src/                             # Core Modular Python Package
│   ├── __init__.py
│   ├── utils.py                     # Logging, config & artifact persistence
│   ├── data_loader.py               # Data ingestion, cleaning & 80/20 train-test split
│   ├── text_preprocessor.py         # Negation-aware NLP text cleaning & lemmatization
│   ├── feature_extraction.py        # Sublinear TF-IDF n-gram vectorizer
│   ├── model_trainer.py             # Multi-algorithm classification trainer
│   ├── hyperparameter_tuner.py      # Grid Search 5-fold cross-validation
│   ├── evaluator.py                 # Performance metrics, confusion matrix & ROC curves
│   └── pipeline.py                  # End-to-end training & artifact serialization orchestrator
├── models/                          # Serialized Production Artifacts
│   ├── best_model.joblib            # Tuned Logistic Regression model
│   ├── tfidf_vectorizer.joblib      # Fitted TF-IDF Vectorizer
│   ├── preprocessor.joblib          # TextPreprocessor instance
│   └── model_metadata.json          # Production evaluation metadata
├── app/                             # Web & API Deployment Layer
│   ├── streamlit_app.py             # Interactive Streamlit Web UI
│   └── api.py                       # FastAPI RESTful Web Service
├── tests/                           # Automated Pytest Suite
│   ├── test_preprocessor.py         # NLP text cleaner unit tests
│   └── test_models.py               # Feature extraction & trainer unit tests
├── notebooks/                       # Executed Visual Analysis Notebook
│   └── Sentiment Analysis on Amazon Product Reviews.ipynb
├── Dockerfile                       # Production Containerization setup
├── .dockerignore
├── .gitignore
├── requirements.txt                 # Pinned project dependencies
└── README.md                        # Portfolio documentation
```

---

## 🔬 NLP Preprocessing & Feature Engineering

### 1. Negation-Aware Stop Word Filtering
Standard NLP stop word removal strips words like `not`, `no`, `never`, `cannot`. However, stripping these words corrupts sentiment polarity (e.g., `"not good"` becomes `"good"`). Our custom `TextPreprocessor` retains negation words to preserve true sentiment context.

### 2. WordNet Lemmatization
Words are reduced to their canonical dictionary root forms (e.g., `running`, `runs` $\to$ `run`), standardizing the vocabulary space.

### 3. Sublinear TF-IDF Unigram & Bigram Features `(1, 2)`
We extract 5,000 max features capturing single words and two-word phrases like `"highly recommend"`, `"bad quality"`, and `"not recommend"`.

---

## 📈 Model Benchmarks & Technical Justification

### Model Benchmark Leaderboard (Evaluated on 4,000 Test Reviews):

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Tuned)** | **90.20%** | **92.29%** | **95.08%** | **0.9366** | **0.9521** | 🏆 **Best Production Model** |
| **Logistic Regression (Base)** | 90.15% | 91.00% | 96.62% | 0.9373 | 0.9554 | Candidate |
| **Linear Support Vector Machine (LinearSVC)** | 89.88% | 92.45% | 94.42% | 0.9342 | 0.9496 | Candidate |
| **Multinomial Naïve Bayes** | 88.73% | 89.00% | 97.21% | 0.9293 | 0.9507 | Candidate |
| **XGBoost Classifier** | 88.35% | 90.19% | 95.04% | 0.9255 | 0.9381 | Candidate |
| **Random Forest Classifier** | 88.03% | 89.34% | 95.70% | 0.9241 | 0.9384 | Candidate |

### Why Logistic Regression Outperformed Tree Ensembles (XGBoost / Random Forest):

1. **High Dimensionality & Sparsity**: TF-IDF text feature matrices contain 5,000 dimensions where 99% of entries are zero. Linear models excel in finding hyperplanes in sparse spaces, whereas Tree ensembles overfit single-word node splits.
2. **Linear Decision Boundary**: Sentiment classification is inherently additive (positive words add positive weights; negative words add negative weights). The underlying decision space is naturally linear.
3. **Probability Confidence Calibration**: Logistic Regression outputs well-calibrated probabilities via the Sigmoid function $P(Y=1|X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$. Linear SVM produces Hinge Loss distance scores rather than calibrated probabilities.
4. **Sub-millisecond Inference Speed**: Inference requires a single dot product taking $<0.1\text{ ms}$, compared to traversing 100+ decision trees in XGBoost.

---

## 🚀 Visitor Quick Start & Live Testing Guide

Follow these simple steps to clone, set up, and test the project live in your browser:

### 1. Clone the Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/Sentiment-Analysis-on-Amazon-Product-Reviews.git
cd Sentiment-Analysis-on-Amazon-Product-Reviews

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web UI

Launch the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```
Open your web browser at: **`http://localhost:8501`**

### 🧪 Sample Test Inputs for Visitors to Try:

| Test Scenario | Review Text | Expected Sentiment |
| :--- | :--- | :---: |
| **Positive Review** | `"This product exceeded my expectations! Battery life is amazing and setup took 2 minutes."` | **POSITIVE 😊** |
| **Negative Review** | `"Terrible quality. Broke after three days of normal use. Highly disappointed."` | **NEGATIVE 🙁** |
| **Mixed / Neutral Review** | `"It's decent for the price. Not great, but gets the job done."` | **POSITIVE / NEUTRAL 😊** |

---

## ⚡ FastAPI REST Service Usage

Launch the RESTful web service:

```bash
uvicorn app.api:app --reload --port 8000
```

- **Interactive Swagger Documentation**: Open **`http://localhost:8000/docs`**
- **Single Review Prediction Endpoint (`POST /predict`)**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/predict' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": "Fast delivery, excellent packaging, and works like a charm!"
  }'
  ```
- **Sample JSON Response**:
  ```json
  {
    "text": "Fast delivery, excellent packaging, and works like a charm!",
    "cleaned_text": "fast delivery excellent packaging work like charm",
    "sentiment": "Positive",
    "label": 1,
    "confidence": 0.9842
  }
  ```

---

## 🧪 Automated Testing & Containerization

### Run Unit Tests (`pytest`)
```bash
pytest tests/
```
Output:
```
tests\test_models.py ..                                                  [ 40%]
tests\test_preprocessor.py ...                                           [100%]
============================= 5 passed in 12.36s ==============================
```

### Docker Container Setup
```bash
# Build Docker image
docker build -t amazon-sentiment-app .

# Run Docker container
docker run -p 8000:8000 amazon-sentiment-app
```

---

## 🔮 Future Roadmap

- **Transformer / BERT Integration**: Fine-tuning HuggingFace `bert-base-uncased` and `RoBERTa` for deep contextual representations.
- **Aspect-Based Sentiment Analysis (ABSA)**: Classifying sentiment across specific product aspects (e.g., *Battery*, *Price*, *Build Quality*, *Shipping*).
- **Multilingual Sentiment Classification**: Expanding to Bengali and multi-language review datasets using `mBERT`.
- **Cloud Auto-Scaling**: Production deployment on AWS EC2 / Kubernetes (K8s).

---

## 📄 License & Author

Distributed under the MIT License. See `LICENSE` for more information.

**Author**: **Ashfakur Rahman**  
*B.Sc. in Computer Science & Engineering (CSE), Green University of Bangladesh*
