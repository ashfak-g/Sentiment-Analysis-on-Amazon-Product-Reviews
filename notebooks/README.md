# Exploratory Notebooks & Visual Analysis

This directory contains interactive Jupyter notebooks exploring the **Amazon Product Reviews Sentiment Analysis** dataset, NLP preprocessing experiments, and model performance comparisons.

---

## Contents

- **[Sentiment Analysis on Amazon Product Reviews.ipynb](./Sentiment%20Analysis%20on%20Amazon%20Product%20Reviews.ipynb)**:
  - **Dataset Ingestion**: Loading and analyzing raw Amazon review text.
  - **Exploratory Data Analysis (EDA)**: Polarity distributions, review length statistics, and vocabulary analysis.
  - **NLP Preprocessing Experiments**: Comparing raw text vs. negation-aware stop-word filtering and lemmatization.
  - **Model Training & Comparison**: Benchmarks across Logistic Regression, Naive Bayes, Linear SVM, Random Forest, and XGBoost.
  - **Model Explainability**: Top positive and negative sentiment token feature weights.

---

## How to Run Locally

1. Activate your virtual environment:
   `ash
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   `

2. Launch Jupyter Notebook or JupyterLab:
   `ash
   jupyter notebook notebooks/
   `
