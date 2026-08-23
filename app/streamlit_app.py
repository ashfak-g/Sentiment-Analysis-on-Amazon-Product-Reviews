"""
Streamlit Web Application for Amazon Product Review Sentiment Analysis.
Provides interactive single review analysis, batch CSV processing, and model benchmarks.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_artifact, load_json
from src.text_preprocessor import TextPreprocessor

MODEL_PATH = "models/best_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METADATA_PATH = "models/model_metadata.json"

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline_artifacts():
    """Caches and loads trained model, vectorizer, preprocessor, and metadata."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None, None, None
    model = load_artifact(MODEL_PATH)
    vectorizer = load_artifact(VECTORIZER_PATH)
    preprocessor = load_artifact(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else TextPreprocessor()
    metadata = load_json(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return model, vectorizer, preprocessor, metadata

model, vectorizer, preprocessor, metadata = load_pipeline_artifacts()

# Sidebar Setup
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=180)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Feature", 
    ["🔍 Live Sentiment Analyzer", "📁 Batch CSV Analyzer", "📊 Model Metrics & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**End-to-End Sentiment Analysis System**\n\n"
    "Built with Python, Scikit-Learn, TF-IDF, NLTK, Streamlit & FastAPI."
)

# Header
st.title("Amazon Product Review Sentiment Analysis")
st.caption("AI-powered NLP Classification System for E-Commerce Product Feedback")

if model is None or vectorizer is None:
    st.warning("⚠️ Model artifacts not found in `models/` directory. Please run `python -m src.pipeline` first.")
    st.stop()

# --- PAGE 1: Live Sentiment Analyzer ---
if page == "🔍 Live Sentiment Analyzer":
    st.subheader("Interactive Single Review Prediction")
    st.markdown("Enter an Amazon product review below to analyze its sentiment polarity in real-time.")
    
    sample_reviews = [
        "This product exceeded my expectations! Battery life is amazing and setup took 2 minutes.",
        "Terrible quality. Broke after three days of normal use. Highly disappointed.",
        "It's decent for the price. Not great, but gets the job done."
    ]
    
    selected_sample = st.selectbox("Or choose a sample review:", ["Custom input..."] + sample_reviews)
    
    if selected_sample != "Custom input...":
        default_text = selected_sample
    else:
        default_text = ""
        
    user_review = st.text_area("Review Content:", value=default_text, height=120, placeholder="Write or paste review text here...")
    
    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if not user_review.strip():
            st.error("Please enter a valid review text.")
        else:
            with st.spinner("Analyzing text with NLP pipeline..."):
                cleaned = preprocessor.clean_text(user_review)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                
                # Confidence score calculation
                prob = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec)[0]
                    prob = probs[1] if pred == 1 else probs[0]
                elif hasattr(model, "decision_function"):
                    df_val = model.decision_function(vec)[0]
                    prob = 1 / (1 + pow(2.71828, -abs(df_val)))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if pred == 1:
                        st.success("### Result: POSITIVE SENTIMENT 😊")
                    else:
                        st.error("### Result: NEGATIVE SENTIMENT 🙁")
                        
                    if prob is not None:
                        st.metric(label="Model Confidence", value=f"{prob * 100:.2f}%")
                        st.progress(prob)
                
                with col2:
                    st.markdown("**NLP Preprocessing Breakdown:**")
                    st.text_area("Cleaned & Lemmatized Tokens:", value=cleaned, height=90, disabled=True)
                    st.caption(f"Word count: {len(user_review.split())} words → {len(cleaned.split())} processed tokens")

# --- PAGE 2: Batch CSV Analyzer ---
elif page == "📁 Batch CSV Analyzer":
    st.subheader("Batch Review Processing")
    st.markdown("Upload a CSV file containing a column named `reviewText` or select a column for bulk inference.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Successfully loaded file with **{len(batch_df)}** rows.")
            st.dataframe(batch_df.head(3), use_container_width=True)
            
            text_col = st.selectbox("Select Review Text Column:", batch_df.columns)
            
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch dataset..."):
                    cleaned_texts = preprocessor.transform(batch_df[text_col].astype(str).tolist())
                    vecs = vectorizer.transform(cleaned_texts)
                    preds = model.predict(vecs)
                    
                    batch_df["Predicted_Sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]
                    
                    st.success("Batch classification complete!")
                    
                    # Sentiment Distribution Summary
                    pos_count = (preds == 1).sum()
                    neg_count = (preds == 0).sum()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Reviews", len(batch_df))
                    m2.metric("Positive Reviews", f"{pos_count} ({pos_count/len(preds)*100:.1f}%)")
                    m3.metric("Negative Reviews", f"{neg_count} ({neg_count/len(preds)*100:.1f}%)")
                    
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# --- PAGE 3: Model Metrics & Insights ---
elif page == "📊 Model Metrics & Insights":
    st.subheader("Model Performance & Production Metrics")
    
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Algorithm", metadata.get("best_model_name", "N/A"))
        col2.metric("Accuracy", f"{metadata.get('metrics', {}).get('accuracy', 0)*100:.2f}%")
        col3.metric("F1 Score", f"{metadata.get('metrics', {}).get('f1_score', 0)*100:.2f}%")
        col4.metric("TF-IDF Features", metadata.get("num_features", 0))
        
        st.markdown("### Production Details")
        st.json(metadata)
    else:
        st.info("Metadata file not found.")
