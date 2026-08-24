"""
FastAPI REST Service for Amazon Product Review Sentiment Analysis.
"""

import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_artifact, load_json
from src.text_preprocessor import TextPreprocessor

MODEL_PATH = "models/best_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METADATA_PATH = "models/model_metadata.json"

app = FastAPI(
    title="Amazon Review Sentiment Analysis API",
    description="Production REST API for classifying Amazon product review sentiment using NLP and Machine Learning.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded artifacts
model = None
vectorizer = None
preprocessor = None
metadata = None

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, preprocessor, metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = load_artifact(MODEL_PATH)
        vectorizer = load_artifact(VECTORIZER_PATH)
        preprocessor = load_artifact(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else TextPreprocessor()
        metadata = load_json(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    else:
        print("Warning: Model artifacts not found. API endpoints will fail until trained.")

class PredictRequest(BaseModel):
    text: str = Field(..., example="This product is amazing, fast shipping and works great!")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., example=[
        "Great quality item, highly recommend!",
        "Worst purchase ever. Completely broken upon arrival."
    ])

class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    label: int
    confidence: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    total_reviews: int
    predictions: List[PredictionResponse]

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Welcome to the Amazon Product Review Sentiment Analysis API",
        "docs_url": "/docs",
        "status": "online"
    }

@app.get("/health", tags=["Health"])
def health_check():
    model_loaded = model is not None and vectorizer is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_name": metadata.get("best_model_name") if metadata else None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_sentiment(request: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
        
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    cleaned = preprocessor.clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = float(probs[1] if pred == 1 else probs[0])
        
    sentiment = "Positive" if pred == 1 else "Negative"
    
    return PredictionResponse(
        text=text,
        cleaned_text=cleaned,
        sentiment=sentiment,
        label=pred,
        confidence=round(confidence, 4) if confidence is not None else None
    )

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(request: BatchPredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
        
    if not request.texts:
        raise HTTPException(status_code=400, detail="List of review texts cannot be empty.")
        
    results = []
    cleaned_texts = preprocessor.transform(request.texts)
    vecs = vectorizer.transform(cleaned_texts)
    preds = model.predict(vecs)
    
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vecs)
        
    for i, orig_text in enumerate(request.texts):
        pred_label = int(preds[i])
        sentiment = "Positive" if pred_label == 1 else "Negative"
        conf = float(probs[i][1] if pred_label == 1 else probs[i][0]) if probs is not None else None
        
        results.append(PredictionResponse(
            text=orig_text,
            cleaned_text=cleaned_texts[i],
            sentiment=sentiment,
            label=pred_label,
            confidence=round(conf, 4) if conf is not None else None
        ))
        
    return BatchPredictionResponse(
        total_reviews=len(results),
        predictions=results
    )
