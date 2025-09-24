
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np

# Load Logistic Regression model and TF-IDF vectorizer
MODEL_PATH = "f:\\Programming\\Artificial intelligence projects\\Svana AI\\Project\\Models\\lr_model.joblib"
VECTORIZER_PATH = "f:\\Programming\\Artificial intelligence projects\\Svana AI\\Project\\Models\\tfidf_vectorizer.joblib"
lr_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

label_map = {0: "positive", 1: "neutral", 2: "negative"}

def clean_text(text):
	text = text.lower()
	text = re.sub(r'[^a-z\s]', '', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

class TextRequest(BaseModel):
	text: str

app = FastAPI()

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
	cleaned = clean_text(request.text)
	X = vectorizer.transform([cleaned])
	pred = lr_model.predict(X)[0]
	label = label_map.get(pred, "unknown")
	return {"label": label, "label_id": int(pred)}
