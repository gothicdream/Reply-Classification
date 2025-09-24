📨 Reply Classification Project

This project classifies email replies into positive, negative, or neutral categories. It includes both a baseline ML model and a fine-tuned transformer model, and provides a FastAPI service for real-time predictions.

📂 Project Structure
Reply-Classification/
│
├─ env/                    # Python virtual environment (optional to include in repo)
├─ data/                   # CSV dataset of email replies
├─ models/                 # Saved model files (Logistic Regression & DistilBERT)
├─ notebooks/              # Jupyter notebooks for experimentation
├─ main.py                 # FastAPI application
├─ requirements.txt        # Python dependencies
├─ Dockerfile              # Optional: for containerization
├─ README.md               # Project documentation
└─ answers.md              # Short answer reasoning for Part C

⚙️ Setup Instructions
1. Python Environment

Install Python 3.11.9.

Create a virtual environment:

python -m venv env


Activate the environment:

Windows (PowerShell):

.\env\Scripts\Activate.ps1


macOS/Linux:

source env/bin/activate


Install required libraries:

pip install -r requirements.txt

2. Install Dependencies

Libraries used in the project:

pandas, numpy – Data processing

scikit-learn – Logistic Regression & metrics

torch, transformers – Fine-tuning DistilBERT

fastapi, uvicorn – API deployment

python-multipart – For FastAPI JSON parsing (if needed)

🧩 ML/NLP Pipeline
Steps:

Load and preprocess the CSV dataset:

Clean text

Handle missing values

Train baseline model:

Logistic Regression

LightGBM (optional)

Fine-tune transformer:

distilbert-base-uncased using Hugging Face

Evaluate models:

Accuracy

F1-score

Comparison:

Logistic Regression: stable, faster, good for small datasets

DistilBERT: higher accuracy but sometimes hallucinate, slower inference

✅ Production choice: Logistic Regression model for reliability.

🚀 FastAPI Deployment
Run the API
uvicorn main:app --reload


Open in browser: http://127.0.0.1:8000/docs

Example input:

{
  "text": "Looking forward to the demo!"
}


Example output:

{
  "label": "positive",
  "confidence": 0.87
}

📖 Use Case Guide

Open reply_classification_report.html to view model evaluation metrics.

Start the Uvicorn server.

Navigate to /docs to test API predictions.
