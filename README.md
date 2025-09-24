# SvaraAI — Reply Classification


**Task**: classify prospect replies into `positive`, `negative`, or `neutral`.


This repo contains a baseline ML pipeline (TF-IDF + Logistic Regression), an optional DistilBERT fine-tune script, and a FastAPI service to serve predictions.


## Deliverables included
- `src/train_and_eval.py` — training & evaluation script (produces models in `models/`).
- `notebook_export.py` — notebook-style Python script you can run or import into Jupyter.
- `src/fine_tune_distilbert.py` — example Hugging Face fine-tune script (optional).
- `app.py` — FastAPI app exposing `/predict`.
- `answers.md` — short answers (2–3 sentences each) for Part C.
- `requirements.txt`, `Dockerfile`, `.gitignore`.


## Quickstart (local)
1. Place your dataset CSV at `data/replies.csv` with **columns**: `text`, `label`.
2. Create a virtualenv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
