# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Import libraries

# Load dataset
DATA_PATH = "F:\\Programming\\Artificial intelligence projects\\Svana AI\\reply_classification_dataset.csv"
data = pd.read_csv(DATA_PATH)

# Normalize labels
data['label'] = data['label'].str.lower().str.strip().str.replace(r'[^a-z]', '', regex=True)

# Check label distribution
print(data['label'].value_counts())

# Map labels to numerical values
label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
data['label'] = data['label'].map(label_map)

# Handle missing values
data = data.dropna()

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['clean_text'] = data['reply'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state=42)

# Baseline Model: Logistic Regression with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)

# Evaluate baseline
lr_accuracy = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds, average='weighted')
print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1 Score: {lr_f1:.4f}")


# Save Logistic Regression model and TF-IDF vectorizer
import joblib
joblib.dump(lr_model, "f:\\Programming\\Artificial intelligence projects\\Svana AI\\Project\\Models\\lr_model.joblib")
joblib.dump(vectorizer, "f:\\Programming\\Artificial intelligence projects\\Svana AI\\Project\\Models\\tfidf_vectorizer.joblib")

print("Logistic Regression model saved for API use.")
