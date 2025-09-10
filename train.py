import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset (tab-separated file: label \t text)
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

X = df["text"]
y = df["label"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)),
    ("clf", MultinomialNB())
])

# Train
pipeline.fit(X_train, y_train)

# Save model
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as spam_classifier.pkl")
