# spam_classifier.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# 1. Load Dataset
# -------------------------------
# Download dataset from Kaggle (SMS Spam Collection Dataset)
# Ensure file name is "spam.csv" in the same directory
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset loaded: {df.shape[0]} messages")
print(df.head())

# -------------------------------
# 2. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# -------------------------------
# 3. Feature Extraction
# -------------------------------
# Option A: Bag of Words
# vectorizer = CountVectorizer(stop_words='english')

# Option B: TF-IDF (recommended for better performance)
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 4. Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------------
# 5. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 6. Test Custom Messages
# -------------------------------
test_messages = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "Hey, are we still on for dinner tonight?",
    "URGENT! Your account has been compromised. Reset your password immediately."
]

test_vec = vectorizer.transform(test_messages)
predictions = model.predict(test_vec)

print("\n🔍 Custom Message Predictions:")
for msg, pred in zip(test_messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Message: {msg}\nPrediction: {label}\n")
