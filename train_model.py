import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("Twitter_Data.csv")  # Change to your file's name if needed

# Check basic info
print("Initial shape:", df.shape)
print(df.head())

# Drop any missing rows (if any)
df = df.dropna()

# Keep only relevant columns
df = df[['clean_text', 'category']]
df = df[df['category'].isin([-1, 0, 1])]  # Filter valid labels only

# Optional: map labels to strings (you can skip if you want to use ints)
label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
df['label'] = df['category'].map(label_map)

# Final preview
# print("After cleaning:")
# print(df.head())

# Step 1: Split the data
X = df['clean_text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build a pipeline with TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Step 3: Train the model
pipeline.fit(X_train, y_train)

# Step 4: Evaluate on test data
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Save the trained pipeline
joblib.dump(pipeline, 'sentiment_model.pkl')
print("✅ Model saved as sentiment_model.pkl")
