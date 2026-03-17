import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🔹 Loading data...")

# Load processed data
df = pd.read_csv(
    r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\processed_resumes.csv"
)

# Remove empty values
df = df.dropna(subset=["cleaned_text"])

print("✅ Data loaded")

# ---------------------------
# Features & Labels
# ---------------------------
X = df["cleaned_text"]
y = df["label"]

# ---------------------------
# TF-IDF (Feature Engineering)
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
X_tfidf = vectorizer.fit_transform(X)

print("✅ TF-IDF created")

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("✅ Data split completed")

# ---------------------------
# Model 1: Logistic Regression
# ---------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\n📊 Logistic Regression Results:")
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n📊 Logistic Regression Evaluation:\n")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(10, 6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# ---------------------------
# Model 2: Random Forest
# ---------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n🌲 Random Forest Results:")
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


print("\n🌲 Random Forest Evaluation:\n")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# ---------------------------
# Save Best Model (choose LR)
# ---------------------------
joblib.dump(lr_model, r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai/models/model.pkl")
joblib.dump(vectorizer, r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai/vectorizer.pkl")

print("\n✅ Model & vectorizer saved!")

