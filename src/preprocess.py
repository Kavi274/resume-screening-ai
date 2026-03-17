import pandas as pd
import spacy
import nltk
import string
import os

from nltk.corpus import stopwords

# ---------------------------
# Step 1: Setup
# ---------------------------

print("🔹 Step 1: Loading resources...")

# Download stopwords (only first time)
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

print("✅ spaCy model loaded")

# ---------------------------
# Step 2: Load Dataset
# ---------------------------

file_path = r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\resumes.csv"

df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully")
print(df.head())

# ---------------------------
# Step 3: Stopwords
# ---------------------------

custom_stopwords = {
    "resume", "cv", "experience", "work", "job",
    "skills", "education", "project", "projects"
}

stop_words = set(stopwords.words('english')).union(custom_stopwords)

# ---------------------------
# Step 4: Preprocessing Function
# ---------------------------

def preprocess(text):
    text = str(text).lower()

    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # Process with spaCy
    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_alpha and token.text not in stop_words:
            tokens.append(token.lemma_)

    return " ".join(tokens)

print("🔹 Step 2: Preprocessing started...")

# Apply preprocessing
df["cleaned_text"] = df["resume_text"].apply(preprocess)

print("✅ Preprocessing completed")

# ---------------------------
# Step 5: Save Output
# ---------------------------

output_path = r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\processed_resumes.csv"

df.to_csv(output_path, index=False)

print("✅ Processed dataset saved at:")
print(output_path)

# ---------------------------
# Step 6: Sample Output
# ---------------------------

print("\n🔹 Sample Cleaned Text:\n")
print(df["cleaned_text"].head())