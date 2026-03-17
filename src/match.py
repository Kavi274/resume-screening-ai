import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🔹 Loading processed data...")

# Load processed dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\processed_resumes.csv")

print("✅ Data loaded")

# ---------------------------
# Job Description
# ---------------------------
job_description = """
Looking for a Data Scientist with strong knowledge in Python,
machine learning, data analysis, pandas, and statistics.
"""

# ---------------------------
# Combine text
# ---------------------------
df = df.dropna(subset=["cleaned_text"])
texts = df["cleaned_text"].tolist()
texts.append(job_description)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

print("✅ TF-IDF created")

# ---------------------------
# Cosine Similarity
# ---------------------------
similarity_scores = cosine_similarity(
    tfidf_matrix[-1],   # job description
    tfidf_matrix[:-1]   # resumes
)

# ---------------------------
# Add scores
# ---------------------------
df["similarity"] = similarity_scores.flatten()

# ---------------------------
# Sort results
# ---------------------------
top_matches = df.sort_values(by="similarity", ascending=False)

# ---------------------------
# Show results
# ---------------------------
print("\n🔥 Top 5 Matching Resumes:\n")
print(top_matches[["label", "similarity"]].head())

# ---------------------------
# Save output
# ---------------------------
top_matches.to_csv(
    r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\matched_resumes.csv",
    index=False
)

print("\n✅ Matching completed & saved!")