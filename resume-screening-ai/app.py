import streamlit as st
import joblib

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
model = joblib.load(r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai/models/model.pkl")
vectorizer = joblib.load(r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai/vectorizer.pkl")

# ---------------------------
# UI Title
# ---------------------------
st.title("📄 Resume Screening AI")

st.write("Upload or paste a resume to predict job category")

# ---------------------------
# Input Text
# ---------------------------
resume_text = st.text_area("Paste Resume Text Here")

# ---------------------------
# Dropdown
# ---------------------------
job_role = st.selectbox(
    "🎯 Select Target Job Role",
    model.classes_
)


# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Category"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text")
    else:
        # Preprocess
        processed = resume_text.lower()

        # Vectorize
        vector = vectorizer.transform([processed])

        # Predict
        prediction = model.predict(vector)[0]

        # Probabilities
        proba = model.predict_proba(vector)[0]

        # -------------------------
        # 🥇 MAIN RESULT
        # -------------------------
        st.subheader("📊 Result")
        st.success(f"Category: {prediction}")

        # Confidence bar
        st.progress(int(max(proba) * 100))
        st.write(f"Confidence: {round(max(proba) * 100, 2)}%")

        # 🔥 MATCH CHECK
        if prediction == job_role:
            st.success("✅ Good match for selected role")
        else:
            st.warning("⚠️ Not a strong match")

        # -------------------------
        # 🏆 TOP 3 PREDICTIONS (ADD HERE)
        # -------------------------
        top_indices = proba.argsort()[-3:][::-1]
        classes = model.classes_

        st.subheader("🏆 Top 3 Matches")

        for i in top_indices:
            st.write(f"{classes[i]} → {round(proba[i] * 100, 2)}%")