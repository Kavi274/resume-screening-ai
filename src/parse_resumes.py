import pandas as pd

file_path = r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\Resume.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!\n")
print(df.head())

print("\nColumns:")
print(df.columns)

# Rename columns
df = df.rename(columns={
    'Resume_str': 'resume_text',
    'Category': 'label'
})

df = df[['resume_text', 'label']]

# Save output
df.to_csv(r"C:\Users\DELL\Desktop\Resume_Screening_AI\resume-screening-ai\data\resumes.csv", index=False)

print("\nDataset saved successfully!")