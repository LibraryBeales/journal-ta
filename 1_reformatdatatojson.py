import os
import pandas as pd

BASE_DIR = r"C:\Users\rdb104\Documents\caserepos\journal-ta\data"

records = []

for root, dirs, files in os.walk(BASE_DIR):
    folder_name = os.path.basename(root)
    
    # Remove trailing " txt" if present
    if folder_name.endswith(" txt"):
        folder_name = folder_name.replace(" txt", "")
    
    # If what's left is digits, treat as year
    if folder_name.isdigit():
        year = int(folder_name)
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1") as f:
                        text = f.read()

                records.append({
                    "year": year,
                    "filename": file,
                    "text": text
                })

# Save into DataFrame
df = pd.DataFrame(records)
df.to_csv("documents_with_year.csv", index=False, encoding="utf-8")
df.to_json("documents_with_year.json", orient="records", force_ascii=False, indent=2)

print(f"Saved {len(df)} documents with year metadata.")
