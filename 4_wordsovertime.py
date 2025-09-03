import json
import pandas as pd
from collections import Counter
import plotly.express as px
import re
from nltk.corpus import stopwords

# Load your JSON file (replace with your filename)
with open("documents_with_year.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

df = pd.DataFrame(docs)

# Simple tokenizer (regex, lowercased, words only)
def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

# Load English stopwords
stop_words = set(stopwords.words("english"))

# ======== ADD YOUR OWN STOPWORDS HERE =========
# Example: ["diary", "today", "said"]
custom_stopwords = [
    
]
stop_words.update(custom_stopwords)
# ==============================================

# Tokenize all docs, remove stopwords
df["tokens"] = df["text"].apply(lambda x: [t for t in tokenize(x) if t not in stop_words])

# Flatten tokens across all years to get global frequency
all_tokens = [t for tokens in df["tokens"] for t in tokens]
top_25 = [w for w, _ in Counter(all_tokens).most_common(25)]

print("Top 25 words:", top_25)

# Count frequencies per year for those 25 words
records = []
for year, group in df.groupby("year"):
    yearly_tokens = [t for tokens in group["tokens"] for t in tokens]
    counts = Counter(yearly_tokens)
    for word in top_25:
        records.append({"year": year, "word": word, "count": counts[word]})

freq_df = pd.DataFrame(records)

# Normalize (optional): relative frequency instead of raw count
# freq_df["relative_freq"] = freq_df.groupby("year")["count"].transform(lambda x: x / x.sum())

# Plotly line chart (one line per word)
fig = px.line(
    freq_df,
    x="year",
    y="count",  # or "relative_freq" if you normalize
    color="word",
    title="Top 25 Word Frequencies Over Time"
)

# Save to file
fig.write_html("top25_word_frequencies.html")
# fig.write_image("top25_word_frequencies.png")  # requires kaleido

print("Graph saved: top25_word_frequencies.html")
