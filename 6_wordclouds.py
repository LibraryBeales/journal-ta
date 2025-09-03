import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ========== USER SETTINGS ==========
DOCS_JSON_PATH = r"documents_with_year.json"   # your dataset
OUTPUT_PNG     = Path("wordclouds_by_decade.png")
MIN_TOKEN_LEN  = 3
MAX_WORDS      = 200
BACKGROUND     = "white"
GRID_COLS      = 3  # number of columns in the tiled grid
# ===================================

# Ensure NLTK stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Base stopwords from NLTK
stop_words = set(stopwords.words("english"))

# ======== ADD YOUR OWN STOPWORDS HERE =========
# Example: ["diary", "today", "said"]
custom_stopwords = [
    
]
stop_words.update(custom_stopwords)
# ==============================================

# Simple tokenizer
token_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
def tokenize(text: str):
    return token_re.findall(str(text).lower())

def read_json(path):
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding="latin-1"))

def to_decade(year: int) -> str:
    base = (int(year) // 10) * 10
    return f"{base}s"

# --- Load & prepare ---
docs = read_json(DOCS_JSON_PATH)

decade_tokens = defaultdict(list)
for doc in docs:
    y = doc.get("year", None)
    txt = doc.get("text", "")
    if y is None:
        continue
    decade = to_decade(int(y))
    toks = tokenize(txt)
    toks = [
        t for t in toks
        if t not in stop_words
        and len(t) >= MIN_TOKEN_LEN
        and not t.isdigit()
    ]
    decade_tokens[decade].extend(toks)

# Sort decades
decades = sorted(decade_tokens.keys())

# Determine grid size
n = len(decades)
cols = GRID_COLS
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
axes = axes.flatten()

for i, decade in enumerate(decades):
    toks = decade_tokens[decade]
    if not toks:
        continue
    freqs = Counter(toks)
    wc = WordCloud(
        width=800,
        height=600,
        background_color=BACKGROUND,
        max_words=MAX_WORDS,
        collocations=False
    ).generate_from_frequencies(freqs)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].set_title(decade, fontsize=16)
    axes[i].axis("off")

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()

print(f"Saved tiled wordclouds: {OUTPUT_PNG}")
