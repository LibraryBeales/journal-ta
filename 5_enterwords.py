import json
import re
import pandas as pd
import plotly.express as px
from collections import Counter
from pathlib import Path

# ======== USER SETTINGS ========
DOCS_JSON_PATH = r"documents_with_year.json"     # your dataset with year/filename/text
WORDLIST_PATH  = r"target_words.txt"             # plain text file, one word per line
OUTPUT_CSV     = r"word_frequencies_by_year.csv"
OUTPUT_HTML    = r"word_frequencies_by_year.html"

# If True, also compute relative frequencies (per-year counts divided by total tokens that year)
ADD_RELATIVE   = True
# =================================

def read_json(path):
    # robust UTF-8 reading, fallback to latin-1
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding="latin-1"))

def read_wordlist(path):
    # one word per line; ignore blanks and lines starting with '#'
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.append(w.lower())
    # deduplicate, keep original order
    seen = set()
    deduped = []
    for w in words:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped

# Simple, fast tokenizer: words composed of letters/numbers/underscore
token_re = re.compile(r"\b\w+\b", flags=re.UNICODE)

def tokenize(text):
    return token_re.findall(str(text).lower())

# ---- Load data and word list ----
docs = read_json(DOCS_JSON_PATH)
df = pd.DataFrame(docs)

if "year" not in df or "text" not in df:
    raise ValueError("Input JSON must contain at least 'year' and 'text' fields.")

target_words = read_wordlist(WORDLIST_PATH)
if not target_words:
    raise ValueError("Your word list is empty. Put one word per line in the file.")

# ---- Compute per-year counts ----
# tokenize each document once
df["tokens"] = df["text"].apply(tokenize)
# total tokens per year (for optional relative frequencies)
year_total_tokens = (
    df[["year", "tokens"]]
    .assign(token_counts=lambda x: x["tokens"].apply(len))
    .groupby("year", as_index=False)["token_counts"].sum()
    .rename(columns={"token_counts": "total_tokens"})
)

# counts per year for listed words
records = []
for year, group in df.groupby("year"):
    all_tokens = [t for toks in group["tokens"] for t in toks]
    counts = Counter(all_tokens)
    for w in target_words:
        records.append({"year": int(year), "word": w, "count": counts[w]})

freq_df = pd.DataFrame(records)

# optional relative frequency
if ADD_RELATIVE:
    freq_df = freq_df.merge(year_total_tokens, on="year", how="left")
    # avoid divide-by-zero
    freq_df["relative_freq"] = freq_df.apply(
        lambda r: (r["count"] / r["total_tokens"]) if r["total_tokens"] else 0.0, axis=1
    )

# ---- Save CSV ----
freq_df.sort_values(["word", "year"], inplace=True)
freq_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved CSV: {OUTPUT_CSV}")

# ---- Plotly chart (interactive HTML) ----
y_col = "relative_freq" if ADD_RELATIVE else "count"
y_title = "Relative Frequency (per total tokens)" if ADD_RELATIVE else "Count"

fig = px.line(
    freq_df,
    x="year",
    y=y_col,
    color="word",
    markers=True,
    title="Listed Word Frequencies Over Time",
    labels={"year": "Year", y_col: y_title, "word": "Word"}
)
fig.update_layout(legend_title_text="Word")
fig.write_html(OUTPUT_HTML)
print(f"Saved HTML: {OUTPUT_HTML}")

# Optional: also print a small summary of totals across all years
totals = (
    freq_df.groupby("word")["count"].sum().reset_index().sort_values("count", ascending=False)
)
print("\nTop words by total count (from your list):")
print(totals.to_string(index=False))
