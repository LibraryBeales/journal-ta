# bert_topic_by_decade.py
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# ========== USER SETTINGS ==========
DOCS_JSON_PATH = Path("documents_with_year.json")
OUT_DIR        = Path("bertopic_by_decade")
MIN_DOCS_PER_DECADE = 10      # skip decades with fewer docs
N_TOP_WORDS_PER_TOPIC = 15    # how many words to export per topic
# ===================================

def to_decade(year: int) -> str:
    base = (int(year) // 10) * 10
    return f"{base}s"

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin-1"))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = read_json(DOCS_JSON_PATH)
    df = pd.DataFrame(docs)
    if not {"year", "text"}.issubset(df.columns):
        raise ValueError("JSON must include 'year' and 'text' fields.")

    # bucket texts by decade
    buckets = defaultdict(list)
    for _, row in df.iterrows():
        decade = to_decade(int(row["year"]))
        text = str(row.get("text", "")).strip()
        if text:
            buckets[decade].append(text)

    decades = sorted(buckets.keys())
    print(f"Found decades: {decades}")

    # Vectorizer for BERTopic (English stopwords, unigrams+bigrams)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    # Tip: to control embeddings (e.g., smaller model), pass embedding_model="all-MiniLM-L6-v2"
    # topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", vectorizer_model=vectorizer, verbose=True)

    for decade in decades:
        texts = buckets[decade]
        if len(texts) < MIN_DOCS_PER_DECADE:
            print(f"Skipping {decade}: only {len(texts)} docs (< {MIN_DOCS_PER_DECADE}).")
            continue

        print(f"Running BERTopic for {decade} on {len(texts)} docs...")

        topic_model = BERTopic(vectorizer_model=vectorizer, verbose=False)
        topics, probs = topic_model.fit_transform(texts)

        # Save document-topic assignments
        assign_df = pd.DataFrame({
            "doc_id": range(len(texts)),
            "topic": topics,
            "probability": [float(p) if p is not None else None for p in probs]
        })
        assign_df.to_csv(OUT_DIR / f"{decade}_doc_topics.csv", index=False, encoding="utf-8")

        # Save topic overview
        info_df = topic_model.get_topic_info()
        info_df.to_csv(OUT_DIR / f"{decade}_topic_info.csv", index=False, encoding="utf-8")

        # Save top words per topic
        rows = []
        for tid in info_df["Topic"].tolist():
            if tid == -1:
                # -1 is usually outliers; include or skip as you prefer
                continue
            terms = topic_model.get_topic(tid) or []
            for rank, (word, score) in enumerate(terms[:N_TOP_WORDS_PER_TOPIC], start=1):
                rows.append({"topic": tid, "rank": rank, "term": word, "score": float(score)})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{decade}_topic_terms.csv", index=False, encoding="utf-8")

        # Save interactive topic visualization (HTML)
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(str(OUT_DIR / f"{decade}_topics.html"))
        except Exception as e:
            print(f"Could not write visualization for {decade}: {e}")

    print(f"Done. Outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
