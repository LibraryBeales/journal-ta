# tfidf_by_decade_GLOBAL_idf_with_facets.py
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from sklearn.feature_extraction import text as sktext

# ========== USER SETTINGS ==========
DOCS_JSON_PATH = Path("documents_with_year.json")
OUT_DIR        = Path("tfidf_by_decade")
TOP_N_TERMS    = 30              # top terms to export/plot per decade
NGRAM_RANGE    = (1, 2)          # (1,1)=unigrams; (1,2)=uni+bigrams
MIN_DF         = 2               # ignore terms seen in < MIN_DF docs (global)
MAX_DF         = 0.60            # drop terms in > 60% of docs (global ubiquity filter)
MAX_FEATURES   = 20000
MAKE_FACETED_HTML = True

# Extra domain stopwords (merged with scikit-learn's English list)
INLINE_STOPWORDS = [
    # Add diary boilerplate or domain-specific terms here:
    "today","yesterday","tomorrow","monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july","august","september","october","november","december"
]
STOPWORDS_FILE = Path("my_stopwords.txt")  # optional, one term per line
# ===================================

def to_decade(year: int) -> str:
    base = (int(year) // 10) * 10
    return f"{base}s"

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin-1"))

def load_custom_stopwords(inline_list, file_path: Path):
    sw = set(w.strip().lower() for w in inline_list if str(w).strip())
    if file_path and file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sw.add(line.lower())
    return sw

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = read_json(DOCS_JSON_PATH)
    df = pd.DataFrame(docs)
    if not {"year", "text"}.issubset(df.columns):
        raise ValueError("JSON must include 'year' and 'text' fields.")

    # Bucket texts by decade and collect all texts (for global vectorizer)
    buckets = defaultdict(list)
    all_texts = []
    for _, row in df.iterrows():
        txt = str(row.get("text", "")).strip()
        if not txt:
            continue
        decade = to_decade(int(row["year"]))
        buckets[decade].append(txt)
        all_texts.append(txt)

    # Build global stopword set
    custom_sw = load_custom_stopwords(INLINE_STOPWORDS, STOPWORDS_FILE)
    stop_words = set(sktext.ENGLISH_STOP_WORDS) | custom_sw
    print(f"Decades: {sorted(buckets.keys())}")
    print(f"Custom stopwords loaded: {len(custom_sw)} (total stopwords used: {len(stop_words)})")

    # ---- GLOBAL TF-IDF: fit once on ALL documents ----
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        lowercase=True,
        sublinear_tf=True,   # damp extremely frequent terms
        norm="l2",
    )
    X_all = vectorizer.fit_transform(all_texts)
    terms = np.array(vectorizer.get_feature_names_out())

    # Map each doc to its decade index for efficient slicing
    # (We rebuild the same order used to build X_all)
    doc_decades = []
    for dec in sorted(buckets.keys()):
        doc_decades.extend([dec] * len(buckets[dec]))

    # Compute top terms per decade using the SAME global vectorizer
    combined_rows = []
    start = 0
    for dec in sorted(buckets.keys()):
        texts = buckets[dec]
        n = len(texts)
        if n == 0:
            continue

        # Slice the corresponding rows from X_all (because we built all_texts in the same loop)
        X_dec = X_all[start:start+n, :]
        start += n

        # Mean TF-IDF score across docs in this decade
        mean_scores = np.asarray(X_dec.mean(axis=0)).ravel()
        order = np.argsort(-mean_scores)[:min(TOP_N_TERMS, len(mean_scores))]
        top_df = pd.DataFrame({
            "decade": dec,
            "term": terms[order],
            "mean_tfidf": mean_scores[order]
        })
        top_df.to_csv(OUT_DIR / f"{dec}_tfidf_top_terms.csv", index=False, encoding="utf-8")
        combined_rows.append(top_df)

    if not combined_rows:
        print("No decade had text. Exiting.")
        return

    all_df = pd.concat(combined_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_decades_tfidf_top_terms.csv", index=False, encoding="utf-8")
    print(f"Wrote combined CSV: {OUT_DIR / 'all_decades_tfidf_top_terms.csv'}")

    # ---- Faceted bars (Option #2) ----
    if MAKE_FACETED_HTML:
        df_ranked = (
            all_df.sort_values(["decade", "mean_tfidf"], ascending=[True, False])
                  .groupby("decade")
                  .head(TOP_N_TERMS)
        )
        fig = px.bar(
            df_ranked.sort_values(["decade", "mean_tfidf"], ascending=[True, True]),
            x="mean_tfidf",
            y="term",
            facet_col="decade",
            facet_col_wrap=3,
            orientation="h",
            title=f"Top TF-IDF Terms per Decade (Global IDF, Top {TOP_N_TERMS})",
            labels={"mean_tfidf": "Mean TF-IDF", "term": "Term"},
        )
        fig.update_layout(margin=dict(l=60, r=20, t=60, b=40))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        out_html = OUT_DIR / "tfidf_faceted_bars.html"
        fig.write_html(str(out_html))
        print(f"Wrote faceted bars HTML: {out_html}")

    print(f"Done. Outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
