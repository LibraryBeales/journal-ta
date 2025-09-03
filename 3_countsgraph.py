import json
import pandas as pd
import plotly.graph_objects as go

# Load your JSON file (replace with your filename)
with open("documents_with_year.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(docs)

# Count entries per year
year_counts = df.groupby("year")["filename"].count().reset_index(name="num_documents")

# Count words per year
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
year_words = df.groupby("year")["word_count"].sum().reset_index(name="num_words")

# Merge both
year_stats = pd.merge(year_counts, year_words, on="year")

# Create Plotly chart with two y-axes
fig = go.Figure()

# Documents per year
fig.add_trace(go.Bar(
    x=year_stats["year"],
    y=year_stats["num_documents"],
    name="Documents",
    yaxis="y1"
))

# Words per year
fig.add_trace(go.Scatter(
    x=year_stats["year"],
    y=year_stats["num_words"],
    name="Words",
    yaxis="y2",
    mode="lines+markers"
))

# Layout with two y-axes
fig.update_layout(
    title="Documents and Word Counts per Year",
    xaxis=dict(title="Year"),
    yaxis=dict(title="Number of Documents"),
    yaxis2=dict(title="Number of Words",
                overlaying="y",
                side="right"),
    barmode="group",
    legend=dict(x=0.1, y=1.1, orientation="h")
)

# Save interactive HTML file
fig.write_html("yearly_documents_words.html")

# Or save as static PNG (requires: pip install -U kaleido)
# fig.write_image("yearly_documents_words.png")

print("Graph saved to file.")