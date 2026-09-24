import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

INPUT_CSV  = "iphone16_modeled.csv"
OUTPUT_CSV = "iphone16_topics.csv"
MODEL_PKL  = "lda_model.pkl"
REPORT_PNG = "topic_modeling_report.png"

TEXT_COL    = "clean_text"
LABEL_COL   = "final_label"
TOKENS_COL  = "tokens"

NUM_TOPICS  = 6       # change after reviewing coherence curve
RANDOM_SEED = 42

COLORS = {
    "positive": "#3B8BD4",
    "neutral":  "#888780",
    "negative": "#E24B4A",
}
TOPIC_PALETTE = ["#185FA5","#1D9E75","#BA7517","#A32D2D","#534AB7","#993556"]

# ─────────────────────────────────────────────────────────────────────────────
#  IPHONE-SPECIFIC STOPWORDS
#  Generic stopwords were removed in preprocessing.
#  These domain terms appear in almost every comment so they carry no topic
#  signal — removing them lets LDA find more meaningful distinctions.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_STOPWORDS = {
    "iphone", "apple", "phone", "device", "get", "got", "one",
    "use", "used", "using", "like", "just", "really", "good",
    "think", "know", "want", "make", "thing", "way", "time",
    "new", "also", "well", "much", "even", "still", "great",
    "would", "could", "going",
}


TOPIC_LABELS = {
    0: "Camera & Photography",
    1: "Battery & Charging",
    2: "Performance & Speed",
    3: "Price & Value",
    4: "Display & Design",
    5: "Software & iOS",
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column '{TEXT_COL}'. Run model_training.py first.")

    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.strip().ne("")]
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} comments loaded.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENIZATION FOR GENSIM  (used for coherence scoring)
# ─────────────────────────────────────────────────────────────────────────────

def get_token_lists(df: pd.DataFrame) -> list[list[str]]:

    if TOKENS_COL in df.columns:
        token_lists = [
            [t for t in str(row).split("|") if t and t not in DOMAIN_STOPWORDS]
            for row in df[TOKENS_COL]
        ]
    else:
        token_lists = [
            [t for t in str(row).split() if len(t) > 2 and t not in DOMAIN_STOPWORDS]
            for row in df[TEXT_COL]
        ]
    return token_lists


# ─────────────────────────────────────────────────────────────────────────────
#  COHERENCE SEARCH — find the optimal number of topics
#  Coherence score (c_v) measures how semantically similar the top words
#  within each topic are. Higher = more meaningful topics.
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_topics(token_lists: list[list[str]],
                        min_topics: int = 3,
                        max_topics: int = 12) -> tuple[list, list]:
    print(f"\nSearching for optimal number of topics ({min_topics}–{max_topics})...")
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]

    coherence_scores = []
    k_range = range(min_topics, max_topics + 1)

    for k in tqdm(k_range, desc="  Coherence search"):
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=10,
            random_state=RANDOM_SEED,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=model, texts=token_lists,
            dictionary=dictionary, coherence="c_v",
        )
        coherence_scores.append(cm.get_coherence())
        print(f"    k={k:>2}  coherence={coherence_scores[-1]:.4f}")

    best_k = list(k_range)[np.argmax(coherence_scores)]
    print(f"\n  Best k = {best_k}  (coherence={max(coherence_scores):.4f})")
    print(f"  Using NUM_TOPICS={NUM_TOPICS} (override in config if needed)")

    return list(k_range), coherence_scores


# ─────────────────────────────────────────────────────────────────────────────
#  LDA MODEL  (scikit-learn — integrates with the rest of the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def build_lda(df: pd.DataFrame, n_topics: int) -> tuple:
    """
    Build count vectorizer + LDA model.
    Returns: vectorizer, lda, doc_topic_matrix, feature_names
    """
    print(f"\nFitting LDA with {n_topics} topics...")

    # Build document-term matrix using raw counts (LDA needs counts, not TF-IDF)
    vectorizer = CountVectorizer(
        max_features  = 10_000,
        min_df        = 2,
        max_df        = 0.90,
        ngram_range   = (1, 2),
        stop_words    = list(DOMAIN_STOPWORDS),
        token_pattern = r"\b[a-z][a-z0-9]*\b",
    )
    dtm = vectorizer.fit_transform(df[TEXT_COL])
    print(f"  Document-term matrix: {dtm.shape[0]:,} docs × {dtm.shape[1]:,} terms")

    lda = LatentDirichletAllocation(
        n_components      = n_topics,
        max_iter          = 20,
        learning_method   = "online",      # faster for large corpora
        learning_offset   = 50.0,
        doc_topic_prior   = None,          # auto (1/n_topics)
        topic_word_prior  = None,          # auto (1/n_topics)
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )
    doc_topic = lda.fit_transform(dtm)
    print(f"  Log-likelihood: {lda.score(dtm):.1f}")
    print(f"  Perplexity    : {lda.perplexity(dtm):.1f}  (lower = better fit)")

    return vectorizer, lda, doc_topic, vectorizer.get_feature_names_out()


# ─────────────────────────────────────────────────────────────────────────────
#  TOPIC ASSIGNMENT  — assign the dominant topic to each comment
# ─────────────────────────────────────────────────────────────────────────────

def assign_topics(df: pd.DataFrame,
                  doc_topic: np.ndarray,
                  n_topics: int) -> pd.DataFrame:
    df = df.copy()
    df["dominant_topic"]       = doc_topic.argmax(axis=1)
    df["topic_label"]          = df["dominant_topic"].map(TOPIC_LABELS)
    df["topic_confidence"]     = doc_topic.max(axis=1).round(3)

    # Store full topic distribution per comment
    for i in range(n_topics):
        df[f"topic_{i}_prob"] = doc_topic[:, i].round(4)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TOP WORDS PER TOPIC
# ─────────────────────────────────────────────────────────────────────────────

def get_top_words(lda, feature_names: np.ndarray,
                  n_words: int = 15) -> dict[int, list[tuple]]:
    top_words = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[:-n_words - 1:-1]
        top_words[topic_idx] = [
            (feature_names[i], round(topic[i], 2)) for i in top_idx
        ]
    return top_words


def print_topics(top_words: dict, n_topics: int) -> None:
    print("\n── Topic top words ─────────────────────────────────────────────")
    for i in range(n_topics):
        label = TOPIC_LABELS.get(i, f"Topic {i}")
        words = ", ".join(w for w, _ in top_words[i][:10])
        print(f"  Topic {i} [{label}]")
        print(f"    {words}\n")


def sentiment_by_topic(df: pd.DataFrame, n_topics: int) -> pd.DataFrame:
    if LABEL_COL not in df.columns:
        print("  (No sentiment labels found — skipping cross-tab)")
        return pd.DataFrame()

    cross = pd.crosstab(
        df["dominant_topic"],
        df[LABEL_COL],
        normalize="index",          # row-normalise → proportions per topic
    ).round(3)

    cross.index = [TOPIC_LABELS.get(i, f"Topic {i}") for i in cross.index]

    print("\n── Sentiment breakdown per topic (row %) ───────────────────────")
    print(f"  {'Topic':<28} {'Positive':>9} {'Neutral':>9} {'Negative':>9}")
    print(f"  {'-'*57}")
    for topic, row in cross.iterrows():
        pos = row.get("positive", 0)
        neu = row.get("neutral",  0)
        neg = row.get("negative", 0)
        flag = "  ← high negativity" if neg > 0.5 else ""
        print(f"  {topic:<28} {pos:>8.1%} {neu:>9.1%} {neg:>9.1%}{flag}")

    return cross


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, lda, top_words: dict,
             feature_names, k_range, coherence_scores,
             cross_tab: pd.DataFrame, n_topics: int) -> None:

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("iPhone 16 — LDA Topic Modeling Report",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── 1. Coherence curve ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if k_range and coherence_scores:
        ax1.plot(k_range, coherence_scores, "o-", color="#185FA5", linewidth=2)
        best_k = k_range[np.argmax(coherence_scores)]
        ax1.axvline(best_k, color="#E24B4A", linestyle="--", linewidth=1)
        ax1.axvline(n_topics, color="#1D9E75", linestyle="--", linewidth=1,
                    label=f"Used k={n_topics}")
        ax1.set_xlabel("Number of topics (k)")
        ax1.set_ylabel("Coherence score (c_v)")
        ax1.set_title("Coherence vs number of topics", fontweight="bold")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "Run coherence search\nto see this chart",
                 ha="center", va="center", transform=ax1.transAxes, color="gray")
        ax1.set_title("Coherence curve", fontweight="bold")

    # ── 2. Topic size (# documents per topic) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    topic_counts = df["dominant_topic"].value_counts().sort_index()
    labels_short = [TOPIC_LABELS.get(i, f"T{i}").split("&")[0].strip()
                    for i in topic_counts.index]
    bars = ax2.barh(labels_short, topic_counts.values,
                    color=[TOPIC_PALETTE[i % len(TOPIC_PALETTE)]
                           for i in topic_counts.index],
                    edgecolor="white")
    for bar, val in zip(bars, topic_counts.values):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=9)
    ax2.set_xlabel("Number of comments")
    ax2.set_title("Comments per topic", fontweight="bold")

    # ── 3. Topic confidence distribution ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["topic_confidence"], bins=30, color="#185FA5", edgecolor="white",
             linewidth=0.4)
    ax3.axvline(df["topic_confidence"].mean(), color="#E24B4A",
                linestyle="--", linewidth=1,
                label=f"Mean={df['topic_confidence'].mean():.2f}")
    ax3.set_xlabel("Topic confidence (dominant topic probability)")
    ax3.set_ylabel("Count")
    ax3.set_title("Assignment confidence", fontweight="bold")
    ax3.legend(fontsize=9)

    # ── 4–5. Sentiment × topic stacked bar ───────────────────────────────────
    if not cross_tab.empty:
        ax4 = fig.add_subplot(gs[1, 0:2])
        label_cols = [c for c in ["positive", "neutral", "negative"]
                      if c in cross_tab.columns]
        bar_colors = [COLORS[c] for c in label_cols]
        cross_tab[label_cols].plot(
            kind="barh", stacked=True, ax=ax4,
            color=bar_colors, edgecolor="white", linewidth=0.3,
        )
        ax4.set_xlabel("Proportion of comments")
        ax4.set_title("Sentiment breakdown per topic", fontweight="bold")
        ax4.legend(loc="lower right", fontsize=9)
        ax4.axvline(0.5, color="gray", linestyle="--", linewidth=0.7)
        for label in ax4.get_yticklabels():
            label.set_fontsize(9)

    # ── 6. Topic-sentiment heat map (counts) ─────────────────────────────────
    if LABEL_COL in df.columns:
        ax5 = fig.add_subplot(gs[1, 2])
        count_cross = pd.crosstab(df["topic_label"], df[LABEL_COL])
        col_order   = [c for c in ["positive", "neutral", "negative"]
                       if c in count_cross.columns]
        sns.heatmap(
            count_cross[col_order], annot=True, fmt="d",
            cmap="YlOrBr", linewidths=0.4, ax=ax5, cbar=False,
        )
        ax5.set_xlabel("")
        ax5.set_ylabel("")
        ax5.set_title("Comment counts: topic × sentiment", fontweight="bold")
        for label in ax5.get_yticklabels():
            label.set_fontsize(8)

    # ── 7–9. Word clouds per selected topic ───────────────────────────────────
    topics_to_show = list(range(min(3, n_topics)))
    for subplot_col, topic_idx in enumerate(topics_to_show):
        ax = fig.add_subplot(gs[2, subplot_col])
        word_freq = {w: score for w, score in top_words[topic_idx]}
        wc = WordCloud(
            width=400, height=250,
            background_color="white",
            colormap="Blues",
            max_words=40,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(word_freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        label = TOPIC_LABELS.get(topic_idx, f"Topic {topic_idx}")
        ax.set_title(f"Topic {topic_idx}: {label}", fontweight="bold", fontsize=10)

    plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n  Report saved → {REPORT_PNG}")
    plt.show()




def print_insights(df: pd.DataFrame, cross_tab: pd.DataFrame,
                   top_words: dict, n_topics: int) -> None:
    print("\n" + "=" * 62)
    print("  INSIGHT SUMMARY")
    print("=" * 62)

    total = len(df)

    # Q1: Overall sentiment
    if LABEL_COL in df.columns:
        dist = df[LABEL_COL].value_counts(normalize=True)
        pos  = dist.get("positive", 0)
        neg  = dist.get("negative", 0)
        neu  = dist.get("neutral",  0)
        mood = "mostly positive" if pos > 0.5 else \
               "mostly negative" if neg > 0.5 else "mixed"
        print(f"\n  Q1 — Overall public sentiment: {mood.upper()}")
        print(f"       Positive {pos:.0%}  |  Neutral {neu:.0%}  |  Negative {neg:.0%}")

    # Q2: Key issues and positive drivers
    if not cross_tab.empty:
        print(f"\n  Q2 — Key issues (highest negative %):")
        neg_col = cross_tab.get("negative", pd.Series(dtype=float))
        top_neg = neg_col.sort_values(ascending=False).head(3)
        for topic, val in top_neg.items():
            words = ", ".join(w for w, _ in top_words[
                [k for k, v in TOPIC_LABELS.items() if v == topic][0]
            ][:5]) if topic in TOPIC_LABELS.values() else ""
            print(f"       {topic:<28} {val:.0%} negative   [{words}]")

        print(f"\n       Positive drivers (highest positive %):")
        pos_col = cross_tab.get("positive", pd.Series(dtype=float))
        top_pos = pos_col.sort_values(ascending=False).head(3)
        for topic, val in top_pos.items():
            print(f"       {topic:<28} {val:.0%} positive")

    # Q3 & Q4: Derived from topics
    print(f"\n  Q3 — Product improvement opportunities:")
    print(f"       Fix the highest-negativity topics above — those are your")
    print(f"       users' biggest pain points. Check top words for specifics.")

    print(f"\n  Q4 — Strategic decisions:")
    print(f"       Topics with high positive rates = marketing messaging.")
    print(f"       Topics with high negative rates = R&D + support priority.")
    print(f"       Monitor topic share over time to track sentiment shifts.")

    print("\n" + "=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data(INPUT_CSV)

    # Optional: run coherence search to validate NUM_TOPICS
    run_coherence = True       # set False to skip (saves ~2 min)
    k_range, coherence_scores = [], []

    if run_coherence:
        token_lists = get_token_lists(df)
        k_range, coherence_scores = find_optimal_topics(
            token_lists, min_topics=3, max_topics=10
        )

    # Fit LDA
    vectorizer, lda, doc_topic, feature_names = build_lda(df, NUM_TOPICS)

    # Assign dominant topic to each comment
    df = assign_topics(df, doc_topic, NUM_TOPICS)

    # Top words per topic
    top_words = get_top_words(lda, feature_names, n_words=20)
    print_topics(top_words, NUM_TOPICS)

    # Sentiment × topic breakdown
    cross_tab = sentiment_by_topic(df, NUM_TOPICS)

    # Save outputs
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    joblib.dump({"lda": lda, "vectorizer": vectorizer}, MODEL_PKL)
    print(f"\n  Saved → {OUTPUT_CSV}")
    print(f"  Saved → {MODEL_PKL}")

    # Visualize
    plot_all(df, lda, top_words, feature_names,
             k_range, coherence_scores, cross_tab, NUM_TOPICS)

    # Insights
    print_insights(df, cross_tab, top_words, NUM_TOPICS)

    # Sample: show representative comments per topic
    print("\n── Sample comments per topic ───────────────────────────────────")
    for i in range(NUM_TOPICS):
        label  = TOPIC_LABELS.get(i, f"Topic {i}")
        subset = df[df["dominant_topic"] == i].nlargest(2, "topic_confidence")
        print(f"\n  [{label}]")
        for _, row in subset.iterrows():
            sentiment = f"[{row[LABEL_COL]}]" if LABEL_COL in df.columns else ""
            print(f"    {sentiment} {row['comment_text'][:100]}")


if __name__ == "__main__":
    main()