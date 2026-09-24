import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

INPUT_CSV  = "iphone16_preprocessed.csv"
OUTPUT_CSV = "iphone16_sentiment.csv"

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ─────────────────────────────────────────────────────────────────────────────
#  VADER LABELING
#  Rule-based, fast, no GPU needed. Good baseline.
#  compound score: >= 0.05 → positive, <= -0.05 → negative, else neutral
# ─────────────────────────────────────────────────────────────────────────────

def label_with_vader(texts: pd.Series) -> pd.DataFrame:
    print("\n[1/2] Running VADER...")
    analyzer = SentimentIntensityAnalyzer()

    # VADER works better on raw text than cleaned — we feed original comments
    records = []
    for text in tqdm(texts, desc="  VADER"):
        scores = analyzer.polarity_scores(str(text))
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        records.append({
            "vader_compound":  round(compound, 4),
            "vader_pos":       round(scores["pos"], 4),
            "vader_neu":       round(scores["neu"], 4),
            "vader_neg":       round(scores["neg"], 4),
            "vader_label":     label,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  ROBERTA LABELING
#  Transformer model trained on 124M tweets. Much better at sarcasm,
#  slang, and context. Slower — batched for efficiency.
# ─────────────────────────────────────────────────────────────────────────────

def truncate_for_roberta(text: str, max_tokens: int = 512) -> str:
    """RoBERTa has a 512-token limit. Truncate by word count as a safe proxy."""
    words = str(text).split()
    return " ".join(words[:max_tokens])


def label_with_roberta(texts: pd.Series, batch_size: int = 32) -> pd.DataFrame:
    print("\n[2/2] Running RoBERTa...")
    print(f"  Model : {ROBERTA_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
    model.eval()

    # Label order from this model: 0=negative, 1=neutral, 2=positive
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    import torch
    records  = []
    text_list = [truncate_for_roberta(t) for t in texts]

    for i in tqdm(range(0, len(text_list), batch_size), desc="  RoBERTa batches"):
        batch = text_list[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            output = model(**encoded)

        probs = softmax(output.logits.numpy(), axis=1)

        for prob in probs:
            pred_idx = int(np.argmax(prob))
            records.append({
                "roberta_neg":   round(float(prob[0]), 4),
                "roberta_neu":   round(float(prob[1]), 4),
                "roberta_pos":   round(float(prob[2]), 4),
                "roberta_label": id2label[pred_idx],
                "roberta_confidence": round(float(prob[pred_idx]), 4),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  ENSEMBLE LABEL
#  When VADER and RoBERTa agree → use that label (high confidence)
#  When they disagree → trust RoBERTa (it's more accurate), flag the conflict
# ─────────────────────────────────────────────────────────────────────────────

def make_ensemble_label(row: pd.Series) -> pd.Series:
    v = row["vader_label"]
    r = row["roberta_label"]

    if v == r:
        label      = r
        confidence = "high"
    else:
        label      = r          # RoBERTa wins on disagreement
        confidence = "low"      # flagged for optional manual review

    return pd.Series({"final_label": label, "label_confidence": confidence})


# ─────────────────────────────────────────────────────────────────────────────
#  AGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def print_agreement(df: pd.DataFrame) -> None:
    total     = len(df)
    agree     = (df["vader_label"] == df["roberta_label"]).sum()
    kappa     = cohen_kappa_score(df["vader_label"], df["roberta_label"])

    print("\n── Model agreement ─────────────────────────────────────────────")
    print(f"  Agreement rate : {agree}/{total}  ({agree/total*100:.1f}%)")
    print(f"  Cohen's Kappa  : {kappa:.3f}  "
          f"({'strong' if kappa > 0.6 else 'moderate' if kappa > 0.4 else 'weak'})")

    print("\n  VADER distribution:")
    for label, count in df["vader_label"].value_counts().items():
        print(f"    {label:<10} {count:>5}  ({count/total*100:.1f}%)")

    print("\n  RoBERTa distribution:")
    for label, count in df["roberta_label"].value_counts().items():
        print(f"    {label:<10} {count:>5}  ({count/total*100:.1f}%)")

    print("\n  Final label distribution:")
    for label, count in df["final_label"].value_counts().items():
        print(f"    {label:<10} {count:>5}  ({count/total*100:.1f}%)")

    low_conf = (df["label_confidence"] == "low").sum()
    print(f"\n  Conflicting (low confidence) : {low_conf} comments "
          f"({low_conf/total*100:.1f}%) — consider manual review")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"positive": "#3B8BD4", "neutral": "#888780", "negative": "#E24B4A"}

def plot_all(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("iPhone 16 — Sentiment Labeling Results", fontsize=15, fontweight="bold")

    # ── 1. Final sentiment distribution (donut) ──────────────────────────────
    ax = axes[0, 0]
    counts = df["final_label"].value_counts()
    colors = [COLORS[l] for l in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"width": 0.55},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Overall sentiment (final label)", fontweight="bold")

    # ── 2. VADER vs RoBERTa agreement heatmap ───────────────────────────────
    ax = axes[0, 1]
    order = ["positive", "neutral", "negative"]
    cm = pd.crosstab(
        df["vader_label"], df["roberta_label"],
        rownames=["VADER"], colnames=["RoBERTa"],
    ).reindex(index=order, columns=order, fill_value=0)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, ax=ax, cbar=False,
    )
    ax.set_title("VADER vs RoBERTa agreement", fontweight="bold")

    # ── 3. RoBERTa confidence distribution ───────────────────────────────────
    ax = axes[1, 0]
    for label in order:
        subset = df[df["roberta_label"] == label]["roberta_confidence"]
        ax.hist(subset, bins=20, alpha=0.65, label=label, color=COLORS[label])
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Number of comments")
    ax.set_title("RoBERTa confidence by label", fontweight="bold")
    ax.legend()

    # ── 4. VADER compound score distribution ─────────────────────────────────
    ax = axes[1, 1]
    ax.axvspan(-1.0, -0.05, alpha=0.08, color=COLORS["negative"])
    ax.axvspan(-0.05, 0.05, alpha=0.08, color=COLORS["neutral"])
    ax.axvspan(0.05, 1.0,  alpha=0.08, color=COLORS["positive"])
    ax.hist(df["vader_compound"], bins=40, color="#185FA5", edgecolor="white", linewidth=0.3)
    ax.axvline(-0.05, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline( 0.05, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Number of comments")
    ax.set_title("VADER score distribution", fontweight="bold")
    ax.text(-0.8, ax.get_ylim()[1]*0.9, "negative", color=COLORS["negative"], fontsize=9)
    ax.text(-0.02, ax.get_ylim()[1]*0.9, "neu", color="gray", fontsize=9)
    ax.text(0.4, ax.get_ylim()[1]*0.9, "positive", color=COLORS["positive"], fontsize=9)

    plt.tight_layout()
    plt.savefig("sentiment_labeling_report.png", dpi=150, bbox_inches="tight")
    print("\n  Chart saved → sentiment_labeling_report.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} comments loaded.")

    # VADER runs on raw text (better results than cleaned text for rule-based)
    vader_df = label_with_vader(df["comment_text"])

    # RoBERTa runs on clean_text (noise already removed)
    roberta_df = label_with_roberta(df["clean_text"])

    # Merge all columns back
    df = pd.concat([df, vader_df, roberta_df], axis=1)

    # Ensemble
    ensemble = df.apply(make_ensemble_label, axis=1)
    df = pd.concat([df, ensemble], axis=1)

    # Agreement analysis
    print_agreement(df)

    # Save
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Columns added: vader_label, roberta_label, final_label, label_confidence")

    # Visualize
    plot_all(df)

    # ── Sample: show a few from each category ────────────────────────────────
    print("\n── Sample labeled comments ─────────────────────────────────────")
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["final_label"] == label].head(2)
        print(f"\n  [{label.upper()}]")
        for _, row in subset.iterrows():
            print(f"    {row['comment_text'][:100]}")
            print(f"    VADER={row['vader_label']} ({row['vader_compound']:+.2f})  "
                  f"RoBERTa={row['roberta_label']} ({row['roberta_confidence']:.2f})  "
                  f"→ {row['final_label']} [{row['label_confidence']} conf]")


if __name__ == "__main__":
    main()