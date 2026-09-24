import re
import string
import pandas as pd
import nltk
import spacy
import contractions
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  

INPUT_CSV  = "iphone16_filtered.csv"
OUTPUT_CSV = "iphone16_preprocessed.csv"

PRESERVE_TERMS = {
    "iphone", "iphone16", "iphone 16", "ios", "ios18", "ios 18",
    "a18", "a18 pro", "magsafe", "faceid", "face id", "touchid",
    "dynamic island", "action button", "capture button",
    "pro max", "ultra wide", "ultrawide",
}

SENTIMENT_PRESERVING = {
    "not", "no", "never", "without", "cant", "cannot", "wont",
    "worst", "best", "love", "hate", "amazing", "terrible",
    "overheating", "laggy", "smooth", "fast", "slow", "worth",
    "expensive", "cheap", "upgrade", "downgrade",
}

_base_stopwords = set(stopwords.words("english"))
STOPWORDS = _base_stopwords - SENTIMENT_PRESERVING



def step1_expand_contractions(text: str) -> str:
    """
    "don't" → "do not", "it's" → "it is"
    Crucial before stopword removal so "don't" isn't mangled into "dont"
    and "not" is preserved.
    """
    return contractions.fix(text)


def step2_convert_emoji(text: str) -> str:
    """
    Convert emoji to text description so sentiment is not lost.
    😍 → "smiling face with heart eyes"
    🔥 → "fire"
    """
    return emoji.demojize(text, delimiters=(" ", " ")).strip()


def step3_lowercase(text: str) -> str:
    return text.lower()


def step4_remove_noise(text: str) -> str:
    """
    Remove URLs, @mentions, #hashtags, HTML entities, and
    non-alphabetic characters (but keep spaces).
    Preserve numbers that are part of product names (handled by preserve step).
    """
    text = re.sub(r"https?://\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                         # @mentions
    text = re.sub(r"#\w+", "", text)                         # #hashtags
    text = re.sub(r"&[a-z]+;", "", text)                     # HTML entities (&amp;)
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # punctuation / special chars
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def step5_tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def step6_remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords but keep sentiment-critical negations and tech terms."""
    return [t for t in tokens if t not in STOPWORDS or t in SENTIMENT_PRESERVING]


def step7_lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduce words to base form using spaCy.
    "cameras" → "camera", "heating" → "heat", "bought" → "buy"
    Skips tokens that are iPhone-specific terms to avoid mangling them.
    """
    text = " ".join(tokens)
    doc  = nlp(text)
    return [
        token.lemma_ if token.text not in PRESERVE_TERMS else token.text
        for token in doc
        if token.text.strip()
    ]


def step8_remove_short_tokens(tokens: list[str], min_len: int = 2) -> list[str]:
    """Drop single-character tokens (noise after cleaning)."""
    return [t for t in tokens if len(t) >= min_len]


def preprocess(text: str) -> tuple[str, str]:
    """
    Run all steps and return:
      - clean_text  : space-joined string  (for transformer models / display)
      - tokens_str  : pipe-joined tokens   (for TF-IDF / bag-of-words models)
    """
    if not isinstance(text, str) or not text.strip():
        return "", ""

    text = step1_expand_contractions(text)
    text = step2_convert_emoji(text)
    text = step3_lowercase(text)
    text = step4_remove_noise(text)

    tokens = step5_tokenize(text)
    tokens = step6_remove_stopwords(tokens)
    tokens = step7_lemmatize(tokens)
    tokens = step8_remove_short_tokens(tokens)

    clean_text = " ".join(tokens)
    tokens_str = "|".join(tokens)        # pipe-separated for easy splitting later

    return clean_text, tokens_str


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} comments to process.\n")

    print("Preprocessing...")
    results = df["comment_text"].apply(preprocess)

    df["clean_text"]  = results.apply(lambda r: r[0])
    df["tokens"]      = results.apply(lambda r: r[1])
    df["token_count"] = df["tokens"].apply(lambda t: len(t.split("|")) if t else 0)

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["clean_text"].str.strip().ne("")]
    df = df[df["token_count"] >= 3]
    df.reset_index(drop=True, inplace=True)
    print(f"  Dropped {before - len(df)} empty/too-short after cleaning.")
    print(f"  {len(df):,} comments ready.\n")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved → {OUTPUT_CSV}")

    # ── Quality check ────────────────────────────────────────────────────────
    print("\n── Sample transformations ──────────────────────────────────────")
    sample = df[["comment_text", "clean_text"]].head(5)
    for _, row in sample.iterrows():
        print(f"  RAW  : {row['comment_text'][:90]}")
        print(f"  CLEAN: {row['clean_text'][:90]}")
        print()

    print("── Token stats ─────────────────────────────────────────────────")
    print(f"  Avg tokens per comment : {df['token_count'].mean():.1f}")
    print(f"  Min tokens             : {df['token_count'].min()}")
    print(f"  Max tokens             : {df['token_count'].max()}")

    # Top 20 most frequent tokens across all comments
    from collections import Counter
    all_tokens = [t for row in df["tokens"] for t in row.split("|") if t]
    top20 = Counter(all_tokens).most_common(20)
    print("\n── Top 20 tokens ───────────────────────────────────────────────")
    for token, count in top20:
        bar = "█" * (count * 30 // top20[0][1])
        print(f"  {token:<20} {count:>5}  {bar}")


if __name__ == "__main__":
    main()