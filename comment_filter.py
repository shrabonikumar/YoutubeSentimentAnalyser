"""
iPhone 16 Comment Relevance Filter
====================================
Run this AFTER youtube_data_collector.py has saved your CSV.

Usage:
    python filter_iphone16_comments.py

Input  : youtube_comments.csv  (raw collected data)
Output : iphone16_filtered.csv (relevant comments only)
         iphone16_filtered.xlsx
"""

import re
import pandas as pd

INPUT_CSV   = "youtube_comments.csv"
OUTPUT_CSV  = "iphone16_filtered.csv"
OUTPUT_XLSX = "iphone16_filtered.xlsx"

# ─────────────────────────────────────────────────────────────
#  KEYWORD SETS  (extend these freely)
# ─────────────────────────────────────────────────────────────

# Must match at least ONE of these to even be considered
CORE_KEYWORDS = [
    "iphone 16", "iphone16", "ip16",
    "apple", "ios 18", "ios18",
    "a18", "a18 pro",                       # chip names
]

# Topic-specific feature / issue keywords — comments mentioning these
# are almost certainly about the phone itself
FEATURE_KEYWORDS = [
    # hardware
    "camera", "ultra wide", "ultrawide", "zoom", "portrait",
    "battery", "charging", "magsafe", "wireless charge",
    "display", "screen", "oled", "brightness", "refresh rate",
    "speaker", "microphone", "haptic",
    "action button", "capture button", "camera control",
    "titanium", "aluminium", "aluminum", "glass back",
    "face id", "touch id",
    "sim", "esim", "5g",
    "overheating", "heating", "warm", "hot",
    "performance", "speed", "benchmark", "geekbench",
    "storage", "128gb", "256gb", "512gb", "1tb",
    # software / ecosystem
    "update", "ios", "app", "feature", "bug", "crash",
    "siri", "apple intelligence", "ai",
    "dynamic island",
    # purchase / value
    "price", "cost", "expensive", "worth", "value", "deal",
    "upgrade", "switch", "switched",
    "buy", "bought", "purchased", "order", "ordered",
    "pre-order", "preorder",
    # comparison
    "vs", "versus", "compared", "better than", "worse than",
    "samsung", "pixel", "android",
    "pro", "pro max", "plus", "base model",
    "iphone 15", "iphone 14", "iphone 13",
]

# If a comment ONLY matches these without any feature keyword,
# it is likely off-topic (e.g. "apple is a great company" with no phone context)
WEAK_ONLY_TERMS = {"apple", "ios", "app"}

# Hard exclusion — any comment containing these is almost certainly off-topic
EXCLUDE_KEYWORDS = [
    "apple pie", "apple juice", "apple cider", "apple tree",
    "apple tv show", "ted lasso",                       # Apple TV+ content
    "macbook", "mac pro", "imac", "mac mini",           # other Apple products
    "ipad", "apple watch", "airpods", "homepod",        # unless combined with iphone
    "stock", "aapl", "share price", "earnings", "revenue",  # finance
    "minecraft", "fortnite", "roblox",                  # gaming off-topic
]

# ─────────────────────────────────────────────────────────────
#  SCORER
# ─────────────────────────────────────────────────────────────

def make_pattern(terms: list[str]) -> re.Pattern:
    escaped = sorted([re.escape(t) for t in terms], key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

CORE_PAT    = make_pattern(CORE_KEYWORDS)
FEATURE_PAT = make_pattern(FEATURE_KEYWORDS)
EXCLUDE_PAT = make_pattern(EXCLUDE_KEYWORDS)
WEAK_PAT    = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in WEAK_ONLY_TERMS) + r")\b",
    re.IGNORECASE,
)


def score_comment(text: str) -> tuple[bool, str, int]:
    """
    Returns (is_relevant, reason, score).
    score  = rough relevance signal (higher = more on-topic)
    reason = short label for transparency
    """
    t = text.lower()

    # Hard exclude first
    if EXCLUDE_PAT.search(t):
        excl = EXCLUDE_PAT.search(t).group()
        # Allow if it also contains an iPhone 16 core keyword
        # e.g. "comparing iphone 16 to airpods max audio" — still relevant
        if not CORE_PAT.search(t):
            return False, f"excluded:{excl}", 0

    core_hits    = CORE_PAT.findall(t)
    feature_hits = FEATURE_PAT.findall(t)

    # Must have at least one core or feature keyword
    if not core_hits and not feature_hits:
        return False, "no_keyword_match", 0

    # Weak-only: has "apple"/"ios"/"app" but no specific iPhone feature
    all_matches = set(m.lower() for m in core_hits + feature_hits)
    if all_matches and all_matches.issubset(WEAK_ONLY_TERMS) and not feature_hits:
        return False, "weak_match_only", 0

    # Score: weighted sum
    score = (len(core_hits) * 3) + (len(feature_hits) * 1)

    # Bonus for direct iPhone 16 mention
    if re.search(r"\biphone\s*16\b", t, re.IGNORECASE):
        score += 5

    # Bonus for question/opinion markers — good sentiment signal
    if re.search(r"\?|should i|is it worth|do you think|i think|i feel|"
                 r"i love|i hate|best|worst", t, re.IGNORECASE):
        score += 2

    return True, "relevant", score


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def filter_comments(input_csv: str) -> pd.DataFrame:
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} comments loaded.\n")

    results = df["comment_text"].apply(score_comment)
    df["relevant"]  = results.apply(lambda r: r[0])
    df["filter_reason"] = results.apply(lambda r: r[1])
    df["relevance_score"] = results.apply(lambda r: r[2])

    # ── Breakdown of what was removed ──
    removed = df[~df["relevant"]]
    print("Removed comments by reason:")
    for reason, count in removed["filter_reason"].value_counts().items():
        print(f"  {reason:<30} {count:>5}")

    # ── Keep only relevant, sort by score ──
    filtered = (
        df[df["relevant"]]
        .drop(columns=["relevant", "filter_reason"])
        .sort_values("relevance_score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nResult : {len(df):,} → {len(filtered):,} relevant comments kept "
          f"({len(df) - len(filtered):,} removed, "
          f"{len(filtered)/len(df)*100:.1f}% retention)")

    return filtered


def save(df: pd.DataFrame, csv_path: str, xlsx_path: str) -> None:
    df.to_csv(csv_path,  index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {xlsx_path}")


def print_samples(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n── Top {n} most relevant comments ──")
    for _, row in df.head(n).iterrows():
        print(f"  [score {row['relevance_score']}] {row['comment_text'][:100]}")

    print(f"\n── Bottom {n} (lowest scoring but still kept) ──")
    for _, row in df.tail(n).iterrows():
        print(f"  [score {row['relevance_score']}] {row['comment_text'][:100]}")


if __name__ == "__main__":
    filtered_df = filter_comments(INPUT_CSV)
    save(filtered_df, OUTPUT_CSV, OUTPUT_XLSX)
    print_samples(filtered_df)
    print("\nNext: run your sentiment analysis on iphone16_filtered.csv → 'comment_text' column.")