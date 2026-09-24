import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Reddit (Pushshift)
from pmaw import PushshiftAPI

# Twitter/X
import snscrape.modules.twitter as sntwitter


# =========================
# CONFIG
# =========================

TARGET_SAMPLES = 1000

REDDIT_QUERIES = [
    "zomato delivery",
    "zomato late",
    "zomato refund",
    "zomato experience",
]

TWITTER_QUERY = "zomato lang:en"


# =========================
# REDDIT COLLECTION
# =========================

def collect_reddit(max_samples=500):
    print("\n🔴 Collecting Reddit data (Pushshift)...")

    api = PushshiftAPI()
    data = []
    seen = set()

    # -------- POSTS --------
    for query in REDDIT_QUERIES:
        submissions = api.search_submissions(q=query, limit=200)

        for post in submissions:
            if len(data) >= max_samples:
                break

            text = (post.title + " " + str(post.selftext)).strip()

            if text and text not in seen:
                data.append({
                    "text": text,
                    "source": "reddit_post",
                    "created_at": datetime.fromtimestamp(post.created_utc)
                })
                seen.add(text)

    # -------- COMMENTS --------
    for query in REDDIT_QUERIES:
        comments = api.search_comments(q=query, limit=300)

        for comment in comments:
            if len(data) >= max_samples:
                break

            text = comment.body.strip()

            if text and len(text) > 20 and text not in seen:
                data.append({
                    "text": text,
                    "source": "reddit_comment",
                    "created_at": datetime.fromtimestamp(comment.created_utc)
                })
                seen.add(text)

    print(f"✅ Reddit samples: {len(data)}")
    return data


# =========================
# TWITTER COLLECTION
# =========================

def collect_twitter(max_samples=500):
    print("\n🐦 Collecting tweets using snscrape...")

    data = []
    seen = set()

    try:
        for tweet in sntwitter.TwitterSearchScraper(TWITTER_QUERY).get_items():
            if len(data) >= max_samples:
                break

            text = tweet.content.strip()

            if text and text not in seen:
                data.append({
                    "text": text,
                    "source": "twitter",
                    "created_at": tweet.date
                })
                seen.add(text)

    except Exception as e:
        print("⚠️ Twitter scraping failed:", e)

    print(f"✅ Twitter samples: {len(data)}")
    return data


# =========================
# MAIN
# =========================

def main():
    print("=" * 60)
    print("🚀 Zomato Sentiment Data Collection")
    print("=" * 60)

    reddit_data = collect_reddit(max_samples=500)
    twitter_data = collect_twitter(max_samples=500)

    combined = reddit_data + twitter_data
    df = pd.DataFrame(combined)

    if df.empty:
        print("❌ No data collected")
        return

    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1).reset_index(drop=True)

    print(f"\n📊 Total samples: {len(df)}")

    df.to_csv("zomato_combined_data.csv", index=False)
    print("💾 Saved to zomato_combined_data.csv")


if __name__ == "__main__":
    main()