import os
import time
import csv
import re
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

API_KEY       = "AIzaSyAzPEeSMdD9N9wqSv6dNA0kU2E-HdTjvbk" 
SEARCH_QUERY  = "iPhone 16 detailed review english"    
MAX_VIDEOS    = 30                   
TARGET_TOTAL  = 3000                  
OUTPUT_CSV    = "youtube_comments.csv"
OUTPUT_EXCEL  = "youtube_comments.xlsx"
# ─────────────────────────────────────────────

QUOTA_COST = {
    "search":           100,   
    "video_details":    1,     
    "comment_threads":  1,     
}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def search_videos(youtube, query: str, max_results: int) -> list[dict]:
    print(f"\n[1/3] Searching YouTube for: '{query}'")
    results = []
    page_token = None
    fetched = 0

    while fetched < max_results:
        batch = min(50, max_results - fetched)
        try:
            resp = youtube.search().list(
                q=query,
                part="id,snippet",
                type="video",
                maxResults=batch,
                pageToken=page_token,
                relevanceLanguage="en",
                order="relevance",
            ).execute()
        except HttpError as e:
            print(f"  [!] Search API error: {e}")
            break

        for item in resp.get("items", []):
            vid_id = item["id"].get("videoId")
            if not vid_id:
                continue
            results.append({
                "video_id":    vid_id,
                "video_title": item["snippet"]["title"],
                "channel":     item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
            })

        fetched += len(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print(f"  Found {len(results)} videos.")
    return results


def get_video_stats(youtube, video_ids: list[str]) -> dict:
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            resp = youtube.videos().list(
                part="statistics",
                id=",".join(batch),
            ).execute()
            for item in resp.get("items", []):
                stats[item["id"]] = item.get("statistics", {})
        except HttpError:
            pass
    return stats


def fetch_comments(youtube, video_id: str, max_comments: int = 200) -> list[dict]:

    comments = []
    page_token = None

    while len(comments) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=page_token,
                textFormat="plainText",
                order="relevance",          
            ).execute()
        except HttpError as e:
            if "commentsDisabled" in str(e) or e.resp.status == 403:
                print(f"    [!] Comments disabled — skipping.")
            else:
                print(f"    [!] HTTP {e.resp.status} — skipping.")
            break

        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_text":     clean_text(top["textDisplay"]),
                "author":           top.get("authorDisplayName", ""),
                "like_count":       top.get("likeCount", 0),
                "reply_count":      item["snippet"].get("totalReplyCount", 0),
                "published_at":     top.get("publishedAt", ""),
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)       

    return comments


def collect_dataset(api_key: str, query: str,
                    max_videos: int, target_total: int) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)

    videos = search_videos(youtube, query, max_results=max_videos)
    if not videos:
        raise RuntimeError("No videos found. Check your query or API key.")

    print("\n[2/3] Fetching video statistics...")
    vid_ids = [v["video_id"] for v in videos]
    stats   = get_video_stats(youtube, vid_ids)

    def comment_count(v):
        return int(stats.get(v["video_id"], {}).get("commentCount", 0))
    videos.sort(key=comment_count, reverse=True)

    for v in videos[:5]:
        cnt = comment_count(v)
        print(f"  [{cnt:>6} comments] {v['video_title'][:70]}")

    print(f"\n[3/3] Collecting up to {target_total} comments...")
    all_rows   = []
    per_video  = max(100, target_total // max(1, len(videos)))

    for i, video in enumerate(videos):
        remaining = target_total - len(all_rows)
        if remaining <= 0:
            break

        batch_limit = min(per_video, remaining, 500)
        print(f"  Video {i+1}/{len(videos)}: {video['video_title'][:55]}...")
        comments = fetch_comments(youtube, video["video_id"], max_comments=batch_limit)
        print(f"    → {len(comments)} comments collected")

        for c in comments:
            all_rows.append({
                "video_id":       video["video_id"],
                "video_title":    video["video_title"],
                "channel":        video["channel"],
                "video_published": video["published_at"],
                "comment_text":   c["comment_text"],
                "author":         c["author"],
                "like_count":     c["like_count"],
                "reply_count":    c["reply_count"],
                "comment_date":   c["published_at"],
                "topic":          query,
                "collected_at":   datetime.utcnow().isoformat(timespec="seconds"),
            })

        time.sleep(0.5)

    df = pd.DataFrame(all_rows)

    # Basic cleanup
    df.drop_duplicates(subset=["comment_text"], inplace=True)
    df = df[df["comment_text"].str.len() > 5]   # drop near-empty comments
    df.reset_index(drop=True, inplace=True)

    return df


def save_outputs(df: pd.DataFrame, csv_path: str, xlsx_path: str) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")   # utf-8-sig for Excel compat
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"\n  Saved {len(df):,} rows → {csv_path}")
    print(f"  Saved {len(df):,} rows → {xlsx_path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total comments collected : {len(df):,}")
    print(f"  Unique videos            : {df['video_id'].nunique()}")
    print(f"  Avg comment length       : {df['comment_text'].str.len().mean():.0f} chars")
    print(f"  Most liked comment (top) : {df['like_count'].max()} likes")
    print(f"\n  Columns: {', '.join(df.columns.tolist())}")
    print("\n  Sample comments:")
    for _, row in df.head(3).iterrows():
        print(f"    • {row['comment_text'][:90]}")
    print("=" * 55)



if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your API_KEY at the top of the script.")
        exit(1)

    df = collect_dataset(
        api_key=API_KEY,
        query=SEARCH_QUERY,
        max_videos=MAX_VIDEOS,
        target_total=TARGET_TOTAL,
    )

    save_outputs(df, OUTPUT_CSV, OUTPUT_EXCEL)
    print_summary(df)